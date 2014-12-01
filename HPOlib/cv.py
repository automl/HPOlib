##
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
from logging.handlers import SocketHandler, DEFAULT_TCP_LOGGING_PORT
import os
import subprocess
import sys
import time
import warnings

import numpy as np

from HPOlib.Experiment import Experiment
from HPOlib.wrapping_util import format_traceback, load_experiment_config_file


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__license__ = "3-clause BSD License"
__contact__ = "automl.org"

hpolib_logger = logging.getLogger("HPOlib")
logger = logging.getLogger("HPOlib.cv")


# TODO: This should be in a util function sometime in the future
#       Is duplicated in runsolver_wrapper.py
def get_optimizer():
    return "_".join(os.getcwd().split("/")[-1].split("_")[0:-2])


def load_experiment_file():
    optimizer = get_optimizer()
    experiment = Experiment(".", optimizer)
    return experiment


def do_cv(params, folds=10):
    logger.info("Starting Cross validation")
    sys.stdout.flush()
    optimizer = get_optimizer()
    cfg = load_experiment_config_file()

    # Store the results to hand them back to tpe and spearmint
    results = []

    try:
        param_array = ["-" + str(param_name) + " " + str(params[param_name]) for param_name in params]
        param_string = " ".join(param_array)
        
        for fold in range(folds):
            # "Usage: runsolver_wrapper <instancename> " + \
            # "<instancespecificinformation> <cutofftime> <cutofflength> " + \
            # "<seed> <param> <param> <param>"
            # Cutofftime, cutofflength and seed can be safely ignored since they
            # are read in runsolver_wrapper
            runsolver_wrapper_script = "python " + \
                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "dispatcher/dispatcher.py")
            cmd = "%s %d %s %d %d %d %s" % \
                (runsolver_wrapper_script, fold, optimizer, 0, 0, 0, param_string)
            logger.info("Calling command:\n%s", cmd)

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, shell=True, executable="/bin/bash")
            logger.info("-------------- DISPATCHING JOB --------------")
            stdoutdata, stderrdata = process.communicate()
            if stdoutdata:
                logger.info(stdoutdata)
            if stderrdata:
                logger.error(stderrdata)

            # Read the runsolver_wrapper output
            lines = stdoutdata.split("\n")
            result_string = None
            for line in lines:
                pos = line.find("Result for ParamILS: SAT")
                if pos != -1:
                    result_string = line[pos:]
                    result_array = result_string.split()
                    results.append(float(result_array[6].strip(",")))
                    break

            if result_string is None:
                raise NotImplementedError("No result string available or result string doesn't contain SAT")

            # If a specified number of runs crashed, quit the whole cross validation
            # in order to save time.
            worst_possible = cfg.getfloat("HPOLIB", "result_on_terminate")
            # So far, this was a nansum, but there must be no NaNs at this
            # point of the program flow!
            assert np.isfinite(results).all()
            crashed_runs = np.sum([0 if res != worst_possible else 1 for res in results])
            if crashed_runs >= cfg.getint("HPOLIB", "max_crash_per_cv"):
                logger.warning("Aborting CV because the number of crashes "
                               "exceeds the configured max_crash_per_cv value")
                return worst_possible

            # TODO: Error Handling
        
        assert(len(results) == folds)
        mean = np.mean(results)

    except Exception as e:
        logger.error(format_traceback(sys.exc_info()))
        logger.error("CV failed %s %s", sys.exc_info()[0], e)
        # status = "CRASHED"
        # status = "SAT"
        mean = np.NaN
        
    # Do not return any kind of nan because this would break spearmint
    with warnings.catch_warnings():
        if not np.isfinite(mean):
            mean = float(cfg.get("HPOLIB", "result_on_terminate"))

    logger.info("Finished CV")
    return mean


def flatten_parameter_dict(params):
    """
    TODO: Generalize this, every optimizer should do this by itself
    """
    # Flat nested dicts and shorten lists
    # FORCES SMAC TO FOLLOW SOME CONVENTIONS, e.g.
    # lr_penalty': hp.choice('lr_penalty', [{
    #    'lr_penalty' : 'zero'}, {  | Former this line was 'type' : 'zero'
    #    'lr_penalty' : 'notZero',
    #    'l2_penalty_nz': hp.lognormal('l2_penalty_nz', np.log(1.0e-6), 3.)}])
    # Lists cannot be forwarded via the command line, therefore the list has to
    # be unpacked. ONLY THE FIRST VALUE IS FORWARDED!

    # This class enables us to distinguish tuples, which are parameters from
    # tuples which contain different things...
    class Parameter:
        def __init__(self, pparam):
            self.pparam = pparam

    params_to_check = list()
    params_to_check.append(params)

    new_dict = dict()
    while len(params_to_check) != 0:
        param = params_to_check.pop()

        if type(param) in (list, tuple, np.ndarray):

            for sub_param in param:
                params_to_check.append(sub_param)

        elif isinstance(param, dict):
            params_to_check.extend([Parameter(tmp_param) for tmp_param in zip(param.keys(), param.values())])

        elif isinstance(param, Parameter):
            key = param.pparam[0]
            value = param.pparam[1]
            if type(value) == dict:
                params_to_check.append(value)
            elif type(value) in (list, tuple, np.ndarray) and \
                    all([type(v) not in (list, tuple, np.ndarray) for v in value]):
                # Spearmint special case, keep only the first element
                # Adding: variable_id = val
                if len(value) == 1:
                    new_dict[key] = value[0]
                else:
                    for v_idx, v in enumerate(value):
                        new_dict[key + "_%s" % v_idx] = v
                #new_dict[key] = value[0]
            elif type(value) in (list, tuple, np.ndarray):
                for v in value:
                    params_to_check.append(v)
            else:
                new_dict[key] = value

        else:
            raise Exception("Invalid params, cannot be flattened: \n%s." % params)
    params = new_dict
    return params


def main(*args, **kwargs):
    hpolib_logger.setLevel(logging.INFO)
    host = 'localhost'
    port = DEFAULT_TCP_LOGGING_PORT
    socketh = SocketHandler(host, port)
    logger.addHandler(socketh)

    logger.critical('args: %s kwargs: %s', str(args), str(kwargs))

    params = None

    if 'params' in kwargs.keys():
        params = kwargs['params']
    else:
        for arg in args:
            if type(arg) == dict:
                params = arg
                break

    if params is None:
        logger.critical("No parameter dict found in cv.py.\n"
                        "args: %s\n kwargs: %s", args, kwargs)
        # TODO: Hack for TPE and AUTOWeka
        params = args

    # Load the experiment to do time-keeping
    cv_starttime = time.time()
    experiment = load_experiment_file()
    # experiment.next_iteration()
    experiment.start_cv(cv_starttime)
    experiment._save_jobs()
    del experiment

    # cfg_filename = "config.cfg"
    cfg = load_experiment_config_file()

    # Load number of folds
    folds = cfg.getint('HPOLIB', 'number_cv_folds')

    params = flatten_parameter_dict(params)

    res = do_cv(params, folds=folds)
    logger.info("Result: %f", res)
    
    # Load the experiment to do time-keeping
    experiment = load_experiment_file()
    experiment.end_cv(time.time())
    experiment._save_jobs()
    del experiment
    
    return res


def read_params_from_command_line():
    params = dict()
    param_list = sys.argv[1:]
    for idx, i in enumerate(param_list[0::2]):
        key = param_list[idx*2]
        value = param_list[idx*2+1]

        if key[0] != "-":
            raise ValueError("Expected a parameter name that start with '-' at "
                             "%d position, instead got %s" % (idx*2, key))
        else:
            key = key[1:]

        if value[0] != "'" or value[-1] != "'":
            raise ValueError("Expected parameter value %s to be inside single"
                             "quotation marks." % value)
        else:
            value = value.strip("'")
        params[key] = value
    return params


if __name__ == "__main__":
    cli_params = read_params_from_command_line()
    main(params=cli_params)
