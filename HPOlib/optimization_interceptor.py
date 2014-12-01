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

from argparse import ArgumentParser
from collections import OrderedDict
import logging
import os
import sys
import time
import warnings

import numpy as np

from HPOlib.dispatcher import dispatcher
from HPOlib.Experiment import Experiment
from HPOlib.wrapping_util import format_traceback, \
    load_experiment_config_file, remove_param_metadata


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__license__ = "3-clause BSD License"
__contact__ = "automl.org"

logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
hpolib_logger = logging.getLogger("HPOlib")
hpolib_logger.setLevel(logging.INFO)
logger = logging.getLogger("HPOlib.optimization_interceptor")


# TODO: This should be in a util function sometime in the future
#       Is duplicated in runsolver_wrapper.py
def get_optimizer():
    return "_".join(os.getcwd().split("/")[-1].split("_")[0:-2])


def load_experiment_file():
    optimizer = get_optimizer()
    experiment = Experiment(".", optimizer)
    return experiment


def do_cv(arguments, parameters, folds=10):
    logger.info("Starting Cross validation")
    logger.info("Parameters: %s", str(parameters))
    sys.stdout.flush()
    cfg = load_experiment_config_file()

    # Store the results to hand them back to tpe and spearmint
    results = []

    try:
        for fold in range(folds):
            logger.info("Starting fold %d" % fold)
            status, wallclock_time, result, additional_data = \
                dispatcher.main(arguments, parameters, fold)
            results.append(result)
            logger.info("Finished fold %d, result: %f, duration: %f" %
                        (fold, result, wallclock_time))

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


def run_one_instance(arguments, parameters):
    """Execute one instance."""
    if arguments.instance is not None:
        instance = int(arguments.instance)
    else:
        instance = 0
    logger.info("Starting instance evaluation for: %s" % str(instance))
    logger.info("Parameters: %s", str(parameters))
    cfg = load_experiment_config_file()

    try:
        status, wallclock_time, result, additional_data = \
            dispatcher.main(arguments, parameters, instance)

        # TODO: Error Handling

    except Exception as e:
        logger.error(format_traceback(sys.exc_info()))
        logger.error("Instance evaluation failed: %s %s", sys.exc_info()[0], e)
        result = np.NaN
        status = "CRASHED"
        wallclock_time = np.NaN
        additional_data = str(e)

    # Do not return any kind of nan because this would break spearmint
    with warnings.catch_warnings():
        if not np.isfinite(result):
            result = float(cfg.get("HPOLIB", "result_on_terminate"))

    logger.info("Finished instance Evaluation: %s; result: %f, duration: %f" %
                (str(instance), result, wallclock_time))
    return result


def parse_params(params_list):
    """Parse a list of parameters which was given on the command line.

    Parameters
    ----------
        params : list
            A list of string. Values at a position with an even index are
            considered to be the parameter name (starting with a '-'),
            followed by its value.

    Returns
    -------
        dict
            A dictionary where the keys are the parameter names and the
            values their value.
    """
    # TODO more restrictive parsing by using HPOlibConfigSpace to check if we
    #  got a valid configuration!
    params_dict = OrderedDict()
    for idx, i in enumerate(params_list[0::2]):
        key = params_list[idx * 2]
        value = params_list[idx * 2 + 1]
        key = key.strip("'").strip('"')
        value = value.strip("'").strip('"')

        if key[0] != "-":
            raise ValueError("Expected a parameter name that start with '-' at "
                             "%d position, instead got %s" % (idx * 2, key))
        else:
            key = key[1:]

        params_dict[key] = value

    remove_param_metadata(params_dict)
    return params_dict


def parse_cli():
    """Parse the arguments of the optimization interceptor."""
    # First, check if --params or --parameters is on the command line,
    # treat everything which is before that as arguments to the optimization
    # interceptor, everything behind as parameters for the target algorithm
    arguments = sys.argv[1:]
    parameters = None
    for idx, value in enumerate(arguments):
        if value == '--params':
            parameters = arguments[idx + 1:]
            arguments = arguments[:idx+1]
            break

    parser = ArgumentParser(description="Internal script of the HPOlib to be "
                                        "called by an optimization algorithm.")
    parser.add_argument('--instance', '--fold', default=None,
                        dest='instance', help='Only use if cross validation '
                                              'is handled by the optimization '
                                              'algorithm. Tell HPOlib that it '
                                              'should not perform cross '
                                              'validation.')
    parser.add_argument('--params', action='store_true', required=True)
    arguments = parser.parse_args(args=arguments)
    params = parse_params(parameters)
    return arguments, params


def main(arguments, parameters):
    """
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
    """

    # Load the experiment to do time-keeping
    cv_starttime = time.time()
    experiment = load_experiment_file()
    # experiment.next_iteration()
    experiment.start_cv(cv_starttime)
    experiment._save_jobs()
    del experiment

    # cfg_filename = "config.cfg"
    cfg = load_experiment_config_file()
    folds = cfg.getint('HPOLIB', 'number_cv_folds')

    """
    params = flatten_parameter_dict(params)
    """

    if arguments.instance is None and folds > 1:
        result = do_cv(arguments, parameters, folds=folds)
    else:
        result = run_one_instance(arguments, parameters)

    # Load the experiment to do time-keeping
    experiment = load_experiment_file()
    experiment.end_cv(time.time())
    experiment._save_jobs()
    del experiment

    sys.stdout.write("Result: %f" % float(result))
    sys.stdout.flush()
    return result


if __name__ == "__main__":
    arguments, parameters = parse_cli()
    main(arguments, parameters)
