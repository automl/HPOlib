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
from logging.handlers import SocketHandler
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

hpolib_logger = logging.getLogger("HPOlib")
logger = logging.getLogger("HPOlib.optimization_interceptor")


# TODO: This should be in a util function sometime in the future
#       Is duplicated in runsolver_wrapper.py
def get_optimizer():
    optimizer_dir = os.getcwd().split("/")[-1]
    if optimizer_dir.count("_") == 1:
        return optimizer_dir.split("_")[0]
    else:
        return "_".join(optimizer_dir.split("_")[0:-2])


def load_experiment_file():
    optimizer = get_optimizer()
    print os.getcwd().split("/")[-1]
    experiment = Experiment(".", optimizer)
    return experiment


def do_cv(arguments, parameters, experiment, folds=10):
    logger.info("Starting Cross validation")
    sys.stdout.flush()
    cfg = load_experiment_config_file()

    # Store the results to hand them back to tpe and spearmint
    results = []
    times = []

    try:
        for fold in range(folds):
            arguments.instance = fold
            result, wallclock_time = run_one_instance(arguments, parameters, experiment)
            results.append(result)
            times.append(wallclock_time)

            worst_possible = cfg.getfloat("HPOLIB", "result_on_terminate")
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
        mean = np.NaN
        
    # Do not return any kind of nan because this would break spearmint
    with warnings.catch_warnings():
        if not np.isfinite(mean):
            mean = float(cfg.get("HPOLIB", "result_on_terminate"))

    logger.info("Finished CV")
    return mean, np.nansum(times)


def run_one_instance(arguments, parameters, experiment):
    """Execute one instance."""
    if arguments.instance is not None:
        instance = int(arguments.instance)
    else:
        instance = 0

    if experiment.is_closed():
        experiment = load_experiment_file()

    # Side-effect: adds a job if it is not yet in the experiments file
    trial_index = get_trial_index(experiment, instance, parameters)
    experiment.set_one_fold_running(trial_index, instance)
    experiment._save_jobs()
    experiment.close()  # release Experiment lock

    logger.info("Starting instance evaluation for configuration: %s, "
                "instance: %s" % (str(trial_index), str(instance)))
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

    # Do bookkeeping!
    if experiment.is_closed():
        experiment = load_experiment_file()

    if status == "SAT":
        experiment.set_one_fold_complete(trial_index, instance, result,
                                         wallclock_time, additional_data)
    elif status == "CRASHED" or status == "UNSAT":
        result = cfg.getfloat("HPOLIB", "result_on_terminate")
        experiment.set_one_fold_crashed(trial_index, instance, result,
                                        wallclock_time, additional_data)
        status = "SAT"
    else:
        # TODO: We need a global stopping mechanism
        pass
    experiment._save_jobs()
    experiment.close()  # release lock

    logger.info("Finished instance Evaluation for configuration: %s, "
                "instance %s; result: %f, duration: %f" %
                (str(trial_index), str(instance), result, wallclock_time))
    return result, wallclock_time


def get_trial_index(experiment, fold, params):
    # Check whether we are in a new configuration; This has to check whether
    # the params were already inserted but also whether the fold already run
    # This is checked twice; the instance_result has to be not NaN and the
    # entry in instance_order has to exist
    new = True
    trial_index = float("NaN")
    for idx, trial in enumerate(experiment.trials):
        exp = trial['params']
        if exp == params and (idx, fold) not in experiment.instance_order and \
                (experiment.get_trial_from_id(idx)['instance_results'][fold] ==
                     np.NaN or
                         experiment.get_trial_from_id(idx)['instance_results'][
                             fold] !=
                         experiment.get_trial_from_id(idx)['instance_results'][
                             fold]):
            new = False
            trial_index = idx
            break
    if new:
        trial_index = experiment.add_job(params)
    return trial_index


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
    config = load_experiment_config_file()

    loglevel = config.getint("HPOLIB", "HPOlib_loglevel")
    hpolib_logger.setLevel(loglevel)
    host = config.get("HPOLIB", "logging_host")
    if host:
        port = config.getint("HPOLIB", "logging_port")
        socketh = SocketHandler(host, port)
        hpolib_logger.addHandler(socketh)
    else:
        streamh = logging.StreamHandler(sys.stdout)
        hpolib_logger.addHandler(streamh)

    # Load the experiment to do time-keeping
    cv_starttime = time.time()
    experiment = load_experiment_file()
    experiment.start_cv(cv_starttime)
    experiment._save_jobs()
    experiment.close()

    folds = config.getint('HPOLIB', 'number_cv_folds')

    if arguments.instance is None and folds > 1:
        result, wallclock_time = \
            do_cv(arguments, parameters, experiment, folds=folds)
    else:
        result, wallclock_time = \
            run_one_instance(arguments, parameters, experiment)

    # Load the experiment to do time-keeping
    if experiment.is_closed():
        experiment = load_experiment_file()

    experiment.end_cv(time.time())
    experiment._save_jobs()
    experiment.close()

    sys.stdout.write("Result: %f, Runtime: %f\n" %
                     (float(result), float(wallclock_time)))
    sys.stdout.flush()
    return result


if __name__ == "__main__":
    arguments, parameters = parse_cli()
    main(arguments, parameters)
