# #
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
import logging
import os
import sys
import time

import HPOlib
import HPOlib.check_before_start as check_before_start
import HPOlib.wrapping_util as wrapping_util
import HPOlib.dispatcher.runsolver_wrapper as runsolver_wrapper
import HPOlib.dispatcher.python_file as python_file
# Import experiment only after the check for numpy succeeded

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

hpolib_logger = logging.getLogger("HPOlib")
logger = logging.getLogger("HPOlib.testing")


def use_arg_parser():
    """Parse all options which can be handled by the wrapping script.
    Unknown arguments are ignored and returned as a list. It is useful to
    check this list in your program to handle typos etc.

    Returns:
        a tuple. The first element is an argparse.Namespace object,
        the second a list with all unknown arguments.
    """
    description = "Perform an experiment with HPOlib. " \
                  "Call this script from experiment directory (containing 'config.cfg')"
    epilog = "Your are using HPOlib " + HPOlib.__version__
    prog = "path/from/Experiment/to/HPOlib/wrapping.py"

    parser = ArgumentParser(description=description, prog=prog, epilog=epilog)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                       help="Call the test function for all hyperparameter "
                            "configurations of an experiment.")
    group.add_argument("--best", action="store_true",
                       help="Call the test function for the best "
                            "hyperparameter configuration of an experiment.")
    group.add_argument("--trajectory", action="store_true",
                       help="Call the test function for the trajectory of best "
                            "hyperparameter configurations of an experiment.")

    parser.add_argument("--cwd", action="store", type=str, dest="working_dir",
                        default=None, help="Change the working directory to "
                                           "<working_directory> prior to running the experiment")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-q", "--silent", action="store_true",
                       dest="silent", default=False,
                       help="Don't print anything during optimization")
    group.add_argument("-v", "--verbose", action="store_true",
                       dest="verbose", default=False,
                       help="Print stderr/stdout for optimizer")

    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    """Test the best algorithm of a previous HPOlib optimization run."""
    formatter = logging.Formatter('[%(levelname)s] [%(asctime)s:%(name)s] %('
                                  'message)s', datefmt='%H:%M:%S')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    hpolib_logger.addHandler(handler)

    args, unknown_arguments = use_arg_parser()

    if args.working_dir:
        experiment_dir = args.working_dir
    else:
        experiment_dir = os.getcwd()

    config = wrapping_util.get_configuration(experiment_dir,
                                             None,
                                             unknown_arguments)
    log_level = config.getint("HPOLIB", "loglevel")
    hpolib_logger.setLevel(log_level)

    os.chdir(experiment_dir)

    # TODO check if the testing directory exists
    # TODO check if the test function is there!
    check_before_start.check_first(experiment_dir)

    # Now we can safely import non standard things
    import numpy as np
    import HPOlib.Experiment as Experiment  # Wants numpy and scipy


    if not config.has_option("HPOLIB", "is_not_original_config_file"):
        logger.critical("The directory you're in seems to be no directory in "
                        "which an HPOlib run was executed: %s" % experiment_dir)
    is_not_original_config_file = config.get("HPOLIB", "is_not_original_config_file")
    if not is_not_original_config_file:
        logger.critical("The directory you're in seems to be no directory in "
                        "which an HPOlib run was executed: %s" % experiment_dir)

    experiment_directory_prefix = config.get("HPOLIB", "experiment_directory_prefix")
    optimizer = wrapping_util.get_optimizer()
    # This is a really bad hack...
    optimizer = optimizer.replace(experiment_directory_prefix, "")
    trials = Experiment.Experiment(expt_dir=".",
                                   expt_name=experiment_directory_prefix + optimizer)

    # TODO: do we need a setup for the testing?
    fn_setup = config.get("HPOLIB", "function_setup")
    if fn_setup:

        fn_setup_output = os.path.join(os.getcwd(),
                                       "function_setup.out")
        runsolver_cmd = runsolver_wrapper._make_runsolver_command(
            config, fn_setup_output)
        setup_cmd = runsolver_cmd + " " + fn_setup
        #runsolver_output = subprocess.STDOUT
        runsolver_output = open("/dev/null")
        runsolver_wrapper._run_command_with_shell(setup_cmd,
                                                  runsolver_output)

    configurations_to_test = []
    # TODO this should not rerun configurations which were already run!
    # Find the configurations to test on!
    if args.all:
        for idx in range(len(trials.trials)):
            configurations_to_test.append(idx)
    elif args.best:
        id_ = trials.get_arg_best(consider_incomplete=False)
        configurations_to_test.append(id_)
    elif args.trajectory:
        raise NotImplementedError("Evaluating the runs along a trajectory is "
                                  "not implemented yet!")
    else:
        raise ValueError()

    trials._save_jobs()
    del trials

    for id_ in configurations_to_test:
        for fold in range(1): # number of test folds, currently this is only one!
            trials = Experiment.Experiment(expt_dir=".",
                                           expt_name=experiment_directory_prefix + optimizer)

            # Check if this configuration was already run
            trial = trials.get_trial_from_id(id_)
            if np.isfinite(trial['test_instance_results'][fold]):
                logger.info("Trial #%d, fold %d already evaluated, continuing "
                            "with next." % (id_, fold))
                del trials
                continue

            logger.info("Evaluating trial #%d, fold %d." % (id_, fold))

            configuration = trial["params"]

            trials.set_one_test_fold_running(id_, fold)
            trials._save_jobs()
            del trials

            dispatch_function = config.get("HPOLIB", "dispatcher")
            if dispatch_function == "runsolver_wrapper.py":
                additional_data, result, status, wallclock_time = \
                    runsolver_wrapper.dispatch(config, fold, configuration,
                                               test=True)
            elif dispatch_function == "python_function.py":
                additional_data, result, status, wallclock_time = \
                    python_file.dispatch(config, fold, configuration, test=True)
            else:
                additional_data = ""
                result = float("NaN")
                status = "CRASHED"
                wallclock_time = 0.
                logger.error(
                    "Invalid value %s for HPOLIB:dispatcher" % dispatch_function)

            trials = Experiment.Experiment(expt_dir=".",
                                           expt_name=experiment_directory_prefix + optimizer)

            if status == "SAT":
                trials.set_one_test_fold_complete(id_, fold, result,
                                             wallclock_time, additional_data)
            elif status == "CRASHED" or status == "UNSAT":
                result = config.getfloat("HPOLIB", "result_on_terminate")
                trials.set_one_test_fold_crashed(id_, fold, result,
                                                wallclock_time, additional_data)
                status = "SAT"
            else:
                # TODO: We need a global stopping mechanism
                pass
            trials._save_jobs()
            del trials  # release lock


            logger.info("Finished: Result: %s, Runtime: %s, Status: %s",
                        str(result),
                        str(wallclock_time),
                        str(status))

    # TODO: do we need a teardown for testing?
    fn_teardown = config.get("HPOLIB", "function_teardown")
    if fn_teardown:

        fn_teardown_output = os.path.join(os.getcwd(),
                                          "function_teardown.out")
        runsolver_cmd = runsolver_wrapper._make_runsolver_command(
            config, fn_teardown_output)
        teardown_cmd = runsolver_cmd + " " + fn_teardown
        runsolver_output = open("/dev/null")
        runsolver_wrapper._run_command_with_shell(teardown_cmd,
                                                  runsolver_output)

    trials = Experiment.Experiment(expt_dir=".",
                                   expt_name=experiment_directory_prefix + optimizer)
    trials.endtime.append(time.time())
    trials._save_jobs()

    del trials
    logger.info("Finished HPOlib-testbest.")
    return 0


if __name__ == "__main__":
    main()