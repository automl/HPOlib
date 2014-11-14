import logging
import numpy as np
import time

import HPOlib.wrapping_util as wrapping_util
import HPOlib.dispatcher.runsolver_wrapper as runsolver_wrapper
import HPOlib.dispatcher.python_file as python_file
from HPOlib.dispatcher import mysqldbtae
import HPOlib.Experiment as Experiment


logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
hpolib_logger = logging.getLogger("HPOlib")
hpolib_logger.setLevel(logging.INFO)
logger = logging.getLogger("HPOlib.dispatcher.dispatcher")


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
                 experiment.get_trial_from_id(idx)['instance_results'][fold] !=
                 experiment.get_trial_from_id(idx)['instance_results'][fold]):
            new = False
            trial_index = idx
            break
    if new:
        trial_index = experiment.add_job(params)
    return trial_index


"""
def format_return_string(status, runtime, runlength, quality, seed,
                         additional_data):
    return_string = "Result for ParamILS: %s, %f, %d, %f, %d, %s" % \
                    (status, runtime, runlength, quality, seed, additional_data)
    return return_string
"""


def main(arguments, parameters, fold):
    """
    If we are not called from cv means we are called from CLI. This means
    the optimizer itself handles crossvalidation (smac). To keep a nice .pkl we
    have to do some bookkeeping here
    """

    cfg = wrapping_util.load_experiment_config_file()
    called_from_cv = True
    if cfg.getint('HPOLIB', 'handles_cv') == 1:
        # If Our Optimizer can handle crossvalidation,
        # we are called from CLI. To keep a sane nice .pkl
        # we have to do some bookkeeping here
        called_from_cv = False

    # This has to be done here for SMAC, since smac does not call cv.py
    if not called_from_cv:
        cv_starttime = time.time()
        experiment = Experiment.load_experiment_file()
        experiment.start_cv(cv_starttime)
        experiment._save_jobs()
        del experiment

    experiment = Experiment.load_experiment_file()
    # Side-effect: adds a job if it is not yet in the experiments file
    trial_index = get_trial_index(experiment, fold, parameters)
    experiment.set_one_fold_running(trial_index, fold)
    experiment._save_jobs()
    del experiment  # release Experiment lock

    dispatch_function = cfg.get("HPOLIB", "dispatcher")
    if dispatch_function == "runsolver_wrapper.py":
        additional_data, result, status, wallclock_time = \
            runsolver_wrapper.dispatch(cfg, fold, parameters)
    elif dispatch_function == "python_function.py":
        additional_data, result, status, wallclock_time = \
            python_file.dispatch(cfg, fold, parameters)
    elif dispatch_function == "mysqldbtae.py":
        additional_data, result, status, wallclock_time = \
            mysqldbtae.dispatch(cfg, fold, parameters)

    else:
        additional_data = ""
        result = float("NaN")
        status = "CRASHED"
        wallclock_time = 0.
        logger.error("Invalid value %s for HPOLIB:dispatcher" %
                     dispatch_function)

    experiment = Experiment.load_experiment_file()
    if status == "SAT":
        experiment.set_one_fold_complete(trial_index, fold, result,
                                         wallclock_time, additional_data)
    elif status == "CRASHED" or status == "UNSAT":
        result = cfg.getfloat("HPOLIB", "result_on_terminate")
        experiment.set_one_fold_crashed(trial_index, fold, result,
                                        wallclock_time, additional_data)
        status = "SAT"
    else:
        # TODO: We need a global stopping mechanism
        pass
    experiment._save_jobs()
    del experiment  # release lock

    if not called_from_cv:
        experiment = Experiment.load_experiment_file()
        experiment.end_cv(time.time())
        experiment._save_jobs()
        del experiment

    return status, wallclock_time, result, additional_data
