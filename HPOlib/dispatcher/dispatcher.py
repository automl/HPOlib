from collections import OrderedDict
import logging
import numpy as np
import re
import sys
import time

import HPOlib.wrapping_util as wrapping_util
import HPOlib.dispatcher.runsolver_wrapper as runsolver_wrapper
import HPOlib.dispatcher.python_file as python_file
import HPOlib.Experiment as Experiment


logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
hpolib_logger = logging.getLogger("HPOlib")
hpolib_logger.setLevel(logging.INFO)
logger = logging.getLogger("HPOlib.dispatcher.dispatcher")


def remove_param_metadata(params):
    """
    Check whether some params are defined on the Log scale or with a Q value,
    must be marked with "LOG$_{paramname}" or Q[0-999]_$paramname
    LOG_/Q_ will be removed from the paramname
    """
    for para in params:
        new_name = para

        if isinstance(params[para], str):
            params[para] = params[para].strip("'")
        if "LOG10_" in para:
            pos = para.find("LOG10_")
            new_name = para[0:pos] + para[pos + 6:]
            params[new_name] = np.power(10, float(params[para]))
            del params[para]
        elif "LOG2_" in para:
            pos = para.find("LOG2_")
            new_name = para[0:pos] + para[pos + 5:]
            params[new_name] = np.power(2, float(params[para]))
            del params[para]
        elif "LOG_" in para:
            pos = para.find("LOG_")
            new_name = para[0:pos] + para[pos + 4:]
            params[new_name] = np.exp(float(params[para]))
            del params[para]
            #Check for Q value, returns round(x/q)*q
        m = re.search(r'Q[0-999\.]{1,10}_', para)
        if m is not None:
            pos = new_name.find(m.group(0))
            tmp = new_name[0:pos] + new_name[pos + len(m.group(0)):]
            q = float(m.group(0)[1:-1])
            params[tmp] = round(float(params[new_name]) / q) * q
            del params[new_name]


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


def parse_command_line():
    # Parse options and arguments
    usage = "This script pickles the params and runs the runsolver with " + \
            "run_instance and extract the output for the optimizer \n" + \
            "The output is printed im a SMACish way: \n\n" + \
            "'Result for ParamILS: <solved>, <runtime>, <runlength>, " + \
            "<quality>, <seed>, <additional rundata>' \n\n" + \
            "Usage: runsolver_wrapper <instancename> " + \
            "<instancespecificinformation> <cutofftime> <cutofflength> " + \
            "<seed> <param> <param> <param>\n" + \
            "<instancename> might be the optimizer name if not" + \
            " called by smac\n"
    if len(sys.argv) < 7:
        sys.stdout.write(usage)
        exit(1)

    # Then get some information for run_instance
    fold = int(sys.argv[1])
    seed = int(sys.argv[5])
    return fold, seed


def get_parameters():
    params = dict(zip(sys.argv[6::2], sys.argv[7::2]))
    # Now remove the leading minus
    for key in params.keys():
        new_key = re.sub('^-', '', key)
        params[new_key] = params[key]
        del params[key]
    remove_param_metadata(params)
    params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))
    return params


def format_return_string(status, runtime, runlength, quality, seed,
                         additional_data):
    return_string = "Result for ParamILS: %s, %f, %d, %f, %d, %s" % \
                    (status, runtime, runlength, quality, seed, additional_data)
    return return_string


def main():
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

    fold, seed = parse_command_line()
    # Side-effect: removes all additional information like log and applies
    # transformations to the parameters
    params = get_parameters()

    experiment = Experiment.load_experiment_file()
    # Side-effect: adds a job if it is not yet in the experiments file
    trial_index = get_trial_index(experiment, fold, params)
    experiment.set_one_fold_running(trial_index, fold)
    experiment._save_jobs()
    del experiment  # release Experiment lock

    dispatch_function = cfg.get("HPOLIB", "dispatcher")
    if dispatch_function == "runsolver_wrapper.py":
        additional_data, result, status, wallclock_time = \
            runsolver_wrapper.dispatch(cfg, fold, params)
    elif dispatch_function == "python_function.py":
        additional_data, result, status, wallclock_time = \
            python_file.dispatch(cfg, fold, params)

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

    return_string = format_return_string(status, wallclock_time, 1, result,
                                         seed, additional_data)

    if not called_from_cv:
        experiment = Experiment.load_experiment_file()
        experiment.end_cv(time.time())
        experiment._save_jobs()
        del experiment

    print return_string
    logger.debug(return_string)
    return return_string

if __name__ == "__main__":
    main()
