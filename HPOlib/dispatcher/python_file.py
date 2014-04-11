import importlib
import logging
import sys
import time

import HPOlib.wrapping_util as wrapping_util

logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
hpolib_logger = logging.getLogger("HPOlib")
hpolib_logger.setLevel(logging.INFO)
logger = logging.getLogger("HPOlib.dispatcher.python_file")


def dispatch(cfg, fold, params):
    starttime = time.time()
    wallclock_time = None
    result = float("NaN")

    fn_module = cfg.get("HPOLIB", "python_module")
    fn_name = cfg.get("HPOLIB", "python_function")
    folds = cfg.getint("HPOLIB", "number_cv_folds")

    if not cfg.getboolean("HPOLIB", "use_own_time_measurement"):
        logger.warn("The configuration HPOLIB:use_own_time_measurment False "
                    "has no effect for the python function dispatcher.")

    try:
        modules = fn_module.rsplit(".", 1)
        fromlist = [] if len(modules) == 1 else [modules[1]]
        module = importlib.import_module(fn_module, fromlist)
        fn = getattr(module, fn_name)
    except Exception as e:
        logger.error(wrapping_util.format_traceback(sys.exc_info()))
        logger.error("Could not import function %s due to exception %s",
                     fn_name, str(e))
        return "", result, "UNSAT", time.time() - starttime

    try:
        # TODO: remove this hackines
        fixed_params = dict()
        for param in params:
            if param[0] == "-":
                fixed_params[param[1:]] = params[param]
            else:
                fixed_params[param] = params[param]
        retval = fn(fixed_params, fold=fold, folds=folds)
        status = "SAT"

        if isinstance(retval, float):
            result = retval
        elif isinstance(retval, dict):
            result = retval["result"]
            wallclock_time = retval["duration"]
        else:
            status = "UNSAT"
            logger.error("Return type %s of target function %s is not "
                            "supported", str(type(retval)), str(fn_name))
    except Exception as e:
        status = "UNSAT"
        logger.error("Target function evaluation raised exception %s.", str(e))

    if wallclock_time is None:
        wallclock_time = time.time() - starttime


    return "", result, status, wallclock_time
