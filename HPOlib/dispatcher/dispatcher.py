import importlib
import logging
import numpy as np
import re
import time

import HPOlib.wrapping_util as wrapping_util
import HPOlib.Experiment as Experiment


hpolib_logger = logging.getLogger("HPOlib")
logger = logging.getLogger("HPOlib.dispatcher.dispatcher")


def main(arguments, parameters, fold):
    """
    If we are not called from cv means we are called from CLI. This means
    the optimizer itself handles crossvalidation (smac). To keep a nice .pkl we
    have to do some bookkeeping here
    """

    cfg = wrapping_util.load_experiment_config_file()

    dispatch_function_name = cfg.get("HPOLIB", "dispatcher")
    dispatch_function_name = re.sub("(\.py)$", "", dispatch_function_name)
    try:
        dispatch_function = importlib.import_module("HPOlib.dispatcher.%s" %
                                                    dispatch_function_name)

        additional_data, result, status, wallclock_time = \
            dispatch_function.dispatch(cfg, fold, parameters)
    except ImportError:
        additional_data = ""
        result = float("NaN")
        status = "CRASHED"
        wallclock_time = 0.
        logger.error("Invalid value %s for HPOLIB:dispatcher" %
                     dispatch_function_name)

    return status, wallclock_time, result, additional_data

if __name__ == "__main__":
    main()
