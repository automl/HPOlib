__author__ = 'alis'

import logging
import os
import sys

logger = logging.getLogger("HPOlib.optimizers.irace_1_07_parser.py")


def manipulate_config(config):
    if not config.has_section('irace'):
        config.add_section('irace')

    # optional cases
    if not config.has_option('irace', 'params'):
        raise Exception("irace:params not specified in .cfg")

    # number_of_jobs = config.getint('HPOLIB', 'number_of_jobs')
    # if not config.has_option('irace', 'number_evals'):
    #     config.set('irace', 'number_evals', config.get('HPOLIB', 'number_of_jobs'))
    # elif config.getint('irace', 'number_evals') != number_of_jobs:
    #     logger.warning("Found a total_num_runs_limit (%d) which differs from "
    #                    "the one read from the config (%d). This can e.g. "
    #                    "happen when restoring a irace run" %
    #                    (config.getint('irace', 'number_evals'),
    #                     number_of_jobs))
    #     config.set('irace', 'number_evals', str(number_of_jobs))

    path_to_optimizer = config.get('irace', 'path_to_optimizer')
    if not os.path.isabs(path_to_optimizer):
        path_to_optimizer = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_to_optimizer)

    path_to_optimizer = os.path.normpath(path_to_optimizer)
    if not os.path.exists(path_to_optimizer):
        logger.critical("Path to optimizer not found: %s" % path_to_optimizer)
        sys.exit(1)

    config.set('irace', 'path_to_optimizer', path_to_optimizer)

    return config
