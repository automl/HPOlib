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
import os
import sys
import time

import ConfigParser

logger = logging.getLogger("HPOlib.optimizers.smac.smac_2_08_00-master_parser")


def manipulate_config(config):
    if not config.has_option('SMAC', 'cutoff_time'):
        print config.get('HPOLIB', 'runsolver_time_limit')
        if config.get('HPOLIB', 'runsolver_time_limit'):
            config.set('SMAC', 'cutoff_time',
                   str(config.getint('HPOLIB', 'runsolver_time_limit') + 100))
        else:
            # SMACs maxint
            config.set('SMAC', 'cutoff_time', "2147483647")
    if not config.has_option('SMAC', 'total_num_runs_limit'):
        config.set('SMAC', 'total_num_runs_limit',
                   str(config.getint('HPOLIB', 'number_of_jobs') *
                       config.getint('HPOLIB', 'number_cv_folds')))
    if not config.has_option('SMAC', 'num_concurrent_algo_execs'):
        config.set('SMAC', 'num_concurrent_algo_execs',
                   config.get('HPOLIB', 'number_of_concurrent_jobs'))

    path_to_optimizer = config.get('SMAC', 'path_to_optimizer')
    if not os.path.isabs(path_to_optimizer):
        path_to_optimizer = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_to_optimizer)

    path_to_optimizer = os.path.normpath(path_to_optimizer)
    if not os.path.exists(path_to_optimizer):
        logger.critical("Path to optimizer not found: %s" % path_to_optimizer)
        sys.exit(1)

    config.set('SMAC', 'path_to_optimizer', path_to_optimizer)
    config.set('SMAC', 'exec_mode', 'SMAC')

    shared_model = config.get('SMAC', 'shared_model')
    wait_time = config.getint('SMAC', 'wait_for_shared_model')
    if shared_model != 'False':
        config.getint('SMAC', 'shared_model')
        if not os.path.isdir(shared_model):
            config.set('SMAC', 'shared_model_scenario_file', os.path.join(shared_model, 'scenario.txt'))

        if config.get('HPOLIB', 'temporary_output_directory') != '':
            logger.critical('Using a temp_out_dir and a shared model is not possible')
            sys.exit(1)
    return config