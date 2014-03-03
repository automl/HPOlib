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

import ConfigParser

logger = logging.getLogger("HPOlib.optimizers.smac.smac_2_06_01-dev_parser")


def add_default(config):
    # This module reads smacDefault.cfg and adds this defaults to a given config

    assert isinstance(config, ConfigParser.RawConfigParser), \
        'config is not a valid instance'

     # Find out name of .cfg, we are in anything_parser.py[c]
    optimizer_config_fn = os.path.splitext(__file__)[0][:-7] + "Default.cfg"
    if not os.path.exists(optimizer_config_fn):
        logger.critical("No default config %s found" % optimizer_config_fn)
        sys.exit(1)

    smac_config = ConfigParser.SafeConfigParser(allow_no_value=True)
    smac_config.read(optimizer_config_fn)
    # --------------------------------------------------------------------------
    # SMAC
    # --------------------------------------------------------------------------
    # Set defaults for SMAC
    if not config.has_section('SMAC'):
        config.add_section('SMAC')
    # optional arguments (exec_dir taken out as is does not seem to be used)
    for option in ('numRun', 'instanceFile', 'intraInstanceObj', 'runObj',
                   'testInstanceFile', 'p', 'rf_full_tree_bootstrap',
                   'rf_split_min', 'adaptiveCapping', 'maxIncumbentRuns',
                   'numIterations', 'runtimeLimit', 'deterministic',
                   'retryTargetAlgorithmRunCount',
                   'intensification_percentage'):
        if not config.has_option('SMAC', option):
            config.set('SMAC', option,
                       smac_config.get('SMAC', option))

    # special cases
    if not config.has_option('SMAC', 'cutoffTime'):
        config.set('SMAC', 'cutoffTime',
                   str(config.getint('HPOLIB', 'runsolver_time_limit') + 100))
    if not config.has_option('SMAC', 'algoExec'):
        config.set('SMAC', 'algoExec',
                   config.get('HPOLIB', 'run_instance'))
    if not config.has_option('SMAC', 'totalNumRunsLimit'):
        config.set('SMAC', 'totalNumRunsLimit',
                   str(config.getint('HPOLIB', 'numberOfJobs') *
                       config.getint('HPOLIB', 'numberCV')))
    if not config.has_option('SMAC', 'numConcurrentAlgoExecs'):
        config.set('SMAC', 'numConcurrentAlgoExecs',
                   config.get('HPOLIB', 'numberOfConcurrentJobs'))

    path_to_optimizer = smac_config.get('SMAC', 'path_to_optimizer')
    if not os.path.isabs(path_to_optimizer):
        path_to_optimizer = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_to_optimizer)

    path_to_optimizer = os.path.normpath(path_to_optimizer)
    if not os.path.exists(path_to_optimizer):
        logger.critical("Path to optimizer not found: %s" % path_to_optimizer)

    config.set('SMAC', 'path_to_optimizer', path_to_optimizer)

    return config