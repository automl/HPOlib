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

logger = logging.getLogger("HPOlib.optimizers.spearmint.spearmint_april2013_mod_parser")


def add_default(config):
   # This module reads __file__Default.cfg and adds this defaults to a given config
    assert isinstance(config, ConfigParser.RawConfigParser), \
        "config is not a valid instance"

    # Find out name of .cfg, we are in anything_parser.py[c]
    optimizer_config_fn = os.path.splitext(__file__)[0][:-7] + "Default.cfg"
    if not os.path.exists(optimizer_config_fn):
        logger.critical("No default config %s found" % optimizer_config_fn)
        sys.exit(1)

    spearmint_config = ConfigParser.SafeConfigParser(allow_no_value=True)
    spearmint_config.read(optimizer_config_fn)

    # --------------------------------------------------------------------------
    # SPEARMINT
    # --------------------------------------------------------------------------
    # Set default for SPEARMINT
    if not config.has_section('SPEARMINT'):
        config.add_section('SPEARMINT')
    # optional arguments
    for option in ('script', 'method', 'grid_size', 'config', 'grid_seed',
                   'spearmint_polling_time', 'max_concurrent', 'method_args'):
        if not config.has_option('SPEARMINT', option):
            config.set('SPEARMINT', option,
                       spearmint_config.get('SPEARMINT', option))

    # special cases
    if not config.has_option('SPEARMINT', 'method'):
        config.set('SPEARMINT', 'method',
                   spearmint_config.get('SPEARMINT', 'method'))
        config.set('SPEARMINT', 'method-args',
                   spearmint_config.get('SPEARMINT', 'method-args'))

    # GENERAL
    if not config.has_option('SPEARMINT', 'max_finished_jobs'):
        config.set('SPEARMINT', 'max_finished_jobs',
                   config.get('HPOLIB', 'numberOfJobs'))

    path_to_optimizer = spearmint_config.get('SPEARMINT', 'path_to_optimizer')
    if not os.path.isabs(path_to_optimizer):
        path_to_optimizer = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_to_optimizer)

    path_to_optimizer = os.path.normpath(path_to_optimizer)
    if not os.path.exists(path_to_optimizer):
        logger.critical("Path to optimizer not found: %s" % path_to_optimizer)
        sys.exit(1)

    config.set('SPEARMINT', 'path_to_optimizer', path_to_optimizer)

    return config