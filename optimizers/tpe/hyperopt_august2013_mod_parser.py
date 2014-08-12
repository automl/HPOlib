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

logger = logging.getLogger("HPOlib.optimizers.tpe.hyperopt_august2013_mod_parser")


def manipulate_config(config):
    if not config.has_section('TPE'):
        config.add_section('TPE')

    # optional cases
    if not config.has_option('TPE', 'space'):
        raise Exception("TPE:space not specified in .cfg")

    number_of_jobs = config.getint('HPOLIB', 'number_of_jobs')
    if not config.has_option('TPE', 'number_evals'):
        config.set('TPE', 'number_evals', config.get('HPOLIB', 'number_of_jobs'))
    elif config.getint('TPE', 'number_evals') != number_of_jobs:
        logger.warning("Found a total_num_runs_limit (%d) which differs from "
                       "the one read from the config (%d). This can e.g. "
                       "happen when restoring a TPE run" %
                       (config.getint('TPE', 'number_evals'),
                        number_of_jobs))
        config.set('TPE', 'number_evals', str(number_of_jobs))

    path_to_optimizer = config.get('TPE', 'path_to_optimizer')
    if not os.path.isabs(path_to_optimizer):
        path_to_optimizer = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_to_optimizer)

    path_to_optimizer = os.path.normpath(path_to_optimizer)
    if not os.path.exists(path_to_optimizer):
        logger.critical("Path to optimizer not found: %s" % path_to_optimizer)
        sys.exit(1)

    config.set('TPE', 'path_to_optimizer', path_to_optimizer)

    return config