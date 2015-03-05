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

sys.path.append(os.path.dirname(__file__))
smac_2_06_01_dev_parser = __import__('smac_2_06_01-dev_parser')

logger = logging.getLogger("HPOlib.optimizers.smac.ROAR_smac_2_06_01-dev_parser")


def manipulate_config(config):
    '''
    This method wraps the smac config parser in order to run ROAR
    '''

    logger.debug("Running in ROAR mode")
    config = smac_2_06_01_dev_parser.manipulate_config(config=config)
    config.set('SMAC', 'exec_mode', 'ROAR')

    return config