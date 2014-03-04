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

import os
import logging
import imp
import sys
import ConfigParser


logger = logging.getLogger("HPOlib.config_parser.parse")


def parse_config(config_files, allow_no_value=True, optimizer_version="",
                 cli_values=None):
    if type(config_files) is str:
        if not os.path.isfile(config_files):
            raise Exception('%s is not a valid file\n' % os.path.join(
                            os.getcwd(), config_files))
    else:
        for config_file in config_files:
            if not os.path.isfile(config_file):
                raise Exception('%s is not a valid file\n' % os.path.join(
                                os.getcwd(), config_file))

    config = ConfigParser.SafeConfigParser(allow_no_value=allow_no_value)
    config.read(config_files)
    if cli_values is not None:
        config.readfp(cli_values)

    # --------------------------------------------------------------------------
    # Check critical values
    # --------------------------------------------------------------------------
    if not config.has_option('HPOLIB', 'numberOfJobs') or \
            config.get('HPOLIB', 'numberOfJobs') == '':
        raise Exception('numberOfJobs not specified in .cfg')
    if not config.has_option('HPOLIB', 'result_on_terminate') or \
            config.get('HPOLIB', 'result_on_terminate') == '':
        raise Exception('No result_on_terminate specified in .cfg')
    if not config.has_option('HPOLIB', 'function') or \
            config.get('HPOLIB', 'function') == '':
        raise Exception('No function specified in .cfg')

    return config
