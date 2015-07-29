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

import ConfigParser


def manipulate_config(config):
    # This module reads sequentialDefault.cfg and adds this defaults to a
    # given config

    assert isinstance(config, ConfigParser.RawConfigParser), \
        'config is not a valid instance'

    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             'ConfigurationRunnerDefault.cfg')
    if not os.path.isfile(config_file):
        raise Exception('%s is not a valid file\n' % config_file)

    sequential_config = ConfigParser.SafeConfigParser(allow_no_value=True)
    sequential_config.read(config_file)

    if not config.has_section('ConfigurationRunner'):
        config.add_section('ConfigurationRunner')

    config.set("DEFAULT", "optimizer_version", "0.1")

    return config