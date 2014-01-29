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


def add_default(config):
    # This module reads tpeDefault.cfg and adds this defaults to a given config
    assert isinstance(config, ConfigParser.RawConfigParser), \
        "config is not a valid instance"

    config_fn = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             "tpeDefault.cfg")
    if not os.path.isfile(config_fn):
        raise Exception('%s is not a valid file\n' % config_fn)

    tpe_config = ConfigParser.SafeConfigParser(allow_no_value=True)
    tpe_config.read(config_fn)
    # --------------------------------------------------------------------------
    # TPE
    # --------------------------------------------------------------------------
    # Set default for TPE
    if not config.has_section('TPE'):
        config.add_section('TPE')

    # optional cases
    if not config.has_option('TPE', 'space'):
            config.set('TPE', 'space',
                       tpe_config.get('TPE', 'space'))

    if not config.has_option('TPE', 'numberEvals'):
            config.set('TPE', 'numberEvals',
                       config.get('DEFAULT', 'numberOfJobs'))

    # Anyway set this
    # Makes it possible to call, e.g. hyperopt_august2013_mod via simply tpe
    config.set('DEFAULT', 'optimizer_version', tpe_config.get('TPE', 'version'))

    return config