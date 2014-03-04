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


def parse_config(config_file, allow_no_value=True, optimizer_version="",
                 cli_values=None):
    # Reads config_file
    # Overwrites with default values from generalDefault.cfg
    # Loads optimizer specific parser, called 'optimizer_version'_parser.py, which can read its own default config
    if not os.path.isfile(config_file):
        raise Exception('%s is not a valid file\n' % os.path.join(
                        os.getcwd(), config_file))

    config = ConfigParser.SafeConfigParser(allow_no_value=allow_no_value)
    config.read(config_file)
    if cli_values is not None:
        config.readfp(cli_values)

    # Load general default configs
    config_fn_default = \
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "generalDefault.cfg")
    config_default = ConfigParser.SafeConfigParser(allow_no_value=True)

    if not os.path.isfile(config_fn_default):
        raise Exception('%s does not exist\n' % config_fn_default)
    config_default.read(config_fn_default)

    # --------------------------------------------------------------------------
    # Defaults for HPOLIB
    # --------------------------------------------------------------------------
    # Set defaults for section HPOLIB
    for option in ('algorithm', 'run_instance', 'numberCV',
                   'leading_algo_info', 'numberOfConcurrentJobs',
                   'runsolver_time_limit', 'total_time_limit', 'memory_limit',
                   'cpu_limit', 'leading_runsolver_info', 'max_crash_per_cv'):
        if not config.has_option('HPOLIB', option):
            config.set('HPOLIB', option,
                       config_default.get('HPOLIB', option))

    if not config.has_option('HPOLIB', 'numberOfJobs') or \
            config.get('HPOLIB', 'numberOfJobs') == '':
        raise Exception('numberOfJobs not specified in .cfg')
    if not config.has_option('HPOLIB', 'result_on_terminate') or \
            config.get('HPOLIB', 'result_on_terminate') == '':
        raise Exception('No result_on_terminate specified in .cfg')
    if not config.has_option('HPOLIB', 'function') or \
            config.get('HPOLIB', 'function') == '':
        raise Exception('No function specified in .cfg')

    # Load optimizer parsing module
    if optimizer_version == "":
        return config
    optimizer_version_parser = optimizer_version + "_parser"
    optimizer_version_parser_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                 optimizer_version_parser + ".py")
    # noinspection PyBroadException,PyUnusedLocal
    try:
        optimizer_module_loaded = imp.load_source(optimizer_version_parser, optimizer_version_parser_path)
    except Exception as e:
        logger.critical('Could not find\n%s\n\tin\n%s\n\t relative to\n%s' %
                        (optimizer_version_parser, optimizer_version_parser_path, os.getcwd()))
        import traceback
        logger.critical(traceback.format_exc())
        sys.exit(1)

    # Add optimizer specific defaults
    config = optimizer_module_loaded.add_default(config)

    return config

# TODO: remove, does not seem to be useful anymore
# def main():
#     config_fn = sys.argv[1]
#     logger.info('Read config from %s..' % config_fn)
#     logger.info('\twith spearmint..',)
#     parse_config(config_fn, optimizer_module="spearmint")
#     logger.info('\t..finished')
#     logger.info('\twith smac..')
#     parse_config(config_fn, optimizer_module="smac")
#     logger.info('\t..finished')
#     logger.info('\twith tpe..')
#     parse_config(config_fn, optimizer_module="tpe")
#     logger.info('\t..finished')
#     logger.info('..finished')
#
# if __name__ == "__main__":
#     main()


