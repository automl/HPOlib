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

from argparse import ArgumentParser
from ConfigParser import SafeConfigParser
import datetime
import logging
import imp
import math
import traceback
import os
from StringIO import StringIO
import sys

import config_parser.parse as parse

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


logger = logging.getLogger("HPOlib.wrapping_util")


def get_time_string():
    local_time = datetime.datetime.today()
    time_string = "%d-%d-%d--%d-%d-%d-%d" % (local_time.year, local_time.month,
                  local_time.day, local_time.hour, local_time.minute,
                  local_time.second, local_time.microsecond)
    return time_string


def float_eq(a, b, eps=0.0001):
    return abs(math.log(a+1) - math.log(b+1)) <= eps


def format_traceback(exc_info):
    traceback_template = '''Traceback (most recent call last):
    File "%(filename)s", line %(lineno)s, in %(name)s
    %(type)s: %(message)s\n''' # Skipping the "actual line" item

    # Also note: we don't walk all the way through the frame stack in this example
    # see hg.python.org/cpython/file/8dffb76faacc/Lib/traceback.py#l280
    # (Imagine if the 1/0, below, were replaced by a call to test() which did 1/0.)

    exc_type, exc_value, exc_traceback = exc_info  # most recent (if any) by default

    '''
    Reason this _can_ be bad: If an (unhandled) exception happens AFTER this,
    or if we do not delete the labels on (not much) older versions of Py, the
    reference we created can linger.

    traceback.format_exc/print_exc do this very thing, BUT note this creates a
    temp scope within the function.
    '''

    traceback_details = {
                         'filename': exc_traceback.tb_frame.f_code.co_filename,
                         'lineno'  : exc_traceback.tb_lineno,
                         'name'    : exc_traceback.tb_frame.f_code.co_name,
                         'type'    : exc_type.__name__,
                         'message' : exc_value.message, # or see traceback._some_str()
                        }

    del(exc_type, exc_value, exc_traceback) # So we don't leave our local labels/objects dangling
    # This still isn't "completely safe", though!
    # "Best (recommended) practice: replace all exc_type, exc_value, exc_traceback
    # with sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]

    return "\n" + traceback.format_exc() + "\n\n" + traceback_template % traceback_details + "\n\n"


def load_experiment_config_file():
    # Load the config file, this holds information about data, black box fn etc.
    try:
        cfg_filename = "config.cfg"
        config = SafeConfigParser(allow_no_value=True)
        config.read(cfg_filename)
        if not config.has_option("HPOLIB", "is_not_original_config_file"):
            logger.critical("Config file in directory %s seems to be an"
                " original config which was not created by wrapping.py. "
                "Please contact the HPOlib maintainer to solve this issue.")
            sys.exit(1)
        return config
    except IOError as e:
        logger.critical("Could not open config file in directory %s" %
                        os.getcwd())
        sys.exit(1)


def get_configuration(experiment_dir, optimizer_version, unknown_arguments):
    """How the configuration is parsed:
    1. The command line is already parsed, we have a list of unknown arguments
    2. Assemble list of all config files related to this experiment, these are
        1. the general default config
        2. the optimizer default config
        3. the experiment configuration
    3. call parse_config to find out all allowed keys. Ignore the values.
    4. call parse_config_values_from_unknown_arguments to get parameters for
       the optimization software from the command line in a config object
    5. call parse_config again. Read config files in the order specified
       under point 2. Then read the command line arguments.
    """
    config_files = list()
    general_default = os.path.join(os.path.dirname(parse.__file__),
                                   "generalDefault.cfg")
    config_files.append(general_default)
    # Load optimizer parsing module
    if optimizer_version != "" and optimizer_version is not None:
        optimizer_version_parser = optimizer_version + "_parser"
        # If optimizer_version_parser is an absolute path, the path of
        # __file__ will be ignored
        optimizer_version_parser_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            optimizer_version_parser + ".py")
        # noinspection PyBroadException,PyUnusedLocal
        try:
            optimizer_module_parser = imp.load_source(optimizer_version_parser,
                                                      optimizer_version_parser_path)
        except Exception as e:
            logger.critical('Could not find\n%s\n\tin\n%s\n\t relative to\n%s' %
                            (optimizer_version_parser,
                             optimizer_version_parser_path, os.getcwd()))
            import traceback

            logger.critical(traceback.format_exc())
            sys.exit(1)

        optimizer_config_fn = os.path.splitext(optimizer_module_parser
                                               .__file__)[0][
                              :-7] + "Default.cfg"
        if not os.path.exists(optimizer_config_fn):
            logger.critical("No default config %s found" % optimizer_config_fn)
            sys.exit(1)
        config_files.append(optimizer_config_fn)

    config_files.append(os.path.join(experiment_dir, "config.cfg"))
    # Is the config file really the right place to get the allowed keys for a
    #  hyperparameter optimizer?
    config = parse.parse_config(config_files, allow_no_value=True,
                                optimizer_version=optimizer_version)

    if unknown_arguments is not None:
        # Parse command line arguments
        config_overrides = parse_config_values_from_unknown_arguments(
            unknown_arguments, config)
        # Remove every option from the configuration which was not present on the
        #  command line
        config = config_with_cli_arguments(config, config_overrides)
        # Transform to a string so we can pass it to parse_config
        fh = StringIO()
        save_config_to_file(fh, config, write_nones=False)
        fh.seek(0)
    else:
        fh = None
    config = parse.parse_config(config_files, allow_no_value=True,
                                optimizer_version=optimizer_version,
                                cli_values=fh)
    if fh is not None:
        fh.close()
    if optimizer_version != "" and optimizer_version is not None:
        config = optimizer_module_parser.manipulate_config(config)
    return config


def parse_config_values_from_unknown_arguments(unknown_arguments, config):
    """Parse values not recognized by use_option_parser for config values.

    Args:
       unknown_arguments: A list of arguments which is returned by
           use_option_parser. It should only contain keys which are allowed
           in config files.
        config: A ConfigParser.SafeConfigParser object which contains all keys
           should be parsed from the unknown_arguments list.
    Returns:
        an argparse.Namespace object containing the parsed values.
    Raises:
        an error if an argument from unknown_arguments is not a key in config
    """
    further_possible_command_line_arguments = []
    for section in config.sections():
        for key in config.options(section):
            further_possible_command_line_arguments.append("--" + section +
                                                           ":" + key)
    for key in config.defaults():
        further_possible_command_line_arguments.append("--DEFAULT:" + key)

    parser = ArgumentParser()
    for argument in further_possible_command_line_arguments:
        parser.add_argument(argument)

    return parser.parse_args(unknown_arguments)


def config_with_cli_arguments(config, config_overrides):
    arg_dict = vars(config_overrides)
    for section in config.sections():
        for key in config.options(section):
            cli_key = "%s:%s" % (section, key)
            if cli_key in arg_dict:
                config.set(section, key, arg_dict[cli_key])
            else:
                config.remove_option(section, key)
    return config


def save_config_to_file(file_handle, config, write_nones=True):
    if len(config.defaults()) > 0:
        file_handle.write("[DEFAULT]\n")
        for key in config.defaults():
            if (config.get("DEFAULT", key) is None and write_nones) or \
                    config.get("DEFAULT", key) is not None:
                file_handle.write(key + " = " + config.get("DEFAULT", key) + "\n")

    for section in config.sections():
        file_handle.write("[" + section + "]\n")
        for key in config.options(section):
            if (config.get(section, key) is None and write_nones) or \
                config.get(section, key) is not None:
                    file_handle.write("%s = %s\n" % (key, config.get(section, key)))