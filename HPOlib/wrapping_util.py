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
import numpy as np
import traceback
import os
import psutil
import re
import signal
from StringIO import StringIO
import sys
import types
import inspect
import config_parser.parse as parse


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


logger = logging.getLogger("HPOlib.wrapping_util")


def nan_mean(arr):
    # First: Sum all finite elements
    arr = np.array(arr)
    res = sum([ele for ele in arr if np.isfinite(ele)])
    num_ele = (arr.size - np.count_nonzero(~np.isfinite(arr)))
    if num_ele == 0:
        return np.nan
    if num_ele != 0 and res == 0:
        return 0
    # Second: divide with number of finite elements
    res /= num_ele
    return res


def nan_std(arr):
    # First: Sum all finite elements
    arr = np.array(arr)
    vals = ([ele for ele in arr if np.isfinite(ele)])
    if len(vals) == 0:
        return np.nan
    try:
        return np.std(vals)
    except:
        return 0


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
    %(type)s: %(message)s\n'''# Skipping the "actual line" item

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


def get_optimizer():
    return "_".join(os.getcwd().split("/")[-1].split("_")[0:-2])


def load_experiment_config_file():
    # Load the config file, this holds information about data, black box fn etc.
    try:
        cfg_filename = "config.cfg"
        config = SafeConfigParser(allow_no_value=True)
        config.read(cfg_filename)
        if not config.has_option("HPOLIB", "is_not_original_config_file"):
            logger.critical("Config file in directory %s seems to be an"
                            " original config which was not created by wrapping.py. "
                            "Are you sure that you are in the right directory?" %
                            os.getcwd())
            sys.exit(1)
        return config
    except IOError as e:
        logger.critical("Could not open config file in directory %s",
                        os.getcwd())
        sys.exit(1)


def get_configuration(experiment_dir, optimizer_version, unknown_arguments, opt_obj,
                      strict=True):
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
        optimizer_version_parser = optimizer_version # + "_parser"
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
            logger.critical('Could not find\n%s\n\tin\n%s\n\t relative to\n%s',
                            optimizer_version_parser,
                            optimizer_version_parser_path, os.getcwd())
            import traceback

            logger.critical(traceback.format_exc())
            sys.exit(1)
        # print(os.path.splitext(optimizer_module_parser.__file__)[0])
        optimizer_config_fn = os.path.splitext(optimizer_module_parser
                                               .__file__)[0] + "Default.cfg"
        if not os.path.exists(optimizer_config_fn):
            logger.critical("No default config %s found", optimizer_config_fn)
            sys.exit(1)
        config_files.append(optimizer_config_fn)

    path_to_current_config = os.path.join(experiment_dir, "config.cfg")
    if os.path.exists(path_to_current_config):
        config_files.append(os.path.join(experiment_dir, "config.cfg"))
    else:
        logger.info("No config file found. Only considering CLI arguments.")

    # TODO Is the config file really the right place to get the allowed keys for a hyperparameter optimizer?
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
        config = opt_obj.manipulate_config(config)

    # Check whether we have all necessary options
    parse.check_config(config)
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
        an argparse.Namespace object containing the parsed values. These are
        packed inside a python list or None if not present.
    Raises:
        an error if an argument from unknown_arguments is not a key in config
    """
    further_possible_command_line_arguments = []
    for section in config.sections():
        for key in config.options(section):
            further_possible_command_line_arguments.append("--" + section +
                                                           ":" + key)

    parser = ArgumentParser()
    for argument in further_possible_command_line_arguments:
        parser.add_argument(argument, nargs="+")

    return parser.parse_args(unknown_arguments)


def config_with_cli_arguments(config, config_overrides):
    arg_dict = vars(config_overrides)
    for section in config.sections():
        for key in config.options(section):
            cli_key = "%s:%s" % (section, key)
            if cli_key in arg_dict:
                value = arg_dict[cli_key]
                if value is not None and not isinstance(value, types.StringTypes):
                    value = " ".join(value)
                config.set(section, key, value)
            else:
                config.remove_option(section, key)
    return config


def save_config_to_file(file_handle, config, write_nones=True):
    for section in config.sections():
        file_handle.write("[" + section + "]\n")
        for key in config.options(section):
            if (config.get(section, key) is None and write_nones) or \
                    config.get(section, key) is not None:
                        file_handle.write("%s = %s\n" % (key, config.get(section, key)))


def kill_processes(sig, processes):
    """Kill a number of processes with a specified signal.

    Parameters
    ----------
    sig : int
        Send this signal to the processes

    processes : list of psutil.Process
    """
    # TODO: somehow wait, until the Experiment pickle is written to disk

    pids_with_commands = []
    for process in processes:
        try:
            pids_with_commands.append((process.pid, process.cmdline()))
        except:
            pass

    logger.debug("Running: %s" % "\n".join(pids_with_commands))
    for process in processes:
        try:
            os.kill(process.pid, sig)
        except Exception as e:
            logger.error(type(e))
            logger.error(e)


class Exit:
    def __init__(self):
        self.exit_flag = False
        self.signal = None

    def true(self):
        self.exit_flag = True

    def false(self):
        self.exit_flag = False

    def set_exit_flag(self, exit):
        self.exit_flag = exit

    def get_exit(self):
        return self.exit_flag

    def signal_callback(self, signal_, frame):
        SIGNALS_TO_NAMES_DICT = dict((getattr(signal, n), n) \
                                     for n in dir(signal) if
                                     n.startswith('SIG') and '_' not in n)

        logger.critical("Received signal %s" % SIGNALS_TO_NAMES_DICT[signal_])
        self.true()
        self.signal = signal_


def remove_param_metadata(params):
    """
    Check whether some params are defined on the Log scale or with a Q value,
    must be marked with "LOG$_{paramname}" or Q[0-999]_$paramname
    LOG_/Q_ will be removed from the paramname
    """
    for para in params:
        new_name = para

        if isinstance(params[para], str):
            params[para] = params[para].strip("'")
            params[para] = params[para].strip('"')
        if "LOG10_" in para:
            pos = para.find("LOG10_")
            new_name = para[0:pos] + para[pos + 6:]
            params[new_name] = np.power(10, float(params[para]))
            del params[para]
        elif "LOG2_" in para:
            pos = para.find("LOG2_")
            new_name = para[0:pos] + para[pos + 5:]
            params[new_name] = np.power(2, float(params[para]))
            del params[para]
        elif "LOG_" in para:
            pos = para.find("LOG_")
            new_name = para[0:pos] + para[pos + 4:]
            params[new_name] = np.exp(float(params[para]))
            del params[para]
            # Check for Q value, returns round(x/q)*q
        m = re.search(r'Q[0-999\.]{1,10}_', para)
        if m is not None:
            pos = new_name.find(m.group(0))
            tmp = new_name[0:pos] + new_name[pos + len(m.group(0)):]
            q = float(m.group(0)[1:-1])
            params[tmp] = round(float(params[new_name]) / q) * q
            del params[new_name]


def flatten_parameter_dict(params):
    """
    TODO: Generalize this, every optimizer should do this by itself
    """
    # Flat nested dicts and shorten lists
    # FORCES SMAC TO FOLLOW SOME CONVENTIONS, e.g.
    # lr_penalty': hp.choice('lr_penalty', [{
    # 'lr_penalty' : 'zero'}, {  | Former this line was 'type' : 'zero'
    # 'lr_penalty' : 'notZero',
    #    'l2_penalty_nz': hp.lognormal('l2_penalty_nz', np.log(1.0e-6), 3.)}])
    # Lists cannot be forwarded via the command line, therefore the list has to
    # be unpacked. ONLY THE FIRST VALUE IS FORWARDED!

    # This class enables us to distinguish tuples, which are parameters from
    # tuples which contain different things...
    class Parameter:
        def __init__(self, pparam):
            self.pparam = pparam

    params_to_check = list()
    params_to_check.append(params)

    new_dict = dict()
    while len(params_to_check) != 0:
        param = params_to_check.pop()

        if type(param) in (list, tuple, np.ndarray):

            for sub_param in param:
                params_to_check.append(sub_param)

        elif isinstance(param, dict):
            params_to_check.extend([Parameter(tmp_param) for tmp_param in
                                    zip(param.keys(), param.values())])

        elif isinstance(param, Parameter):
            key = param.pparam[0]
            value = param.pparam[1]
            if type(value) == dict:
                params_to_check.append(value)
            elif type(value) in (list, tuple, np.ndarray) and \
                    all([type(v) not in (list, tuple, np.ndarray) for v in
                         value]):
                # Spearmint special case, keep only the first element
                # Adding: variable_id = val
                if len(value) == 1:
                    new_dict[key] = value[0]
                else:
                    for v_idx, v in enumerate(value):
                        new_dict[key + "_%s" % v_idx] = v
                        # new_dict[key] = value[0]
            elif type(value) in (list, tuple, np.ndarray):
                for v in value:
                    params_to_check.append(v)
            else:
                new_dict[key] = value

        else:
            raise Exception(
                "Invalid params, cannot be flattened: \n%s." % params)
    params = new_dict
    return params