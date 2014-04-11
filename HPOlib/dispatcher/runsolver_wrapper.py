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
import numpy as np
import os
import re
import subprocess

import HPOlib.wrapping_util as wrapping_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
hpolib_logger = logging.getLogger("HPOlib")
hpolib_logger.setLevel(logging.INFO)
logger = logging.getLogger("HPOlib.dispatcher.runsolver_wrapper")


def read_runsolver_output(runsolver_output_file):
    """
    Read the runsolver output, watch out for
    Mem limit exceeded: sending SIGTERM then SIGKILL
    Maximum CPU time exceeded: sending SIGTERM then SIGKILL
    Maximum wall clock time exceeded: sending SIGTERM then SIGKILL
    Maximum VSize exceeded: sending SIGTERM then SIGKILL
    In case one of these happened, send back the worst possible result
    as specified in the config
    """
    limit_exceeded = None
    cpu_time = None
    wallclock_time = None
    solver_ended_section = False
    with open(runsolver_output_file, 'r') as f:
        runsolver_output_content = f.readlines()
        for line in runsolver_output_content:
            if "Maximum CPU time exceeded" in line:
                limit_exceeded = "CPU time exceeded"
            if "Maximum wall clock time exceeded" in line:
                limit_exceeded = "Wall clock time exceeded"
            if "Maximum VSize exceeded" in line:
                limit_exceeded = "VSize exceeded"
            if "Mem limit exceeded" in line:
                limit_exceeded = "Memory exceeded"
            if "Solver just ended. Dumping a history of the last" in line:
                solver_ended_section = True
            if re.search(r"Real time \(s\): ", line) and solver_ended_section:
                wallclock_time = float(line.split()[3])
            if re.search(r"^CPU time \(s\): ", line) and solver_ended_section:
                cpu_time = float(line.split()[3])
    return cpu_time, wallclock_time, limit_exceeded


def read_run_instance_output(run_instance_output):
    """
    Read the run_instance output file
    """
    result_array = None
    fh = open(run_instance_output, "r")
    run_instance_content = fh.readlines()
    fh.close()
    result_string = None
    for line in run_instance_content:
        match = re.search(r"\s*[Rr]esult\s+(?:([Ff]or)|([oO]f))\s"
                          r"+(?:(HAL)|(ParamILS)|(SMAC)|([tT]his [wW]rapper))",
                          line)
        if match:
            pos = match.start(0)
            result_string = line[pos:].strip()
            result_array = result_string.split()
            result_array = [value.strip(",") for value in result_array]
            break

    # If we do not find a result string, return the last three lines of the
    # run_instance output
    # TODO: there must be some better way to tell the user what happend
    if not result_string and len(run_instance_content) >= 3:
        result_string = "".join(run_instance_content[-3:])

    return result_array, result_string


def make_command(cfg, fold, param_string, run_instance_output):
    fn = cfg.get("HPOLIB", "function")
    python_cmd = cfg.get("HPOLIB", "leading_algo_info") + " " + fn
    python_cmd += " --fold %d --folds %d --params %s" % (fold, cfg.getint(
        "HPOLIB", "number_cv_folds"), param_string)
    # Do not write the actual task in quotes because runsolver will not work
    # then; also we need use-pty and timestamp so that the "solver" output
    # is flushed to the output directory
    delay = 0
    cmd = cfg.get("HPOLIB", "leading_runsolver_info")
    cmd += " runsolver -o %s --timestamp --use-pty" % run_instance_output
    if cfg.get('HPOLIB', 'runsolver_time_limit'):
        cmd += " -W %d" % cfg.getint('HPOLIB', 'runsolver_time_limit')
    if cfg.get('HPOLIB', 'cpu_limit'):
        cmd += " -C %d" % cfg.getint('HPOLIB', 'cpu_limit')
    if cfg.get('HPOLIB', 'memory_limit'):
        cmd += " -M %d" % cfg.getint('HPOLIB', 'memory_limit')
    if delay is not None:
        cmd += " -d %d" % int(delay)
    cmd += " " + python_cmd
    return cmd


def parse_output_files(cfg, run_instance_output, runsolver_output_file):
    cpu_time, measured_wallclock_time, error = read_runsolver_output(
        runsolver_output_file)
    result_array, result_string = read_run_instance_output(run_instance_output)
    if not result_array:
        logger.critical("We could not find anything matching our regexp. "
                        "Setting the target algorithm runtime to the time "
                        "measured by the runsolver. Last lines of your "
                        "output:\n%s"
                        % result_string)
        instance_wallclock_time = measured_wallclock_time
        result_array = [None]*7
    else:
        instance_wallclock_time = float(result_array[4])

    if cfg.getboolean("HPOLIB", "use_own_time_measurement") is True:
        wallclock_time = measured_wallclock_time
    else:
        wallclock_time = instance_wallclock_time

    if error is None and result_string is None:
        additional_data = "No result string returned. Please have a look " \
                          "at " + run_instance_output
        rval = (cpu_time, wallclock_time, "CRASHED", cfg.getfloat("HPOLIB",
                                                                  "result_on_terminate"), additional_data)
        os.remove(runsolver_output_file)

    elif error is None and result_array[3] != "SAT":
        additional_data = "Please have a look at " + run_instance_output + "." \
                                                                           "The output status is not \"SAT\""
        rval = (cpu_time, wallclock_time, "CRASHED", cfg.getfloat("HPOLIB",
                                                                  "result_on_terminate"), additional_data)
        os.remove(runsolver_output_file)

    elif error is None and not np.isfinite(float(result_array[6].strip(","))):
        additional_data = "Response value is not finite. Please have a look " \
                          "at " + run_instance_output
        rval = (cpu_time, wallclock_time, "UNSAT", cfg.getfloat("HPOLIB",
                                                                "result_on_terminate"), additional_data)

    elif error is None:
        if cfg.getboolean("HPOLIB", "remove_target_algorithm_output"):
            os.remove(run_instance_output)
        os.remove(runsolver_output_file)
        rval = (cpu_time, wallclock_time, "SAT", float(result_array[6].strip(",")),
                cfg.get("HPOLIB", "function"))

    else:
        rval = (cpu_time, wallclock_time, "CRASHED", cfg.getfloat("HPOLIB",
                                                                  "result_on_terminate"),
                error + " Please have a look at " +
                runsolver_output_file)
        # It is useful to have the run_instance_output for debugging
        # os.remove(run_instance_output)

    return rval


def dispatch(cfg, fold, params):
    param_string = " ".join([key + " " + str(params[key]) for key in params])
    time_string = wrapping_util.get_time_string()
    run_instance_output = os.path.join(os.getcwd(),
                                       time_string + "_run_instance.out")
    runsolver_output_file = os.path.join(os.getcwd(),
                                         time_string + "_runsolver.out")
    cmd = make_command(cfg, fold, param_string, run_instance_output)
    fh = open(runsolver_output_file, "w")

    logger.debug("Calling: %s" % cmd)
    process = subprocess.Popen(cmd, stdout=fh,
                               stderr=fh, shell=True, executable="/bin/bash")
    logger.info(
        "-----------------------RUNNING RUNSOLVER----------------------------")
    process.wait()
    fh.close()
    cpu_time, wallclock_time, status, result, additional_data = \
        parse_output_files(cfg, run_instance_output, runsolver_output_file)
    return additional_data, result, status, wallclock_time
