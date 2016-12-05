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
import time

import HPOlib.wrapping_util as wrapping_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


logger = logging.getLogger("HPOlib.dispatcher.runsolver_wrapper")


def read_runsolver_output(runsolver_output_content):
    """
    Parse the output of the runsolver.

    This method scans for the occurence of several errors and tries to keep
    track of the time wallclock time spent (in case the runsolver crashed and
    did not output a final line).

    It looks for the following errors
    * Mem limit exceeded: sending SIGTERM then SIGKILL
    * Maximum CPU time exceeded: sending SIGTERM then SIGKILL
    * Maximum wall clock time exceeded: sending SIGTERM then SIGKILL
    * Maximum VSize exceeded: sending SIGTERM then SIGKILL
    these errors are inserted into the variable limit exceeded

    In case of normal termination, the runsolver prints a line "Solver just
    ended...". If this does not occur, this function return a pessimistic
    estimation of the wallclock time and the cpu time used so far. This is
    double the time the runsolver stated in its last output.
    """
    error = None
    cpu_time = 0
    wallclock_time = 0
    solver_ended_section = False

    for line in runsolver_output_content:
        # Look for error messages
        if "Maximum CPU time exceeded" in line:
            error = "CPU time exceeded"
        if "Maximum wall clock time exceeded" in line:
            error = "Wall clock time exceeded"
        if "Maximum VSize exceeded" in line:
            error = "VSize exceeded"
        if "Mem limit exceeded" in line:
            error = "Memory exceeded"

        # Keep track of the time used so far
        match = re.search(r"\[(startup\+)(\d*\.\d*)( s)\]", line)
        if match:
            wallclock_time = float(match.group(2))
        match = re.search(r"(Current children cumulated CPU time \(s\)) (\d*\.\d*)", line)
        if match:
            cpu_time = float(match.group(2))

        # Find out if the solver ended and the runsolver prints the final
        # statistices
        if "Solver just ended. Dumping a history of the last" in line:
            solver_ended_section = True
        if re.search(r"Real time \(s\): ", line) and solver_ended_section:
            wallclock_time = float(line.split()[3])
        if re.search(r"^CPU time \(s\): ", line) and solver_ended_section:
            cpu_time = float(line.split()[3])

    # According to the runsolver, the period between two prints is
    # maximally the time the runsolver ran so far
    if not solver_ended_section:
        cpu_time *= 2
        wallclock_time *= 2
        error = "Runsolver probably crashed!"

    return cpu_time, wallclock_time, error


def read_run_instance_output(run_instance_output_string):
    result_string = None
    result_array = None
    assert type(run_instance_output_string) == list, type(
        run_instance_output_string)
    for line in run_instance_output_string:
        match = re.search(r"\s*[Rr]esult\s+(?:([Ff]or)|([oO]f))\s"
                          r"+(?:(HAL)|(ParamILS)|(SMAC)|([tT]his "
                          r"[wW]rapper)|(this algorithm run))", line)
        if match:
            pos = match.start(0)
            result_string = line[pos:].strip()
            result_array = result_string.split()
            result_array = [value.strip(",") for value in result_array]
            break

    # If we do not find a result string, return the last three lines of the
    # run_instance output
    # TODO: there must be some better way to tell the user what happend
    if not result_string and len(run_instance_output_string) >= 3:
        result_string = "".join(run_instance_output_string[-3:])

    return result_array, result_string


def make_command(cfg, fold, param_string, run_instance_output, test=False):
    if test:
        fn = cfg.get("HPOLIB", "test_function")
        fn += " --fold 0 --folds 1"
        # TODO: test if test_function exists! probably in the startup script!
    else:
        fn = cfg.get("HPOLIB", "function")
        fn += " --fold %d --folds %d" % (
            fold, cfg.getint("HPOLIB", "number_cv_folds"))

    python_cmd = fn + " --params %s" % param_string
    # Do not write the actual task in quotes because runsolver will not work
    # then; also we need use-pty and timestamp so that the "solver" output
    # is flushed to the output directory
    cmd = _make_runsolver_command(cfg, run_instance_output)
    cmd += " " + python_cmd
    return cmd


def _make_runsolver_command(cfg, output_filename=None):
    cmd = []
    cmd.append(cfg.get("HPOLIB", "leading_runsolver_info"))
    cmd.append("runsolver")
    if output_filename is not None:
        cmd.extend(["-o",  "\"%s\"" % output_filename])
        cmd.extend(["--timestamp", "--use-pty"])
    else:
        cmd.extend(["-w", "/dev/null/"])
    if cfg.get('HPOLIB', 'runsolver_time_limit'):
        cmd.append("-W")
        cmd.append("%d" % cfg.getint('HPOLIB', 'runsolver_time_limit'))
    if cfg.get('HPOLIB', 'cpu_limit'):
        cmd.append("-C")
        cmd.append("%d" % cfg.getint('HPOLIB', 'cpu_limit'))
    if cfg.get('HPOLIB', 'memory_limit'):
        cmd.append("-M")
        cmd.append("%d" % cfg.getint('HPOLIB', 'memory_limit'))
    delay = 0
    if delay is not None:
        cmd.append("-d")
        cmd.append("%d" % int(delay))
    return " ".join(cmd)


def parse_output(cfg, run_instance_content, runsolver_output_content,
                 measured_time):
    cpu_time, measured_wallclock_time, error = \
        read_runsolver_output(runsolver_output_content)
    result_array, result_string = read_run_instance_output(run_instance_content)

    if not result_array:
        logger.critical("We could not find anything matching our regexp. "
                        "Setting the target algorithm runtime to the time "
                        "measured by the runsolver. Last lines of your "
                        "output:\n%s"
                        % result_string)
        instance_wallclock_time = measured_wallclock_time
        result_array = [None] * 8
    else:
        instance_wallclock_time = float(result_array[4])
    additional_data = " ".join(result_array[8:])

    if cfg.getboolean("HPOLIB", "use_HPOlib_time_measurement") is True:
        # if the runsolver time measurement is available
        if error != "Runsolver probably crashed!":
            wallclock_time = measured_wallclock_time
        # if it is not available, we don't use the guess by read runsolver
        # output, but use the own time measurement
        else:
            wallclock_time = measured_time
    else:
        if cfg.getfloat("HPOLIB", "runtime_on_terminate") <= 0:
            raise ValueError('Configuration error: You cannot use the '
                             'option "use_HPOlib_time_measurement = False'
                             ' without setting "runtime_on_terminate" '
                             'or setting a negative value for '
                             '"runtime_on_terminate".')
        if error != "Runsolver probably crashed!":
            wallclock_time = instance_wallclock_time
        else:
            wallclock_time = cfg.getfloat("HPOLIB", "runtime_on_terminate")

    if result_array is not None and result_array[0] is not None:
        if error is None:
            if result_array[3] != "SAT":
                rval = (cpu_time, wallclock_time, "CRASHED",
                        cfg.getfloat("HPOLIB", "result_on_terminate"),
                        additional_data)
            elif not np.isfinite(float(result_array[6].strip(","))):
                rval = (cpu_time, wallclock_time, "CRASHED",
                        cfg.getfloat("HPOLIB", "result_on_terminate"),
                        additional_data)
            else:
                rval = (cpu_time, wallclock_time, "SAT",
                        float(result_array[6].strip(",")),
                        additional_data)
        else:
            if error != "Runsolver probably crashed!":
                # The runsolver terminated the target algorithm, so there should
                # be no additional run info and we can use the field
                # TODO: there should be the runsolver output file in the
                # additional information!
                rval = (cpu_time, wallclock_time, "CRASHED",
                        cfg.getfloat("HPOLIB", "result_on_terminate"),
                        error + " Please have a look at the runsolver output "
                                "file.")
                # It is useful to have the run_instance_output for debugging
                # os.remove(run_instance_output)
            else:
                if result_array[3] != "SAT":
                    rval = (cpu_time, wallclock_time, "CRASHED",
                            cfg.getfloat("HPOLIB", "result_on_terminate"),
                            additional_data)
                elif not np.isfinite(float(result_array[6].strip(","))):
                    rval = (cpu_time, wallclock_time, "CRASHED",
                            cfg.getfloat("HPOLIB", "result_on_terminate"),
                            additional_data)
                else:
                    rval = (cpu_time, wallclock_time, "SAT",
                            float(result_array[6].strip(",")),
                            additional_data)
    else:
        if error is None:
            # There should really be the runinstance output filename here!
            rval = (cpu_time, wallclock_time, "CRASHED",
                    cfg.getfloat("HPOLIB", "result_on_terminate"),
                    "No result string returned. Please have a look "
                    "at the runinstance output")
        else:
            if error != "Runsolver probably crashed!":
                # The runsolver terminated the target algorithm, so there should
                # be no additional run info and we can use the field
                rval = (cpu_time, wallclock_time, "CRASHED",
                        cfg.getfloat("HPOLIB", "result_on_terminate"),
                        error + " Please have a look at the runsolver output "
                        "file.")
                # It is useful to have the run_instance_output for debugging
                # os.remove(run_instance_output)
            else:
                # There is no result string and the runsolver crashed
                rval = (cpu_time, wallclock_time, "CRASHED",
                        cfg.getfloat("HPOLIB", "result_on_terminate"),
                        "There is no result string and it seems that the "
                        "runsolver crashed. Please have a look at the "
                        "runsolver output file.")
                
    return rval


def store_target_algorithm_calls(path, wallclock_time, result, additional_data,
                                 call):
    # Save the call to the target algorithm
    if not os.path.exists(path):
        fh = open(path, 'w')
        fh.write(",".join(["RESULT", "DURATION", "ADDITIONAL_INFO", "CALL"]) +
                 "\n")
    else:
            fh = open(path, "a")

    try:
        fh.write(",".join([str(result), str(wallclock_time),
                               str(additional_data), call]) + "\n")
    finally:
        fh.close()
    return


def dispatch(cfg, fold, params, test=False):
    param_string = " ".join(["-" + key + " " + str(params[key]) for key in params])
    time_string = wrapping_util.get_time_string()
    run_instance_output = os.path.join(os.getcwd(),
                                       time_string + "_run_instance.out")
    runsolver_output_file = os.path.join(os.getcwd(),
                                         time_string + "_runsolver.out")
    cmd = make_command(cfg, fold, param_string, run_instance_output, test=test)

    starttime = time.time()
    fh = open(runsolver_output_file, "w")
    _run_command_with_shell(cmd, fh)
    fh.close()
    endtime = time.time()

    with open(run_instance_output, "r") as fh:
        run_instance_content = fh.readlines()
    with open(runsolver_output_file, "r") as fh:
        runsolver_output_content = fh.readlines()

    cpu_time, wallclock_time, status, result, additional_data = \
        parse_output(cfg, run_instance_content, runsolver_output_content,
                     measured_time=endtime - starttime)

    if status == "SAT":
        if cfg.getboolean("HPOLIB", "remove_target_algorithm_output"):
            os.remove(run_instance_output)
        os.remove(runsolver_output_file)

    if cfg.getboolean("HPOLIB", "store_target_algorithm_calls"):
        store_target_algorithm_calls(
            path=os.path.join(os.getcwd(), "target_algorithm_calls.csv"),
            wallclock_time=wallclock_time, result=result,
            additional_data=additional_data, call=cmd)

    return additional_data, result, status, wallclock_time


def _run_command_with_shell(command, output):
    logger.info("Calling: %s" % command)
    process = subprocess.Popen(command, stdout=output,
                               stderr=output, shell=True,
                               executable="/bin/bash")
    logger.info(
        "-----------------------RUNNING RUNSOLVER----------------------------")

    process.wait()
