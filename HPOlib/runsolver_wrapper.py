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

#!/usr/bin/env python

import cPickle
import numpy as np
import os
import sys
import time
import subprocess

import Experiment
import wrapping_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


# TODO: This should be in a util function sometime in the future
#       Is duplicated in cv.py
def get_optimizer():
    return "_".join(os.getcwd().split("/")[-1].split("_")[0:-2])


def load_experiment_file(seed):
    optimizer = get_optimizer()
    experiment = Experiment.Experiment(".", optimizer)
    return experiment


def replace_nan_in_last_trial(replacement):
    exp = load_experiment_file()
    # TODO: Make a better test to find out if we are supposed to write to exp.
    if len(exp.trials) == 0 and len(exp.instance_order) == 0:
        del exp
    else:
        _id, fold = exp.instance_order[-1]
        if not np.isfinite(exp.trials[_id]['instance_results'][fold]):
            exp.trials[_id]['instance_results'][fold] = replacement
            exp.trials[_id]['instance_status'][fold] == Experiment.BROKEN_STATE
        elif exp.trials[_id]['instance_results'][fold] == replacement:
            pass
        else:
            raise Exception("Trying to replace %f with %f" %
                            (exp.trials[_id]['instance_results'][fold], replacement))
        exp._save_jobs()
        del exp  # This also saves the experiment


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
    with open(runsolver_output_file, 'r') as f:
        runsolver_output_content = f.readlines()
        limit_exceeded = None
        error_time = 0
        for line in runsolver_output_content:
            if "Maximum CPU time exceeded" in line:
                limit_exceeded = "CPU time exceeded"
            if "Maximum wall clock time exceeded" in line:
                limit_exceeded = "Wall clock time exceeded"
            if "Maximum VSize exceeded" in line:
                limit_exceeded = "VSize exceeded"
            if "Mem limit exceeded" in line:
                limit_exceeded = "Memory exceeded"
            if "Real time (s): " in line:
                # If the solver terminated, get the wallclock time of termination
                error_time = float(line.split()[3])
        return limit_exceeded, error_time


def read_run_instance_output(run_instance_output):
    """
    Read the run_instance output file
    """
    result_string = None
    result_array = None
    fh = open(run_instance_output, "r")
    run_instance_content = fh.readlines()
    fh.close()
    result_string = None
    for line in run_instance_content:
        pos = line.find("Result for ParamILS:")
        if pos != -1:
            result_string = line[pos:]
            result_array = result_string.split()
            break

    return result_array, result_string


def main():
    # Parse options and arguments
    usage = "This script pickles the params and runs the runsolver with " +\
            "run_instance and extract the output for the optimizer \n" + \
            "The output is printed im a SMACish way: \n\n" + \
            "'Result for ParamILS: <solved>, <runtime>, <runlength>, " + \
            "<quality>, <seed>, <additional rundata>' \n\n" + \
            "Usage: runsolver_wrapper <instancename> " + \
            "<instancespecificinformation> <cutofftime> <cutofflength> " + \
            "<seed> <param> <param> <param>\n" + \
            "<instancename> might be the optimizer name if not" + \
            " called by smac\n"
    if len(sys.argv) < 7:
        sys.stdout.write(usage)
        exit(1)

    # Then get some information for run_instance
    fold = int(sys.argv[1])
    seed = int(sys.argv[5])

    optimizer = get_optimizer()
    
    # This has to be done here for SMAC, since smac does not call cv.py
    if 'smac' in optimizer:
        cv_starttime = time.time()
        experiment = load_experiment_file(seed)
        experiment.start_cv(cv_starttime)
        del experiment

    # Again we need to find the config.cfg
    cfg = cfg = wrapping_util.load_experiment_config_file()
    cfg_filename = "config.cfg"

    # Ignore smac cutofftime
    time_limit = cfg.getint('DEFAULT', 'runsolver_time_limit')
    memory_limit = cfg.getint('DEFAULT', 'memory_limit')
    cpu_limit = cfg.getint('DEFAULT', 'cpu_limit')

    # Now build param dict
    param_list = sys.argv[6:]
    params = dict()

    for idx, i in enumerate(param_list[0::2]):
        params[param_list[idx*2][1:]] = (param_list[idx*2+1].strip("'"))
    # TODO: remove overhead of pickling the param dict
    time_string = wrapping_util.get_time_string()
    params_filename = os.path.join(os.getcwd(), "params" + time_string)
    params_fh = open(params_filename, 'w')
    print 'Pickling param dict', params_filename, params
    cPickle.dump(params, params_fh)
    params_fh.close()

    # Timestamp and use-pty are options are used so that output of the "solver"
    # is flushed to the solver output file directly
    run_instance_output = os.path.join(os.getcwd(), time_string + "_run_instance.out")
    sys.stdout.write("Using optimizer: " + str(optimizer) + "\n")

    python_cmd = cfg.get("DEFAULT", "leading_algo_info") + " python " + \
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     "run_instance.py") + \
        " --fold %d --config %s --seed %d --pkl %s" % \
        (fold, cfg_filename, seed, optimizer + ".pkl")

    # Do not write the actual task in quotes because runsolver will not work
    # then
    delay = 0
    cmd = cfg.get("DEFAULT", "leading_runsolver_info") + \
        " runsolver -o %s --timestamp --use-pty -W %d -C %d -M %d -d %d %s %s" \
        % (run_instance_output, time_limit, cpu_limit, memory_limit, delay,
        python_cmd, params_filename)

    runsolver_output_file = os.path.join(os.getcwd(),
                                         time_string + "_runsolver.out")
    fh = open(runsolver_output_file, "w")
    process = subprocess.Popen(cmd, stdout=fh,
                               stderr=fh, shell=True, executable="/bin/bash")
                                    
    print
    print cmd
    print "-----------------------RUNNING RUNSOLVER----------------------------"
    process.wait()
    fh.close()

    limit_exceeded, error_time = read_runsolver_output(runsolver_output_file)
    # <solved>, <runtime>, <runlength>, <quality>, <seed>, <additional rundata>
    error_string = "Result for ParamILS: %s, %d, 0, %f, %d, %s"

    result_array, result_string = read_run_instance_output(run_instance_output)

    # Write the SMACish output to the command line and remove temporary files
    # Additionaly, we also have to replace the NaN-value in the result file by
    # the worst possible value
    if result_array is not None and result_array[3].strip(",") not in ("SAT", "CRASHED"):
        raise Exception("Unknown return status %s" % result_array[3].strip(","))

    if (limit_exceeded is None and result_string is None) or\
            (result_string is not None and "CRASHED" in result_string and not limit_exceeded):
        replace_nan_in_last_trial(cfg.getfloat("DEFAULT", "result_on_terminate"))
        return_string = error_string % ("SAT", error_time,
                              cfg.getfloat('DEFAULT', "result_on_terminate"),
                              seed,
                              "Please have a look at " + run_instance_output)
        print return_string
        os.remove(runsolver_output_file)

    elif limit_exceeded is None:
        result_float = float(result_array[6].strip(","))
        if not np.isfinite(result_float):
            replace_nan_in_last_trial()
            experiment = load_experiment_file(seed)
            last_experiment = experiment.instance_order[-1]
            assert last_experiment is not None
            experiment.instance_results[last_experiment[0]][last_experiment[1]]\
                = cfg.getfloat('DEFAULT', "result_on_terminate")
            del experiment  # This also saves the experiment
            
            result_array[6] = (cfg.getfloat("DEFAULT", "result_on_terminate")) + ","
            result_string = " ".join(result_array)
        else:
            # Remove the run_instance_output only if there is a valid result
            os.remove(run_instance_output)
            os.remove(runsolver_output_file)
        return_string = result_string
        print result_string
        
    else:
        return_string = error_string % ("SAT", error_time, cfg.getfloat
                             ('DEFAULT', "result_on_terminate"), seed,
                              limit_exceeded)
        print return_string
        # It is useful to have the run_instance_output for debugging
        #os.remove(run_instance_output)

    os.remove(params_filename)

    # Remove param pkl and runsolver files
    #os.remove(run_instance_output)
    #os.remove(runsolver_output_file)
    #os.remove(params_filename)
    if 'smac' in optimizer:
        experiment = load_experiment_file(seed)
        experiment.end_cv(time.time())
        del experiment
    return return_string
        
if __name__ == "__main__":
    main()
