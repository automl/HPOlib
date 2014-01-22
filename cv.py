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
import os
import subprocess
import sys
import time

from importlib import import_module

import numpy as np

from config_parser.parse import parse_config
from Experiment import Experiment
from run_instance import run_instance
from wrapping_util import format_traceback

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__license__ = "3-clause BSD License"
__contact__ = "automl.org"


def loadExperimentFile():
    optimizer = os.getcwd().split("/")[-1].split("_")[0]
    experiment = Experiment(".", optimizer)
    return experiment


def doCV(params, folds=10):
    print "Starting Cross validation"
    sys.stdout.flush()
    optimizer = os.getcwd().split("/")[-1].split("_")[0]

    # Load the config file, this holds information about data, black box fn etc.
    try:
        cfg_filename = "config.cfg"
        cfg = parse_config(cfg_filename, allow_no_value=True)
    except:
        cfg_filename = "../config.cfg"
        cfg = parse_config(cfg_filename, allow_no_value=True)

    # Now evaluate $fold times
    if folds < 1: return np.NaN
    
    # Store the results to hand them back to tpe and spearmint
    results = []

    try:
        print "###\n", params, "\n###\n"
        param_array = ["-" + str(param_name) + " " + str(params[param_name]) \
            for param_name in params]
        param_string = " ".join(param_array)
        
        for fold in range(folds):
            # "Usage: runsolver_wrapper <instancename> " + \
            # "<instancespecificinformation> <cutofftime> <cutofflength> " + \
            # "<seed> <param> <param> <param>"
            # Cutofftime, cutofflength and seed can be safely ignored since they
            # are read in runsolver_wrapper
            runsolver_wrapper_script = "python " + \
                os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                    "runsolver_wrapper.py")
            cmd = "%s %d %s %d %d %d %s" % \
                (runsolver_wrapper_script, fold, optimizer, 0, 0, 0, \
                    param_string)
            print "Calling command:\n", cmd

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, shell=True, executable="/bin/bash")
            print
            print "----------------RUNNING RUNSOLVER_WRAPPER-------------------"
            stdoutdata, stderrdata = process.communicate()
            if stdoutdata:
                print stdoutdata
            if stderrdata:
                print stderrdata

            # Read the runsolver_wrapper output
            lines = stdoutdata.split("\n")
            result_string = None
            for line in lines:
                pos = line.find("Result for ParamILS: SAT")
                if pos != -1:
                    result_string = line[pos:]
                    result_array = result_string.split()
                    results.append(float(result_array[6].strip(",")))
                    break

            if result_string == None:
                raise NotImplementedError("No result string available or result string doesn't contain SAT")

            # If a specified number of runs crashed, quit the whole cross validation
            # in order to save time.
            worst_possible = cfg.getfloat("DEFAULT", "result_on_terminate")
            crashed_runs = np.nansum([0 if res != worst_possible else 1 for res in results])
            if crashed_runs >= cfg.getint("DEFAULT", "max_crash_per_cv"):
                print "Aborting CV because the number of crashes excceds the " \
                      "configured max_crash_per_cv value"
                return worst_possible

            # TODO: Error Handling
        
        assert(len(results) == folds)
        mean = np.mean(results)

    except Exception as e:
        print format_traceback(sys.exc_info()) 
        print "CV failed", sys.exc_info()[0], e
        # status = "CRASHED"
        status = "SAT"
        mean = np.NaN
        
    # Do not return any kind of nan because this would break spearmint
    if not np.isfinite(mean):
        mean = float(cfg.get("DEFAULT", "result_on_terminate"))

    print "Finished CV"
    return mean


def flatten_parameter_dict(params):
    """
    TODO: Generalize this, every optimizer should do this by itself
    """
    # Flat nested dicts and shorten lists
    # FORCES SMAC TO FOLLOW SOME CONVENTIONS, e.g.
    # lr_penalty': hp.choice('lr_penalty', [{
    #    'lr_penalty' : 'zero'}, {  | Former this line was 'type' : 'zero'
    #    'lr_penalty' : 'notZero',
    #    'l2_penalty_nz': hp.lognormal('l2_penalty_nz', np.log(1.0e-6), 3.)}])
    # Lists cannot be forwarded via the command line, therefore the list has to
    # be unpacked. ONLY THE FIRST VALUE IS FORWARDED!

    # This class enables us to distinguish tuples, which are parameters from
    # tuples which contain different things...
    class Parameter:
        def __init__(self, param):
            self.param = param

    params_to_check = list()
    params_to_check.append(params)

    new_dict = dict()
    while len(params_to_check) != 0:
        param = params_to_check.pop()

        if type(param) in (list, tuple, np.ndarray):
            added_literal = False

            for sub_param in param:
                params_to_check.append(sub_param)

        elif isinstance(param, dict):
            params_to_check.extend([Parameter(tmp_param) for tmp_param in zip(param.keys(), param.values())])

        elif isinstance(param, Parameter):
            key = param.param[0]
            value = param.param[1]
            if type(value) == dict:
                params_to_check.append(value)
            elif type(value) in (list, tuple, np.ndarray) and \
                all([type(v) not in (list, tuple, np.ndarray) for v in value]):
                # Spearmint special case, keep only the first element
                new_dict[key] = value[0]
            elif type(value) in (list, tuple, np.ndarray):
                for v in value:
                    params_to_check.append(v)
            else:
                new_dict[key] = value

        else:
            raise Exception("Invalid params, cannot be flattened.")
    params = new_dict
    return params


def main(job_id, params):
    # Forget job_id which comes from spearmint

    # Load the experiment to do time-keeping
    cv_starttime = time.time()
    experiment = loadExperimentFile()
    # experiment.next_iteration()
    experiment.start_cv(cv_starttime)
    del experiment

    # Load the config file, this holds information about data, black box fn etc.
    try:
        cfg_filename = "config.cfg"
        cfg = parse_config(cfg_filename, allow_no_value=True)
    except:
        cfg_filename = "../config.cfg"
        cfg = parse_config(cfg_filename, allow_no_value=True)

    # Load number of folds
    folds = cfg.getint('DEFAULT', 'numberCV')

    params = flatten_parameter_dict(params)

    res = doCV(params, folds=folds)
    print "Result: ", res
    
    # Load the experiment to do time-keeping
    experiment = loadExperimentFile()
    experiment.end_cv(time.time())
    del experiment
    
    return res


def doForTPE(params):
    return main(42, params)


def read_params_from_command_line():
    params = dict()
    param_list = sys.argv[1:]
    for idx, i in enumerate(param_list[0::2]):
        key = param_list[idx*2]
        value = param_list[idx*2+1]

        if key[0] != "-":
            raise ValueError("Expected a parameter name that start with '-' at "
                             "%d position, instead got %s" % (idx*2, key))
        else:
            key = key[1:]

        if value[0] != "'" or value[-1] != "'":
            raise ValueError("Expected parameter value %d to be inside single"
                             "quotation marks." % (idx*2 + 1))
        else:
            value = value.strip("'")
        params[key] = value
    return params


if __name__ == "__main__":
    params = read_params_from_command_line()
    print main(42, params)
