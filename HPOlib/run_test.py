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
from optparse import OptionParser
import os
import subprocess

import numpy as np

from config_parser.parse import parse_config
from HPOlib.Experiment import Experiment
import HPOlib.wrapping_util as wrapping_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def loadExperimentFile():
    optimizer = os.getcwd().split("/")[-1].split("_")[0]
    experiment = Experiment(".", optimizer)
    return experiment


def main():
    # Parse options and arguments
    usage = "Coming soon"
    parser = OptionParser(usage=usage)
    (options, args) = parser.parse_args()
    
    if len(args) != 1:
        raise Exception("You must specify a directory.")

    # Then load config.cfg and function
    cfg = parse_config("config.cfg", allow_no_value=True)
    module = cfg.get("DEFAULT", "function")

    os.chdir(args[0])

    # Load the experiment pickle and extract best configuration according to the
    # validation loss
    experiment = loadExperimentFile()
    optimizer = experiment.optimizer
    if optimizer == "smac":
        tmp_results = experiment.instance_results
        tmp_results = np.ma.masked_array(tmp_results, np.isnan(tmp_results))
        results = np.mean(tmp_results, axis=1)
    else:
        tmp_results = experiment.results
        tmp_results[np.isnan(tmp_results)] = cfg.get('DEFAULT', 'result_on_terminate')
        results = tmp_results
    best_config_idx = np.argmin(results)
    print "Found best config #%d with validation loss %f" % (best_config_idx, results[best_config_idx])
    params = experiment.params[best_config_idx]   
    del experiment #release lock
    
    # Create a param pickle
    time_string = wrapping_util.get_time_string()
    params_filename = os.path.join(os.getcwd(), "params" + time_string)
    params_fh = open(params_filename, 'w')
    print "Pickling param dict", params_filename, params
    cPickle.dump(params, params_fh)
    params_fh.close()
    
    #Call run_instance.py
    fh = open(args[0][0:-1] + "_test_run.out", "w")
    leading_runsolver_info = cfg.get('DEFAULT', 'leading_runsolver_info')
    leading_algo_info = cfg.get('DEFAULT', 'leading_algo_info')
    leading_algo_info = "optirun"
    cmd = "%s %s python %s --fold 1 --config ../config.cfg --test %s" % (leading_runsolver_info, leading_algo_info, os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/run_instance.py"), params_filename)
    process = subprocess.Popen(cmd, stdout=fh, stderr=fh,
                               shell=True, executable="/bin/bash")
                                    
    print
    print cmd
    print "-----------------------RUNNING TEST----------------------------"
    ret = process.wait()
    fh.close()

    os.remove(params_filename)
    print ret

if __name__ == "__main__":
    main()
    #test()
