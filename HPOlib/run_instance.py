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


import cPickle
import imp
from optparse import OptionParser
import os
import re
import sys
import time

import numpy as np

from config_parser.parse import parse_config
import HPOlib.Experiment as Experiment
import HPOlib.wrapping_util as wrapping_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def loadExperimentFile(pickle):
    experiment = Experiment.Experiment(os.path.split(pickle)[0],
                        os.path.split(pickle)[1].split(".")[0])
    return experiment


def remove_param_metadata(params):
    """
    Check whether some params are defined on the Log scale or with a Q value,
    must be marked with "LOG$_{paramname}" or Q[0-999]_$paramname
    LOG/Q will be removed from the paramname
    """
    for para in params:
        new_name = para
        if "LOG10_" in para:
            pos = para.find("LOG10")
            new_name = para[0:pos] + para[pos+6:]
            # new_name = new_name.strip("_")
            params[new_name] = np.power(10, float(params[para]))
            del params[para]
        elif "LOG2" in para:
            pos = para.find("LOG2_")
            new_name = para[0:pos] + para[pos+5:]
            # new_name = new_name.strip("_")
            params[new_name] = np.power(2, float(params[para]))
            del params[para]
        elif "LOG_" in para:
            pos = para.find("LOG")
            new_name = para[0:pos] + para[pos+4:]
            # new_name = new_name.strip("_")
            params[new_name] = np.exp(float(params[para]))
            del params[para]
        #Check for Q value, returns round(x/q)*q
        m = re.search(r'Q[0-999\.]{1,10}_', para)
        if m is not None:
            pos = new_name.find(m.group(0))
            tmp = new_name[0:pos] + new_name[pos+len(m.group(0)):]
            #tmp = tmp.strip("_")
            q = float(m.group(0)[1:-1])
            params[tmp] = round(float(params[new_name])/q)*q
            del params[new_name]


def run_instance(fold, fn, params, cfg):
    # Run instance
    result = np.NaN
    status = "UNSAT"
    folds = cfg.getint('DEFAULT', 'numberCV')

    # Remove additional information about variables and change them accordingly
    remove_param_metadata(params)

    starttime = time.time()
    try:
        result = fn(params, fold=fold, folds=folds)
        status = "SAT"
    except Exception:
        print wrapping_util.format_traceback(sys.exc_info())
        status = "CRASHED"
    duration = time.time() - starttime
    return result, duration, status


def get_trial_index(experiment, fold, params):
    # Check whether we are in a new configuration; This has to check whether
    # the params were already inserted but also whether the fold already run
    # This is checked twice; the instance_result has to be not NaN and the
    # entry in instance_order has to exist
    new = True
    trial_index = np.NaN
    for idx, trial in enumerate(experiment.trials):
        exp = trial['params']
        if exp == params and (idx, fold) not in experiment.instance_order and \
                (experiment.get_trial_from_id(idx)['instance_results'][fold] == np.NaN or \
                 experiment.get_trial_from_id(idx)['instance_results'][fold] !=
                 experiment.get_trial_from_id(idx)['instance_results'][fold]):
            new = False
            trial_index = idx
            break
    if new:
        trial_index = experiment.add_job(params)
    return trial_index


def main():
    # Parse options and arguments
    usage = "This script loads the module specified in the config.cfg" + \
            "file as 'function'.\nIt will either run the main() method or the"+\
            " run_test() method and print the output as: \n" + \
            "'Result for ParamILS: <solved>, <runtime>, <runlength>, " + \
            "<quality>, <seed>, <additional rundata>' \n\n" + \
            "%prog [-f <fold> -c <config.cfg>| -d] <parampickle>\n" + \
            "where <parampickle> is a pickled dict of parameters\n"
    parser = OptionParser(usage=usage)
    parser.add_option("-f", "--fold",
                      action="store", dest="fold", type="int", default=None,
                      help="Number of fold to execute")
    parser.add_option("-c", "--config",
                      action="store", dest="configFile", default=None,
                      help="path to the config.cfg")
    parser.add_option("-s", "--seed",
                      action="store", dest="seed", type="int", default=1,
                      help="Seed to use for calculations")
    parser.add_option("-p", "--pkl",
                      action="store", dest="pkl", default=None,
                      help="Pickle file for bookkeeping")
    parser.add_option("-t", "--test",
                      action = "store_true", dest = "test", default = False,
                      help = "With this flag activated a function run_test " + \
                      "is called instead of main")
    (options, args) = parser.parse_args()

    if (options.configFile is None and options.fold is not None) or \
       (options.configFile is not None and options.fold is None):
        parser.error("If not using -d, fold and config.cfg need " + \
            "to be specified")
    if options.configFile is None and options.fold is None and \
       not options.debug:
        parser.print_usage()

    seed = options.seed

    # Load the pickled param file...
    if len(args) != 1:
        raise Exception("Too many or less paramfiles specified")
    paramfile = args[0]
    print "Paramfile", paramfile, os.path.exists(paramfile)
    if not os.path.exists(paramfile):
        raise Exception("%s is not a file\n" % paramfile)
    fh = open(paramfile, "r")
    params = cPickle.load(fh)
    fh.close()

    # ...then load config.cfg...
    cfg = parse_config(options.configFile, allow_no_value=True)
    if os.path.isabs(cfg.get("DEFAULT", "function")):
        fn_path = cfg.get("DEFAULT", "function")
    else:
        fn_path = cfg.get("DEFAULT", "function")
        fn_path_parent = os.path.join("..", cfg.get("DEFAULT", "function"))
    fn_name, ext = os.path.splitext(os.path.basename(fn_path))
    try:
        fn = imp.load_source(fn_name, fn_path)
    except (ImportError, IOError) as e:
        print "Raised", e, "trying to recover..."
        try:
            print os.path.exists(fn_path_parent)
            fn = imp.load_source(fn_name, fn_path_parent)
        except (ImportError, IOError):
            print os.path.join(fn_path_parent)
            print(("Could not find\n%s\n\tin\n%s\n\tor its parent directory " +
                   "relative to\n%s")
                  % (fn_name, fn_path, os.getcwd()))
            import traceback
            print traceback.format_exc()
            sys.exit(1)

    if options.test:
        fn = fn.run_test
    else:
        fn = fn.main

    # Do bookkeeping before running the instance
    optimizer = "w/o bookkeeping"

    if options.pkl is not None:
        # Don't do bookkeeping if you don't want to
        if not os.path.isfile(options.pkl):
            raise Exception("%s does not exist\n" % options.pkl)
        experiment = loadExperimentFile(options.pkl)
        optimizer = experiment.optimizer

        # This has the side-effect of adding a job
        trial_index = get_trial_index(experiment, options.fold, params)

        experiment.set_one_fold_running(trial_index, options.fold)
        del experiment # release lock

    # TODO: Forward seed to run_instance, because like this it is useless
    result, duration, status = run_instance(options.fold, fn, params, cfg)

    # Do bookkeeping after the run_instance
    if options.pkl is not None:
        # Don't do bookkeeping if we don't want you to do
        if not os.path.isfile(options.pkl):
            raise Exception("%s does not exist\n" % paramfile)
        experiment = loadExperimentFile(options.pkl)
        optimizer = experiment.optimizer
        if status == "SAT":
            experiment.set_one_fold_complete(trial_index, options.fold, result,
                                         duration)
        elif status == "CRASHED":
            result = cfg.getfloat("DEFAULT", "result_on_terminate")
            experiment.set_one_fold_crashed(trial_index, options.fold, result, duration)
        else:
            # TODO: We need a global stopping mechanism
            sys.exit(1)
        del experiment  #release lock

    print "Result for ParamILS: %s, %d, 1, %f, %d, %s for %s" % \
        (status, abs(duration), result, seed, optimizer, str(fn_path))

if __name__ == "__main__":
    main()
