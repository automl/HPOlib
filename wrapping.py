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

import imp
import numpy as np
import os
import subprocess
import sys
import time

import check_before_start
from config_parser.parse import parse_config
import Experiment
import numpy as np
import argparse
from argparse import ArgumentParser

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def calculate_wrapping_overhead(trials):
    wrapping_time = 0
    for times in zip(trials.cv_starttime, trials.cv_endtime):
        wrapping_time += times[1] - times[0]

    wrapping_time = wrapping_time - np.nansum(trials.instance_durations)
    return wrapping_time


def calculate_optimizer_time(trials):
    optimizer_time = []
    time_idx = 0
    
    optimizer_time.append(trials.cv_starttime[0] - trials.starttime[time_idx])
    
    for i in range(len(trials.cv_starttime[1:])):
        if trials.cv_starttime[i + 1] > trials.endtime[time_idx]:
            optimizer_time.append(trials.endtime[time_idx] -
                                  trials.cv_endtime[i])
            time_idx += 1
            optimizer_time.append(trials.cv_starttime[i + 1] -
                                  trials.starttime[time_idx])
        else:
            optimizer_time.append(trials.cv_starttime[i + 1] -
                                  trials.cv_endtime[i])
                
    optimizer_time.append(trials.endtime[time_idx] - trials.cv_endtime[-1])
    trials.optimizer_time = optimizer_time
    
    return np.nansum(optimizer_time)


def use_option_parser():
    """Parse all options which can be handled by the wrapping script.
    Unknown arguments are ignored and returned as a list. It is useful to
    check this list in your program to handle typos etc.

    Returns:
        a tuple. The first element is an argparse.Namespace object,
        the second a list with all unknown arguments.
    """
    parser = ArgumentParser(description="Perform an experiment with the "
                                        "HPOlib")
    parser.add_argument("-o", "--optimizer", action="store", type=str,
                        help="Specify the optimizer name.", required=True)
    parser.add_argument("-p", "--print", action="store_true", dest="printcmd",
                      default=False,
                      help="If set print the command instead of executing it")
    parser.add_argument("-s", "--seed", dest="seed", default=1, type=int,
                      help="Set the seed of the optimizer")
    parser.add_argument("-t", "--title", dest="title", default=None,
                      help="A title for the experiment")
    restore_help_string = "Restore the state from a given directory"
    parser.add_argument("--restore", default=None, dest="restore",
                      help=restore_help_string)

    args = parser.parse_args()
    return args


def main():
    """Start an optimization of the HPOlib. For documentation see the
    comments inside this function and the general HPOlib documentation."""
    experiment_dir = os.getcwd()
    check_before_start._check_zeroth(experiment_dir)
    args = use_option_parser()
    optimizer = args.optimizer

    config_file = os.path.join(experiment_dir, "config.cfg")
    config = parse_config(config_file, allow_no_value=True, optimizer_module=optimizer)

    # _check_runsolver, _check_modules()
    check_before_start._check_first(experiment_dir)

    # build call
    wrapping_dir = os.path.dirname(os.path.realpath(__file__))
    cmd = "export PYTHONPATH=$PYTHONPATH:" + wrapping_dir + "\n"

    # Load optimizer
    try:
        optimizer_module = imp.load_source(optimizer, wrapping_dir + "/" +
                                           optimizer + ".py")
    except (ImportError, IOError):
        print "Optimizer module", optimizer, "not found"
        import traceback
        print traceback.format_exc()
        sys.exit(1)
    optimizer_call, optimizer_dir = optimizer_module.main(config=config,
                                                          options=args,
                                                          experiment_dir=
                                                          experiment_dir)
    cmd += optimizer_call

    # _check_function
    check_before_start._check_second(experiment_dir, optimizer_dir)

    # initialize/reload pickle file
    if args.restore:
        try:
            os.remove(os.path.join(optimizer_dir, optimizer + ".pkl.lock"))
        except OSError:
            pass
    folds = config.getint('DEFAULT', 'numberCV')
    trials = Experiment.Experiment(optimizer_dir, optimizer, folds=folds,
                                   max_wallclock_time=config.get('DEFAULT',
                                                                 'cpu_limit'),
                                   title=args.title)
    trials.optimizer = optimizer

    if args.restore:
        # Old versions did store NaNs instead of the worst possible result for
        # crashed runs in the instance_results. In order to be able to load
        # these files, these NaNs are replaced
        for i, trial in enumerate(trials.instance_order):
            _id, fold = trial
            instance_result = trials.get_trial_from_id(_id)['instance_results'][fold]
            if not np.isfinite(instance_result):
                # Make sure that we do delete the last job which was running but
                # did not finish
                if i == len(trials.instance_order) - 1 and \
                        len(trials.cv_starttime) != len(trials.cv_endtime):
                    # The last job obviously did not finish correctly, do not
                    # replace it
                    pass
                else:
                    trials.get_trial_from_id(_id)['instance_results'][fold] = \
                        config.getfloat('DEFAULT', 'result_on_terminate')
                    # Pretty sure we need this here:
                    trials.get_trial_from_id(_id)['instance_status'][fold] = \
                        Experiment.BROKEN_STATE

        #noinspection PyBroadException
        try:
            restored_runs = optimizer_module.restore(config=config,
                                                     optimizer_dir=optimizer_dir,
                                                     cmd=cmd)
        except:
            print "Could not restore runs for %s" % args.restore
            import traceback
            print traceback.format_exc()
            sys.exit(1)

        print "Restored %d runs" % restored_runs
        trials.remove_all_but_first_runs(restored_runs)
        fh = open(os.path.join(optimizer_dir, optimizer + ".out"), "a")
        fh.write("#" * 80 + "\n" + "Restart! Restored %d runs.\n" % restored_runs)
        fh.close()

        if len(trials.endtime) < len(trials.starttime):
            trials.endtime.append(trials.cv_endtime[-1])
        trials.starttime.append(time.time())
    else:
        trials.starttime.append(time.time())
    #noinspection PyProtectedMember
    trials._save_jobs()
    del trials
    sys.stdout.flush()

    # Run call
    if args.printcmd:
        print cmd
        return 0
    else:
        print cmd
        output_file = os.path.join(optimizer_dir, optimizer + ".out")
        fh = open(output_file, "a")
        process = subprocess.Popen(cmd, stdout=fh, stderr=fh, shell=True,
                                   executable="/bin/bash")
        print "-----------------------RUNNING----------------------------------"
        ret = process.wait()
        trials = Experiment.Experiment(optimizer_dir, optimizer)
        trials.endtime.append(time.time())
        #noinspection PyProtectedMember
        trials._save_jobs()
        # trials.finish_experiment()
        print "Finished with return code: " + str(ret)
        total_time = 0
        print trials.get_best()
        #noinspection PyBroadException
        try:
            for starttime, endtime in zip(trials.starttime, trials.endtime):
                total_time += endtime - starttime
            print "Needed a total of %f seconds" % total_time
            print "The optimizer %s took %f seconds" % \
                  (optimizer, calculate_optimizer_time(trials))
            print "The overhead of the wrapping software is %f seconds" % \
                  (calculate_wrapping_overhead(trials))
            print "The algorithm itself took %f seconds" % \
                  trials.total_wallclock_time
        except:
            pass
        del trials
        return ret

if __name__ == "__main__":
    main()