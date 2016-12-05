import argparse
from collections import defaultdict
import ConfigParser
import cPickle
import os
import StringIO
import sys

import numpy as np

from HPOlib.Experiment import CANDIDATE_STATE, COMPLETE_STATE, \
    INCOMPLETE_STATE, RUNNING_STATE, BROKEN_STATE
import HPOlib.Plotting.plot_util as plot_util
from HPOlib.Locker import Locker


def get_num_runs_for_state(state, trials):
    terminates = 0
    for trial in trials["trials"]:
        for instance_status in trial["instance_status"]:
            if instance_status == state:
                terminates += 1

    return terminates

def get_num_nans(trials):
    nans = 0
    for trial in trials['trials']:
        for instance_result in trial['instance_results']:
            if np.isnan(instance_result):
                nans += 1

    return  nans

def get_instance_durations(trials):
    instance_durations = []
    for trial in trials["trials"]:
        for instance_duration in trial["instance_durations"]:
            if np.isfinite(instance_duration):
                instance_durations.append(instance_duration)

    return instance_durations

def get_total_time(trials):
    total_time = 0
    for starttime, endtime in zip(trials['starttime'], trials['endtime']):
        total_time += endtime - starttime
    if total_time < 1:
        print "Time might be inaccurate"
        total_time = trials['cv_endtime'][-1] - trials['starttime'][0]
    return total_time

def collect_results(directory):
    locker = Locker()
    errors = []
    sio = StringIO.StringIO()
    sio.write("Statistics for %s\n" % directory)
    sio.write("%30s | %6s | %7s/%7s/%7s/%7s/%7s/%7s/%7s/%7s | %10s | %10s\n" %
             ("Optimizer", "Seed", "#conf", "#runs", "#compl", "#incom",
              "#crashs", "#run", "#notrun", "#NaNs", "best", "AvgRunTime"))


    subdirs = os.listdir(directory)
    subdirs.sort()
    results = defaultdict(list)
    # Look for all sub-experiments
    for subdir in subdirs:
        subdir = os.path.join(directory, subdir)
        if os.path.isdir(subdir):
            # Get the experiment pickle...
            for possible_experiment_pickle in os.listdir(subdir):
                # Some simple checks for an experiment pickle
                if possible_experiment_pickle[-4:] == ".pkl" and \
                        possible_experiment_pickle[:-4] in subdir:
                    exp_pkl = os.path.join(subdir, possible_experiment_pickle)
                    locker.lock(exp_pkl)
                    with open(exp_pkl) as fh:
                        try:
                            pkl = cPickle.load(fh)
                        except Exception as e:
                            errors.append(exp_pkl + ' ' +  str(type(e)))
                            continue
                    locker.unlock(exp_pkl)

                    cfg = ConfigParser.ConfigParser()
                    cfg.read(os.path.join(subdir, "config.cfg"))

                    optimizer = pkl["experiment_name"]
                    optimizer = optimizer.split("/")[-1]

                    configurations = len(pkl["trials"])
                    instance_runs = len(pkl["instance_order"])

                    # HPOlib < 0.1 don't have a seed in the config, try to
                    # infer it
                    try:
                        seed = cfg.getint("HPOLIB", "seed")
                    except:
                        seed = subdir.replace(possible_experiment_pickle[:-4],
                                              "")
                        seed = seed[1:].split("_")[0]
                        seed = int(seed)

                    crashs = get_num_runs_for_state(BROKEN_STATE, pkl)
                    candidates = get_num_runs_for_state(CANDIDATE_STATE, pkl)
                    running = get_num_runs_for_state(RUNNING_STATE, pkl)
                    incomplete = get_num_runs_for_state(INCOMPLETE_STATE, pkl)
                    complete = get_num_runs_for_state(COMPLETE_STATE, pkl)
                    nans = get_num_nans(pkl)

                    try:
                        best_performance = plot_util.get_best(pkl)
                        if best_performance == sys.maxint:
                            # There is not at least one evaluated config
                            best_performance = np.nan
                    except Exception as e:
                        errors.append(str(e) + ' ' + exp_pkl)
                        continue

                    instance_durations = get_instance_durations(pkl)
                    mean_instance_durations = np.mean(instance_durations)

                    #total_runtime = get_total_time(pkl)
                    results[optimizer].append([optimizer, int(seed),
                        configurations, instance_runs, complete, incomplete,
                        crashs, running, candidates, nans, best_performance,
                        mean_instance_durations])

    def comparator(left, right):
        if left[0] < right[0]:
            return -1
        elif left[0] > right[0]:
            return 1
        else:
            if left[1] < right[1]:
                return -1
            elif left[1] > right[1]:
                return 1
            else:
                return 0

    for optimizer in sorted(results):
        results[optimizer].sort(cmp=comparator)

        results_for_mean = []
        runtimes_for_mean = []
        total_times_for_mean = []

        for result in results[optimizer]:
            results_for_mean.append(float(result[10]))
            runtimes_for_mean.append(float(result[11]))
            sio.write("%30s | %6d | %7s/%7s/%7s/%7s/%7s/%7s/%7s/%7s | %10f | %10f\n"
                      % (result[0], result[1], result[2], result[3], result[4],
                         result[5], result[6], result[7], result[8], result[9],
                         result[10], result[11]))

        sio.write("#NumRuns %5d | Mean %5f | Std %5f | Best %5f | Median %5f "
                  "| AvgRunTime %10f | AvgTotTime %10f\n"
                  % (len(results_for_mean), np.mean(results_for_mean),
                     np.std(results_for_mean), np.min(results_for_mean),
                     np.median(results_for_mean), np.mean(runtimes_for_mean), np.mean(total_times_for_mean),))

    if len(errors) > 0:
        sio.write("\nCouldn't read the following .pkl files\n")
        for error in errors:
            sio.write(error)
            sio.write("\n")
    return sio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print several statistics "
                                                 "for an experiment directory.")
    parser.add_argument('directory', type=str, default=os.getcwd(), nargs="?",
                        help="Directory for which statistics should be "
                             "printed. Default is the current working "
                             "directory.")
    args = parser.parse_args()
    sio = collect_results(args.directory)
    print sio.getvalue()
