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
import itertools
import logging
import os
import numpy as np
import sys

import HPOlib.wrapping_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
hpolib_logger = logging.getLogger("HPOlib")
hpolib_logger.setLevel(logging.INFO)
logger = logging.getLogger("HPOlib.plot_util")

# A super-simple cache for unpickled objects...
cache = dict()


def get_empty_iterator():
    return itertools.cycle([None])


def get_plot_markers():
    return itertools.cycle(['o', 's', 'x', '^', 'p', 'v', '>', '<', '8', '*',
                            '+', 'D'])


def get_plot_linestyles():
    return itertools.cycle(['-', '--', '-.', ':', ])


def get_single_linestyle():
    return itertools.cycle(['-'])


def get_plot_colors():
    # color brewer, 2nd qualitative 9 color scheme (http://colorbrewer2.org/)
    return itertools.cycle(["#e41a1c",    # Red
                            "#377eb8",    # Blue
                            "#4daf4a",    # Green
                            "#984ea3",    # Purple
                            "#ff7f00",    # Orange
                            "#ffff33",    # Yellow
                            "#a65628",    # Brown
                            "#f781bf",    # Pink
                            "#999999"])   # Grey


def load_pickles(name_list, pkl_list):
    pickles = dict()
    for i in range(len(name_list)):
        key = name_list[i][0]
        pickles[key] = list()

        for pkl in pkl_list[i]:
            if cache.get(pkl) is None:
                fh = open(pkl)
                pickles[key].append(cPickle.load(fh))
                fh.close()
                cache[pkl] = pickles[key][-1]
            else:
                pickles[key].append(cache.get(pkl))
    return pickles


def get_pkl_and_name_list(argument_list):
    name_list = list()
    pkl_list = list()
    now_data = False
    for i in range(len(argument_list)):
        if not ".pkl" in argument_list[i] and now_data:
            raise ValueError("You need at least on .pkl file per Experiment, %s has none" % name_list[-1])
        elif not ".pkl" in argument_list[i] and not now_data:
            # print "Adding", argument_list[i]
            name_list.append([argument_list[i], 0])
            pkl_list.append(list())
            now_data = True
            continue
        else:
            if os.path.exists(argument_list[i]):
                now_data = False
                name_list[-1][1] += 1
                pkl_list[-1].append(argument_list[i])
            else:
                raise ValueError("%s is not a valid file" % argument_list[i])
    if now_data:
        raise ValueError("You need at least one .pkl file per Experiment,  %s has none" % name_list[-1])
    return pkl_list, name_list


def get_best_dict(name_list, pickles, cut=sys.maxint):
    """
    Get the best values of many experiments.

    Input
    * name_list: A list with of tuples of kind (optimizer_name, num_pickles)
    * pickles: A dictionary with a list of  all pickle files for an
        optimizer_name
    * cut: How many iterations should be considered

    Returns:
    * best_dict: A dictionary with a list of the best response value for every
        optimizer
    * idx_dict: A dictionary with a list the number of iterations needed to
        find the optimum
    * keys: A list with optimizer names.

    """
    best_dict = dict()
    idx_dict = dict()
    keys = list()
    for i in range(len(name_list)):
        keys.append(name_list[i][0])
        best_dict[name_list[i][0]] = list()
        idx_dict[name_list[i][0]] = list()
        for pkl in pickles[name_list[i][0]]:
            best, idx = get_best_value_and_index(pkl, cut)
            best_dict[name_list[i][0]].append(best)
            idx_dict[name_list[i][0]].append(idx)
    return best_dict, idx_dict, keys


def fill_trajectories(trace_list, times_list):
    """ Each trajectory must have the exact same number of entries
    and timestamps

    trace_list: list of n lists with y values
    times_list: list of n lists with x values

    returns a list of n lists where for each x value and each y-list an
    entry exists. time list will always start at 0

    Example:

    trace_list = [[5,3], [5,2,1]]
    times_list = [[1,2], [1,3,5]]

    returns:
    trajectories = [[5, 5, 3, 3, 3], [5, 5, 5, 2, 1]]
    times = [0,1,2,3,5]
    """
    # We need to define the max value =
    # what is measured before the first evaluation
    max_value = np.max([np.max(ls) for ls in trace_list])

    for idx in range(len(trace_list)):
        assert len(trace_list[idx]) == len(times_list[idx]), \
            "%d != %d" % (len(trace_list[idx]), len(times_list[idx]))

    number_exp = len(trace_list)
    new_trajectories = list()
    new_times = list()
    for i in range(number_exp):
        new_trajectories.append(list())
        new_times.append(list())
    # noinspection PyUnusedLocal
    counter = [1 for i in range(number_exp)]
    finish = False

    # We need to insert the max values in the beginning
    # and the min values in the end
    for i in range(number_exp):
        trace_list[i].insert(0, max_value)
        trace_list[i].append(np.min(trace_list[i]))
        times_list[i].insert(0, 0)
        times_list[i].append(sys.maxint)

    # Add all possible time values
    while not finish:
        min_idx = np.argmin([times_list[idx][counter[idx]]
                             for idx in range(number_exp)])
        counter[min_idx] += 1
        for idx in range(number_exp):
            new_times[idx].append(times_list[min_idx][counter[min_idx] - 1])
            new_trajectories[idx].append(trace_list[idx][counter[idx] - 1])
        # Check if we're finished
        for i in range(number_exp):
            finish = True
            if counter[i] < len(trace_list[i]) - 1:
                finish = False
                break

    times = new_times
    trajectories = new_trajectories
    tmp_times = list()

    # Sanitize lists and delete double entries
    for i in range(number_exp):
        tmp_times = list()
        tmp_traj = list()
        for t in range(len(times[i]) - 1):
            if times[i][t + 1] != times[i][t] and not np.isnan(times[i][t]):
                tmp_times.append(times[i][t])
                tmp_traj.append(trajectories[i][t])
        tmp_times.append(times[i][-1])
        tmp_traj.append(trajectories[i][-1])
        times[i] = tmp_times
        trajectories[i] = tmp_traj

    # We need only one list for all times
    times = tmp_times

    # Now clean data as sometimes the best val doesn't change over time
    last_perf = [i*10 for i in range(number_exp)]  # dummy entry
    time_ = list()
    performance = list([list() for i in range(number_exp)])
    for idx, t in enumerate(times):
        # print t, idx, last_perf, perf_list[0][idx], perf_list[1][idx]
        diff = sum([np.abs(last_perf[i] - trajectories[i][idx]) for i in range(number_exp)])
        if diff != 0 or idx == 0 or idx == len(times) - 1:
            # always use first and last entry
            time_.append(t)
            [performance[i].append(trajectories[i][idx]) for i in range(number_exp)]
        last_perf = [p[idx] for p in trajectories]

    trajectories = performance
    times = time_
    return trajectories, times


def extract_trajectory(experiment, cut=sys.maxint, maxvalue=sys.maxint,
                       test=False):
    """
    Extract a list where the value at position i is the current best after i
    configurations. Starts with maxvalue, as at timestep 0 there is no known
    performance value
    """
    if not isinstance(cut, int):
        raise ValueError("Argument cut must be an Integer value but is %s" %
            type(cut))
    if cut <= 0:
        raise ValueError("Argument cut cannot be zero or negative.")

    trace = list([maxvalue, ])
    test_results = None
    if test:
        test_results = list([maxvalue, ])

    currentbest = experiment['trials'][0]["result"]
    if not np.isfinite(currentbest):
        currentbest = maxvalue

    for result in [trial for trial in experiment['trials'][:cut]]:
        if result["status"] != 3 or not np.isfinite(result["result"]):
            # Ignore this trial, it is not valid/finished
            # add previous result
            trace.append(currentbest)
            continue

        if result["result"] < currentbest:
            currentbest = min(maxvalue, result["result"])
        trace.append(currentbest)
        if test and np.isfinite(result["test_result"]):
            test_results.append([len(trace) - 1, min(maxvalue, result["test_result"])])
    if test:
        return trace, test_results
    else:
        return trace

#def extract_trajectory(trials, cut=sys.maxint):
#    trace = list()
#    currentbest = trials['trials'][0]

#    for result in [trial["result"] for trial in trials['trials'][:cut]]:
#        if result < currentbest:
#            currentbest = result
#        trace.append(currentbest)
#    return trace


def extract_results(experiment, cut=sys.maxint):
    """Extract a list with all results.

    If `cut` is given, return up to `cut` results. Raise ValueError if cut
    is equal or less than zero."""
    if not isinstance(cut, int):
        raise ValueError("Argument cut must be an Integer value but is %s" %
            type(cut))
    if cut <= 0:
        raise ValueError("Argument cut cannot be zero or negative.")

    trl = [trial["result"] for trial in experiment['trials'][:cut]]
    return trl


def extract_runtime_timestamps(trials, cut=sys.maxint, conf_overhead=False):
    """Extracts timesteps for a list of trials
    trials = list of trials as in a HPOlib.pkl
    cut = consider only that many trials
    conf_overhead = add conf overhead, if false only add up target algorithm time

    return a list like (0, 20, 53, 101, 200)
    """
    # (TODO): This does not work for crossvalidation + intensify

    time_list = list()
    time_list.append(0)
    for idx, trial in enumerate(trials["trials"][:cut+1]):
        if conf_overhead:
            if len(trials["starttime"]) > 1:
                raise ValueError("Cannot extract runtimes for restarted "
                                 "experiments, please implement me")

            if trial["status"] != 3:
                # Although this trial is crashed, we add some minor timestep
                logger.critical("%d: trial is crashed, status %d" % (idx, trial["status"]))
                t = time_list[-1] + np.sum(trial["instance_durations"])
                if np.isnan(t):
                    logger.critical("Trying to use instance durations failed")
                    if len(trials["trials"]) == len(trials["cv_endtime"]):
                        t = time_list[-1] + trials["cv_endtime"][idx] - trials["cv_starttime"][idx]
                        logging.critical("Use 'cv_starttime' and 'cv_endtime': %d" % t)
                    elif idx == len(trials["trials"][:cut+1]):
                        t = trials["total_wallclock_time"]
                        logging.critical("Assuming last trial, use 'max_wallclock_time' ")

            else:
                t = trials["cv_starttime"][idx] - trials["starttime"][0] + trial["duration"]

            if np.isnan(t):
                logger.critical("%d: Cannot extract sample as trial is broken. "
                                "Assuming it is the last one and returning "
                                "'total_wallclock_time'" % idx)
                logger.critical("%d: Obtain duration failed, use %f" % (idx, 0.1))
                t = time_list[-1] + 0.1
            time_list.append(t)
        else:
            time_list.append(np.sum(trial["instance_durations"]) + time_list[-1])
    return time_list


def get_best(experiment, cut=sys.maxint):
    """Return the best value found in experiment.

    If `cut` is given, look at the first `cut` results. Raise ValueError if cut
    is equal or less than zero."""
    if not isinstance(cut, int):
        raise ValueError("Argument cut must be an Integer value but is %s" %
            type(cut))
    if cut <= 0:
        raise ValueError("Argument cut cannot be zero or negative.")

    # returns the best value found in this experiment
    traj = extract_trajectory(experiment)
    if cut < len(traj):
        best_value = traj[cut-1]
    else:
        best_value = traj[-1]
    return best_value


def get_best_value_and_index(trials, cut=sys.maxint):
    """Return the best value found and its index in experiment.

    If `cut` is given, look at the first `cut` results. Raise ValueError if cut
    is equal or less than zero. Important: The index is zero-based!

    if test is given, look for the best trial and report testperformance
    """
    if not isinstance(cut, int):
        raise ValueError("Argument cut must be an Integer value but is %s" %
            type(cut))
    if cut <= 0:
        raise ValueError("Argument cut cannot be zero or negative.")

    traj = extract_trajectory(experiment=trials, cut=cut, maxvalue=sys.maxint)
    if traj[0] == sys.maxint:
        traj = traj[1:]

    if cut < len(traj):
        best_value = traj[cut-1]
        best_index = np.argmin(traj[:cut])
    else:
        best_value = traj[-1]
        best_index = np.argmin(traj)
    return best_value, best_index


def get_Trace_cv(trials, maxvalue=sys.maxint):
    trace = list()
    trials_list = trials['trials']
    instance_order = trials['instance_order']
    instance_mean = np.ones([len(trials_list), 1]) * np.inf
    instance_val = np.ones([len(trials_list), len(trials_list[0]['instance_results'])]) * np.nan
    for tr_idx, in_idx in instance_order:
        instance_val[tr_idx, in_idx] = trials_list[tr_idx]['instance_results'][in_idx]

        val = HPOlib.wrapping_util.nan_mean(instance_val[tr_idx, :])
        if np.isnan(val):
            val = np.inf
        instance_mean[tr_idx] = val
        trace.append(np.min(instance_mean, axis=0)[0])
    if np.isnan(trace[-1]):
        del trace[-1]

    trace = [min(maxvalue, entry) for entry in trace]
    return trace


def get_defaults():
    default = {"linestyles": get_single_linestyle(),
               "colors": get_plot_colors(),
               "markers": get_empty_iterator(),
               "markersize": 6,
               "labelfontsize": 12,
               "linewidth": 1,
               "titlefontsize": 15,
               "gridcolor": 'lightgrey',
               "gridalpha": 0.5,
               "dpi": 100
               }
    return default


def fill_with_defaults(def_dict):
    defaults = get_defaults()
    for key in defaults:
        if key not in def_dict:
            def_dict[key] = defaults[key]
        elif def_dict[key] is None:
            def_dict[key] = defaults[key]
        else:
            pass
    return def_dict
