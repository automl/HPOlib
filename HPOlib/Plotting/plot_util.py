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
import os
import numpy as np
import sys

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def read_pickles(name_list, pkl_list, cut=sys.maxint):
    best_dict = dict()
    idx_dict = dict()
    keys = list()
    for i in range(len(name_list)):
        keys.append(name_list[i][0])
        best_dict[name_list[i][0]] = list()
        idx_dict[name_list[i][0]] = list()
        for pkl in pkl_list[i]:
            fh = open(pkl)
            trial = cPickle.load(fh)
            fh.close()
            best, idx = get_best_value_and_index(trial, cut)
            best_dict[name_list[i][0]].append(best)
            idx_dict[name_list[i][0]].append(idx)
    return best_dict, idx_dict, keys


def get_pkl_and_name_list(argument_list):
    name_list = list()
    pkl_list = list()
    now_data = False
    for i in range(len(argument_list)):
        if not ".pkl" in argument_list[i] and now_data:
            raise ValueError("You need at least on .pkl file per Experiment, %s has none" % name_list[-1])
        elif not ".pkl" in argument_list[i] and not now_data:
            print "Adding", argument_list[i]
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


def extract_trajectory(trials, cut=sys.maxint):
    trace = list()
    currentbest = trials['trials'][0]

    for result in [trial["result"] for trial in trials['trials'][:cut]]:
        if result < currentbest:
            currentbest = result
        trace.append(currentbest)
    return trace


def extract_trials(trials, cut=sys.maxint):
    trl = [trial["result"] for trial in trials['trials'][:cut]]
    return trl


def extract_runtime_timestamps(trials, cut=sys.maxint):
    # return a list like (20, 53, 101, 200)
    time_list = list()
    time_list.append(0)
    for trial in trials["trials"][:cut]:
        time_list.append(np.sum(trial["instance_durations"]) + time_list[-1])
    return time_list


def get_best(trials, cut=False):
    # returns the best value found in this experiment
    traj = extract_trajectory(trials)
    best_value = sys.maxint
    if cut and 0 < cut < len(traj):
        best_value = traj[cut]
    else:
        best_value = traj[-1]
    return best_value


def get_best_value_and_index(trials, cut=False):
    # returns the best value and corresponding index
    traj = extract_trajectory(trials)
    best_value = sys.maxint
    best_index = -1
    if cut and 0 < cut < len(traj):
        best_value = traj[cut]
        best_index = np.argmin(traj[:cut])
    else:
        best_value = traj[-1]
        best_index = np.argmin(traj)
    return best_value, best_index