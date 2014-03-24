#!/usr/bin/env python

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
from argparse import ArgumentParser

import cPickle
import itertools
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec
import numpy as np

from HPOlib.Plotting import plot_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def plot_time_trace(time_dict, name_list, title="", log=True, save="", y_max=0, y_min=0):
    colors = itertools.cycle(['b', 'g', 'r', 'k'])
    markers = itertools.cycle(['o', 's', 'x', '^'])
    linestyles = itertools.cycle(['-'])

    size = 5
    ratio = 5
    gs = matplotlib.gridspec.GridSpec(ratio, 1)
    fig = plt.figure(1, dpi=100)
    fig.suptitle(title, fontsize=16)
    ax1 = plt.subplot(gs[0:ratio, :])
    ax1.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    min_val = sys.maxint
    max_val = -sys.maxint
    max_trials = 0

    trial_list_means = list()
    trial_list_std = list()
    num_runs_list = list()

    # Get mean and std for all times and optimizers
    for entry in name_list:
        k = entry[0]
        trial_list_std.append(np.std(np.array(time_dict[k]), axis=0))
        if log:
            trial_list_means.append(np.log10(np.mean(np.array(time_dict[k]), axis=0)))
        else:
            trial_list_means.append(np.mean(np.array(time_dict[k]), axis=0))
        num_runs_list.append(len(time_dict[k]))

    for k in range(len(name_list)):
        # Plot mean and std for optimizer duration
        c = colors.next()
        m = markers.next()
        x = range(len(trial_list_means[k]))
        l = linestyles.next()
        ax1.fill_between(x, trial_list_means[k] - trial_list_std[k],
                         trial_list_means[k] + trial_list_std[k],
                         facecolor=c, alpha=0.3, edgecolor=c)
        ax1.plot(x, trial_list_means[k], color=c, linewidth=size, label=name_list[k][0], linestyle=l, marker=m)
        # Plot number of func evals for this experiment

        if min(trial_list_means[k] - trial_list_std[k]) < min_val:
            min_val = min(trial_list_means[k] - trial_list_std[k])
        if max(trial_list_means[k] + trial_list_std[k]) > max_val:
            max_val = max(trial_list_means[k] + trial_list_std[k])
        if len(trial_list_means[k]) > max_trials:
            max_trials = len(trial_list_means[k])

    # Descript and label the stuff
    fig.suptitle(title, fontsize=16)
    leg = ax1.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    if log:
        ax1.set_ylabel("log10(Optimizer time in [sec])")
    else:
        ax1.set_ylabel("Optimizer time in [sec]")
    if y_max == y_min:
        ax1.set_ylim([min_val-2, max_val+2])
    else:
        ax1.set_ylim([y_min, y_max])
    ax1.set_xlim([0, max_trials])

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save != "":
        plt.savefig(save, dpi=100, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()


def main(pkl_list, name_list, autofill, title="", log=False, save="",
         y_min=0, y_max=0, cut=sys.maxint):

    times_dict = dict()
    for exp in range(len(name_list)):
        times_dict[name_list[exp][0]] = list()
        for pkl in pkl_list[exp]:
            fh = open(pkl, "r")
            trials = cPickle.load(fh)
            fh.close()

            # Get all variables from the trials object
            cv_starttime = trials["cv_starttime"][:cut]
            cv_endtime = trials["cv_endtime"][:cut]

            # Get optimizer duration times
            time_list = list()
            # First duration
            time_list.append(cv_starttime[0] - trials["starttime"][0])
            time_idx = 0
            for i in range(len(cv_starttime[1:])):
                # Is there a next restored run?
                # if yes, does next cvstart belong to a restored run?
                if len(trials["endtime"]) > time_idx and \
                        cv_starttime[i+1] > trials["endtime"][time_idx]:
                    # Check whether run crashed/terminated during cv
                    # Equals means that the run crashed
                    if cv_endtime[i] < trials["endtime"][time_idx]:
                        # No .. everything is fine
                        time_list.append((trials["endtime"][time_idx] - cv_endtime[i]))
                        time_list.append((cv_starttime[i + 1] - trials["starttime"][time_idx+1]))
                    elif trials["endtime"][time_idx] == cv_endtime[i]:
                        # Yes, but BBoM managed to set an endtime
                        pass
                    else:
                        # Yes ... trouble
                        print "Help"
                        print trials["endtime"][time_idx]
                        print cv_endtime[i]
                    time_idx += 1
                # everything ...
                else:
                    time_list.append(cv_starttime[i + 1] - cv_endtime[i])
            times_dict[name_list[exp][0]].append(time_list)

    for key in times_dict.keys():
        max_len = max([len(ls) for ls in times_dict[key]])
        for t in range(len(times_dict[key])):
            if len(times_dict[key][t]) < max_len and autofill:
                diff = max_len - len(times_dict[key][t])
                # noinspection PyUnusedLocal
                times_dict[key][t] = np.append(times_dict[key][t], [times_dict[key][t][-1] for x in range(diff)])
            elif len(times_dict[key][t]) < max_len and not autofill:
                raise ValueError("(%s != %s), Traces do not have the same length, please use -a" %
                                 (str(max_len), str(len(times_dict[key][t]))))

    plot_time_trace(times_dict, name_list, title=title, log=log, save=save,
                    y_min=y_min, y_max=y_max)

    if save != "":
        sys.stdout.write("Saved plot to " + save + "\n")
    else:
        sys.stdout.write("..Done\n")

if __name__ == "__main__":
    prog = "python plotOptimizerOverhead.py WhatIsThis <oneOrMorePickles> [WhatIsThis <oneOrMorePickles>]"
    description = "Plot a Trace with std for multiple experiments"

    parser = ArgumentParser(description=description, prog=prog)

    # General Options
    parser.add_argument("-c", "--cut", type=int, default=sys.maxint,
                        help="Cut the experiment pickle length.")
    parser.add_argument("-l", "--log", action="store_true", dest="log",
                        default=False, help="Plot on log scale")
    parser.add_argument("--max", type=float, dest="max",
                        default=0, help="Maximum of the plot")
    parser.add_argument("--min", type=float, dest="min",
                        default=0, help="Minimum of the plot")
    parser.add_argument("-s", "--save", dest="save",
                        default="", help="Where to save plot instead of showing it?")
    parser.add_argument("-t", "--title", dest="title",
                        default="", help="Choose a supertitle for the plot")

    # Options which are available only for this plot
    parser.add_argument("-a", "--autofill", action="store_true", dest="autofill",
                        default=False, help="Fill trace automatically")

    args, unknown = parser.parse_known_args()

    sys.stdout.write("\nFound " + str(len(unknown)) + " arguments\n")

    pkl_list_main, name_list_main = plot_util.get_pkl_and_name_list(unknown)
    main(pkl_list_main, name_list_main, autofill=args.autofill, title=args.title,
         log=args.log, save=args.save, y_min=args.min, y_max=args.max,
         cut=args.cut)
