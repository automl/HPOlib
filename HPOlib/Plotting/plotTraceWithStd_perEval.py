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

from matplotlib.pyplot import tight_layout, figure, subplots_adjust, subplot, savefig, show
import matplotlib.gridspec
import numpy as np

from HPOlib.Plotting import plot_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def plot_optimization_trace(trial_list, name_list, optimum=0, title="",
                            log=True, save="", y_max=0, y_min=0, scale_std=1):
    markers = itertools.cycle(['o', 's', 'x', '^'])
    colors = itertools.cycle(['b', 'g', 'r', 'k'])
    linestyles = itertools.cycle(['-'])
    size = 1

    ratio = 5
    gs = matplotlib.gridspec.GridSpec(ratio, 1)
    fig = figure(1, dpi=100)
    fig.suptitle(title, fontsize=16)
    ax1 = subplot(gs[0:ratio, :])
    ax1.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    min_val = sys.maxint
    max_val = -sys.maxint
    max_trials = 0

    trial_list_means = list()
    trial_list_std = list()

    # One trialList represents all runs from one optimizer
    for i in range(len(trial_list)):
        if log:
            trial_list_means.append(np.log10(np.mean(np.array(trial_list[i]), axis=0)))
        else:
            trial_list_means.append(np.mean(np.array(trial_list[i]), axis=0))
        trial_list_std.append(np.std(np.array(trial_list[i]), axis=0)*scale_std)

    fig.suptitle(title, fontsize=16)

    # Plot the average error and std
    for i in range(len(trial_list_means)):
        x = range(1, len(trial_list_means[i])+1)
        y = trial_list_means[i] - optimum
        m = markers.next()
        c = colors.next()
        l = linestyles.next()
        std_up = y + trial_list_std[i]
        std_down = y - trial_list_std[i]
        ax1.fill_between(x, std_down, std_up,
                         facecolor=c, alpha=0.3, edgecolor=c)
        ax1.plot(x, y, color=c, linewidth=size,
                 label=name_list[i][0] + "(" + str(len(trial_list[i])) + ")",
                 linestyle=l, marker=m)
        if min(std_down) < min_val:
            min_val = min(std_down)
        if max(std_up) > max_val:
            max_val = max(std_up)
        if len(trial_list_means[i]) > max_trials:
            max_trials = len(trial_list_means[i])

    # Maybe plot on logscale
    if scale_std != 1:
        ylabel = ", %s * std" % scale_std
    else:
        ylabel = ""

    if log:
        ax1.set_ylabel("log10(Minfunction value)" + ylabel)
    else:
        ax1.set_ylabel("Minfunction value" + ylabel)

    # Descript and label the stuff
    leg = ax1.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ax1.set_xlabel("#Function evaluations")

    if y_max == y_min:
         # Set axes limits
        ax1.set_ylim([min_val-0.1*abs((max_val-min_val)), max_val+0.1*abs((max_val-min_val))])
    else:
        ax1.set_ylim([y_min, y_max])
    ax1.set_xlim([0, max_trials + 1])

    tight_layout()
    subplots_adjust(top=0.85)
    if save != "":
        savefig(save, dpi=100, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches="tight", pad_inches=0.1)
    else:
        show()


def main(pkl_list, name_list, autofill, optimum=0, save="", title="", log=False,
         y_min=0, y_max=0, scale_std=1, cut=sys.maxint):

    trial_list = list()
    for i in range(len(pkl_list)):
        trial_list.append(list())
        for pkl in pkl_list[i]:
            fh = open(pkl, "r")
            trials = cPickle.load(fh)
            fh.close()

            trace = plot_util.extract_trajectory(trials, cut=cut)
            trial_list[-1].append(np.array(trace))

    for i in range(len(trial_list)):
        max_len = max([len(ls) for ls in trial_list[i]])
        for t in range(len(trial_list[i])):
            if len(trial_list[i][t]) < max_len and autofill:
                diff = max_len - len(trial_list[i][t])
                # noinspection PyUnusedLocal
                trial_list[i][t] = np.append(trial_list[i][t], [trial_list[i][t][-1] for x in range(diff)])
            elif len(trial_list[i][t]) < max_len and not autofill:
                raise ValueError("(%s != %s), Traces do not have the same length, please use -a" %
                                 (str(max_len), str(len(trial_list[i][t]))))

    plot_optimization_trace(trial_list, name_list, optimum, title=title, log=log,
                            save=save, y_min=y_min, y_max=y_max, scale_std=scale_std)

    if save != "":
        sys.stdout.write("Saved plot to " + save + "\n")
    else:
        sys.stdout.write("..Done\n")

if __name__ == "__main__":
    prog = "python plotTraceWithStd.py WhatIsThis <oneOrMorePickles> [WhatIsThis <oneOrMorePickles>]"
    description = "Plot a Trace with std for multiple experiments"

    parser = ArgumentParser(description=description, prog=prog)

    # Options for specific benchmarks
    parser.add_argument("-o", "--optimum", type=float, dest="optimum",
                        default=0, help="If not set, the optimum is supposed to be zero")

    # Options which are available only for this plot
    parser.add_argument("-a", "--autofill", action="store_true", dest="autofill",
                        default=False, help="Fill trace automatically")
    parser.add_argument("-c", "--scale", type=float, dest="scale",
                        default=1, help="Multiply std to get a nicer plot")
    # General Options
    parser.add_argument("-l", "--log", action="store_true", dest="log",
                        default=False, help="Plot on log scale")
    parser.add_argument("--max", dest="max", type=float,
                        default=0, help="Maximum of the plot")
    parser.add_argument("--min", dest="min", type=float,
                        default=0, help="Minimum of the plot")
    parser.add_argument("-s", "--save", dest="save",
                        default="", help="Where to save plot instead of showing it?")
    parser.add_argument("-t", "--title", dest="title",
                        default="", help="Optional supertitle for plot")

    args, unknown = parser.parse_known_args()

    sys.stdout.write("\nFound " + str(len(unknown)) + " arguments\n")

    pkl_list_main, name_list_main = plot_util.get_pkl_and_name_list(unknown)

    main(pkl_list=pkl_list_main, name_list=name_list_main, autofill=args.autofill, optimum=args.optimum,
         save=args.save, title=args.title, log=args.log, y_min=args.min, y_max=args.max, scale_std=args.scale)
