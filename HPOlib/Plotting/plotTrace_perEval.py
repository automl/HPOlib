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

import HPOlib.Plotting.plot_util as plot_util
import HPOlib.Plotting.plot_trajectory as plot_trajectory

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def plot_optimization_trace(trial_list, name_list, optimum=0, title="",
                            log=True, save="", y_max=0, y_min=0, scale_std=1,
                            linewidth=1, linestyles=plot_util.
                            get_single_linestyle(), colors=None,
                            markers=plot_util.get_empty_iterator(),
                            markersize=6, print_lenght_trial_list=True,
                            ylabel=None, xlabel=None):


    if colors is None:
        colors= plot_util.get_plot_colors()

    ratio = 5
    gs = matplotlib.gridspec.GridSpec(ratio, 1)
    fig = plt.figure(dpi=300)
    fig.suptitle(title, fontsize=16)
    ax1 = fig.add_subplot(gs[0:ratio, :])
    ax1.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    min_val = sys.maxint
    max_val = -sys.maxint
    max_trials = 0

    trial_list_means = list()
    trial_list_std = list()

    # One trialList represents all runs from one optimizer
    for i in range(len(trial_list)):
        if log:
            trial_list_means.append(np.log10(np.nanmean(np.array(trial_list[i]),
                                                        axis=0)))
        else:
            trial_list_means.append(np.nanmean(np.array(trial_list[i]), axis=0))
        trial_list_std.append(np.nanstd(np.array(trial_list[i]), axis=0) *
                              scale_std)

    fig.suptitle(title, fontsize=16)

    # Plot the average error and std
    for i in range(len(trial_list_means)):
        x = range(1, len(trial_list_means[i])+1)
        y = trial_list_means[i] - optimum
        m = markers.next()
        c = colors.next()
        l = linestyles.next()
        label = name_list[i][0]
        if print_lenght_trial_list:
            label += "(" + str(len(trial_list[i])) + ")"
        std_up = y + trial_list_std[i]
        std_down = y - trial_list_std[i]
        ax1.fill_between(x, std_down, std_up,
                         facecolor=c, alpha=0.3, edgecolor=c)
        ax1.plot(x, y, linewidth=linewidth, linestyle=l, color=c, marker=m,
                 markersize=markersize, label=label)
        if min(std_down) < min_val:
            min_val = min(std_down)
        if max(std_up) > max_val:
            max_val = max(std_up)
        if len(trial_list_means[i]) > max_trials:
            max_trials = len(trial_list_means[i])

    # Maybe plot on logscale

    if ylabel is None:
        if optimum == 0:
            ylabel = "Min function value"
        else:
            ylabel = "Difference to min function value"
        if scale_std != 1:
            ylabel = "%s, %s * std" % (ylabel, scale_std)
        if log:
            ylabel = "log10(%s)" % ylabel
    ax1.set_ylabel(ylabel)

    # Descript and label the stuff
    leg = ax1.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    if xlabel is None:
        xlabel = "#Function evaluations"
    ax1.set_xlabel(xlabel)

    if y_max == y_min:
         # Set axes limits
        ax1.set_ylim([min_val - 0.1 * abs((max_val - min_val)),
                      max_val + 0.1 * abs((max_val - min_val))])
    else:
        ax1.set_ylim([y_min, y_max])
    ax1.set_xlim([0, max_trials + 1])

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    if save != "":
        plt.savefig(save, dpi=100, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches="tight", pad_inches=0.1)
        fig.clf()
        plt.close(fig)
    else:
        plt.show()


def main(pkl_list, name_list, autofill, optimum=0, save="", title="", log=False,
         y_min=None, y_max=None, scale_std=1, aggregation="mean",
         cut=sys.maxint, xlabel="#Function evaluations", ylabel="Loss",
         properties=None, print_lenght_trial_list=False):

    trial_list = list()
    x_ticks = list()
    for i in range(len(pkl_list)):
        trial_list.append(list())
        for pkl in pkl_list[i]:
            if pkl in plot_util.cache:
                trials = plot_util.cache[pkl]
            else:
                fh = open(pkl, "r")
                trials = cPickle.load(fh)
                fh.close()
                plot_util.cache[pkl] = trials

            trace = plot_util.extract_trajectory(trials, cut=cut)
            trial_list[-1].append(np.array(trace))

    for i in range(len(trial_list)):
        max_len = max([len(ls) for ls in trial_list[i]])
        for t in range(len(trial_list[i])):
            if len(trial_list[i][t]) < max_len and autofill:
                diff = max_len - len(trial_list[i][t])
                # noinspection PyUnusedLocal
                trial_list[i][t] = np.append(trial_list[i][t],
                                             [trial_list[i][t][-1]
                                              for x in range(diff)])
            elif len(trial_list[i][t]) < max_len and not autofill:
                raise ValueError("(%s != %s), Traces do not have the same "
                                 "length, please use -a" %
                                 (str(max_len), str(len(trial_list[i][t]))))
        x_ticks.append(range(np.max([len(ls) for ls in trial_list[i]])))

    plot_trajectory.plot_trajectories(trial_list=trial_list,
                                      name_list=name_list, x_ticks=x_ticks,
                                      optimum=optimum,
                                      y_min=y_min, y_max=y_max,
                                      title=title, ylabel=ylabel, xlabel=xlabel,
                                      log=log, save=save,
                                      aggregation=aggregation,
                                      properties=properties,
                                      scale_std=scale_std,
                                      print_lenght_trial_list=
                                      print_lenght_trial_list)
    return

if __name__ == "__main__":
    prog = "python plotTraceWithStd.py WhatIsThis <oneOrMorePickles> " \
           "[WhatIsThis <oneOrMorePickles>]"
    description = "Plot a Trace with std for multiple experiments"

    parser = ArgumentParser(description=description, prog=prog)

    # Options for specific benchmarks
    parser.add_argument("-o", "--optimum", type=float, dest="optimum",
                        default=0,
                        help="If not set, the optimum is supposed to be zero")

    # Options which are available only for this plot
    parser.add_argument("-a", "--autofill", action="store_true",
                        dest="autofill", default=False,
                        help="Fill trace automatically")
    parser.add_argument("-c", "--scale", type=float, dest="scale",
                        default=1, help="Multiply std to get a nicer plot")

    # General Options
    parser.add_argument("-l", "--log", action="store_true", dest="log",
                        default=False, help="Plot on log scale")
    parser.add_argument("--max", dest="max", type=float,
                        default=None, help="Maximum of the plot")
    parser.add_argument("--min", dest="min", type=float,
                        default=None, help="Minimum of the plot")
    parser.add_argument("-s", "--save", dest="save",
                        default="",
                        help="Where to save plot instead of showing it?")
    parser.add_argument("-t", "--title", dest="title",
                        default="", help="Optional supertitle for plot")
    parser.add_argument("--xlabel", dest="xlabel",
                        default="#Function evaluations",
                        help="x axis")
    parser.add_argument("--ylabel", dest="ylabel",
                        default="#Minfunction value", help="y label")
    parser.add_argument("--printLength", dest="printlength", default=False,
                        action="store_true",
                        help="Print number of runs in brackets (legend)")
    parser.add_argument("--aggregation", dest="aggregation", default="mean",
                        choices=("mean", "median"),
                        help="Print Median/Quantile or Mean/Std")

    # Properties
    # We need this to show defaults for -h
    defaults = plot_util.get_defaults()
    for key in defaults:
        parser.add_argument("--%s" % key, dest=key, default=None,
                            help="%s, default: %s" % (key, str(defaults[key])))


    args, unknown = parser.parse_known_args()

    sys.stdout.write("\nFound " + str(len(unknown)) + " arguments\n")

    pkl_list_main, name_list_main = plot_util.get_pkl_and_name_list(unknown)

    prop = {}
    args_dict = vars(args)
    for key in defaults:
        prop[key] = args_dict[key]

    main(pkl_list=pkl_list_main, name_list=name_list_main,
         autofill=args.autofill, optimum=args.optimum, save=args.save,
         title=args.title, log=args.log,
         y_min=args.min, y_max=args.max, scale_std=args.scale,
         aggregation=args.aggregation,
         xlabel=args.xlabel, ylabel=args.ylabel, properties=prop,
         print_lenght_trial_list=args.printlength)
