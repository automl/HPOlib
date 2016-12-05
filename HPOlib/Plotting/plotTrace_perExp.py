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


def plot_optimization_trace_cv(trial_list, name_list, optimum=0, title="",
                               log=True, save="", y_max=0, y_min=0,
                               linewidth=1,
                               linestyles=plot_util.get_empty_iterator(),
                               colors=None, markers=plot_util.get_empty_iterator(),
                               markersize=6, ylabel=None, xlabel=None):

    if colors is None:
        colors= plot_util.get_plot_colors()

    ratio = 5
    gs = matplotlib.gridspec.GridSpec(ratio, 1)
    fig = figure(1, dpi=100)
    fig.suptitle(title, fontsize=16)
    ax1 = subplot(gs[0:ratio, :])
    ax1.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    min_val = sys.maxint
    max_val = -sys.maxint
    max_trials = 0

    fig.suptitle(title, fontsize=16)

    # Plot the average error and std
    for i in range(len(name_list)):
        m = markers.next()
        c = colors.next()
        l = linestyles.next()
        leg = False
        for tr in trial_list[i]:
            if log:
                tr = np.log10(tr)
            x = range(1, len(tr)+1)
            y = tr
            if not leg:
                ax1.plot(x, y, color=c, linewidth=linewidth,
                         markersize=markersize, label=name_list[i][0])
                leg = True
            ax1.plot(x, y, color=c, linewidth=linewidth, markersize=markersize)
            min_val = min(min_val, min(tr))
            max_val = max(max_val, max(tr))
            max_trials = max(max_trials, len(tr))

    # Maybe plot on logscale
    ylabel = ""
    if ylabel is None:
        if log:
            ylabel = "log10(Minfunction value)"
        else:
            ylabel = "Minfunction value"
    ax1.set_ylabel(ylabel)

    # Descript and label the stuff
    leg = ax1.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    if xlabel is None:
        xlabel = "#Function evaluations"
    ax1.set_xlabel(xlabel)

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


def main(pkl_list, name_list, autofill, optimum=0, maxvalue=sys.maxint,
         save="", title="", log=False,
         y_min=0, y_max=0, linewidth=1, linestyles=plot_util.get_single_linestyle(),
         colors=None, markers=plot_util.get_empty_iterator(),
         markersize=6, ylabel=None, xlabel=None):

    if colors is None:
        colors = plot_util.get_plot_colors()

    trial_list = list()
    for i in range(len(pkl_list)):
        tmp_trial_list = list()
        max_len = -sys.maxint
        for pkl in pkl_list[i]:
            fh = open(pkl, "r")
            trials = cPickle.load(fh)
            fh.close()

            trace = plot_util.get_Trace_cv(trials, maxvalue=maxvalue)
            tmp_trial_list.append(trace)
            max_len = max(max_len, len(trace))
        trial_list.append(list())
        for tr in tmp_trial_list:
        #    if len(tr) < max_len:
        #        tr.extend([tr[-1] for idx in range(abs(max_len - len(tr)))])
            trial_list[-1].append(np.array(tr))

    plot_optimization_trace_cv(trial_list, name_list, optimum, title=title, log=log,
                               save=save, y_min=y_min, y_max=y_max,
                               linewidth=linewidth, linestyles=linestyles,
                               colors=colors, markers=markers,
                               markersize=markersize, ylabel=ylabel, xlabel=xlabel)

    if save != "":
        sys.stdout.write("Saved plot to " + save + "\n")
    else:
        sys.stdout.write("..Done\n")

if __name__ == "__main__":
    prog = "python plotTrace_perExp.py WhatIsThis <oneOrMorePickles> [WhatIsThis <oneOrMorePickles>]"
    description = "Plot a Trace with std for multiple experiments"

    parser = ArgumentParser(description=description, prog=prog)

    # Options for specific benchmarks
    parser.add_argument("-o", "--optimum", type=float, dest="optimum",
                        default=0, help="If not set, the optimum is supposed to be zero")

    # Options which are available only for this plot
    parser.add_argument("-a", "--autofill", action="store_true", dest="autofill",
                        default=False, help="Fill trace automatically")

    # General Options
    parser.add_argument("-l", "--log", action="store_true", dest="log",
                        default=False, help="Plot on log scale")
    parser.add_argument("--max", dest="max", type=float,
                        default=0, help="Maximum of the plot")
    parser.add_argument("--min", dest="min", type=float,
                        default=0, help="Minimum of the plot")
    parser.add_argument("--maxvalue", dest="maxvalue", default=10000,
                        type=float, help="replace all y values higher than this")
    parser.add_argument("-s", "--save", dest="save",
                        default="", help="Where to save plot instead of showing it?")
    parser.add_argument("-t", "--title", dest="title",
                        default="", help="Optional supertitle for plot")

    args, unknown = parser.parse_known_args()

    sys.stdout.write("\nFound " + str(len(unknown)) + " arguments\n")

    pkl_list_main, name_list_main = plot_util.get_pkl_and_name_list(unknown)

    main(pkl_list=pkl_list_main, name_list=name_list_main,
         autofill=args.autofill, optimum=args.optimum, maxvalue=args.maxvalue,
         save=args.save, title=args.title, log=args.log, y_min=args.min,
         y_max=args.max)
