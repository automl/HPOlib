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
import sys
import cPickle

from matplotlib.pyplot import tight_layout, figure, subplots_adjust, subplot, savefig, show, yscale
import matplotlib.gridspec
import numpy as np
import scipy.stats as sc

from HPOlib.Plotting import plot_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def plot_optimization_trace(trial_list, name_list, optimum=0, title="", log=False,
                            save="", y_min=0, y_max=0, cut=sys.maxint,
                            linewidth=1, linestyles=plot_util.get_single_linestyle(),
                            colors=None, markers=plot_util.get_empty_iterator(),
                            ylabel=None, xlabel=None):

    if colors is None:
        colors= plot_util.get_plot_colors()

    size = 1
    # get handles
    ratio = 5
    gs = matplotlib.gridspec.GridSpec(ratio, 1)
    fig = figure(1, dpi=100)
    fig.suptitle(title, fontsize=16)
    ax = subplot(gs[0:ratio, :])
    ax.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    min_val = sys.maxint
    max_val = -sys.maxint
    max_trials = 0

    # This might not do what we actually want; Ideally it would only take the
    # ones that where actually run according to instance_order
    for i in range(len(name_list)):
        print cut, len(trial_list[i])
        num_plotted_trials = np.min([cut, len(trial_list[i])])
        print num_plotted_trials
        x = range(num_plotted_trials)
        y = np.zeros((num_plotted_trials))
        line = np.zeros((num_plotted_trials))
        for j, inst_res in enumerate(trial_list[i][:num_plotted_trials]):
            if j >= len(y):
                break
            if type(inst_res) == np.ndarray and not np.isfinite(inst_res).any():
                inst_res[np.isnan(inst_res)] = 1
            elif type(inst_res) != np.ndarray and np.isnan(inst_res):
                inst_res = 1
            tmp = sc.nanmean(np.array([inst_res, inst_res]).flat)  # Black Magic
            if log:
                y[j] = np.log(tmp - optimum)
                line[j] = np.min(y[:j + 1])
            else:
                y[j] = tmp - optimum
                line[j] = np.min(y[:j + 1])

        # Plot the stuff
        marker = markers.next()
        color = colors.next()
        line_style_ = linestyles.next()
        ax.scatter(np.argmin(line), min(line), facecolor="w", edgecolor=color,
                   s=size*10*15, marker=marker)
        ax.scatter(x, y, color=color, marker=marker, s=size*15)
        ax.plot(x, line, color=color, label=name_list[i][0], linestyle=line_style_,
                linewidth=linewidth)

        if min(y) < min_val:
            min_val = min(y)
        if max(y) > max_val:
            max_val = max(y)
        if num_plotted_trials > max_trials:
            max_trials = num_plotted_trials

    # Describe and label the stuff
    if xlabel is None:
        xlabel = "#Function evaluations"
    ax.set_xlabel(xlabel)

    if ylabel is None:
        if log:
            ylabel = "log10(Minfunction value)"
        else:
            ylabel = "Minfunction value"
    ax.set_ylabel(ylabel)

    if y_min == y_max:
        ax.set_ylim([min_val - 0.1, max_val + 0.1])
    else:
        ax.set_ylim([y_min, y_max])
    ax.set_xlim([0, max_trials])

    leg = ax.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)

    tight_layout()
    subplots_adjust(top=0.85)

    if save != "":
        savefig(save, dpi=100, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches="tight", pad_inches=0.1)
    else:
        show()


def main(pkl_list, name_list, optimum=0, title="", log=False, save="", y_max=0,
         y_min=0, cut=sys.maxint, linewidth=1,
         linestyles=plot_util.get_single_linestyle(), colors=None,
         markers=plot_util.get_empty_iterator(),
         ylabel=None, xlabel=None):

    if colors is None:
        colors= plot_util.get_plot_colors()

    trial_list = list()
    for i in range(len(pkl_list)):
        if len(pkl_list[i]) != 1:
            raise ValueError("%s is more than <onePickle>!" % str(pkl_list))
        fh = open(pkl_list[i][0])
        trl = cPickle.load(fh)
        fh.close()
        trial_list.append(plot_util.extract_results(trl))

    sys.stdout.write("Plotting trace\n")
    plot_optimization_trace(trial_list=trial_list, name_list=name_list, optimum=optimum,
                            title=title, log=log, save=save, y_max=y_max,
                            y_min=y_min, cut=cut, linewidth=linewidth,
                            linestyles=linestyles, colors=colors,
                            markers=markers, ylabel=ylabel, xlabel=xlabel)

    if save != "":
        sys.stdout.write("Saved plot to " + save + "\n")
    else:
        sys.stdout.write("..Done\n")

if __name__ == "__main__":
    prog = "python plotTrace.py WhatIsThis <onePickle> [WhatIsThis <onePickle>]"
    description = "Plot a Trace with evaluated points wrt to performance"

    parser = ArgumentParser(description=description, prog=prog)

    # Options for specific benchmarks
    parser.add_argument("-o", "--optimum", type=float, dest="optimum",
                        default=0, help="If not set, the optimum is supposed to be zero")
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
    parser.add_argument("-c", "--cut", default=sys.maxint, type=int,
                        help="Cut experiment pickles after a specified number of trials.")

    args, unknown = parser.parse_known_args()

    sys.stdout.write("Found " + str(len(unknown)) + " arguments\n")

    pkl_list_main, name_list_main = plot_util.get_pkl_and_name_list(unknown)

    main(pkl_list=pkl_list_main, name_list=name_list_main, optimum=args.optimum,
         title=args.title, log=args.log, save=args.save, y_max=args.max,
         y_min=args.min, cut=args.cut)
