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
import sys

from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import xticks, figure, subplot, savefig, show, tight_layout, subplots_adjust

import numpy as np

from HPOlib.Plotting import plot_util


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__license__ = "3-clause BSD License"
__contact__ = "automl.org"


def plot_box_whisker(best_trials, name_list, title="", save="", y_min=0,
                     y_max=0):
    ratio = 5
    gs = GridSpec(ratio, 1)
    fig = figure(1, dpi=100)
    fig.suptitle(title)
    ax = subplot(gs[0:ratio, :])
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    bp = ax.boxplot(best_trials, 0, 'ok')
    boxlines = bp['boxes']
    for line in boxlines:
        line.set_color('k')
        line.set_linewidth(2)

    min_y = sys.maxint
    max_y = -sys.maxint

    # Get medians and limits
    medians = range(len(name_list))
    for i in range(len(name_list)):
        med = bp['medians'][i]
        median_x = []
        median_y = []
        for j in range(2):
            median_x.append(med.get_xdata()[j])
            median_y.append(med.get_ydata()[j])
            medians[i] = median_y[0]
        if min(best_trials[i]) < min_y:
            min_y = min(best_trials[i])
        if max(best_trials[i]) > max_y:
            max_y = max(best_trials[i])
    print medians
    
    # Plot xticks
    xticks(range(1, len(name_list)+1), name_list)

    # Set limits
    if y_max == y_min:
        # Set axes limit
        ax.set_ylim([min_y-0.1*abs((max_y-min_y)), max_y+0.1*abs((max_y-min_y))])
    else:
        ax.set_ylim([y_min, y_max])
        max_y = y_max
        min_y = y_min

    # Print medians as upper labels
    top = max_y-((max_y-min_y)*0.05)
    pos = np.arange(len(name_list))+1
    upper_labels = [str(np.round(s, 5)) for s in medians]
    upper_labels[0] = "median=[%s," % upper_labels[0]
    for i in range(len(upper_labels[1:-1])):
        upper_labels[i+1] = "%s," % upper_labels[i+1]
    upper_labels[-1] = "%s]" % upper_labels[-1]
    for tick, label in zip(range(len(name_list)), ax.get_xticklabels()):
        ax.text(pos[tick], top, upper_labels[tick],
                horizontalalignment='center', size='x-small')

    ax.set_ylabel('Minfunction')

    tight_layout()
    subplots_adjust(top=0.85)
    if save != "":
        savefig(save, dpi=100, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches="tight", pad_inches=0.1)
    else:
        show()


def main(pkl_list, name_list, title="", save="", y_min=0, y_max=0, cut=sys.maxint):

    best_trials = list()
    for i in range(len(name_list)):
        best_trials.append(list())
        for pkl in pkl_list[i]:
            fh = open(pkl, "r")
            trials = cPickle.load(fh)
            fh.close()
            best_trials[i].append(plot_util.get_best(trials, cut=cut))

    plot_box_whisker(best_trials=best_trials, name_list=name_list,
                     title=title, save=save, y_min=y_min, y_max=y_max)

    if save != "":
        sys.stdout.write("Saving plot to " + save + "\n")
    else:
        sys.stdout.write("..Done\n")

if __name__ == "__main__":
    prog = "python plotBoxWhisker.py WhatIsThis <ManyPickles> [WhatIsThis <ManyPickles>]"
    description = "Plot a Box whisker plot for many experiments. The box covers lower to upper quartile."

    parser = ArgumentParser(description=description, prog=prog)

    # General Options
    parser.add_argument("-t", "--title", dest="title", default="",
                        help="Optional supertitle for plot")
    parser.add_argument("--max", dest="max", default=0,
                        type=float, help="Maximum of the plot")
    parser.add_argument("--min", dest="min", default=0,
                        type=float, help="Minimum of the plot")
    parser.add_argument("-s", "--save", dest="save", default="",
                        help="Where to save plot instead of showing it?")
    parser.add_argument("-c", "--cut", default=sys.maxint, type=int,
                        help="Cut experiment pickles after a specified number of trials.")
    args, unknown = parser.parse_known_args()

    sys.stdout.write("\nFound " + str(len(unknown)) + " arguments...")

    pkl_list_main, name_list_main = plot_util.get_pkl_and_name_list(unknown)

    main(pkl_list=pkl_list_main, name_list=name_list_main, title=args.title, save=args.save,
         y_min=args.min, y_max=args.max, cut=args.cut)
