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

import cPickle
import itertools
import optparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.stats as sc

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def plot_optimization_trace(opts, trialList, nameList, optimum = 0, title="", Ymin=0, Ymax=0):
    markers = itertools.cycle(['o', 's', '^', 'x'])
    colors = itertools.cycle(['b', 'g', 'r', 'k'])
    linestyles = itertools.cycle(['-'])
    size = 1
    labels = list()
    plotH = list()

    # Get handles    
    ratio=5
    gs = gridspec.GridSpec(ratio,1)
    fig = plt.figure(1, dpi=100)
    fig.suptitle(title)
    ax = plt.subplot(gs[0:ratio, :])
    ax.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    # Get values
    minVal = sys.maxint
    maxVal = 0
    maxTrials = 0
    # This might not do what we actually want; Ideally it would only take the
    # ones that where actually run according to instance_order
    for i in range(len(nameList)):
        x = range(len(trialList[i]['trials']))
        y = np.zeros((len(trialList[i]['trials'])))
        line = np.zeros((len(trialList[i]['trials'])))
        for j, trial in enumerate(trialList[i]['trials']):
            inst_res = trial['instance_results']
            if j >= len(y):
                break
            if type(inst_res) == np.ndarray and not np.isfinite(inst_res).any():
                inst_res[np.isnan(inst_res)] = 1
            elif type(inst_res) != np.ndarray and np.isnan(inst_res):
                inst_res = 1
            tmp = sc.nanmean(np.array([inst_res, inst_res]).flat)  # Black Magic
            y[j] = tmp - optimum
            line[j] = np.min(y[:j + 1])
        # Plot the stuff
        marker = markers.next()
        color = colors.next()
        l = linestyles.next()
        ax.scatter(np.argmin(line), min(line), facecolor="w", edgecolor=color, \
                   s=size*10*15, marker=marker)
        ax.scatter(x, y, color=color, marker=marker, s=size*15)
        ax.plot(x, line, color=color, label=nameList[i], linestyle=l, linewidth=size)
        labels.append(trialList[i]['experiment_name'] + " (" + str(min(y)) + ")")
        if min(y) < minVal:
            minVal = min(y)
        if max(y) > maxVal:
            maxVal = max(y)
        if len(trialList[i]) > maxTrials:
            maxTrials = len(trialList[i]['trials'])

    # Describe and label the stuff
    ax.set_xlabel("#Function evaluations")
    ax.set_ylabel("Minfunction value")

    if Ymin == Ymax:
        ax.set_ylim([minVal-0.1, maxVal+0.1])
    else:
        ax.set_ylim([Ymin, Ymax])
    
    leg = ax.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ax.set_xlim([0, maxTrials])
    plt.subplots_adjust(top=0.85)
    if opts.log:
        plt.yscale('log')
    if opts.save != "":
        plt.savefig(opts.save, dpi=100, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()
    
def main():
    parser = optparse.OptionParser()
    # Options for specific functions
    parser.add_option("-o", "--optimum", type = float, dest = "optimum",
            default = 0, help = "Optimum")
    parser.add_option("-b", "--branin", dest = "branin", action = "store_true",
            default = False, help = "Automatic shift for branin function")
    parser.add_option("-c", "--camel", dest="camelback", action="store_true",
            default = False, help="Automatic shift for camelback function")
    parser.add_option("-i", "--har3", dest = "har3", action = "store_true",
            default = False, help = "Automatic shift for hartmann3 function")
    parser.add_option("-j", "--har6", dest = "har6", action = "store_true",
            default = False, help = "Automatic shift for hartmann6 function")
    parser.add_option("-g", "--gold", dest = "gold", action = "store_true",
            default = False, help = "Automatic shift for goldstein function")
    
    # General Options
    parser.add_option("-l", "--log", dest = "log", action = "store_true",
            default = False, help = "Plot on the log scale")
    parser.add_option("--max", dest = "max", action = "store", default = 0,
            type = float, help = "Maximum of the plot")
    parser.add_option("--min", dest = "min", action = "store", default = 0,
            type = float, help = "Minimum of the plot")
    parser.add_option("-s", "--save", dest = "save", default = "", \
            help = "Where to save plot instead of showing it?")
    parser.add_option("-t", "--title", dest="title", \
        default="", help="Optional supertitle for plot")
    (opts, args) = parser.parse_args()
    
    optimum = opts.optimum
    if opts.branin: optimum = 0.397887
    if opts.camelback: optimum = -1.031628453
    if opts.har3: optimum = -3.86278
    if opts.har6: optimum = -3.32237
    if opts.gold: optimum = -3
    
    sys.stdout.write("Found " + str(len(args)) + " pkl file(s)\n")
    trialList = list()
    nameList = list()
    for i in range(len(args)):
        if not ".pkl" in args[i]:
            print "Adding", args[i]
            nameList.append(args[i])
            continue
        result_file = args[i]
        fh = open(result_file, "r")
        trials = cPickle.load(fh)
        fh.close()
        trialList.append(trials)
    sys.stdout.write("Plotting trace\n")
    plot_optimization_trace(opts, trialList, nameList, optimum, title=opts.title, Ymax=opts.max, Ymin=opts.min)

if __name__ == "__main__":
    main()
