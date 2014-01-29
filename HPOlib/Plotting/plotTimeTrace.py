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

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def plot_time_trace(timeDict, nameList, title = "", log=True, save="", verbose=False, \
                    Ymax=0, Ymin=0):
    colors = itertools.cycle(['b', 'g', 'r', 'k'])
    markers = itertools.cycle(['o', 's', 'x', '^'])
    linestyles = itertools.cycle(['-'])
    size = 1
    labels = list()
    ratio = 5
    gs = gridspec.GridSpec(ratio,1)
    fig = plt.figure(figsize=(1,1), dpi=100)
    
    stdDict = dict()
    meanDict = dict()
    numDict = dict()
    minYMean = sys.maxint
    maxYMean = 0
    maxYNum = 0
    maxXNum = 0

    # Get mean and std for all times and optimizers
    for k in nameList:
        stdDict[k] = list()
        meanDict[k] = list()
        numDict[k] = list()
        # iterate over all funEvals
        for i in range(max([len(i) for i in timeDict[k]])):
            timeList = list()
            # check for each #funEval if it exists for this optimizer
            for run in range(len(timeDict[k])):
                if len(timeDict[k][run]) > i:
                    timeList.append(timeDict[k][run][i])
            # calc mean and std
            stdDict[k].append(np.std(np.array(timeList)))
            meanDict[k].append(np.mean(np.array(timeList)))
            numDict[k].append(len(timeList))
        if min(meanDict[k]) < minYMean:
            minYMean = min(meanDict[k])
        if max(meanDict[k]) > maxYMean:
            maxYMean = max(meanDict[k])
        if max(numDict[k]) > maxYNum:
            maxYNum = max(numDict[k])
        if len(numDict[k]) > maxXNum:
            maxXNum = len(numDict[k])

    # Get the handles
    if verbose:
        ax1 = plt.subplot(gs[0:ratio-1, :])

        ax2 = plt.subplot(gs[ratio-1, :], sharex=ax1)
        ax2.set_xlabel("#Function evaluations")
        ax2.set_ylabel("#Runs with #Function evaluations")
        ax2.set_ylim([0, maxYNum+1])
        ax2.set_xlim([0, maxXNum])
        ax2.grid(True)
        fig.subplots_adjust(hspace=0.3)
    else:
        ax1 = plt.subplot(gs[0:ratio, :])
        ax1.set_xlabel("#Function evaluations")
    ax1.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5) 
    
    # Maybe plot on logscale
    if log and minYMean >= 0:
        ax1.set_yscale('log')
    else:
        sys.stdout.write("Plotting on logscale not possible or desired (minVal: %s)\n" % 
                    (minYMean,))
        log = False

    # Plot the stuff
    leg = list()
    for k in nameList:
        # Plot mean and std for optimizer duration
        c = colors.next()
        m = markers.next()
        x = range(len(meanDict[k]))
        l = linestyles.next()
        if not log:
            stdUp = np.array(meanDict[k])+np.array(stdDict[k])
            stdDown = np.array(meanDict[k])-np.array(stdDict[k])
            ax1.fill_between(x, meanDict[k], stdUp, \
                            facecolor=c, alpha=0.3, edgecolor=c)
            ax1.fill_between(x, meanDict[k], stdDown, \
                            facecolor=c, alpha=0.3, edgecolor=c)
        else:
            sys.stdout.write("INFO: Plotting std on a logscale makes no sense\n")
        ax1.plot(x, meanDict[k], color=c, linewidth=size, label=k, linestyle=l, marker=m)
        # Plot number of func evals for this experiment
        x = range(len(numDict[k]))
        if verbose:
            ax2.plot(x, numDict[k], color=c)

    # Descript and label the stuff
    fig.suptitle(title, fontsize=16)
    leg = ax1.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ax1.set_ylabel("Optimizer time in [sec]")
    if Ymax == Ymin:
        ax1.set_ylim([minYMean-2, maxYMean+2])
    else:
        ax1.set_ylim([Ymin, Ymax])
    ax1.set_xlim([0, maxXNum])

    if save != "":
        fig.set_size_inches(10,10)
        plt.savefig(save, dpi=100, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()
    
def main():
    parser = optparse.OptionParser()
    
    # Options which are only available for this plot
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true", \
            default=False, help="Add a plot, which shows #experiments")
    
    # General Options
    parser.add_option("-l", "--nolog", dest="log", action="store_true", \
            default=False, help="Do NOT plot on log scale")
    parser.add_option("--max", dest="max", action="store", default=0,
            type=float, help = "Maximum of the plot")
    parser.add_option("--min", dest="min", action="store", default=0,
            type=float, help="Minimum of the plot")
    parser.add_option("-s", "--save", dest="save", default="", \
            help="Where to save plot instead of showing it?")
    parser.add_option("-t", "--title", dest="title", default="", \
            help="Choose a supertitle for the plot")
    
    
    (opts, args) = parser.parse_args()

    sys.stdout.write("\nFound " + str(len(args)) + " arguments...")
    timeDict = dict()
    nameList = list()
    curOpt = ""
    for i in range(len(args)):
        if not ".pkl" in args[i]:
            print "Adding", args[i]
            timeDict[args[i]] = list()
            nameList.append(args[i])
            curOpt = args[i]
            continue

        result_file = args[i]
        fh = open(result_file, "r")
        trials = cPickle.load(fh)
        fh.close()

        # Get optimizer duration times
        timeList = list()
        # First duration
        timeList.append(trials["cv_starttime"][0] - trials["starttime"][0])
        time_idx = 0
        for i in range(len(trials["cv_starttime"][1:])):
            # Is there a next restored run?
            # if yes, does next cvstart belong to a restored run?
            if len(trials["endtime"]) > time_idx and \
                trials["cv_starttime"][i+1] > trials["endtime"][time_idx]:
                # Check whether run crashed/terminated during cv
                # Equals means that the run crashed
                if trials["cv_endtime"][i] < trials["endtime"][time_idx]:
                    # No .. everything is fine
                    timeList.append((trials["endtime"][time_idx] - trials["cv_endtime"][i]))
                    timeList.append((trials["cv_starttime"][i + 1] - trials["starttime"][time_idx+1]))
                elif trials["endtime"][time_idx] == trials["cv_endtime"][i]:
                    # Yes, but BBoM managed to set an endtime
                    pass
                else:
                    # Yes ... trouble
                    print "Help"
                    print trials["endtime"][time_idx]
                    print trials["cv_endtime"][i]
                time_idx += 1
            # everything ...
            else:
                timeList.append(trials["cv_starttime"][i + 1] - trials["cv_endtime"][i])
        timeDict[curOpt].append(timeList)

    title = opts.title

    plot_time_trace(timeDict, nameList, title=title, log=not opts.log, save=opts.save, \
                    verbose=opts.verbose, Ymin=opts.min, Ymax=opts.max)
    if opts.save != "":
        sys.stdout.write("Saved plot to " + opts.save + "\n")
    else:
        sys.stdout.write("..Done\n")

if __name__ == "__main__":
    main()
