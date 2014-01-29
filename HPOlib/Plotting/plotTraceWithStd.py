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


def plot_optimization_trace(trialList, nameList, optimum=0, title="", \
                            log=True, save="", Ymax=0, Ymin=0):
    markers = itertools.cycle(['o', 's', 'x', '^'])
    colors = itertools.cycle(['b', 'g', 'r', 'k'])
    linestyles = itertools.cycle(['-'])
    size = 1
    labels = list()
    plotH = list()

    ratio = 5
    gs = gridspec.GridSpec(ratio,1)
    fig = plt.figure(1, dpi=100)
    fig.suptitle(title, fontsize=16)
    ax1 = plt.subplot(gs[0:ratio, :])
    ax1.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    minVal = sys.maxint
    maxVal = 0
    maxTrials = 0

    trialListMeans = list()
    trialListStd = list()
    trialListMax = list()
    trialListMin = list()

    # One trialList represents all runs from one optimizer
    for i in range(len(trialList)):
        trialListMeans.append(np.mean(np.array(trialList[i]), axis=0))
        trialListStd.append(np.std(np.array(trialList[i]), axis=0))
        trialListMax.append(np.max(np.array(trialList[i]), axis=0))
        trialListMin.append(np.min(np.array(trialList[i]), axis=0))

    fig.suptitle(title, fontsize=16)

    # Plot the average error and std
    for i in range(len(trialListMeans)):
        x = range(1,len(trialListMeans[i])+1)
        y = trialListMeans[i] - optimum
        m = markers.next()
        c = colors.next()
        l = linestyles.next()
        if log == False:
            stdUp = y+trialListStd[i]
            stdDown = y-trialListStd[i]
            ax1.fill_between(x, y, stdUp, \
                            facecolor=c, alpha=0.3, edgecolor=c)
            ax1.fill_between(x, y, stdDown, \
                            facecolor=c, alpha=0.3, edgecolor=c)
            #ax1.errorbar(x, y, yerr=trialListStd[i], color='k')
        else:
            sys.stdout.write("INFO: Plotting std on a logscale makes no sense\n")
        ax1.plot(x, y, color=c, linewidth=size, label=nameList[i][0], \
                   linestyle=l, marker=m)
      
        if min(y) < minVal:
            minVal = min(y)
        if max(y) > maxVal:
            maxVal = max(y)
        if len(trialListMeans[i]) > maxTrials:
            maxTrials = len(trialListMeans[i])

    # Maybe plot on logscale
    if float(minVal) > 0.0 and log:
        ax1.set_yscale('log')
    else:
        sys.stdout.write("Plotting on logscale not possible or desired (minVal: %s)\n" % 
                    (minVal,))
        log = False

    # Descript and label the stuff
    leg = ax1.legend(loc='best', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    ax1.set_xlabel("#Function evaluations")
    ax1.set_ylabel("Minfunction value")
    if Ymax == Ymin:
        ax1.set_ylim([minVal-0.1, maxVal+1])
    else:
        ax1.set_ylim([Ymin, Ymax])
    ax1.set_xlim([0, 100])# maxTrials])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save != "":
        plt.savefig(save, dpi=100, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()
    
def main():
    parser = optparse.OptionParser()
    # Options for specific functions
    parser.add_option("-b", "--branin", dest="branin", action="store_true",
            default = False, help="Automatic shift for branin function")
    parser.add_option("-c", "--camel", dest="camelback", action="store_true",
            default = False, help="Automatic shift for camelback function")
    parser.add_option("-i", "--har3", dest="har3", action="store_true",
            default = False, help="Automatic shift for hartmann3 function")
    parser.add_option("-j", "--har6", dest="har6", action="store_true",
            default = False, help="Automatic shift for hartmann6 function")
    parser.add_option("-g", "--gold", dest="gold", action="store_true",
            default = False, help="Automatic shift for goldstein function")
    parser.add_option("-o", "--optimum", type=float, dest="optimum",
            default=0, help="Optimum")
    
    # Options which are available only for this plot
    parser.add_option("-a", "--autofill", dest="autofill", default=False, \
            action="store_true", help="Fill trace automatically")
    parser.add_option("--fill", dest="fill", type=int, default=0,
            help="In case that a trace is not long enough, fill it with the" \
                    + " best result so far until there are <fill> entries")
                    
    # General Options
    parser.add_option("-l", "--nolog", dest="log", action="store_true", \
            default=False, help="Do NOT plot on log scale")
    parser.add_option("--max", dest="max", action="store", default=0,
            type=float, help="Maximum of the plot")
    parser.add_option("--min", dest="min", action="store", default=0,
            type=float, help="Minimum of the plot")
    parser.add_option("-s", "--save", dest = "save", default="", \
            help = "Where to save plot instead of showing it?")
    parser.add_option("-t", "--title", dest="title", \
        default="", help="Optional supertitle for plot")

    (opts, args) = parser.parse_args()

    sys.stdout.write("Generate plot for ")
    optimum = opts.optimum
    title = opts.title
    if opts.branin: 
        optimum = 0.397887
        sys.stdout.write("branin")
    if opts.camelback:
        optimum = -1.031628453
        sys.stdout.write("camelback")
    if opts.har3:
        optimum = -3.86278
        sys.stdout.write("hartmann3")
    if opts.har6:
        optimum = -3.32237
        sys.stdout.write("hartmann6")
    if opts.gold: 
        optimum = 3
        sys.stdout.write("goldstein")

    sys.stdout.write("\nFound " + str(len(args)) + " arguments...")
    nameList = list()
    trialList = list()
    lenList = list()
    for i in range(len(args)):
        if not ".pkl" in args[i]:
            print "Adding", args[i]
            nameList.append([args[i],0])
            trialList.append(list())
            lenList.append(0)
            continue
        else:
            nameList[-1][1] += 1

        result_file = args[i]
        fh = open(result_file, "r")
        trials = cPickle.load(fh)
        fh.close()

        trace = list()
        currentbest = np.nanmax(np.array([trial["result"] for trial in trials['trials']]))

        for result in [trial["result"] for trial in trials['trials']]:
            if result < currentbest:
                currentbest = result
            trace.append(currentbest)
        """
        # Make sure that trace is long enough
        if opts.fill != 0 and opts.fill > len(trace):
            print "Pickle file", args[i], "does only contain", len(trace), "entries"
            trace.extend([currentbest for i in range(opts.fill - len(trace))])
            print len(trace)
        """
        if len(trace) > lenList[-1]:
            lenList[-1] = len(trace)
        trialList[-1].append(np.array(trace))

    if opts.autofill:
        assert(len(trialList) == len(lenList))
        print lenList
        for i in range(len(trialList)):
            for t in range(len(trialList[i])):
                if len(trialList[i][t]) < lenList[i]:
                    diff = lenList[i] - len(trialList[i][t])
                    trialList[i][t] = np.append(trialList[i][t], [trialList[i][t][-1] for x in range(diff)])
                assert len(trialList[i][t]) == lenList[i], \
                        "%s, %s" % (len(trialList[i][t]), lenList[i])

    plot_optimization_trace(trialList, nameList, optimum, title=title, \
                 log=not opts.log, save=opts.save, Ymin=opts.min, Ymax=opts.max)

    if opts.save != "":
        sys.stdout.write("Saved plot to " + opts.save + "\n")
    else:
        sys.stdout.write("..Done\n")

if __name__ == "__main__":
    main()
