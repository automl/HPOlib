#!/usr/bin/env python

import cPickle
import itertools
import optparse
import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import numpy as np

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__license__ = "3-clause BSD License"
__contact__ = "automl.org"


def plot_boxWhisker(bestDict, title = "", save = "", Ymin=0, Ymax=0):
    ratio = 5
    gs = gridspec.GridSpec(ratio,1)
    fig = plt.figure(1, dpi=100)
    fig.suptitle(title)
    ax = plt.subplot(gs[0:ratio, :])
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    
    labels = bestDict.keys()
    bp = ax.boxplot([bestDict[i] for i in labels], 0, 'ok')    
    boxlines = bp['boxes']
    for line in boxlines:
        line.set_color('k')
        line.set_linewidth(2)

    if Ymin != Ymax:
        ax.set_ylim([Ymin, Ymax])
    # Get medians
    medians = range(len(labels))
    for i in range(len(labels)):
        med = bp['medians'][i]
        medianX = []
        medianY = []
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            medians[i] = medianY[0]
    print medians
    
    # Plot xticks
    plt.xticks(range(1, len(bestDict.keys())+1), labels)
    
    # Print medians as upper labels
    top = Ymax-((Ymax-Ymin)*0.05)
    pos = np.arange(len(labels))+1
    upperLabels = [str(np.round(s, 5)) for s in medians]
    for tick,label in zip(range(len(labels)), ax.get_xticklabels()):
       k = tick % 2
       ax.text(pos[tick], top, upperLabels[tick],
            horizontalalignment='center', size='x-small')

    ax.set_ylabel('Minfunction evaluation')
    if save != "":
        plt.savefig(save, dpi=100, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=True, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()
    
def main():
    parser = optparse.OptionParser()
    parser.add_option("-t", "--title", dest = "title", \
        default = "TBA", help = "Optional supertitle for plot")
    parser.add_option("--max", dest = "max", action = "store", default = 0,
            type = float, help = "Maximum of the plot")
    parser.add_option("--min", dest = "min", action = "store", default = 0,
            type = float, help = "Minimum of the plot")
    parser.add_option("-s", "--save", dest = "save", default = "", \
            help = "Where to save plot instead of showing it?")
    (opts, args) = parser.parse_args()

    title = ""
    if opts.title == "TBA":
        title = "Results for Optimizer(s)"

    sys.stdout.write("\nFound " + str(len(args)) + " arguments...")
    bestDict = dict()
    curOpt = ""
    for i in range(len(args)):
        if not ".pkl" in args[i]:
            print "Adding", args[i]
            bestDict[args[i]] = list()
            curOpt = args[i]
            continue

        result_file = args[i]
        fh = open(result_file, "r")
        trials = cPickle.load(fh)
        fh.close()
        bestDict[curOpt].append(min([trial['result'] for trial in trials['trials']]))

    plot_boxWhisker(bestDict=bestDict, title=title, save=opts.save, \
                        Ymin=opts.min, Ymax=opts.max)

    if opts.save != "":
        sys.stdout.write("Saving plot to " + opts.save + "\n")
    else:
        sys.stdout.write("..Done\n")

if __name__ == "__main__":
    main()
