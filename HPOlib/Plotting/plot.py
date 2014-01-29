#!/usr/bin/env python

import cPickle
import optparse
import os
import re
import sys

from matplotlib.colors import LogNorm
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import numpy as np

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__license__ = "3-clause BSD License"
__contact__ = "automl.org"


def translatePara(key, value):
    # sanitize all params
    newName = key
    if "LOG10" in key:
        pos = key.find("LOG10")
        newName = key[0:pos] + key[pos+5:]
        newName = newName.strip("_")
        value = np.power(10, float(value))
    elif "LOG2" in key:
        pos = key.find("LOG2")
        newName = key[0:pos] + key[pos+4:]
        newName = newName.strip("_")
        value = np.power(2, float(value))
    elif "LOG" in key:
        pos = key.find("LOG")
        newName = key[0:pos] + key[pos+3:]
        newName = newName.strip("_")
        value = np.exp(float(value))
    #Check for Q value, returns round(x/q)*q
    m = re.search('Q[0-999]{1,3}', key)
    if m != None:
        pos = newName.find(m.group(0))
        tmp = newName[0:pos] + newName[pos+3:]
        newName = tmp.strip("_")
        q = float(m.group(0)[1:])
        value = round(float(value)/q)*q
    return (newName, value)

def plot_params(paramDict, save="", title=""):    
    numPara = len(paramDict.keys())
    size = 10
    
    if save == "":
        # Show all paras in one plot
        f = plt.figure(figsize = (1, 5), dpi = 100)
        f.suptitle(title)
        
        plotNum = 1

        for para in paramDict.keys():
          plt.subplot(numPara, 1, plotNum)
          plotNum += 1
          plt.scatter(paramDict[para][:,0], paramDict[para][:,1], marker="^", \
            s=size, facecolors="none", edgecolors="k")

        plt.subplots_adjust(hspace=0.5)
        plt.show()
    else:
        # Make one plot for each para        
        for para in paramDict.keys():
            fig, ax = plt.subplots(dpi=100)
            fig.suptitle(title + " " + para)
            ax.scatter(paramDict[para][:,0], paramDict[para][:,1], marker="^", \
                        s=size, facecolors="none", edgecolors="k")
            
            save_fn = save + para
            try:
                fig.savefig(save_fn + ".png", dpi=100, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches="tight", pad_inches=0.1)
                fig.savefig(save_fn + ".pdf", dpi=100, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches="tight", pad_inches=0.1)
                fig.clear()
            except Exception as e:
                print e
    
def main():
    parser = optparse.OptionParser()
    parser.add_option("-s", "--save", dest = "save", default = "", \
            help = "Where to save plot instead of showing it?")
    parser.add_option("-t", "--title", dest="title", default="TBA", \
            help = "Choose a supertitle for the plot")
    (opts, args) = parser.parse_args()
    sys.stdout.write("\nFound " + str(len(args)) + " arguments...\n")
    paramDict = dict()
    curOpt = args[0]
    
    if opts.title == "TBA":
        title = curOpt
    else:
        title = opts.title

    for fn in args[1:]:
        if not ".pkl" in fn:
            print "This is not a .pkl file: %s" % (fn,)
            continue
        result_file = fn
        fh = open(result_file, "r")
        trials = cPickle.load(fh)
        fh.close()

        keys = set()
        for i in range(len(trials["trials"])):
            for k in trials["trials"][i]["params"].keys():
                key, value = translatePara(k, trials["trials"][i]["params"][k])
                if key not in paramDict:
                    paramDict[key] = list()
                else:
                    paramDict[key].append((value, trials["trials"][i]["result"]))

    for p in paramDict.keys():
        paramDict[p] = np.array(paramDict[p])

    plot_params(paramDict, save=opts.save, title=title)

if __name__ == "__main__":
    main()

