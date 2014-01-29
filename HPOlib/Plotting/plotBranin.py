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

from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../benchmarks/branin")
import branin

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def plot_contour(trialList, nameList, save = "", title=""):
    # constraints:
    # -5 <= x <= 10, 0 <= y <= 15
    # three global optima:  (-pi, 12.275), (pi, 2.275), (9.42478, 2.475), where
    # branin = 0.397887
    
    markers = itertools.cycle(['o', 's', '^', 'x'])
    colors = itertools.cycle(['b', 'g', 'r', 'k'])
    # linestyles = itertools.cycle(['-'])
    size = 5
    
    # Get handles    
    ratio=5
    gs = gridspec.GridSpec(ratio,1)
    fig = plt.figure(1, dpi=100)
    fig.suptitle(title)
    ax = plt.subplot(gs[0:ratio, :])
    ax.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    xopt = [-np.pi, np.pi, 9.42478]
    yopt = [12.275, 2.275, 2.475]
    
    # Plot Branin
    step = 0.1
    xi = np.arange(-5, 10 + step, step)
    yi = np.arange(-0, 15 + step, step)
    
    z = np.zeros([len(xi), len(yi)])
    for i in range(len(xi)):
        for j in range(len(yi)):
            #z[j, i] = np.power(np.e, branin.branin({"x":xi[i], "y":yi[j]}))
            z[j, i] = branin.branin({"x":xi[i], "y":yi[j]})
    XI, YI = np.meshgrid(xi, yi)
    cax = ax.contourf(XI, YI, z, 50, cmap=cm.gray)
    fig.colorbar(cax)

    # Plot Optimums after all work is done
    d = plt.scatter(xopt, yopt, marker="o", facecolor='w', edgecolor='w', s = 20*size, label="Optimum")

    # Get values
    for opt in range(len(nameList)):
        print nameList[opt], "has", len(trialList[opt]['trials']), "samples"
        m = markers.next()
        c = colors.next()
        x = np.zeros(len(trialList[opt]["trials"]))
        y = np.zeros(len(trialList[opt]["trials"]))
        for i in range(len(x)):
            x[i] = trialList[opt]["trials"][i]["params"]["x"]
            y[i] = trialList[opt]["trials"][i]["params"]["y"]
        a = plt.scatter(x[0:10], y[0:10], marker=m,
            s=size, facecolors=c, linewidth=0.1)  # label=nameList[opt]) # + " first 10 points")
        b = plt.scatter(x[10:-10], y[10:-10], marker=m,
            linewidth=0.1, s=4*size, facecolors=c)  # label="points in the middle")
        c = plt.scatter(x[-10:-1], y[-10:-1], marker=m,
            linewidth=0.1, s=6*size, facecolors=c, label=nameList[opt])  # label="last 10 points")
        
        plt.xlim([-5, 10])
        plt.ylim([-0, 15])
        plt.xlabel("X")
        plt.ylabel("Y")
        if [trial["result"] for trial in trialList[opt]["trials"]]:
            minimum = str(min([trial["result"] for trial in trialList[opt]["trials"]]))
        else:
            minimum = "NaN"
    
    # Describe the plot
    plt.title(title)
    leg = plt.legend(loc="best", fancybox=True)
    leg.get_frame().set_alpha(0.5)
    
    if save != "":
        plt.subplots_adjust(top=0.85)
        plt.savefig(save, dpi=600, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()

def main():
    usage = "Usage: python plotBranin.py SMAC <pathToONEsmacPkl> TPE <..> ..." + \
                        "[-s <whereToSave> ] [-t <something>]"
    parser = optparse.OptionParser(usage)
    parser.add_option("-s", "--save", dest = "save", default = "", \
            help = "Where to save plot instead of showing it?")
    parser.add_option("-t", "--title", dest="title", \
        default="", help="Optional supertitle for plot")
    (opts, args) = parser.parse_args()

    if len(args)%2 != 0:
        print "Wrong number of arguments", len(args)
        print usage
        sys.exit(1)

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

    assert len(nameList) == len(trialList), ("Optimizers: %s != %s" % (len(nameList), len(trialList)))
    plot_contour(trialList=trialList, nameList=nameList, save=opts.save, title=opts.title)
   

if __name__ == "__main__":
    main()

