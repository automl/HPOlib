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
import optparse
import os
import sys

from matplotlib.colors import LogNorm
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import numpy as np
import math

sys.path.append("../")
import functions.har6 as har6

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def plot_contour(trials, save="", title=""):
    # 6d Hartmann function 
    # constraints:
    # 0 <= xi <= 1, i = 1..6
    # global optimum at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
    # where har6 = -3.32237

    xopt = 0.20169
    yopt = 0.150011
    zopt = 0.476874
    aopt = 0.275332
    bopt = 0.311652
    copt = 0.6573
    optimum = -3.32237

    x = np.zeros(len(trials["trials"]))
    y = np.zeros(len(trials["trials"]))
    z = np.zeros(len(trials["trials"]))
    a = np.zeros(len(trials["trials"]))
    b = np.zeros(len(trials["trials"]))
    c = np.zeros(len(trials["trials"]))
    result = np.zeros(len(trials["params"]))
    for i in range(len(x)):
        x[i] = trials["trials"][i]["params"]["x"]
        y[i] = trials["trials"][i]["params"]["y"]
        z[i] = trials["trials"][i]["params"]["z"]
        a[i] = trials["trials"][i]["params"]["a"]
        b[i] = trials["trials"][i]["params"]["b"]
        c[i] = trials["trials"][i]["params"]["c"]
        result[i] = trials['results'][i] - optimum
        if math.isnan(x[i]) or math.isnan(y[i]) or math.isnan(result[i]):
            print "#################sdfkdfs!"
    maxRes = max(result)
    minRes = min(result)

    f = plt.figure(1, figsize = (10, 10), dpi = 100)
    
    # Find minimum to generate title
    if trials["results"] != []:
        minimum = str(min(trials["results"]))
    else:
        minimum = "NaN"
    f.suptitle(title + \
        " with Hartmann6d\nUsed " + str(len(x)) + " trials, " \
           + "best found value: " + minimum + " (-3.32237)", fontsize=16)
    
    plt.subplot(611)
    size = 10
    aplt = plt.scatter(x[0:10], result[0:10], marker="^", \
        s=size, facecolors="none", edgecolors="k", label="first 10 points")
    bplt = plt.scatter(x[10:-10], result[10:-10], marker="o", \
        s=4*size, facecolors="none", edgecolors="k", label="points in the middle")
    cplt = plt.scatter(x[-10:-1], result[-10:-1], marker="*", \
        s=4*4*size, facecolors="none", edgecolors="k", label="last 10 points")
    plt.plot([xopt, xopt], [0, maxRes], color="r")
    plt.xlim([0, 1])
    plt.ylim([0, maxRes])
    plt.xlabel("X")
    plt.ylabel("Result - Optimum")
    #plt.yscale('log')
    leg = plt.legend((aplt, bplt, cplt), (aplt.get_label(), bplt.get_label(), \
            cplt.get_label()), fancybox=True, loc="best")
    leg.get_frame().set_alpha(0.5)

    plt.subplot(612)
    aplt = plt.scatter(y[0:10], result[0:10], marker="^", \
        s=size, facecolors="none", edgecolors="k", label="first 10 points")
    bplt = plt.scatter(y[10:-10], result[10:-10], marker="o", \
        s=4*size, facecolors="none", edgecolors="k", label="points in the middle")
    cplt = plt.scatter(y[-10:-1], result[-10:-1], marker="*", \
        s=4*4*size, facecolors="none", edgecolors="k", label="last 10 points")
    plt.plot([yopt, yopt], [0, maxRes], color="r")
    plt.xlim([0, 1])
    plt.ylim([0, maxRes])
    plt.xlabel("Y")
    plt.ylabel("Result - Optimum")
    #plt.yscale('log')
    leg = plt.legend((aplt, bplt, cplt), (aplt.get_label(), bplt.get_label(), \
        cplt.get_label()), fancybox=True, loc="best")
    leg.get_frame().set_alpha(0.5)

    plt.subplot(613)
    aplt = plt.scatter(z[0:10], result[0:10], marker="^", \
        s=size, facecolors="none", edgecolors="k", label="first 10 points")
    bplt = plt.scatter(z[10:-10], result[10:-10], marker="o", \
        s=4*size, facecolors="none", edgecolors="k", label="points in the middle")
    cplt = plt.scatter(z[-10:-1], result[-10:-1], marker="*", \
        s=4*4*size, facecolors="none", edgecolors="k", label="last 10 points")
    plt.plot([zopt, zopt], [0, maxRes], color="r")

    plt.xlim([0, 1])
    plt.ylim([0, maxRes])
    plt.xlabel("Z")
    plt.ylabel("Result - Optimum")
    #plt.yscale('log')
    leg = plt.legend((aplt, bplt, cplt), (aplt.get_label(), bplt.get_label(), \
        cplt.get_label()), fancybox=True, loc="best")
    leg.get_frame().set_alpha(0.5)

    plt.subplot(614)
    aplt = plt.scatter(a[0:10], result[0:10], marker="^", \
        s=size, facecolors="none", edgecolors="k", label="first 10 points")
    bplt = plt.scatter(a[10:-10], result[10:-10], marker="o", \
        s=4*size, facecolors="none", edgecolors="k", label="points in the middle")
    cplt = plt.scatter(a[-10:-1], result[-10:-1], marker="*", \
        s=4*4*size, facecolors="none", edgecolors="k", label="last 10 points")
    plt.plot([zopt, zopt], [0, maxRes], color="r")

    plt.xlim([0, 1])
    plt.ylim([0, maxRes])
    plt.xlabel("A")
    plt.ylabel("Result - Optimum")
    #plt.yscale('log')
    leg = plt.legend((aplt, bplt, cplt), (aplt.get_label(), bplt.get_label(), \
        cplt.get_label()), fancybox=True, loc="best")
    leg.get_frame().set_alpha(0.5)
    
    
    plt.subplot(615)
    aplt = plt.scatter(b[0:10], result[0:10], marker="^", \
        s=size, facecolors="none", edgecolors="k", label="first 10 points")
    bplt = plt.scatter(b[10:-10], result[10:-10], marker="o", \
        s=4*size, facecolors="none", edgecolors="k", label="points in the middle")
    cplt = plt.scatter(b[-10:-1], result[-10:-1], marker="*", \
        s=4*4*size, facecolors="none", edgecolors="k", label="last 10 points")
    plt.plot([zopt, zopt], [0, maxRes], color="r")

    plt.xlim([0, 1])
    plt.ylim([0, maxRes])
    plt.xlabel("B")
    plt.ylabel("Result - Optimum")
    #plt.yscale('log')
    leg = plt.legend((aplt, bplt, cplt), (aplt.get_label(), bplt.get_label(), \
        cplt.get_label()), fancybox=True, loc="best")
    leg.get_frame().set_alpha(0.5)
    
    
    plt.subplot(616)
    aplt = plt.scatter(c[0:10], result[0:10], marker="^", \
        s=size, facecolors="none", edgecolors="k", label="first 10 points")
    bplt = plt.scatter(c[10:-10], result[10:-10], marker="o", \
        s=4*size, facecolors="none", edgecolors="k", label="points in the middle")
    cplt = plt.scatter(c[-10:-1], result[-10:-1], marker="*", \
        s=4*4*size, facecolors="none", edgecolors="k", label="last 10 points")
    plt.plot([zopt, zopt], [0, maxRes], color="r")

    plt.xlim([0, 1])
    plt.ylim([0, maxRes])
    plt.xlabel("C")
    plt.ylabel("Result - Optimum")
    #plt.yscale('log')
    leg = plt.legend((aplt, bplt, cplt), (aplt.get_label(), bplt.get_label(), \
        cplt.get_label()), fancybox=True, loc="best")
    leg.get_frame().set_alpha(0.5)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save != "":
        plt.savefig(save, dpi=100, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()
    f.clear()

def main():
    parser = optparse.OptionParser()
    parser.add_option("-s", "--save", dest = "save", default = "", \
            help = "Where to save plot instead of showing it?")
    (opts, args) = parser.parse_args()

    if opts.save != "":
        sys.stdout.write("Creating %s plots\n" % (len(args),) )
        for idx,i in enumerate(args):
            result_file = i
            print i
            if os.path.exists(result_file) != True:
                sys.stderr.write("No valid file: %s\n" % (result_file,) )
            fh = open(result_file, "r")
            trials = cPickle.load(fh)
            fh.close()
            idxx = result_file.find(trials["experiment_name"])
            title = result_file[idxx:-(len(trials["experiment_name"])+5)]
            plot_contour(trials, save=opts.save + title, title=title)
            print len(trials['results'])
    else:
        result_file = args[0]
        fh = open(result_file, "r")
        trials = cPickle.load(fh)
        fh.close()

        """
        for i in range(10):
            print "x:" + str(trials["params"][i]["x"]) + \
                  " y:" + str(trials["params"][i]["y"]) + \
                  " z:" + str(trials["params"][i]["z"]) + \
                  " a:" + str(trials["params"][i]["a"]) + \
                  " b:" + str(trials["params"][i]["b"]) + \
                  " c:" + str(trials["params"][i]["c"]) + \
                  " res:" + str(trials["results"][i])
        """
        idxx = result_file.find(trials["experiment_name"])
        title = result_file[idxx:-(len(trials["experiment_name"])+5)]          
        plot_contour(trials, title=title)

if __name__ == "__main__":
    main()
