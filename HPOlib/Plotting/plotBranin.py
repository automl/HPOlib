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

import matplotlib.cm
import matplotlib.gridspec as gridSpec
import matplotlib.pyplot
import numpy as np

import HPOlib.benchmarks.benchmark_functions
import HPOlib.Plotting.plot_util as plotUtil

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def plot_contour(trial_list, name_list, save="", title=""):
    # constraints:
    # -5 <= x <= 10, 0 <= y <= 15
    # three global optima:  (-pi, 12.275), (pi, 2.275), (9.42478, 2.475), where
    # branin = 0.397887
    
    markers = itertools.cycle(['o', 's', '^', 'x'])
    colors = itertools.cycle(['b', 'g', 'r', 'k'])
    size = 5
    
    # Get handles    
    ratio = 5
    gs = gridSpec.GridSpec(ratio, 1)
    fig = matplotlib.pyplot.figure(1, dpi=100)
    fig.suptitle(title)
    ax = matplotlib.pyplot.subplot(gs[0:ratio, :])
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
            z[j, i] = HPOlib.benchmarks.benchmark_functions.branin(x=xi[i], y=yi[j])
    xi, yi = np.meshgrid(xi, yi)
    cax = ax.contourf(xi, yi, z, 50, cmap=matplotlib.cm.gray)
    fig.colorbar(cax)

    # Plot Optimums after all work is done
    matplotlib.pyplot.scatter(xopt, yopt, marker="o", facecolor='w', edgecolor='w', s=20*size, label="Optimum")

    # Get values
    for opt in range(len(name_list)):
        print name_list[opt], "has", len(trial_list[opt]['trials']), "samples"
        m = markers.next()
        c = colors.next()
        x = np.zeros(len(trial_list[opt]["trials"]))
        y = np.zeros(len(trial_list[opt]["trials"]))
        for i in range(len(x)):
            if '-x' in trial_list[opt]["trials"][i]["params"]:
                x[i] = float(trial_list[opt]["trials"][i]["params"]["-x"].strip("'"))
                y[i] = float(trial_list[opt]["trials"][i]["params"]["-y"].strip("'"))
            else:
                x[i] = float(trial_list[opt]["trials"][i]["params"]["x"].strip("'"))
                y[i] = float(trial_list[opt]["trials"][i]["params"]["y"].strip("'"))
        matplotlib.pyplot.scatter(x[0:10], y[0:10], marker=m,
                                  s=size, facecolors=c, linewidth=0.1)
        matplotlib.pyplot.scatter(x[10:-10], y[10:-10], marker=m,
                                  linewidth=0.1, s=4*size, facecolors=c)
        matplotlib.pyplot.scatter(x[-10:-1], y[-10:-1], marker=m,
                                  linewidth=0.1, s=6*size, facecolors=c, label=name_list[opt][0])
        
        matplotlib.pyplot.xlim([-5, 10])
        matplotlib.pyplot.ylim([-0, 15])
        matplotlib.pyplot.xlabel("X")
        matplotlib.pyplot.ylabel("Y")
    
    # Describe the plot
    matplotlib.pyplot.title(title)
    leg = matplotlib.pyplot.legend(loc="best", fancybox=True)
    leg.get_frame().set_alpha(0.5)
    
    if save != "":
        matplotlib.pyplot.subplots_adjust(top=0.85)
        matplotlib.pyplot.savefig(save, dpi=600, facecolor='w', edgecolor='w',
                                  orientation='portrait', papertype=None, format=None,
                                  transparent=False, bbox_inches="tight", pad_inches=0.1)
    else:
        matplotlib.pyplot.show()


def main():
    prog = "python plotBranin.py whatIsThis <onepkl> [whatIsThis] <onepkl>]"
    description = "Plot a Trace with std for multiple experiments"

    parser = ArgumentParser(description=description, prog=prog)

    parser.add_argument("-s", "--save", dest="save", default="",
                        help="Where to save plot instead of showing it?")
    parser.add_argument("-t", "--title", dest="title", default="",
                        help="Optional supertitle for plot")

    args, unknown = parser.parse_known_args()

    if len(unknown) % 2 != 0:
        print "Wrong number of arguments", len(args)
        print prog
        sys.exit(1)

    pkl_list, name_list = plotUtil.get_pkl_and_name_list(unknown)
    trial_list = list()
    for i in range(len(name_list)):
        result_file = pkl_list[i][0]
        fh = open(result_file, "r")
        trials = cPickle.load(fh)
        fh.close()
        trial_list.append(trials)

    plot_contour(trial_list=trial_list, name_list=name_list, save=args.save, title=args.title)
   

if __name__ == "__main__":
    main()