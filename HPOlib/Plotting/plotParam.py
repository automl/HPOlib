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
import re
import os
import sys

from matplotlib.pyplot import tight_layout, figure, subplots_adjust, subplot,\
    savefig, show, xticks
from matplotlib.gridspec import GridSpec

import numpy as np

from HPOlib.Plotting import plot_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def translate_para(key, value):
    # sanitize all params
    new_name = key
    if "LOG10" in key:
        pos = key.find("LOG10")
        new_name = key[0:pos] + key[pos+5:]
        new_name = new_name.strip("_")
        value = np.power(10, float(value))
    elif "LOG2" in key:
        pos = key.find("LOG2")
        new_name = key[0:pos] + key[pos+4:]
        new_name = new_name.strip("_")
        value = np.power(2, float(value))
    elif "LOG" in key:
        pos = key.find("LOG")
        new_name = key[0:pos] + key[pos+3:]
        new_name = new_name.strip("_")
        value = np.exp(float(value))
    #Check for Q value, returns round(x/q)*q
    m = re.search('Q[0-999]{1,3}', key)
    if m is not None:
        pos = new_name.find(m.group(0))
        tmp = new_name[0:pos] + new_name[pos+3:]
        new_name = tmp.strip("_")
        q = float(m.group(0)[1:])
        value = round(float(value)/q)*q
    return new_name, value


def plot_params(value_list, result_list, name, min_=0.0, max_=0.0, save=None,
                title="", jitter=0, xticks_=None, rotation=0):
    """
    Parameters
    ----------
    value_list : list
        List of the parameter values. Will be plotted on the x axis.

    result_list : list
        List of the accompanying results. Will be plotted on the y axis.

    name : str
        Name of the hyperparameter.

    min : float, optional
        Minimum value on the y axis. If min_ == max_, the minumum and maximum
        of result_list are used.

    max : float, optional
        Maximum value on the y axis.

    save : str or None
        Save the plot to disc. Use save as filename.

    title : str, optional
        Optional suptitle.

    jitter : float, optional
        Jitter the data to make it easier to display.

    xticks_ : list, optional
        List of hyperparameter values in sorted order. This should only be
        used for categorical hyperparameters.

    rotation : int, optional
        Rotate the xticks by this many degrees.
    """
    color = 'k'
    marker = 'o'
    size = 1

    # Get handles
    ratio = 5
    gs = GridSpec(ratio, 4)
    fig = figure(1, dpi=100)
    fig.suptitle(title, fontsize=16)
    ax = subplot(gs[:, :])
    ax.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # Define xlims
    if min_ == max_:
        min_y = min(result_list)
        max_y = max(result_list)
        y_offset = np.abs(max_y - min_y) * 0.1
        min_y -= y_offset
        max_y += y_offset
    else:
        min_y = min_
        max_y = max_
        result_list = np.array(result_list)
        result_list[result_list > max_y] = max_y

    min_x = min(value_list)
    max_x = max(value_list)
    x_offset = np.abs(max_x - min_x) * 0.1
    min_x -= x_offset
    max_x += x_offset

    # Maybe jitter data
    # But before save best values
    best_value = value_list[np.nanargmin(result_list)]
    best_result = np.nanmin(result_list)

    for idx in range(len(result_list)):
        # noinspection PyArgumentList
        result_list[idx] += float((np.random.rand(1) - 0.5) * jitter)
        # noinspection PyArgumentList
        value_list[idx] += float((np.random.rand(1) - 0.5) * jitter)

    # Plot
    ax.scatter(value_list, result_list, facecolor=color, edgecolor='', marker=marker, s=size*50, alpha=0.5)
    ax.scatter(best_value, best_result, facecolor='r',
               edgecolor='r', s=size*150, marker='o', alpha=0.5,
               label="f[%5.3f]=%5.3f" % (best_value, best_result))

    # Describe and label the stuff
    ax.set_xlabel("Value of %s, jitter=%f" % (name, jitter))
    ax.set_ylabel("Minfunction value, jitter=%f" % jitter)

    leg = ax.legend(loc='best', fancybox=True, scatterpoints=1)
    leg.get_frame().set_alpha(0.5)

    # Apply the xlims
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])

    if xticks_:
        xticks(range(1, 1 + len(xticks_)), xticks_, rotation=rotation)

    # Save Plot
    tight_layout()
    subplots_adjust(top=0.85)
    if save != "":
        savefig(save, dpi=100, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches="tight", pad_inches=0.1)
    else:
        show()


def main(pkl_list, name_list, param=None, min_=0.0, max_=0.0,
         save=None, title="", jitter=0, rotation=0):
    """
    Parameters
    ----------
    pkl_list : list of lists
        The first list contains a list for each optimization algorithm. Each
        of these lists lists HPOlib experiment pickles. plotParam will read the
        configuration/result pairs from these pickles. Currently, the first
        list is only allowed to have one entry.

    name_list : list
        Names of the optimizers in pkl_list. Currently, only one optimizer is
        supported.

    param : str or None
        If this is a string, plot a single hyperparameter. Otherwise,
        plot all hyperparameters.

    min : float
        See documentation of :meth:`HPOlib.Plotting.plotParam.plot_params`

    max : float
        See documentation of :meth:`HPOlib.Plotting.plotParam.plot_params`

    save : str or None
        If param is None, save should either be pdf or png. Otherwise,
        see documentation of :meth:`HPOlib.Plotting.plotParam.plot_params`

    title : str
        See documentation of :meth:`HPOlib.Plotting.plotParam.plot_params`

    jitter : float
        See documentation of :meth:`HPOlib.Plotting.plotParam.plot_params`

    rotation : int, optional
        See documentation of :meth:`HPOlib.Plotting.plotParam.plot_params`
    """
    if len(name_list) > 1:
        raise NotImplementedError("No support for more than one experiment")

    if not param:
        params = set()

        for pkl in pkl_list[0]:
            fh = open(pkl, "r")
            trials = cPickle.load(fh)
            fh.close()
            for t in trials["trials"]:
                for k in t["params"]:
                    k = re.sub('^-', '', k)
                    params.add(k)

    else:
        params = [param]

    for param in sorted(params):
        if len(params) > 1:
            try:
                os.mkdir("params")
            except:
                pass

            if save is None or save.lower() not in ["png", "pdf"]:
                save = "png"

            save = "params/%s.%s" % (param, save)

        mod_param = param

        string_to_value_map = dict()

        value_list = list()
        result_list = list()
        param_set = set()
        for pkl in pkl_list[0]:
            fh = open(pkl, "r")
            trials = cPickle.load(fh)
            fh.close()
            for t in trials["trials"]:
                if mod_param in t["params"]:
                    k, value = translate_para(param, t["params"][mod_param])
                    k = re.sub('^-', '', k)
                    # print k, value
                    value = value.strip()
                    try:
                        value = float(value)
                    except:
                        if value in string_to_value_map:
                            value = string_to_value_map[value]
                        else:
                            key = len(string_to_value_map.keys()) + 1
                            string_to_value_map[value] = key
                            value = key
                    if np.isnan(t["result"]):
                        continue
                    value_list.append(value)
                    result_list.append(t["result"])
                    param_set.add(k)

        if len(value_list) == 0:
            print("No values found for param '%s', Available params:\n%s" %
                  (param, "\n".join([p for p in param_set])))
            sys.exit(1)
        else:
            print "Found %s values for %s" % (str(len(value_list)), param)

        plot_params(value_list=value_list, result_list=result_list, name=param,
                    min_=min_, max_=max_, save=save, title=title,
                    jitter=jitter,
                    xticks_=[key for key, value in
                             sorted(string_to_value_map.items(), key=lambda
                                 t: t[1])],
                    rotation=rotation)

if __name__ == "__main__":
    prog = "python plot_param.py WhatIsThis <pathTo.pkl>* "
    description = "Plot one param wrt to minfunction value"

    parser = ArgumentParser(description=description, prog=prog)
    parser.add_argument("-s", "--save", dest="save", default=None,
                        help="Where to save plot instead of showing it?")
    parser.add_argument("--max", dest="max", type=float,
                        default=0, help="Maximum value of the y axis.")
    parser.add_argument("--min", dest="min", type=float,
                        default=0, help="Minimum value of the y axis.")
    parser.add_argument("-p", "--parameter", dest="param", default="",
                        help="Which parameter to plot")
    parser.add_argument("-j", "--jitter", dest="jitter", default=0, type=float,
                        help="Jitter data?")
    parser.add_argument("-t", "--title", dest="title", default="",
                        help="Choose a supertitle for the plot")
    parser.add_argument("-r", "--rotation", default=0, type=int)

    args, unknown = parser.parse_known_args()

    sys.stdout.write("\nFound " + str(len(unknown)) + " argument[s]...\n")
    pkl_list_main, name_list_main = plot_util.get_pkl_and_name_list(unknown)
    main(pkl_list=pkl_list_main, name_list=name_list_main, param=args.param,
         min_=args.min, max_=args.max, save=args.save, title=args.title,
         jitter=args.jitter, rotation=args.rotation)
