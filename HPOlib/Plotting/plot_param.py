#!/usr/bin/env python

import cPickle
import optparse
import re
import sys

from matplotlib.pyplot import tight_layout, figure, subplots_adjust, subplot, savefig, show
from matplotlib.gridspec import GridSpec

import numpy as np


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__license__ = "3-clause BSD License"
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


def plot_params(value_list, result_list, name, save="", title="", jitter=0):
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
    min_y = min(result_list)
    max_y = max(result_list)
    y_offset = np.abs(max_y - min_y) * 0.1
    min_y -= y_offset
    max_y += y_offset

    min_x = min(value_list)
    max_x = max(value_list)
    x_offset = np.abs(max_x - min_x) * 0.1
    min_x -= x_offset
    max_x += x_offset

    # Maybe jitter data
    # But before save best values
    best_value = value_list[np.argmin(result_list)]
    best_result = min(result_list)

    for idx in range(len(result_list)):
        result_list[idx] += (np.random.rand(1) - 0.5) * jitter
        value_list[idx] += (np.random.rand(1) - 0.5) * jitter

    # Plot
    ax.scatter(value_list, result_list, facecolor=color, edgecolor=color, marker=marker, s=size*50)
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

    # Save Plot
    tight_layout()
    subplots_adjust(top=0.85)
    if save != "":
        savefig(save, dpi=100, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches="tight", pad_inches=0.1)
    else:
        show()


def main():
    usage = "python plot_param.py WhatIsThis <pathTo.pkl>* [options]"
    parser = optparse.OptionParser()
    parser.add_option("-s", "--save", dest="save", default="",
                      help="Where to save plot instead of showing it?")
    parser.add_option("-p", "--parameter", dest="param", default="",
                      help="Which parameter to plot")
    parser.add_option("-j", "--jitter", dest="jitter", default=0, type=float,
                      help="Jitter data?")
    parser.add_option("-t", "--title", dest="title", default="",
                      help="Choose a supertitle for the plot")
    (opts, args) = parser.parse_args()
    sys.stdout.write("\nFound " + str(len(args)) + " argument[s]...\n")

    title = opts.title

    value_list = list()
    result_list = list()
    param_set = set()
    for fn in args:
        if not ".pkl" in fn:
            print "This is not a .pkl file: %s" % (fn,)
            continue
        print "Loading ", fn
        fh = open(fn, "r")
        trials = cPickle.load(fh)
        fh.close()

        for t in trials["trials"]:
            if opts.param in t["params"]:
                k, value = translate_para(opts.param, t["params"][opts.param])
                value_list.append(float(value.strip("'")))
                result_list.append(t["result"])
            param_set.update(t["params"].keys())

    if len(value_list) == 0:
        print("No values found for param '%s', Available params:\n%s" % (opts.param, "\n".join([p for p in param_set])))
        sys.exit(1)
    else:
        print "Found %s values for %s" % (str(len(value_list)), opts.param)

    plot_params(value_list=value_list, result_list=result_list, name=opts.param, save=opts.save, title=title,
                jitter=opts.jitter)

if __name__ == "__main__":
    main()
