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
import sys


import numpy as np

import HPOlib.Plotting.plot_util as plot_util
import HPOlib.Plotting.plot_trajectory as plot_trajectory

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def main(pkl_list, name_list, autofill, optimum=0, save="", title="",
         maxvalue=sys.maxint, logy=False, logx=False,
         y_min=None, y_max=None, x_min=None, x_max=None,
         scale_std=1, properties=None,
         aggregation="mean", print_length_trial_list=True,
         ylabel="Minfunction value", xlabel="Duration [sec]"):

    trial_list = list()
    times_list = list()

    for i in range(len(pkl_list)):
        tmp_trial_list = list()
        tmp_times_list = list()
        for pkl in pkl_list[i]:
            fh = open(pkl, "r")
            trials = cPickle.load(fh)
            fh.close()
            trace = plot_util.extract_trajectory(trials, maxvalue=maxvalue)
            times = plot_util.extract_runtime_timestamps(trials=trials)
            if np.isnan(times[-1]):
                print "Last time is nan, removing trial"
                times = times[:-1]
                trace = trace[:-1]
            tmp_times_list.append(times)
            tmp_trial_list.append(trace)
        # We feed this function with two lists of lists and get
        # one list of lists and one list
        tmp_trial_list, tmp_times_list = plot_util.\
            fill_trajectories(tmp_trial_list, tmp_times_list)
        trial_list.append(tmp_trial_list)
        times_list.append(tmp_times_list)

    for i in range(len(trial_list)):
        max_len = max([len(ls) for ls in trial_list[i]])
        for t in range(len(trial_list[i])):
            if len(trial_list[i][t]) < max_len and autofill:
                diff = max_len - len(trial_list[i][t])
                # noinspection PyUnusedLocal
                trial_list[i][t] = np.append(trial_list[i][t],
                                             [trial_list[i][t][-1]
                                              for x in range(diff)])
            elif len(trial_list[i][t]) < max_len and not autofill:
                raise ValueError("(%s != %s), Traces do not have the same "
                                 "length, please use -a" %
                                 (str(max_len), str(len(trial_list[i][t]))))

    plot_trajectory.plot_trajectories(trial_list=trial_list,
                                      name_list=name_list,
                                      x_ticks=times_list,
                                      optimum=optimum,
                                      logy=logy, logx=logx,
                                      aggregation=aggregation,
                                      scale_std=scale_std,
                                      y_max=y_max, y_min=y_min,
                                      x_max=x_max, x_min=x_min,
                                      properties=properties,
                                      print_length_trial_list=
                                      print_length_trial_list,
                                      title=title, save=save,
                                      ylabel=ylabel, xlabel=xlabel)
    return

if __name__ == "__main__":
    prog = "python plotTrace_perTime.py WhatIsThis <oneOrMorePickles> " \
           "[WhatIsThis <oneOrMorePickles>]"
    description = "Plot a Trace with std for multiple experiments"

    parser = ArgumentParser(description=description, prog=prog)

    # Options for specific benchmarks
    parser.add_argument("-o", "--optimum", type=float, dest="optimum",
                        default=0,
                        help="If not set, the optimum is supposed to be zero")

    # Options which are available only for this plot
    parser.add_argument("-a", "--autofill", action="store_true",
                        dest="autofill", default=False,
                        help="Fill trace automatically")
    parser.add_argument("-c", "--scale", type=float, dest="scale",
                        default=1, help="Multiply std to get a nicer plot")
    # General Options
    parser.add_argument("--logy", action="store_true", dest="logy",
                        default=False, help="Plot y on log scale")
    parser.add_argument("--logx", action="store_true", dest="logx",
                        default=False, help="Plot x on log scale")
    parser.add_argument("--ymax", dest="ymax", type=float,
                        default=None, help="Y-Maximum of the plot")
    parser.add_argument("--ymin", dest="ymin", type=float,
                        default=None, help="Y-Minimum of the plot")
    parser.add_argument("--xmax", dest="xmax", type=float,
                        default=None, help="X-Maximum of the plot")
    parser.add_argument("--xmin", dest="xmin", type=float,
                        default=None, help="X-Minimum of the plot")
    parser.add_argument("-s", "--save", dest="save",
                        default="",
                        help="Where to save plot instead of showing it?")
    parser.add_argument("-t", "--title", dest="title",
                        default="", help="Optional supertitle for plot")
    parser.add_argument("--xlabel", dest="xlabel",
                        default="Duration [sec]", help="x axis")
    parser.add_argument("--ylabel", dest="ylabel",
                        default="Minfunction value", help="y label")
    parser.add_argument("--printLength", dest="printlength", default=False,
                        action="store_true",
                        help="Print number of optimizer runs "
                             "in brackets (legend)")
    parser.add_argument("--maxvalue", dest="maxvalue", default=10000,
                        type=float, help="replace all y values higher than this")
    parser.add_argument("--aggregation", dest="aggregation", default="mean",
                        choices=("mean", "median"),
                        help="Print Median/Quantile or Mean/Std")

    # Properties
    # We need this to show defaults for -h
    defaults = plot_util.get_defaults()
    for key in defaults:
        parser.add_argument("--%s" % key, dest=key, default=None,
                            help="%s, default: %s" % (key, str(defaults[key])))

    args, unknown = parser.parse_known_args()

    sys.stdout.write("\nFound " + str(len(unknown)) + " arguments\n")

    pkl_list_main, name_list_main = plot_util.get_pkl_and_name_list(unknown)

    prop = {}
    args_dict = vars(args)
    for key in defaults:
        prop[key] = args_dict[key]

    main(pkl_list_main, name_list_main, autofill=args.autofill,
         optimum=args.optimum, save=args.save, title=args.title,
         logy=args.logy, logx=args.logx,
         maxvalue=args.maxvalue,
         y_min=args.ymin, y_max=args.ymax,
         x_min=args.xmin, x_max=args.xmax,
         scale_std=args.scale,
         aggregation=args.aggregation,
         xlabel=args.xlabel, ylabel=args.ylabel, properties=prop,
         print_length_trial_list=args.printlength)
