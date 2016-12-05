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

import matplotlib.pyplot as plt
import matplotlib.gridspec
import numpy as np

import HPOlib.Plotting.plot_util as plot_util
import HPOlib.Plotting.plot_trajectory as plot_trajectory

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def main(pkl_list, name_list, autofill, optimum=0, save="", title="",
         log=False, maxvalue=sys.maxint,
         y_min=None, y_max=None, scale_std=1,
         aggregation="mean", cut=sys.maxint,
         xlabel="#Function evaluations", ylabel="Loss",
         properties=None, print_lenght_trial_list=False,
         plot_test_performance=False):

    trial_list = list()
    test_list = list()

    x_ticks = list()
    for i in range(len(pkl_list)):
        trial_list.append(list())
        test_list.append(list())

        for pkl in pkl_list[i]:
            if pkl in plot_util.cache:
                trials = plot_util.cache[pkl]
            else:
                fh = open(pkl, "r")
                trials = cPickle.load(fh)
                fh.close()
                plot_util.cache[pkl] = trials

            if plot_test_performance:
                trace, test = plot_util.extract_trajectory(trials, cut=cut,
                                                           maxvalue=maxvalue,
                                                           test=True)
                # TODO: We can only plot one test result
                if len(test) > 1:
                    raise NotImplementedError("Cannot yet plot more than one testresult")
                else:
                    test = test[0]
                print test_list
                if len(test_list) == 0 or len(test_list[-1]) == 0:
                    test_list[-1].extend([[test[0], ], [test[1], ]])
                else:
                    test_list[-1][0].append(test[0])
                    test_list[-1][1].append(test[1])
            else:
                trace = plot_util.extract_trajectory(trials, cut=cut,
                                                     maxvalue=maxvalue,
                                                     test=False)
            trial_list[-1].append(np.array(trace))

    if not plot_test_performance:
        test_list = None

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
        x_ticks.append(range(np.max([len(ls) for ls in trial_list[i]])))

    plot_trajectory.plot_trajectories(trial_list=trial_list,
                                      name_list=name_list, x_ticks=x_ticks,
                                      optimum=optimum,
                                      y_min=y_min, y_max=y_max,
                                      title=title, ylabel=ylabel, xlabel=xlabel,
                                      logy=log, save=save,
                                      aggregation=aggregation,
                                      properties=properties,
                                      scale_std=scale_std,
                                      test_trials=test_list,
                                      print_length_trial_list=
                                      print_lenght_trial_list)
    return

if __name__ == "__main__":
    prog = "python plotTrace_perEval.py WhatIsThis <oneOrMorePickles> " \
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
    parser.add_argument("-l", "--log", action="store_true", dest="log",
                        default=False, help="Plot on log scale")
    parser.add_argument("--max", dest="max", type=float,
                        default=None, help="Maximum of the plot")
    parser.add_argument("--min", dest="min", type=float,
                        default=None, help="Minimum of the plot")
    parser.add_argument("-s", "--save", dest="save",
                        default="",
                        help="Where to save plot instead of showing it?")
    parser.add_argument("-t", "--title", dest="title",
                        default="", help="Optional supertitle for plot")
    parser.add_argument("--xlabel", dest="xlabel",
                        default="#Function evaluations",
                        help="x axis")
    parser.add_argument("--ylabel", dest="ylabel",
                        default="#Minfunction value", help="y label")
    parser.add_argument("--printLength", dest="printlength", default=False,
                        action="store_true",
                        help="Print number of runs in brackets (legend)")
    parser.add_argument("--aggregation", dest="aggregation", default="mean",
                        choices=("mean", "median"),
                        help="Print Median/Quantile or Mean/Std")
    parser.add_argument("--maxvalue", dest="maxvalue", default=10000,
                        type=float, help="replace all y values higher than this")
    parser.add_argument("--test", dest="test", default=False, action="store_true",
                        help="Print test performances?")

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

    main(pkl_list=pkl_list_main, name_list=name_list_main,
         autofill=args.autofill, optimum=args.optimum, save=args.save,
         title=args.title, log=args.log, maxvalue=args.maxvalue,
         y_min=args.min, y_max=args.max, scale_std=args.scale,
         aggregation=args.aggregation,
         xlabel=args.xlabel, ylabel=args.ylabel, properties=prop,
         print_lenght_trial_list=args.printlength, plot_test_performance=args.test)
