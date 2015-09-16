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

import numpy as numpy

import HPOlib.Plotting.plot_util as plot_util
import HPOlib.Plotting.plot_trajectory as plot_trajectory

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def main(pkl_list, name_list, autofill, title="", log=False, save="",
         y_min=None, y_max=None, cut=sys.maxint, scale_std=1,
         ylabel="Time [sec]", xlabel="#Function evaluations",
         aggregation="mean", properties=None,
         print_lenght_trial_list=True):

    x_ticks = list()
    overhead_list = list()
    for exp in range(len(name_list)):
        tmp_overhead_list = list()
        for pkl in pkl_list[exp]:
            fh = open(pkl, "r")
            trials = cPickle.load(fh)
            fh.close()

            # Get all variables from the trials object
            cv_starttime = trials["cv_starttime"][:cut]
            cv_endtime = trials["cv_endtime"][:cut]

            # Get optimizer duration times
            overhead = list()
            # First duration
            overhead.append(cv_starttime[0] - trials["starttime"][0])
            time_idx = 0
            for i in range(len(cv_starttime[1:])):
                # Is there a next restored run?
                # if yes, does next cvstart belong to a restored run?
                if len(trials["endtime"]) > time_idx and \
                        cv_starttime[i+1] > trials["endtime"][time_idx]:
                    # Check whether run crashed/terminated during cv
                    # Equals means that the run crashed
                    if cv_endtime[i] < trials["endtime"][time_idx]:
                        # No .. everything is fine
                        overhead.append((trials["endtime"][time_idx] -
                                         cv_endtime[i]))
                        overhead.append((cv_starttime[i + 1] -
                                         trials["starttime"][time_idx+1]))
                    elif trials["endtime"][time_idx] == cv_endtime[i]:
                        # Yes, but HPOlib managed to set an endtime
                        pass
                    else:
                        # Yes ... trouble
                        print "Help"
                        print trials["endtime"][time_idx]
                        print cv_endtime[i]
                    time_idx += 1
                # everything ...
                else:
                    overhead.append(cv_starttime[i + 1] - cv_endtime[i])
            tmp_overhead_list.append(overhead)
        overhead_list.append(tmp_overhead_list)

    for i in range(len(overhead_list)):
        max_len = max([len(ls) for ls in overhead_list[i]])
        for t in range(len(overhead_list[i])):
            if len(overhead_list[i][t]) < max_len and autofill:
                diff = max_len - len(overhead_list[i][t])
                # noinspection PyUnusedLocal
                overhead_list[i][t] = numpy.append(overhead_list[i][t],
                                             [overhead_list[i][t][-1]
                                              for x in range(diff)])
            elif len(overhead_list[i][t]) < max_len and not autofill:
                raise ValueError("(%s != %s), Traces do not have the same "
                                 "length, please use -a" %
                                 (str(max_len), str(len(overhead_list[i][t]))))
        x_ticks.append(range(numpy.max([len(ls) for ls in overhead_list[i]])))


    plot_trajectory.plot_trajectories(trial_list=overhead_list,
                                      name_list=name_list, x_ticks=x_ticks,
                                      optimum=0, aggregation=aggregation,
                                      scale_std=scale_std, logy=log,
                                      properties=properties,
                                      y_max=y_max, y_min=y_min,
                                      print_length_trial_list=
                                      print_lenght_trial_list,
                                      ylabel=ylabel, xlabel=xlabel,
                                      title=title, save=save)

if __name__ == "__main__":
    prog = "python plotOptimizerOverhead.py WhatIsThis <oneOrMorePickles> [WhatIsThis <oneOrMorePickles>]"
    description = "Plot a Trace with std for multiple experiments"

    parser = ArgumentParser(description=description, prog=prog)

    # General Options
    parser.add_argument("-c", "--cut", type=int, default=sys.maxint,
                        help="Cut the experiment pickle length.")
    parser.add_argument("-t", "--title", dest="title",
                        default="", help="Optional supertitle for plot")
    parser.add_argument("--xlabel", dest="xlabel",
                        default="Duration [sec]", help="x axis")
    parser.add_argument("--ylabel", dest="ylabel",
                        default="Minfunction value", help="y label")
    parser.add_argument("-s", "--save", dest="save",
                        default="",
                        help="Where to save plot instead of showing it?")
    parser.add_argument("-l", "--log", action="store_true", dest="log",
                        default=False, help="Plot on log scale")
    parser.add_argument("--max", dest="max", type=float,
                        default=None, help="Maximum of the plot")
    parser.add_argument("--min", dest="min", type=float,
                        default=None, help="Minimum of the plot")

    # Options which are available only for this plot
    parser.add_argument("--scale", type=float, dest="scale",
                        default=1, help="Multiply std to get a nicer plot")
    parser.add_argument("-a", "--autofill", action="store_true", dest="autofill",
                        default=False, help="Fill trace automatically")
    parser.add_argument("--printLength", dest="printlength", default=False,
                        action="store_true",
                        help="Print number of optimizer runs "
                             "in brackets (legend)")
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
         save=args.save, title=args.title, log=args.log,
         y_min=args.min, y_max=args.max, cut=args.cut,
         aggregation=args.aggregation, scale_std=args.scale,
         xlabel=args.xlabel, ylabel=args.ylabel, properties=prop,
         print_lenght_trial_list=args.printlength)