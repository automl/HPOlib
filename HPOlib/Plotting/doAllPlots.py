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
import os
import subprocess
import sys
import time

from HPOlib.Plotting import plot_util
from HPOlib.wrapping_util import format_traceback


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def _plot_trace(pkl_list, name_list, save="", cut=sys.maxint, log=False):
    # We have one pkl per experiment

    plotting_dir = os.path.dirname(os.path.realpath(__file__))
    cur_dir = os.getcwd()

    # noinspection PyBroadException
    try:
        os.chdir(plotting_dir)
        import plotTrace
        plotTrace.main(pkl_list, name_list, save=save, log=log, cut=cut)
        os.chdir(cur_dir)
        sys.stdout.write("passed\n")
    except Exception, e:
        sys.stderr.write(format_traceback(sys.exc_info()))
        sys.stderr.write("failed: %s %s" % (sys.exc_info()[0], e))


def _trace_with_std_per_eval(pkl_list, name_list, maxvalue, save="",
                             cut=sys.maxint, log=False, aggregation="mean"):
    plotting_dir = os.path.dirname(os.path.realpath(__file__))
    cur_dir = os.getcwd()

    # noinspection PyBroadException
    try:
        os.chdir(plotting_dir)
        import plotTrace_perEval
        plotTrace_perEval.main(pkl_list, name_list, autofill=True,
                               aggregation=aggregation, optimum=0,
                               maxvalue=maxvalue, save=save, log=log,
                               print_lenght_trial_list=True)
        os.chdir(cur_dir)
        sys.stdout.write("passed\n")
    except Exception, e:
        sys.stderr.write(format_traceback(sys.exc_info()))
        sys.stderr.write("failed: %s %s" % (sys.exc_info()[0], e))


def _trace_with_std_per_time(pkl_list, name_list, maxvalue, save="",
                             cut=sys.maxint, log=False, aggregation="mean"):
    plotting_dir = os.path.dirname(os.path.realpath(__file__))
    cur_dir = os.getcwd()

    # noinspection PyBroadException
    try:
        os.chdir(plotting_dir)
        import plotTrace_perTime
        plotTrace_perTime.main(pkl_list, name_list, autofill=True,
                               aggregation=aggregation, optimum=0,
                               maxvalue=maxvalue, save=save, logy=log)
        os.chdir(cur_dir)
        sys.stdout.write("passed\n")
    except Exception, e:
        sys.stderr.write(format_traceback(sys.exc_info()))
        sys.stderr.write("failed: %s %s" % (sys.exc_info()[0], e))


def _optimizer_overhead(pkl_list, name_list, save="",
                        cut=sys.maxint, log=False):
    plotting_dir = os.path.dirname(os.path.realpath(__file__))
    cur_dir = os.getcwd()

    # noinspection PyBroadException
    try:
        os.chdir(plotting_dir)
        import plotOptimizerOverhead
        plotOptimizerOverhead.main(pkl_list=pkl_list, name_list=name_list,
                                   autofill=True, title="", log=log,
                                   save=save, cut=cut, ylabel="Time [sec]",
                                   xlabel="#Function evaluations",
                                   aggregation="mean", properties=None,
                                   print_lenght_trial_list=True)
        os.chdir(cur_dir)
        sys.stdout.write("passed\n")
    except Exception, e:
        sys.stderr.write(format_traceback(sys.exc_info()))
        sys.stderr.write("failed: %s %s" % (sys.exc_info()[0], e))


def _box_whisker(pkl_list, name_list, save="", cut=sys.maxint, log=False):
    plotting_dir = os.path.dirname(os.path.realpath(__file__))
    cur_dir = os.getcwd()

    # noinspection PyBroadException
    try:
        os.chdir(plotting_dir)
        import plotBoxWhisker
        plotBoxWhisker.main(pkl_list, name_list, save=save, cut=cut)
        os.chdir(cur_dir)
        sys.stdout.write("passed\n")
    except Exception, e:
        sys.stderr.write(format_traceback(sys.exc_info()))
        sys.stderr.write("failed: %s %s" % (sys.exc_info()[0], e))


def _generate_tex_table(pkl_list, name_list, save="", cut=sys.maxint,
                        log=False):
    plotting_dir = os.path.dirname(os.path.realpath(__file__))
    cur_dir = os.getcwd()
    # noinspection PyBroadException
    try:
        os.chdir(plotting_dir)
        import generateTexTable
        generateTexTable.main(pkl_list, name_list, save, cut)
        os.chdir(cur_dir)
        sys.stdout.write("passed\n")
    except Exception, e:
        sys.stderr.write(format_traceback(sys.exc_info()))
        sys.stderr.write("failed: %s %s" % (sys.exc_info()[0], e))


def _statistics(pkl_list, name_list, save="", cut=sys.maxint, log=False):
    plotting_dir = os.path.dirname(os.path.realpath(__file__))
    cur_dir = os.getcwd()

    # noinspection PyBroadException
    try:
        os.chdir(plotting_dir)
        import statistics
        stringIO = statistics.get_statistics_as_text(pkl_list, name_list, cut)
        os.chdir(plotting_dir)

        if save is not "":
            with open(save, "w") as fh:
                fh.write(stringIO.getvalue())
        else:
            print stringIO.getvalue()
        sys.stdout.write("passed\n")
    except Exception, e:
        sys.stderr.write(format_traceback(sys.exc_info()))
        sys.stderr.write("failed: %s %s" % (sys.exc_info()[0], e))


def main():

    prog = "python doAllPlots.py WhatIsThis <oneOrMorePickles> [WhatIsThis <oneOrMorePickles>]"
    description = "Tries to save as many plots as possible"

    parser = ArgumentParser(description=description, prog=prog)

    # General Options
    parser.add_argument("-l", "--log", action="store_true", dest="log",
                        default=False, help="Plot on log scale")
    parser.add_argument("-s", "--save", dest="save",
                        default="", help="Where to save plots? (directory)")
    parser.add_argument("-f", "--file", dest="file",
                        default="png", help="File ending")
    parser.add_argument("--maxvalue", dest="maxvalue", type=float, required=True,
                        help="Replace all y values higher than this")
    parser.add_argument("-c", "--cut", dest="cut", default=sys.maxint,
                        type=int, help="Cut experiment pickles after a specified number of trials.")

    args, unknown = parser.parse_known_args()

    sys.stdout.write("Found " + str(len(unknown)) + " arguments\n")

    save_dir = os.path.realpath(args.save)

    log = args.log

    pkl_list, name_list = plot_util.get_pkl_and_name_list(unknown)

    if len(pkl_list) == 0:
        raise ValueError("You must at least provide one experiment pickle.")

    time_str = int(time.time() % 1000)

    if not os.path.isdir(save_dir) and save_dir is not "":
        os.mkdir(save_dir)

    for i in range(len(pkl_list)):
        for j in range(len(pkl_list[i])):
            if os.path.exists(pkl_list[i][j]):
                pkl_list[i][j] = os.path.abspath(pkl_list[i][j])
            else:
                raise NotImplementedError("%s is not a valid file" % pkl_list[i][j])

    if len(name_list) == 1 and name_list[0][1] == 1:
        # We have one exp and one pkl
        if save_dir is not "":
            tmp_save = os.path.join(save_dir, "plotTrace_%s.%s" % (time_str, args.file))
        else:
            tmp_save = save_dir
        sys.stdout.write("plotTrace.py ... %s ..." % tmp_save)
        _plot_trace(pkl_list=pkl_list, name_list=name_list, save=tmp_save,
                    log=log, cut=args.cut)

    if len(name_list) > 1:
        # Some plots only make sense, if there are many experiments
        # BoxWhisker
        if save_dir is not "":
            tmp_save = os.path.join(save_dir, "BoxWhisker_%s.%s" % (time_str, args.file))
        else:
            tmp_save = save_dir
        sys.stdout.write("plotBoxWhisker.py ... %s ..." % tmp_save)
        _box_whisker(pkl_list=pkl_list, name_list=name_list, save=tmp_save,
                     log=log, cut=args.cut)

        # LaTeX table
        if save_dir is not "":
            tmp_save = os.path.join(save_dir, "table_%s.tex" % time_str)
        else:
            tmp_save = save_dir
        sys.stdout.write("generateTexTable.py ... %s ..." % tmp_save)
        ret = _generate_tex_table(pkl_list=pkl_list, name_list=name_list,
                                  save=tmp_save, log=log, cut=args.cut)
        if ret is not None:
            print ret


    # We can always plot this
    # OptimizerOverhead
    if save_dir is not "":
        tmp_save = os.path.join(save_dir, "OptimizerOverhead_%s.%s" % (time_str, args.file))
    else:
        tmp_save = save_dir
    sys.stdout.write("plotOptimizerOverhead.py ... %s ..." % tmp_save)
    _optimizer_overhead(pkl_list=pkl_list, name_list=name_list, save=tmp_save,
                        log=log, cut=args.cut)

    # statistics
    if save_dir is not "":
        tmp_save = os.path.join(save_dir, "statistics_%s.txt" % time_str)
    else:
        tmp_save = save_dir
    sys.stdout.write("statistics.py ... %s ..." % tmp_save)
    _statistics(pkl_list=pkl_list, name_list=name_list, save=tmp_save,
                log=log, cut=args.cut)

    # Error Trace with Std
    if save_dir is not "":
        tmp_save = os.path.join(save_dir, "MeanTrace_perEval_%s.%s" % (time_str, args.file))
    else:
        tmp_save = save_dir
    sys.stdout.write("MeanTrace_perEval.py ... %s ..." % tmp_save)
    _trace_with_std_per_eval(pkl_list=pkl_list, name_list=name_list,
                             save=tmp_save, log=log, cut=args.cut,
                             maxvalue=args.maxvalue)

    if save_dir is not "":
        tmp_save = os.path.join(save_dir, "MeanTrace_perTime_%s.%s" % (time_str, args.file))
    else:
        tmp_save = save_dir
    sys.stdout.write("MeanTrace_perTime.py ... %s ..." % tmp_save)
    _trace_with_std_per_time(pkl_list=pkl_list, name_list=name_list,
                             save=tmp_save, log=log, cut=args.cut,
                             maxvalue=args.maxvalue)

if __name__ == "__main__":
    main()