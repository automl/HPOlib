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

import sys
import logging

import matplotlib.pyplot
import matplotlib.gridspec
import numpy

import HPOlib.Plotting.plot_util as plot_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
hpolib_logger = logging.getLogger("HPOlib")
hpolib_logger.setLevel(logging.INFO)
logger = logging.getLogger("HPOlib.plot_trajectory")


def plot_trajectories(trial_list, name_list, x_ticks,
                      test_trials=None, optimum=0,
                      aggregation="mean", scale_std=1, legend_loc='best',
                      logy=False, logx=False, properties=None,
                      y_max=None, y_min=None, x_max=None, x_min=0,
                      print_length_trial_list=True, baseline=None,
                      ylabel="Loss", xlabel=None, title="", save=""):
    """Plot trajectory
    Plots a given
    Parameters
    ----------
    trial_list: list of lists, [n x m]
    name_list: list of strings, [n]
    x_ticks: list of lists, [n x m] OR list, [m]
    optimum: float, optional
    log: bool, optional
    aggregation: {'mean', 'median'}, optional
    scale_std: bool, optional (ignored if !std)
    legend_loc: Location for legend in plot
    y_max, y_min: float, optional
    properties: dictionary, optional (for keys see plot_util)
    print_lenght_trial_list: bool, optional
    baseline: if not None, plot line parallel to x-axis with this value
    maxvalue: float, replace all performances higher than this
    save: str, optional
    title, ylabel, xlabel: str, optional

    Returns
    ----------
    nothing
    """

    logger.info("Aggregate %s" % aggregation)

    # complete properties
    if properties is None:
        properties = dict()
    properties = plot_util.fill_with_defaults(properties)

    ratio = 5
    gs = matplotlib.gridspec.GridSpec(ratio, 1)
    fig = matplotlib.pyplot.figure(1, dpi=100)
    fig.suptitle(title, fontsize=properties['titlefontsize'])
    ax1 = matplotlib.pyplot.subplot(gs[0:ratio, :])
    ax1.grid(True, linestyle='-', which='major', color=properties["gridcolor"],
             alpha=properties["gridalpha"])

    # Values to autoscale
    min_val = sys.maxint
    max_val = -sys.maxint
    max_trials = 0

    # Sanity check
    x_ticks = numpy.array(x_ticks)
    if len(x_ticks) > 1 and (isinstance(x_ticks[0], numpy.ndarray) or
                                 isinstance(x_ticks[0], list)):
        # We have different xticks for each trial
        assert x_ticks.shape[0] == len(trial_list)

    fig.suptitle(title, fontsize=properties["titlefontsize"])

    # One trialList represents all runs from one optimizer
    for i in range(len(trial_list)):
        if isinstance(x_ticks[0], numpy.ndarray) or \
                isinstance(x_ticks[0], list):
            x = x_ticks[i]
        else:
            x = x_ticks

        performance = numpy.array(trial_list[i]) - optimum

        if aggregation == "mean":
            m = numpy.mean(performance, axis=0)
        else:
            m = numpy.median(performance, axis=0)

        if logy:
            m = numpy.log10(m)

        if logx:
            x = [max(0.00001, xe) for xe in x]

        if aggregation == "mean":
            lower = m - numpy.std(performance, axis=0) * scale_std
            upper = m + numpy.std(performance, axis=0) * scale_std
        else:
            lower = numpy.percentile(performance, axis=0, q=25)
            upper = numpy.percentile(performance, axis=0, q=75)

        assert len(m) == len(x), "%d != %d" % (len(m), len(x))

        marker = properties["markers"].next()
        color = properties["colors"].next()
        linestyle = properties["linestyles"].next()
        label = name_list[i][0]

        if print_length_trial_list:
            label += "(" + str(len(trial_list[i])) + ")"

        ax1.fill_between(x, lower, upper, facecolor=color, alpha=0.3,
                         edgecolor=color)
        ax1.plot(x, m, linewidth=properties["linewidth"],
                 linestyle=linestyle, color=color,
                 marker=marker, markersize=properties["markersize"],
                 label=label)
        if test_trials is not None:
            ax1.scatter(x=test_trials[i][:][0], y=test_trials[i][:][1],
                        color=color, marker='o', s=20)

        # Start from second entry as first entry is always maxint
        min_val = numpy.min([numpy.min(lower[1:]), min_val])
        max_val = numpy.max([numpy.max(upper[1:]), max_val])
        max_trials = numpy.max([x[-1], max_trials])
        if test_trials is not None:
            min_val = numpy.min([numpy.min(test_trials[i][1]), min_val])
            max_val = numpy.max([numpy.max(test_trials[i][1]), max_val])

    # Set y, x label
    if logy:
        ylabel = "log10(%s)" % ylabel
        if y_max is not None:
            y_max = numpy.log10(y_max)
        if y_min is not None:
            y_min = numpy.log10(y_min)

    if scale_std != 1:
       ylabel = "%s, %s * std" % (ylabel, scale_std)
    if logx:
        ax1.set_xscale('log')

    ax1.set_ylabel(ylabel, fontsize=properties["labelfontsize"])
    ax1.set_xlabel(xlabel, fontsize=properties["labelfontsize"])

    # Set legend
    leg = ax1.legend(loc=legend_loc, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    if y_max is None and y_min is None:
        # autoscale upper bound of y axis
        ax1.set_ylim([min_val - 0.1 * abs((max_val - min_val)),
                      max_val + 0.1 * abs((max_val - min_val))])
    elif y_max is None and y_min is not None:
        assert y_min < max_val, "%f < %f" % (y_min, max_val)
        ax1.set_ylim([y_min,
                      max_val + 0.1 * abs((max_val - min_val))])
    elif y_max is not None and y_min is None:
        assert min_val < y_max, "%f < %f" % (min_val, y_max)
        ax1.set_ylim([min_val - 0.1 * abs((max_val - min_val)),
                      y_max])
    else:
        assert y_min < y_max, "%f < %f" % (y_min, y_max)
        ax1.set_ylim([y_min, y_max])

    ax1.set_xlim([0, max_trials])
    if x_max is not None:
        ax1.set_xlim([ax1.get_xlim()[0], x_max])
    if x_min is not None:
        ax1.set_xlim([x_min, ax1.get_xlim()[1]])

    if baseline is not None:
        for base in baseline:
            ax1.plot(ax1.get_xlim(), [base[0], base[0]], c="k", linestyle="--",
                     linewidth=properties["linewidth"], zorder=1)
            ax1.text(ax1.get_xlim()[0] * 1.01,
                     base[0] + 0.02 * (ax1.get_ylim()[1] - ax1.get_ylim()[0]),
                     base[1])

    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.subplots_adjust(top=0.85)
    if save != "":
        logger.info("Save plot to %s" % save)
        matplotlib.pyplot.savefig(save, dpi=int(properties["dpi"]),
                                  facecolor='w', edgecolor='w',
                                  orientation='portrait',
                                  papertype=None, format=None,
                                  transparent=False, bbox_inches="tight",
                                  pad_inches=0.1)
        matplotlib.pyplot.clf()
        matplotlib.pyplot.close()
    else:
        matplotlib.pyplot.show()
    return