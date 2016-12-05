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
from collections import defaultdict
import cPickle
import StringIO
import sys

import numpy

from scipy import stats

from HPOlib.Plotting import plot_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def _get_best_trial(filename, cut=None):
    try:
        fh = open(filename, "r")
        trials = cPickle.load(fh)
        fh.close()

        current_best = numpy.Inf
        best_idx = 0
        if cut is None:
            cut = len(trials['trials'])
        print filename, "#Trials", len(trials['trials'])
        for i, trial in enumerate(trials['trials'][:cut]):
            result = trial['result']
            if result < current_best:
                best_idx = i
                current_best = result
        if current_best == numpy.Inf:
            raise Exception("%s does not contain any results" % filename)
        return current_best, best_idx
    except Exception as e:
        print "Problem with ", filename, e
        sys.stdout.flush()
        return None, None

# TODO: Don't know whether this is longer needed
"""
def _mann_whitney_u(x, y=None):
    # Calculate the Mann-Whitney-U test.
    # The Wilcoxon signed-rank test tests the null hypothesis that two related paired
    # samples come from the same distribution. In particular, it tests whether the
    # distribution of the differences x - y is symmetric about zero.
    # It is a non-parametric version of the paired T-test.

    # A significance-dict for a two-tailed test with 0.05 confidence
    # from http://de.wikipedia.org/wiki/Wilcoxon-Mann-Whitney-Test
    significance_table = \
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
        [0, 0, 0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 13, 14],
        [0, 0, 0, 0, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20],
        [0, 0, 0, 0, 0, 5, 6, 8, 10, 11, 13, 14, 16, 17, 19, 21, 22, 24, 25, 27],
        [0, 0, 0, 0, 0, 0, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34],
        [0, 0, 0, 0, 0, 0, 0, 13, 15, 17, 19, 22, 24, 26, 29, 31, 34, 36, 38, 41],
        [0, 0, 0, 0, 0, 0, 0, 0, 17, 20, 23, 26, 28, 31, 34, 37, 39, 42, 45, 48],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 26, 29, 33, 36, 39, 42, 45, 48, 52, 55],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 33, 37, 40, 44, 47, 51, 55, 58, 62],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 41, 45, 49, 53, 57, 61, 65, 69],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 50, 54, 59, 63, 67, 72, 76],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 59, 64, 69, 74, 78, 83],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 70, 75, 80, 85, 90],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 75, 81, 86, 92, 98],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 87, 93, 99, 105],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 99, 106, 112],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 113, 119],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 127]]
    
    u_value, p = scipy.stats.mannwhitneyu(x, y)

    # noinspection PyNoneFunctionAssignment
    x = asarray(x).compressed().view(numpy.ndarray)
    y = asarray(y).compressed().view(numpy.ndarray)

    n1 = len(x)
    n2 = len(y)
    print n1, n2
    if n1 > n2:
        tmp = n2
        n2 = n1
        n1 = tmp
    
    if n1 < 5 or n2 < 5:
        return 10000, 10000
    if n1 < 20 or n2 < 20:
        print "WARNING: scipy.stat might not be accurate, p value is %f and significance according to table is %s" % \
              (p, u_value <= significance_table[n1 - 1][n2 - 1])
    return u_value, p
"""


# This are some artefacts from the welch test (t-test with unequal variance)
"""if tuple(map(int, (scipy.__version__.split(".")))) >= (0, 11, 0):
                    # print scipy.__version__ >= '0.11.0'
                    t_false, p_false = stats.ttest_ind(best_dict[k], best_dict[j], equal_var=False)
                    rounded_t_false, rounded_p_false = stats.ttest_ind(numpy.round(best_dict[k], 3),
                                                                       numpy.round(best_dict[j], 3),
                                                                       equal_var=False)

                    if p_false < 0.05:
                        if best_dict[k] < best_dict[j]:
                            wins_of_optimizer_welch[k][j] += 1
                        elif best_dict[j] < best_dict[k]:
                            wins_of_optimizer_welch[j][k] += 1

                    output.write("Welch's t-test, no equal population variance\n")
                    output.write(" "*24)
                    output.write(": T: %10.5e, p-value: %10.5e (%5.3f%%)\n" %
                                    (t_false, p_false, p_false*100))
                    output.write("Rounded:                ")
                    output.write(": T: %10.5e, p-value: %10.5e (%5.3f%%)\n" %
                                    (rounded_t_false, rounded_p_false, rounded_p_false*100))
"""


def get_statistics_as_text(pkl_list, name_list, cut=sys.maxint, round_=0):
    pickles = plot_util.load_pickles(name_list, pkl_list)
    best_dict, idx_dict, keys = plot_util.get_best_dict(name_list, pickles,
                                                        cut=cut)

    p_values = calculate_statistics(best_dict, keys, round_=round_)
    output = StringIO.StringIO()
    output.write("Unpaired t-tests-----------------------------------------------------\n")
    output.write("Standard independent 2 sample test, equal population variance\n")

    for key in keys:
        output.write("%10s: %s experiment(s)\n" % (key, len(best_dict[key])))

    for idx, key0 in enumerate(p_values):
        if len(keys) > 1:
            for j, key1 in enumerate(p_values[key0]):
                output.write("%10s vs %10s" % (key0, key1))
                output.write("      p-value: %10.5e (%5.3f%%) \n" %
                                (p_values[key0][key1], p_values[key0][key1]*100))
                output.write("\n")

    output.write("Best Value-----------------------------------------------------------\n")
    for k in keys:
        output.write("%10s: %10.5f (min: %10.5f, max: %10.5f, std: %5.3f)\n" %
                        (k, float(numpy.mean(best_dict[k])), float(numpy.min(best_dict[k])),
                         numpy.max(best_dict[k]), float(numpy.std(best_dict[k]))))

    output.write("Needed Trials--------------------------------------------------------\n")
    for k in keys:
        output.write("%10s: %10.5f (min: %10.5f, max: %10.5f, std: %5.3f)\n" %
                        (k, float(numpy.mean(idx_dict[k])), float(numpy.min(idx_dict[k])),
                         numpy.max(idx_dict[k]), float(numpy.std(idx_dict[k]))))

    output.write("------------------------------------------------------------------------\n")
    output.seek(0)
    return output


def get_p_values(pkl_list, name_list, cut=sys.maxint, round_=0):
    pickles = plot_util.load_pickles(name_list, pkl_list)
    best_dict, idx_dict, keys = plot_util.get_best_dict(name_list, pickles,
                                                       cut=cut)
    p_values = calculate_statistics(best_dict, keys, round_=round_)
    return p_values


def get_pairwise_wins(pkl_list, name_list, cut=sys.maxint, round_=0):
    pickles = plot_util.load_pickles(name_list, pkl_list)
    best_dict, idx_dict, keys = plot_util.get_best_dict(name_list, pickles,
                                                       cut=cut)
    p_values = calculate_statistics(best_dict, keys, round_=round_)

    wins_of_optimizer = dict()
    for key in p_values:
        wins_of_optimizer[key] = defaultdict(int)

    for idx, key0 in enumerate(p_values):
        if len(keys) > 1:
            for j, key1 in enumerate(p_values[key0]):
                if p_values[key0][key1] < 0.05:
                    if best_dict[key0] < best_dict[key1]:
                        wins_of_optimizer[key0][key1] += 1
                    elif best_dict[key1] < best_dict[key0]:
                        wins_of_optimizer[key1][key0] += 1

    return wins_of_optimizer


def calculate_statistics(best_dict, keys, round_=0):
    p_values = dict()
    for key in keys:
        p_values[key] = defaultdict(float)

    for idx, key0 in enumerate(keys):
        if len(keys) > 1:
            for j, key1 in enumerate(keys[idx+1:]):
                if round_ > 0:
                    t_true, p_true = stats.ttest_ind(numpy.round(best_dict[key0], round_),
                                                        numpy.round(best_dict[key1], round_))
                else:
                    t_true, p_true = stats.ttest_ind(best_dict[key0], best_dict[key1])
                p_values[key0][key1] = p_true

    return p_values


if __name__ == "__main__":
    prog = "python statistics.py WhatIsThis <manyPickles> WhatIsThis <manyPickles> [WhatIsThis <manyPickles>]"
    description = "Return some statistical information"

    parser = ArgumentParser(description=description, prog=prog)

    parser.add_argument("-c", "--cut", dest="cut", default=sys.maxint,
                        type=int, help="Only consider that many evaluations.")
    parser.add_argument("--round", dest="round_", default=0, type=int,
                        help="Round the best result before performing the"
                             "statistical test.")
    args, unknown = parser.parse_known_args()

    pkl_list_main, name_list_main = plot_util.get_pkl_and_name_list(unknown)
    output = get_statistics_as_text(pkl_list=pkl_list_main,
                                    name_list=name_list_main,
                                    cut=args.cut, round_=args.round_)
    print output.getvalue()
