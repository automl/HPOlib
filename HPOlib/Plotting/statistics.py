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

import numpy
import scipy
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


def main(pkl_list, name_list, cut=sys.maxint):
    pickles = plot_util.load_pickles(name_list, pkl_list)
    best_dict, idx_dict, keys = plot_util.get_best_dict(name_list, pickles,
                                                       cut=cut)

    for k in keys:
        sys.stdout.write("%10s: %s experiment(s)\n" % (k, len(best_dict[k])))

    sys.stdout.write("Unpaired t-tests-----------------------------------------------------\n")
    # TODO: replace by itertools
    for idx, k in enumerate(keys):
        if len(keys) > 1:
            for j in keys[idx+1:]:
                t_true, p_true = stats.ttest_ind(best_dict[k], best_dict[j])
                rounded_t_true, rounded_p_true = stats.ttest_ind(numpy.round(best_dict[k], 3),
                                                                 numpy.round(best_dict[j], 3))

                sys.stdout.write("%10s vs %10s\n" % (k, j))
                sys.stdout.write("Standard independent 2 sample test, equal population variance\n")
                sys.stdout.write(" "*24 + "  T: %10.5e, p-value: %10.5e (%5.3f%%) \n" %
                                (t_true, p_true, p_true*100))
                sys.stdout.write("Rounded:                ")
                sys.stdout.write("  T: %10.5e, p-value: %10.5e (%5.3f%%)\n" %
                                (rounded_t_true, rounded_p_true, rounded_p_true*100))
                if tuple(map(int, (scipy.__version__.split(".")))) >= (0, 11, 0):
                    # print scipy.__version__ >= '0.11.0'
                    t_false, p_false = stats.ttest_ind(best_dict[k], best_dict[j], equal_var=False)
                    rounded_t_false, rounded_p_false = stats.ttest_ind(numpy.round(best_dict[k], 3),
                                                                       numpy.round(best_dict[j], 3),
                                                                       equal_var=False)
                    sys.stdout.write("Welch's t-test, no equal population variance\n")
                    sys.stdout.write(" "*24)
                    sys.stdout.write(": T: %10.5e, p-value: %10.5e (%5.3f%%)\n" %
                                    (t_false, p_false, p_false*100))
                    sys.stdout.write("Rounded:                ")
                    sys.stdout.write(": T: %10.5e, p-value: %10.5e (%5.3f%%)\n" %
                                    (rounded_t_false, rounded_p_false, rounded_p_false*100))
                sys.stdout.write("\n")

    sys.stdout.write("Best Value-----------------------------------------------------------\n")
    for k in keys:
        sys.stdout.write("%10s: %10.5f (min: %10.5f, max: %10.5f, std: %5.3f)\n" %
                        (k, float(numpy.mean(best_dict[k])), float(numpy.min(best_dict[k])),
                         numpy.max(best_dict[k]), float(numpy.std(best_dict[k]))))

    sys.stdout.write("Needed Trials--------------------------------------------------------\n")
    for k in keys:
        sys.stdout.write("%10s: %10.5f (min: %10.5f, max: %10.5f, std: %5.3f)\n" %
                        (k, float(numpy.mean(idx_dict[k])), float(numpy.min(idx_dict[k])),
                         numpy.max(idx_dict[k]), float(numpy.std(idx_dict[k]))))

    sys.stdout.write("------------------------------------------------------------------------\n")

if __name__ == "__main__":
    prog = "python statistics.py WhatIsThis <manyPickles> WhatIsThis <manyPickles> [WhatIsThis <manyPickles>]"
    description = "Return some statistical information"

    parser = ArgumentParser(description=description, prog=prog)

    parser.add_argument("-c", "--cut", dest="cut", default=sys.maxint,
                        type=int, help="Only consider that many evaluations")
    args, unknown = parser.parse_known_args()

    pkl_list_main, name_list_main = plot_util.get_pkl_and_name_list(unknown)
    main(pkl_list=pkl_list_main, name_list=name_list_main, cut=args.cut)
