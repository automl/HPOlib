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

#!/usr/bin/env python

import cPickle
import optparse
import sys

import numpy
import numpy.ma as ma
from scipy import stats

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def _getBestTrial(filename, cut=None):
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


def _mann_whitney_u(x, y=None):
    """
    Calculate the Mann-Whitney-U test.

    The Wilcoxon signed-rank test tests the null hypothesis that two related paired 
    samples come from the same distribution. In particular, it tests whether the 
    distribution of the differences x - y is symmetric about zero. 
    It is a non-parametric version of the paired T-test.
    """
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
    
    U, p = stats.mannwhitneyu(x, y)
    
    x = ma.asarray(x).compressed().view(numpy.ndarray)
    y = ma.asarray(y).compressed().view(numpy.ndarray)
    
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
                (p, U <= significance_table[n1 - 1][n2 - 1])
    return U, p

def _unpaired_ttest(x, y, equal_var=True):
    U, p = stats.ttest_ind(x, y, equal_var=equal_var)
    return U, p

def main():
    usage = "python statistics.py SMAC smac_*/smac.pkl TPE tpe_*/tpe.pkl\n"
    parser = optparse.OptionParser(usage)
    parser.add_option("-c", "--c", type=int, dest="cut",
                      default=None,
                      help="Only consider that many evaluations")
    (opts, args) = parser.parse_args()

    bestDict = dict()
    idxDict = dict()
    keys = list()
    curOpt = ""
    for i in range(len(args)):
        if not ".pkl" in args[i]:
            bestDict[args[i]] = list()
            idxDict[args[i]] = list()
            keys.append(args[i])
            curOpt = args[i]
            continue
        best, idx =_getBestTrial(args[i], opts.cut)
        if best is None and idx is None:
            continue

        # best = best * 100

        bestDict[curOpt].append(best)
        idxDict[curOpt].append(idx)

    for k in keys:
        sys.stdout.write("%10s: %s experiments\n" % (k, len(bestDict[k])))
    """
    sys.stdout.write("Wilcoxon Test--------------------------------------------------------\n")
    for idx, k in enumerate(keys):
        if len(keys) > 1:
            for jdx, j in enumerate(keys[idx+1:]):
                #if len(bestDict[k]) == len(bestDict[j]):
                T, p = _mann_whitney_u(bestDict[k], bestDict[j])
                sys.stdout.write("%10s vs %10s" % (k,j))
                sys.stdout.write(": T: %10.5e, p-value: %10.5e (%5.3f%%)\n" % 
                    (T, p, p*100))
                #else:
                #    print "WARNING: %s has not the same number of function evals than %s" % \
                #            (k, j)
                #    continue
        else:
            T, p = _mann_whitney_u(bestDict[k], bestDict[k])
            sys.stdout.write("%10s vs. %10s" % (k,k))
            sys.stdout.write(": T: %10.5e, p-value: %10.5e (%5.3f%%)\n" % 
                    (T, p, p*100))
    """
    sys.stdout.write("Unpaired t-tests-----------------------------------------------------\n")
    for idx, k in enumerate(keys):
        if len(keys) > 1:
            for jdx, j in enumerate(keys[idx+1:]):
                #if len(bestDict[k]) == len(bestDict[j]):
                T_t, p_t = _unpaired_ttest(bestDict[k], bestDict[j], equal_var=True)
                T_f, p_f = _unpaired_ttest(bestDict[k], bestDict[j], equal_var=False)
                T_rt, p_rt = _unpaired_ttest(numpy.round(bestDict[k], 3), numpy.round(bestDict[j], 3), equal_var=True)
                T_rf, p_rf = _unpaired_ttest(numpy.round(bestDict[k], 3), numpy.round(bestDict[j], 3), equal_var=False)
                sys.stdout.write("Standard independent 2 sample test, equal population variance\n")
                sys.stdout.write("%10s vs %10s" % (k,j))
                sys.stdout.write(": T: %10.5e, p-value: %10.5e (%5.3f%%) \n" % 
                    (T_t, p_t, p_t*100))
                sys.stdout.write("Rounded:                ")
                sys.stdout.write(": T: %10.5e, p-value: %10.5e (%5.3f%%)\n" % 
                    (T_rt, p_rt, p_rt*100))
                sys.stdout.write("Welch's t-test, no equal population variance\n")
                sys.stdout.write("%10s vs %10s" % (k,j))
                sys.stdout.write(": T: %10.5e, p-value: %10.5e (%5.3f%%)\n" % 
                    (T_f, p_f, p_f*100))
                sys.stdout.write("Rounded:                ")
                sys.stdout.write(": T: %10.5e, p-value: %10.5e (%5.3f%%)\n" % 
                    (T_rf, p_rf, p_rf*100))
    sys.stdout.write("Best Value-----------------------------------------------------------\n")
    for k in keys:
        stdBest = numpy.std(bestDict[k])
        minBest = numpy.min(bestDict[k])
        maxBest = numpy.max(bestDict[k])
        meanBest = numpy.mean(bestDict[k])
        sys.stdout.write("%10s: %10.5f (min: %10.5f, max: %10.5f, std: %5.3f)\n" %
            (k, meanBest, minBest, maxBest, stdBest))

    sys.stdout.write("Needed Trials--------------------------------------------------------\n")
    for k in keys:
        stdIdx = numpy.std(idxDict[k])
        meanIdx = numpy.mean(idxDict[k])
        minIdx = numpy.min(idxDict[k])
        maxIdx = numpy.max(idxDict[k])
        sys.stdout.write("%10s: %10.5f (min: %10.5f, max: %10.5f, std: %5.3f)\n" %
            (k, meanIdx, minIdx, maxIdx, stdIdx))

    sys.stdout.write("------------------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
