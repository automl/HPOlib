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

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

from argparse import ArgumentParser
import cPickle
import os
import sys

import numpy as np

from HPOlib.Plotting import plot_util


if __name__ == "__main__":
    prog = "python getTopK.py <oneOrMorePickles>"
    description = "Outputs top K configurations"

    parser = ArgumentParser(description=description, prog=prog)

    parser.add_argument("-k", dest="k", type=int,
                        default=10, help="How many outputs?")
    parser.add_argument("-i", dest="invers", action="store_true",
                        default=False, help="Plot worst k configs")
    parser.add_argument("-a", dest="additional", action="store_true",
                        default=False, help="Show additional info")

    args, unknown = parser.parse_known_args()

    sys.stdout.write("\nFound " + str(len(unknown)) + " arguments\n")

    pkl_list = list()
    result_dict = dict()
    print unknown
    for pkl in unknown:
        if not os.path.exists(pkl):
            print "%s does not exist" % pkl
        else:
            trials = cPickle.load(file(pkl))
            for trial in trials['trials']:
                if not np.isfinite(trial['result']):
                    continue
                if trial['result'] in result_dict:
                    result_dict[trial['result']].append(trial)
                else:
                    result_dict[trial['result']] = list([trial, ])
    results = result_dict.keys()
    results.sort()

    if args.invers:
        results.reverse()

    topK = list()
    ct = 0

    if args.k > len(result_dict):
        args.k = len(result_dict)
        print "Reduce number of configurations"

    while len(topK) < args.k:
        topK.append(result_dict[results[ct]])
        ct += 1
    print "Found %d different results" % len(result_dict)
    for k in topK:
        print "Result = %10f, Time = %10f, " % (k[0]['result'], k[0]['duration']),\
            ", ".join(["%s = %3s" % (key.strip('-'), k[0]['params'][key]) for key in k[0]['params']]),
        if args.additional:
            print ", additional_info = %s" % str(k[0]['additional_data'])
        else:
            print
