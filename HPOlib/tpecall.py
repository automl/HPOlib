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
from optparse import OptionParser
from functools import partial
from importlib import import_module
import os

import hyperopt

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

"""
def pyll_replace_list_with_dict(search_space, indent = 0):
    ""
    Recursively traverses a pyll search space and replaces pos_args nodes with
    dict nodes.
    ""

    # Convert to apply first. This makes sure every node of the search space is
    # an apply or literal node which makes it easier to traverse the tree
    if not isinstance(search_space, hyperopt.pyll.Apply):
        search_space = hyperopt.pyll.as_apply(search_space)

    if search_space.name == "pos_args":
        print " " * indent + search_space.name, search_space.__dict__
        param_dict = {}
        for pos_arg in search_space.pos_args:
            print " " * indent + pos_arg.name
            #param_dict["key"] = pos_arg
    for param in search_space.inputs():
        pyll_replace_list_with_dict(param, indent=indent+2)

    return search_space
"""


def main():
    # Parse options and arguments
    parser = OptionParser()
    parser.add_option("-p", "--space",
                      dest="spaceFile",
                      help="Where is the space.py located?")
    parser.add_option("-a", "--algoExec",
                      dest="algoExec",
                      help="Which function to load located?")
    parser.add_option("-m", "--maxEvals",
                      dest="maxEvals",
                      help="How many evaluations?")
    parser.add_option("-s", "--seed",
                      dest="seed",
                      default="123",
                      type=int,
                      help="Seed for the TPE algorithm")
    parser.add_option("-r", "--restore",
                      dest="restore",
                      action="store_true",
                      help="When this flag is set state.pkl is restored in " +
                             "the current working directory")
    parser.add_option("--random", default=False, 
                      dest="random",
                      action="store_true",
                      help="Use a random search")
    (options, args) = parser.parse_args()

    # First remove ".py"
    algo, ext = os.path.splitext(os.path.basename(options.algoExec))
    space, ext = os.path.splitext(os.path.basename(options.spaceFile))

    # Then load dict searchSpace and out function cv.py
    import sys
    sys.path.append("./")
    sys.path.append("")
    print os.getcwd()
    module = import_module(space)
    search_space = module.space
    fn = import_module(algo)
    fn = fn.doForTPE
    
    if options.random:
        # We use a random search
        tpe_with_seed = partial(hyperopt.tpe.rand.suggest, seed=int(options.seed))
    else:
        tpe_with_seed = partial(hyperopt.tpe.suggest, seed=int(options.seed))
    
    # Now run TPE, emulate fmin.fmin()
    state_filename = "state.pkl"
    if options.restore:
        # We do not need to care about the state of the trials object since it
        # is only serialized in a synchronized state, there will never be a save
        # with a running experiment
        fh = open(state_filename)
        tmp_dict = cPickle.load(fh)
        domain = tmp_dict['domain']
        trials = tmp_dict['trials']
        print trials.__dict__
    else:
        domain = hyperopt.Domain(fn, search_space, rseed=int(options.seed))
        trials = hyperopt.Trials()
        fh = open(state_filename, "w")
        # By this we probably loose the seed; not too critical for a restart
        cPickle.dump({"trials": trials, "domain": domain}, fh)
        fh.close()
    
    for i in range(int(options.maxEvals) + 1):
        # in exhaust, the number of evaluations is max_evals - num_done
        rval = hyperopt.FMinIter(tpe_with_seed, domain, trials, max_evals=i)
        rval.exhaust()
        fh = open(state_filename, "w")
        cPickle.dump({"trials": trials, "domain": domain}, fh)
        fh.close()

    best = trials.argmin
    print "Best Value found for params:", best

if __name__ == "__main__":
    main()