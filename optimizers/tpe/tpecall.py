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
from collections import OrderedDict
import cPickle
from functools import partial
from importlib import import_module
import logging
import os
import subprocess
import StringIO
import sys

import numpy as np

import hyperopt

from HPOlib.wrapping_util import flatten_parameter_dict, load_experiment_config_file

logger = logging.getLogger("tpecall")

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def construct_cli_call(cli_target, params):
    cli_call = StringIO.StringIO()
    cli_call.write("python -m ")
    cli_call.write(cli_target)
    cli_call.write(" --params")
    params = flatten_parameter_dict(params)
    params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))
    for param in params:
        cli_call.write(" -" + param + " \"'" + str(params[param]) + "'\"")
    return cli_call.getvalue()


def command_line_function(params, cli_target):
    call = construct_cli_call(cli_target, params)
    proc = subprocess.Popen(call, shell=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    logger.info(stdout)
    if stderr:
        logger.error("STDERR:")
        logger.error(stderr)

    lines = stdout.split("\n")

    result = np.Inf
    for line in lines:
        pos = line.find("Result:")
        if pos != -1:
            result_string = line[pos:]
            result_array = result_string.split()
            result = float(result_array[1].strip(","))
            break

    # Parse the CLI
    return result


def main():
    prog = "python statistics.py WhatIsThis <manyPickles> WhatIsThis <manyPickles> [WhatIsThis <manyPickles>]"
    description = "Return some statistical information"

    parser = ArgumentParser(description=description, prog=prog)

    parser.add_argument("-p", "--space",
                        dest="spaceFile", help="Where is the space.py located?")
    parser.add_argument("-m", "--maxEvals",
                        dest="maxEvals", help="How many evaluations?")
    parser.add_argument("-s", "--seed", default="1",
                        dest="seed", type=int, help="Seed for the TPE algorithm")
    parser.add_argument("-r", "--restore", action="store_true",
                        dest="restore", help="When this flag is set state.pkl is restored in " +
                             "the current working directory")
    parser.add_argument("--random", default=False, action="store_true",
                        dest="random", help="Use a random search")
    parser.add_argument("--cwd", help="Change the working directory before "
                                      "optimizing.")

    args, unknown = parser.parse_known_args()

    if args.cwd:
        os.chdir(args.cwd)

    cfg = load_experiment_config_file()
    log_level = cfg.getint("HPOLIB", "HPOlib_loglevel")
    logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                               'message)s', datefmt='%H:%M:%S')
    logger.setLevel(log_level)

    if not os.path.exists(args.spaceFile):
        logger.critical("Search space not found: %s" % args.spaceFile)
        sys.exit(1)

    # First remove ".py"
    space, ext = os.path.splitext(os.path.basename(args.spaceFile))

    # Then load dict searchSpace and out function cv.py
    sys.path.append("./")
    sys.path.append("")

    module = import_module(space)
    search_space = module.space

    cli_target = "HPOlib.optimization_interceptor"
    fn = partial(command_line_function, cli_target=cli_target)
    
    if args.random:
        # We use a random search
        tpe_with_seed = partial(hyperopt.tpe.rand.suggest, seed=int(args.seed))
        logger.info("Using Random Search")
    else:
        tpe_with_seed = partial(hyperopt.tpe.suggest, seed=int(args.seed))
    
    # Now run TPE, emulate fmin.fmin()
    state_filename = "state.pkl"
    if args.restore:
        # We do not need to care about the state of the trials object since it
        # is only serialized in a synchronized state, there will never be a save
        # with a running experiment
        fh = open(state_filename)
        tmp_dict = cPickle.load(fh)
        domain = tmp_dict['domain']
        trials = tmp_dict['trials']
        print trials.__dict__
    else:
        domain = hyperopt.Domain(fn, search_space, rseed=int(args.seed))
        trials = hyperopt.Trials()
        fh = open(state_filename, "w")
        # By this we probably loose the seed; not too critical for a restart
        cPickle.dump({"trials": trials, "domain": domain}, fh)
        fh.close()
    
    for i in range(int(args.maxEvals) + 1):
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