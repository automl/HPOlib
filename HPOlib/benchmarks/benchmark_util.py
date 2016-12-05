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

import logging
import sys


logger = logging.getLogger("HPOlib.benchmark_util")


def parse_cli():
    """
    Provide a generic command line interface for benchmarks. It will just parse
    the command line according to simple rules and return two dictionaries, one
    containing all arguments for the benchmark algorithm like dataset,
    crossvalidation metadata etc. and the containing all learning algorithm
    hyperparameters.

    Parsing rules:
    - Arguments with two minus signs are treated as benchmark arguments, Xalues
     are not allowed to start with a minus. The last argument must --params,
     starting the hyperparameter arguments.
    - All arguments after --params are treated as hyperparameters to the
     learning algorithm. Every parameter name must start with one minus and must
     have exactly one value which has to be given in single quotes.
    - Arguments with no value before --params are treated as boolean arguments

    Example:
    python neural_network.py --folds 10 --fold 1 --dataset convex  --params
        -depth '3' -n_hid_0 '1024' -n_hid_1 '1024' -n_hid_2 '1024' -lr '0.01'
    """
    args = {}
    arg_name = None
    arg_values = None
    parameters = {}

    cli_args = sys.argv
    found_params = False
    skip = True
    iterator = enumerate(cli_args)

    for idx, arg in iterator:
        if skip:
            skip = False
            continue
        else:
            skip = True

        if arg == "--params":
            if arg_name:
                args[arg_name] = " ".join(arg_values)
            found_params = True
            skip = False

        elif arg[0:2] == "--" and not found_params:
            if arg_name:
                args[arg_name] = " ".join(arg_values)
            arg_name = arg[2:]
            arg_values = []
            skip = False

        elif arg[0:2] == "--" and found_params:
            raise ValueError("You are trying to specify an argument after the "
                             "--params argument. Please change the order.")

        elif arg[0] == "-" and arg[0:2] != "--" and found_params:
            parameters[cli_args[idx][1:]] = cli_args[idx+1]

        elif arg[0] == "-" and arg[0:2] != "--" and not found_params:
            raise ValueError("You either try to use arguments with only one lea"
                             "ding minus or try to specify a hyperparameter bef"
                             "ore the --params argument. %s" %
                             " ".join(cli_args))
        elif arg[0:2] != "--" and not found_params:
            arg_values.append(arg)
            skip = False

        elif not found_params:
            raise ValueError("Illegal command line string, expected an argument"
                             " starting with -- but found %s" % (arg,))

        else:
            raise ValueError("Illegal command line string, expected a hyperpara"
                             "meter starting with - but found %s" % (arg,))

    return args, parameters
