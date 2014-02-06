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


import HPOlib.benchmark_util as benchmark_util

import sys
from numpy import pi, cos, ndarray
import time

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


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

    Example:
    python neural_network.py --folds 10 --fold 1 --dataset convex  --params
        -depth '3' -n_hid_0 '1024' -n_hid_1 '1024' -n_hid_2 '1024' -lr '0.01'
    """
    args = {}
    parameters = {}

    cli_args = sys.argv
    found_params = False
    skip = True
    for idx, arg in enumerate(cli_args):
        if skip:
            skip = False
            continue
        else:
            skip = True

        if arg == "--params":
            found_params = True
            skip = False

        elif arg[0:2] == "--" and not found_params:
            if cli_args[idx+1][0] == "-":
                raise ValueError("Argument name is not allowed to have a "
                                 "leading minus %s" % cli_args[idx + 1])
            args[cli_args[idx][2:]] = cli_args[idx+1]

        elif arg[0:2] == "--" and found_params:
            raise ValueError("You are trying to specify an argument after the "
                             "--params argument. Please change the order.")

        elif arg[0] == "-" and arg[0:2] != "--" and found_params:
            if cli_args[idx+1][0] == "-":
                raise ValueError("Hyperparameter name is not allowed to have a "
                                 "leading minus %s" % cli_args[idx + 1])
            parameters[cli_args[idx][1:]] = cli_args[idx+1]

        elif arg[0] == "-" and arg[0:2] != "--" and not found_params:
            raise ValueError("You either try to use arguments with only one lea"
                             "ding minus or try to specify a hyperparameter bef"
                             "ore the --params argument.")

        elif not found_params:
            raise ValueError("Illegal command line string, expected an argument"
                             " starting with -- but found %s" % (arg,))

        else:
            raise ValueError("Illegal command line string, expected a hyperpara"
                             "meter starting with - but found %s" % (arg,))

    return args, parameters

def branin(params, **kwargs):
# branin function
# The number of variables n = 2.
# constraints:
# -5 <= x <= 10, 0 <= y <= 15
# three global optima:  (-pi, 12.275), (pi, 2.275), (9.42478, 2.475), where
# branin = 0.397887
    if "x" not in params or "y" not in params:
        raise ValueError("No params found ['x', 'y']\n")
    x = float(params["x"])
    y = float(params["y"])
    if type(x) == ndarray or type(y) == ndarray:
        x = x[0]
        y = y[0]

    if -5 > x or x > 10:
        raise ValueError("X value not in between -5 and 10")
    if 0 > y or y > 15:
        raise ValueError("Y value not in between 0 and 15")

    result = (y-(5.1/(4*pi**2))*x**2+5*x/pi-6)**2
    result += 10*(1-1/(8*pi))*cos(x)+10
    return result


def main(params, **kwargs):
    print 'Params: ', params,
    y = branin(params, **kwargs)
    print 'Result: ', y
    return y

if __name__ == "__main__":
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %d, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
