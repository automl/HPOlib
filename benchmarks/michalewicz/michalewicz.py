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

__authors__ = ["Mohsin Ali"]
__contact__ = ["automl.org"]
__credits__ = ["Sonja Surjanovic", "Derek Bingham"]
__function__= ["MICHALEWICZ FUNCTION"]

import time

import HPOlib.benchmarks.benchmark_util as benchmark_util


def michal(xx, m):
    import math

    d = len(xx)
    sum_ = 0
    for ii in range(d):
        xi = xx[ii]
        i = ii+1
        new = math.sin(xi) * \
            math.pow((math.sin(i * math.pow(xi, 2) / math.pi)), (2*m))
        sum_ += new

    return -sum_


def main(params, **kwargs):
    print 'Params: ', params
    print 'kwargs: ', kwargs

    xx = [float(params["x1"]), float(params["x2"]),
          float(params["x3"]), float(params["x4"]),
          float(params["x5"]), float(params["x6"]),
          float(params["x7"]), float(params["x8"]),
          float(params["x9"]), float(params["x10"])]

    y = michal(xx, 10)
    return y


if __name__ == "__main__":
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
