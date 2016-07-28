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

__authors__  = ["Mohsin Ali"]
__contact__  = ["automl.org"]
__credits__  = ["Ziyu Wang", "John Assael", "Nando de Freitas"]
__function__ = ["RKHS"]

import time

import HPOlib.benchmarks.benchmark_util as benchmark_util


def covSEard(hyp, x, z):
    import numpy as np
    from scipy.spatial.distance import cdist
    """ARD covariance:
    x is of dimension n X D
    y is of dimension m X D
    """
    hyp = np.exp(hyp)

    D = x.shape[1]
    X = (1 / hyp[:D]) * x

    Z = (1 / hyp[:D]) * z
    K = cdist(X, Z)

    K = hyp[D] ** 2 * np.exp(-K ** 2 / 2)

    return K


def rkhs_synth(params, **kwargs):
    import numpy as np

    """
    RKHS Function
        Description: Synthetic heteroscedastic function generated from 2 Squared Exponential kernels
                     for Bayesian Optimization method evaluation tasks
        Evaluated: x \in [0,1]
        Global Maximum: x=0.89235, f(x)=5.73839
        Authors: Ziyu Wang, John Assael and Nando de Freitas
    """
    if "x" not in params:
        raise ValueError("No params found ['x']\n")
    x = float(params["x"])
    # y = float(params["y"])

    x = np.atleast_2d(x)
    hyp_1 = np.log(np.array([0.1, 1]))
    hyp_2 = np.log(np.array([0.01, 1]))

    support_1 = [0.1, 0.15, 0.08, 0.3, 0.4]
    support_2 = [0.8, 0.85, 0.9, 0.95, 0.92, 0.74, 0.91, 0.89, 0.79, 0.88, 0.86, 0.96, 0.99, 0.82]
    vals_1 = [4, -1, 2., -2., 1.]
    vals_2 = [3, 4, 2, 1, -1, 2, 2, 3, 3, 2., -1., -2., 4., -3.]

    f = sum([vals_2[i] * covSEard(hyp_2, np.atleast_2d(np.array(s)), x) for i, s in enumerate(support_2)])
    f += sum([vals_1[i] * covSEard(hyp_1, np.atleast_2d(np.array(s)), x) for i, s in enumerate(support_1)])

    return float(f)


def main(params, **kwargs):
    print 'Params: ', params['x'],
    print 'kwargs: ', kwargs,

    y = rkhs_synth(params, **kwargs)
    return -y

if __name__ == "__main__":
    starttime = time.time()
    args, params = benchmark_util.parse_cli()
    result = main(params, **args)
    duration = time.time() - starttime

    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
