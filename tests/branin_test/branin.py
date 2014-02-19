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

import sys
from numpy import NaN, pi, cos, ndarray

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
    y =  branin(params, **kwargs)
    print 'Result: ', y
    return y
