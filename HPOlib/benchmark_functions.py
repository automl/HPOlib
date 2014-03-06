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

from numpy import pi, cos, ndarray, array, exp, NaN
import sys


def save_branin(params, **kwargs):
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
    return branin(x, y)


def branin(x, y):
    """Branin test function

    The number of variables n = 2.
    constraints:
    -5 <= x <= 10, 0 <= y <= 15
    three global optima:  (-pi, 12.275), (pi, 2.275), (9.42478, 2.475), where
    branin = 0.397887"""

    result = (y-(5.1/(4*pi**2))*x**2+5*x/pi-6)**2
    result += 10*(1-1/(8*pi))*cos(x)+10
    return result


def save_har6(params, **kwargs):
    if "x" not in params or "y" not in params or "z" not in params \
     or "a" not in params or "b" not in params or "c" not in params:
        sys.stderr.write("No params found ['x', 'y']\n")
        return NaN
    x = float(params["x"])
    y = float(params["y"])
    z = float(params["z"])
    a = float(params["a"])
    b = float(params["b"])
    c = float(params["c"])
    if type(x) == ndarray:
        x = x[0]
        y = y[0]
        z = z[0]
        a = a[0]
        b = b[0]
        c = c[0]
    return har6(x, y, z, a, b, c)


def har6(x, y, z, a, b, c):
    """6d Hartmann test function

    constraints:
    0 <= xi <= 1, i = 1..6
    global optimum at (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
    where har6 = -3.32236"""

    value = array([x, y, z, a, b, c])

    if 0 > x or x > 1 or 0 > y or y > 1 or 0 > z or z > 1:
        raise ValueError("x=%s, y=%s or z=%s not in between 0 and 1" %
                         (x, y, z))
    if 0 > a or a > 1 or 0 > b or b > 1 or 0 > c or c > 1:
        raise ValueError("a=%s, b=%s or c=%s not in between 0 and 1" %
                         (a, b, c))

    a = array([[10.0,   3.0, 17.0,   3.5,  1.7,  8.0],
               [ 0.05, 10.0, 17.0,   0.1,  8.0, 14.0],
               [ 3.0,   3.5,  1.7,  10.0, 17.0,  8.0],
               [17.0,   8.0,  0.05, 10.0,  0.1, 14.0]])
    c = array([1.0, 1.2, 3.0, 3.2])
    p = array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
               [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
               [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
               [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    s = 0
    for i in [0,1,2,3]:
        sm = a[i,0]*(value[0]-p[i,0])**2
        sm += a[i,1]*(value[1]-p[i,1])**2
        sm += a[i,2]*(value[2]-p[i,2])**2
        sm += a[i,3]*(value[3]-p[i,3])**2
        sm += a[i,4]*(value[4]-p[i,4])**2
        sm += a[i,5]*(value[5]-p[i,5])**2
        s += c[i]*exp(-sm)
    result = -s
    return result