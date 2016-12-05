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


import HPOlib.benchmarks.logreg_on_grid as logreg

import HPOlib.benchmarks.benchmark_util as benchmark_util


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"
__credits__ = ["Jasper Snoek", "Ryan P. Adams", "Hugo Larochelle"]


def main(params, ret_type='validation', **kwargs):
    y = logreg.save_logreg_on_grid(params, ret_type=ret_type, **kwargs)
    return y


if __name__ == "__main__":
    args, params = benchmark_util.parse_cli()
    if 'test' in args:
        result = main(params, ret_type='test', **args)
    else:
        result = main(params, ret_type='validation', **args)
    duration = main(params, ret_type='time')
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))
