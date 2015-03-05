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

import HPOlib.benchmarks.svm_on_grid

import HPOlib.benchmarks.benchmark_util as benchmark_util


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"
__credits__ = ["Jasper Snoek", "Ryan P. Adams", "Hugo Larochelle"]


def main(params, ret_time=False, **kwargs):
    print 'Params: ', params, '\n'
    y = HPOlib.benchmarks.svm_on_grid.save_svm_on_grid(params,
                                                       ret_time=ret_time,
                                                       **kwargs)
    print 'Result: ', y
    return y


if __name__ == "__main__":
    args, cli_params = benchmark_util.parse_cli()
    result = main(cli_params, ret_time=False, **args)
    duration = main(cli_params, ret_time=True, **args)
    print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
        ("SAT", abs(duration), result, -1, str(__file__))