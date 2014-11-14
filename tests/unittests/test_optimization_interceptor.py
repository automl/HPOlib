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

from collections import OrderedDict
import os
import unittest
import sys

from HPOlib import optimization_interceptor


class OptimizationInterceptorTest(unittest.TestCase):
    def setUp(self):
        # Change into the parent of the test directory
        os.chdir(os.path.join("..", os.path.dirname(os.path.realpath(__file__))))

        # Make sure there is no config file
        try:
            os.remove("./config.cfg")
        except:
            pass

    def test_read_parameters_from_command_line(self):
        # Legal call
        sys.argv = ["test_optimization_interceptor.py", "--params",
                    "-x", "'5'",
                    "-quoted_string", "'Koenigsberghausen'",
                    "-unquoted_string", "string",
                    "-y", "5.0",
                    "-z", '"-3.0"']
        args, params = optimization_interceptor.parse_cli()
        self.assertEqual(params,
                         OrderedDict([('x', '5'),
                         ('quoted_string', 'Koenigsberghausen'),
                         ('unquoted_string', 'string'),
                         ('y', '5.0'),
                         ('z', '-3.0')]))

        # illegal call, no - in front of parameter name
        sys.argv = ["test_optimization_interceptor.py", "--params", "x", "'5'"]
        self.assertRaises(ValueError, optimization_interceptor.parse_cli)


if __name__ == "__main__":
    unittest.main()
