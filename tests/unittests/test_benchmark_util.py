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

import os
import unittest
import sys

import HPOlib.benchmarks.benchmark_util as benchmark_util


class BenchmarkUtilTest(unittest.TestCase):
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
        sys.argv = ["test.py", "--folds", "10", "--fold", "0", "--params", "-x",
                    "3"]
        args, params = benchmark_util.parse_cli()
        self.assertEqual(params, {'x': '3'})
        self.assertEqual(args, {'folds': '10', 'fold': '0'})

        sys.argv = ["test.py", "--folds", "10", "--fold", "0", "--long",
                    "long", "long", "li", "long", "long", "long", "HEY",
                    "--params", "-x", "3"]
        args, params = benchmark_util.parse_cli()
        self.assertEqual(params, {'x': '3'})
        self.assertEqual(args, {'folds': '10', 'fold': '0',
                                'long': "long long li long long long HEY"})

        sys.argv = ["test.py", "--folds", "10", "--fold", "0", "--long", '"',
                    "long", "long", "li", "long", "long", "'", "lng", "HEY",
                    "--params", "-x", "3"]
        args, params = benchmark_util.parse_cli()
        self.assertEqual(params, {'x': '3'})
        self.assertEqual(args, {'folds': '10', 'fold': '0',
                                'long': "\" long long li long long \' lng HEY"})

        sys.argv = ["test.py", "--folds", "10", "--fold", "0", "--long",
                    "--params", "-x", "3"]
        args, params = benchmark_util.parse_cli()
        self.assertEqual(params, {'x': '3'})
        self.assertEqual(args, {'folds': '10', 'fold': '0',
                                'long': ""})

        # illegal call, arguments with one minus before --params
        sys.argv = ["test.py", "-folds", "10", "--fold", "0", "--params", "-x",
                    "3"]
        with self.assertRaises(ValueError) as cm1:
            benchmark_util.parse_cli()
        self.assertEqual(cm1.exception.message, "You either try to use arguments"
                         " with only one leading minus or try to specify a "
                         "hyperparameter before the --params argument. test.py"
                         " -folds 10 --fold 0 --params -x 3")

        # illegal call, trying to specify an arguments after --params
        sys.argv = ["test.py", "--folds", "10", "--params", "-x",
                    "'3'", "--fold", "0"]
        with self.assertRaises(ValueError) as cm5:
            benchmark_util.parse_cli()
        self.assertEqual(cm5.exception.message, "You are trying to specify an argument after the "
                             "--params argument. Please change the order.")

        # illegal call, no - in front of parameter name
        sys.argv = ["test_cv.py", "--params", "x", "'5'"]
        with self.assertRaises(ValueError) as cm2:
            benchmark_util.parse_cli()
        self.assertEqual(cm2.exception.message, "Illegal command line string, expected a hyperpara"
                             "meter starting with - but found x")
