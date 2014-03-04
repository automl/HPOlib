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
import shutil
import sys
import unittest
import tempfile
import StringIO

import HPOlib.wrapping as wrapping
import HPOlib.config_parser.parse as parse

class WrappingTest(unittest.TestCase):
    def setUp(self):
        # Change into the test directory
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        
        # Make sure there is no config file
        try:
            os.remove("./config.cfg")
        except:
            pass
            
    @unittest.skip("Not implemented yet")
    def test_calculate_wrapping_overhead(self):
        self.fail()

    @unittest.skip("Not implemented yet")
    def test_calculate_optimizer_time(self):
        self.fail()

    def test_use_option_parser_no_optimizer(self):
        # Test not specifying an optimizer but random other options
        sys.argv = ['wrapping.py', '-s', '1', '-t', 'DBNet']
        self.assertRaises(SystemExit, wrapping.use_arg_parser)

    def test_use_option_parser_the_right_way(self):
        sys.argv = ['wrapping.py', '-s', '1', '-t', 'DBNet', '-o', 'SMAC']
        args, unknown = wrapping.use_arg_parser()
        self.assertEqual(args.optimizer, 'SMAC')
        self.assertEqual(args.seed, 1)
        self.assertEqual(args.title, 'DBNet')
        self.assertEqual(len(unknown), 0)
        
    # General main test
    @unittest.skip("Not implemented yet")
    def test_main(self):
        self.fail()


if __name__ == "__main__": 
    unittest.main()