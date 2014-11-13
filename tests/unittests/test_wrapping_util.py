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

import ConfigParser
import numpy as np
import os
import shutil
import sys
import unittest
import tempfile
import StringIO

import HPOlib.wrapping as wrapping
import HPOlib.wrapping_util as wrapping_util
import HPOlib.config_parser.parse as parse


class WrappingTestUtil(unittest.TestCase):
    def setUp(self):
        # Change into the test directory
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        # Make sure there is no config file
        # noinspection PyBroadException
        try:
            os.remove("./config.cfg")
        except:
            pass

    def test_save_config_to_file(self):
        config = parse.parse_config("dummy_config.cfg", allow_no_value=True)
        config.set("HPOLIB", "total_time_limit", None)
        string_stream = StringIO.StringIO()
        wrapping_util.save_config_to_file(string_stream, config)
        file_content = string_stream.getvalue()
        asserted_file_content = "[HPOLIB]\n" \
                                "number_of_jobs = 1\n" \
                                "result_on_terminate = 1\n"\
                                "function = 1\n"\
                                "use_hpolib_time_measurement = True\n"\
                                "total_time_limit = None\n"\
                                "[GRIDSEARCH]\n" \
                                "params = params.pcs\n"

        self.assertEqual(asserted_file_content, file_content)
        string_stream.close()

    def test_save_config_to_file_ignore_none(self):
        config = parse.parse_config("dummy_config.cfg", allow_no_value=True)
        config.set("HPOLIB", "total_time_limit", None)
        string_stream = StringIO.StringIO()
        wrapping_util.save_config_to_file(string_stream, config,
                                          write_nones=False)
        file_content = string_stream.getvalue()
        asserted_file_content = "[HPOLIB]\n" \
                                "number_of_jobs = 1\n" \
                                "result_on_terminate = 1\n"\
                                "function = 1\n"\
                                "use_hpolib_time_measurement = True\n"\
                                "[GRIDSEARCH]\n" \
                                "params = params.pcs\n"

        self.assertEqual(asserted_file_content, file_content)
        string_stream.close()

    def test_parse_config_values_from_unknown_arguments(self):
        """Test if we can convert a config with Sections and variables into an
        argparser."""
        sys.argv = ['wrapping.py', '-s', '1', '-t', 'DBNet', '-o', 'SMAC',
                    '--HPOLIB:number_of_jobs', '2']
        args, unknown = wrapping.use_arg_parser()
        self.assertEqual(len(unknown), 2)
        config = ConfigParser.SafeConfigParser(allow_no_value=True)
        config.read("dummy_config.cfg")
        config_args = wrapping_util.parse_config_values_from_unknown_arguments(
            unknown, config)
        self.assertListEqual(vars(config_args)['HPOLIB:number_of_jobs'], ['2'])
        self.assertIs(vars(config_args)['GRIDSEARCH:params'], None)
        self.assertIs(vars(config_args)['HPOLIB:function'], None)
        self.assertIs(vars(config_args)['HPOLIB:result_on_terminate'], None)

    def test_parse_config_values_from_unknown_arguments2(self):
        """Test if we can convert a config with Sections and variables into an
        argparser. Test for arguments with whitespaces"""
        sys.argv = ['wrapping.py', '-s', '1', '-t', 'DBNet', '-o', 'SMAC',
                    '--HPOLIB:function', 'python', '../branin.py']
        args, unknown = wrapping.use_arg_parser()
        self.assertEqual(len(unknown), 3)
        config = ConfigParser.SafeConfigParser(allow_no_value=True)
        config.read("dummy_config.cfg")
        config_args = wrapping_util.parse_config_values_from_unknown_arguments(
            unknown, config)
        self.assertListEqual(vars(config_args)['HPOLIB:function'], ['python',
            '../branin.py'])
        self.assertIs(vars(config_args)['GRIDSEARCH:params'], None)
        self.assertIs(vars(config_args)['HPOLIB:result_on_terminate'], None)

    def test_nan_mean(self):
        self.assertEqual(wrapping_util.nan_mean(np.array([1, 5])), 3)
        self.assertEqual(wrapping_util.nan_mean((1, 5)), 3)
        # self.assertEqual(wrapping_util.nan_mean("abc"), 3)
        self.assertRaises(TypeError, wrapping_util.nan_mean, ("abc"))
        self.assertEqual(wrapping_util.nan_mean(np.array([1, 5, np.nan])), 3)
        self.assertEqual(wrapping_util.nan_mean(np.array([1, 5, np.inf])), 3)
        self.assertTrue(np.isnan(wrapping_util.nan_mean(np.array([np.inf]))))
        self.assertEqual(wrapping_util.nan_mean(np.array([-1, 1])), 0)
        self.assertTrue(np.isnan(wrapping_util.nan_mean(np.array([]))))

if __name__ == "__main__":
    unittest.main()
