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

    # Test spearmint
    @unittest.skip("Not implemented yet")
    def test_build_spearmint_call(self):
        self.fail()

    @unittest.skip("Not implemented yet")
    def test_main_spearmint(self):
        self.fail()
        
    # Test tpe
    @unittest.skip("Not implemented yet")
    def test_build_tpe_call(self):
        self.fail()

    @unittest.skip("Not implemented yet")
    def test_main_tpe(self):
        self.fail()
        
    # Test random
    @unittest.skip("Not implemented yet")
    def test_build_random_call(self):
        self.fail()

    @unittest.skip("Not implemented yet")
    def test_main_random(self):
        self.fail()
        
    # Test smac
    @unittest.skip("Not implemented yet")
    def test_build_smac_call(self):
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

    def test_use_option_parser_with_config(self):
        sys.argv = ['wrapping.py', '-s', '1', '-t', 'DBNet', '-o', 'SMAC',
                    '--HPOLIB:total_time_limit', '3600']
        args, unknown = wrapping.use_arg_parser()
        self.assertEqual(len(unknown), 2)
        config = parse.parse_config("dummy_config.cfg", allow_no_value=True)
        config_args = wrapping.parse_config_values_from_unknown_arguments(
            unknown, config)
        self.assertEqual(vars(config_args)['HPOLIB:total_time_limit'],
                         '3600')

    def test_override_config_with_cli_arguments(self):
        unknown = ['--HPOLIB:total_time_limit', '50',
                   '--HPOLIB:numberofjobs', '2']
        config = parse.parse_config("dummy_config.cfg", allow_no_value=True)
        config_args = wrapping.parse_config_values_from_unknown_arguments(
            unknown, config)
        new_config = wrapping.override_config_with_cli_arguments(config,
                                                            config_args)
        self.assertEqual(new_config.get('HPOLIB', 'numberofjobs'), '2')


    def test_save_config_to_file(self):
        unknown = ['--HPOLIB:total_time_limit', '50',
                   '--HPOLIB:numberofjobs', '2']
        config = parse.parse_config("dummy_config.cfg", allow_no_value=True)
        string_stream = StringIO.StringIO()
        wrapping.save_config_to_file(string_stream, config)
        file_content = string_stream.getvalue()
        asserted_file_content = "[HPOLIB]\n" \
                            "numberofjobs = 1\n" \
                            "result_on_terminate = 1\n"\
                            "function = 1\n"\
                            "algorithm = cv.py\n"\
                            "run_instance = runsolver_wrapper.py\n"\
                            "numbercv = 1\n"\
                            "numberofconcurrentjobs = 1\n"\
                            "runsolver_time_limit = 3600\n"\
                            "total_time_limit = 3600\n"\
                            "memory_limit = 2000\n"\
                            "cpu_limit = 14400\n"\
                            "max_crash_per_cv = 3\n"

        self.assertEqual(asserted_file_content, file_content)
        string_stream.close()

    @unittest.skip("Not implemented yet")
    def test_main_smac(self):
        os.chdir("./branin_test/")
        sys.argv = ['wrapping.py', 'smac']
        ret = wrapping.main()
        self.assertEqual(ret, 0, "Return code of SMAC is not 0, please debug")
        
        # TODO: Test what happens when the runsolver limit is exceeded
        # TODO: Clean up the test directory
        
    # General main test
    @unittest.skip("Not implemented yet")
    def test_main(self):
        self.fail()



if __name__ == "__main__": 
    unittest.main()