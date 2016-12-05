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
import os
import unittest

import HPOlib.dispatcher.runsolver_wrapper as runsolver_wrapper


class RunsolverWrapperTest(unittest.TestCase):
    def setUp(self):
        # Change into the test directory
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        # Make sure there is no config file
        try:
            os.remove("config.cfg")
        except OSError:
            pass
        try:
            os.remove("./tests.pkl")
        except OSError:
            pass
        try:
            os.remove("./tests.pkl.lock")
        except OSError:
            pass

        # Read the dummy config
        self.config = ConfigParser.ConfigParser()
        self.config.read("dummy_config.cfg")

    def tearDown(self):
        try:
            os.remove("config.cfg")
        except OSError:
            pass
        try:
            os.remove("./tests.pkl")
        except OSError:
            pass
        try:
            os.remove("./tests.pkl")
        except OSError:
            pass

    def test_read_runsolver_output(self):
        with open("runsolver_positive.txt") as fh:
            runsolver_output_content = fh.readlines()
        cpu_time, wallclock_time, error = runsolver_wrapper \
            .read_runsolver_output(runsolver_output_content)
        self.assertAlmostEqual(cpu_time, 0.188011)
        self.assertAlmostEqual(wallclock_time, 0.259524)
        self.assertTrue(error is None)

    def test_read_runsolver_output_wallclock(self):
        with open("runsolver_wallclock_time_limit.txt") as fh:
            runsolver_output_content = fh.readlines()
        cpu_time, wallclock_time, error = runsolver_wrapper \
            .read_runsolver_output(runsolver_output_content)
        self.assertAlmostEqual(cpu_time, 0.044002)
        self.assertAlmostEqual(wallclock_time, 0.066825)
        self.assertEqual(error, "Wall clock time exceeded")

    def test_read_runsolver_output_vsize(self):
        with open("runsolver_vsize_exceeded.txt") as fh:
            runsolver_output_content = fh.readlines()
        cpu_time, wallclock_time, error = runsolver_wrapper \
            .read_runsolver_output(runsolver_output_content)
        self.assertAlmostEqual(cpu_time,  0.016)
        self.assertAlmostEqual(wallclock_time,0.039276)
        self.assertEqual(error, "VSize exceeded")

    def test_read_runsolver_output_warning(self):
        with open("runsolver_positive_with_warning.txt") as fh:
            runsolver_output_content = fh.readlines()
        cpu_time, wallclock_time, error = runsolver_wrapper \
            .read_runsolver_output(runsolver_output_content)
        self.assertAlmostEqual(cpu_time,  0.820027)
        self.assertAlmostEqual(wallclock_time, 1.00203)
        self.assertTrue(error is None)

    def test_read_runsolver_no_result(self):
        with open("runsolver_did_not_end_properly.txt") as fh:
            runsolver_output_content = fh.readlines()
        cpu_time, wallclock_time, error = runsolver_wrapper \
            .read_runsolver_output(runsolver_output_content)
        self.assertAlmostEqual(cpu_time, 0.66*2)
        self.assertAlmostEqual(wallclock_time, 1.52108*2)
        self.assertEqual(error, "Runsolver probably crashed!")

    def test_read_run_instance_output_no_result(self):
        with open("run_instance_no_result.txt") as fh:
            runinstance_content = fh.readlines()
        result_array, result_string = runsolver_wrapper.\
            read_run_instance_output(runinstance_content)
        # We expect some useful output for the user
        self.assertFalse(result_string is None)
        self.assertTrue(result_array is None)

    ############################################################################
    # A lot of tests for the function which parses the output...
    # These tests are organized in the same structure as in the original
    # if/else tree
    def test_parse_output_resultstring_noerror_notsat(self):
        runinstance_output_content = ["Bla\nBla\nBla\nResult for ParamILS: "
            "UNSAT, 0.35, 1, 0.5, -1, Random file"]
        with open("runsolver_positive.txt") as fh:
            runsolver_output_content = fh.readlines()

        cpu_time, wallclock_time, status, result, additional_data = \
            runsolver_wrapper.parse_output(self.config,
                                           runinstance_output_content,
                                           runsolver_output_content, 0.35)

        self.assertAlmostEqual(0.259524, wallclock_time)
        self.assertAlmostEqual(0.188011, cpu_time)
        self.assertEqual("CRASHED", status)
        self.assertAlmostEqual(result, 1)

    def test_parse_output_resultstring_noerror_notfinite(self):
        runinstance_output_content = ["Bla\nBla\nBla\nResult for ParamILS: "
                                      "SAT, 0.35, 1, NaN, -1, Random file"]
        with open("runsolver_positive.txt") as fh:
            runsolver_output_content = fh.readlines()

        cpu_time, wallclock_time, status, result, additional_data = \
            runsolver_wrapper.parse_output(self.config,
                                           runinstance_output_content,
                                           runsolver_output_content, 0.35)

        self.assertAlmostEqual(0.259524, wallclock_time)
        self.assertAlmostEqual(0.188011, cpu_time)
        self.assertEqual("CRASHED", status)
        self.assertAlmostEqual(result, 1)

    def test_parse_output_resultstring_noerror(self):
        runinstance_output_content = ["Bla\nBla\nBla\nResult for ParamILS: "
                                      "SAT, 0.35, 1, 0.5, -1, Random file"]
        with open("runsolver_positive.txt") as fh:
            runsolver_output_content = fh.readlines()

        cpu_time, wallclock_time, status, result, additional_data = \
            runsolver_wrapper.parse_output(self.config,
                                           runinstance_output_content,
                                           runsolver_output_content, 0.35)

        self.assertAlmostEqual(0.259524, wallclock_time)
        self.assertAlmostEqual(0.188011, cpu_time)
        self.assertEqual("SAT", status)
        self.assertAlmostEqual(result, 0.5)

    def test_parse_output_resultstring_errormemout(self):
        runinstance_output_content = ["Bla\nBla\nBla\nResult for ParamILS: "
                                      "SAT, 0.35, 1, 0.5, -1, Random file"]
        with open("runsolver_vsize_exceeded.txt") as fh:
            runsolver_output_content = fh.readlines()

        cpu_time, wallclock_time, status, result, additional_data = \
            runsolver_wrapper.parse_output(self.config,
                                           runinstance_output_content,
                                           runsolver_output_content, 0.35)

        self.assertAlmostEqual(0.039276, wallclock_time)
        self.assertAlmostEqual(0.016, cpu_time)
        self.assertEqual("CRASHED", status)
        self.assertAlmostEqual(result, 1)

    def test_parse_output_resultstring_errorrunsolver_notsat(self):
        runinstance_output_content = ["Bla\nBla\nBla\nResult for ParamILS: "
                                      "UNSAT, 0.35, 1, 0.5, -1, Random file"]
        with open("runsolver_did_not_end_properly.txt") as fh:
            runsolver_output_content = fh.readlines()

        cpu_time, wallclock_time, status, result, additional_data = \
            runsolver_wrapper.parse_output(self.config,
                                           runinstance_output_content,
                                           runsolver_output_content, 0.35)

        self.assertAlmostEqual(0.35, wallclock_time)
        self.assertAlmostEqual(1.32, cpu_time)
        self.assertEqual("CRASHED", status)
        self.assertAlmostEqual(result, 1)

    def test_parse_output_resultstring_errorrunsolver_notfinite(self):
        runinstance_output_content = ["Bla\nBla\nBla\nResult for ParamILS: "
                                      "SAT, 0.35, 1, NaN, -1, Random file"]
        with open("runsolver_did_not_end_properly.txt") as fh:
            runsolver_output_content = fh.readlines()

        cpu_time, wallclock_time, status, result, additional_data = \
            runsolver_wrapper.parse_output(self.config,
                                           runinstance_output_content,
                                           runsolver_output_content, 0.35)

        self.assertAlmostEqual(0.35, wallclock_time)
        self.assertAlmostEqual(1.32, cpu_time)
        self.assertEqual("CRASHED", status)
        self.assertAlmostEqual(result, 1)

    def test_parse_output_resultstring_errorrunsolver(self):
        runinstance_output_content = ["Bla\nBla\nBla\nResult for ParamILS: "
                                      "SAT, 0.35, 1, 0.5, -1, Random file"]
        with open("runsolver_did_not_end_properly.txt") as fh:
            runsolver_output_content = fh.readlines()

        cpu_time, wallclock_time, status, result, additional_data = \
            runsolver_wrapper.parse_output(self.config,
                                           runinstance_output_content,
                                           runsolver_output_content, 0.35)

        self.assertAlmostEqual(0.35, wallclock_time)
        self.assertAlmostEqual(1.32, cpu_time)
        self.assertEqual("SAT", status)
        self.assertAlmostEqual(result, 0.5)

    def test_parse_output_noresultstring_noerror(self):
        runinstance_output_content = ["Bla\nBla\nBla\n"]
        with open("runsolver_positive.txt") as fh:
            runsolver_output_content = fh.readlines()

        cpu_time, wallclock_time, status, result, additional_data = \
            runsolver_wrapper.parse_output(self.config,
                                           runinstance_output_content,
                                           runsolver_output_content, 0.35)

        self.assertAlmostEqual(0.259524, wallclock_time)
        self.assertAlmostEqual(0.188011, cpu_time)
        self.assertEqual("CRASHED", status)
        self.assertAlmostEqual(result, 1)
        self.assertEqual("No result string returned. Please have a look " \
                         "at the runinstance output", additional_data)

    def test_parse_output_noresultstring_errormemlimit(self):
        runinstance_output_content = ["Bla\nBla\nBla\n"]
        with open("runsolver_vsize_exceeded.txt") as fh:
            runsolver_output_content = fh.readlines()

        cpu_time, wallclock_time, status, result, additional_data = \
            runsolver_wrapper.parse_output(self.config,
                                           runinstance_output_content,
                                           runsolver_output_content, 0.35)

        self.assertAlmostEqual(0.039276, wallclock_time)
        self.assertAlmostEqual(0.016, cpu_time)
        self.assertEqual("CRASHED", status)
        self.assertAlmostEqual(result, 1)
        self.assertEqual("VSize exceeded Please have a look at the runsolver "
                         "output file.", additional_data)

    def test_parse_output_noresultstring_errorrunsolver(self):
        runinstance_output_content = ["Bla\nBla\nBla\n"]
        with open("runsolver_did_not_end_properly.txt") as fh:
            runsolver_output_content = fh.readlines()

        cpu_time, wallclock_time, status, result, additional_data = \
            runsolver_wrapper.parse_output(self.config,
                                           runinstance_output_content,
                                           runsolver_output_content, 0.35)

        self.assertAlmostEqual(0.35, wallclock_time)
        self.assertAlmostEqual(1.32, cpu_time)
        self.assertEqual("CRASHED", status)
        self.assertAlmostEqual(result, 1)
        self.assertEqual("There is no result string and it seems that the "
                         "runsolver crashed. Please have a look at the "
                         "runsolver output file.", additional_data)

    def test_read_run_instance_output_result(self):
        # TODO: add more tests here
        runinstance_output_content = [
            "00.01/00.12     Result for ParamILS: SAT, 0.35, 1, 0.5, -1, Random file"]
        result_array, result_string = runsolver_wrapper.\
            read_run_instance_output(runinstance_output_content)
        self.assertEqual(result_string, "Result for ParamILS: SAT, 0.35, 1, "
                                       "0.5, -1, Random file")
        self.assertListEqual(result_array, ["Result", "for", "ParamILS:",
                                            "SAT", "0.35", "1", "0.5",
                                            "-1", "Random", "file"])


