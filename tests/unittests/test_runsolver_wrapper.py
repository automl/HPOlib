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
import sys
import unittest

import HPOlib.Experiment as Experiment
import HPOlib.runsolver_wrapper as runsolver_wrapper


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
        cpu_time, wallclock_time, error = runsolver_wrapper\
            .read_runsolver_output("runsolver_positive.txt")
        self.assertAlmostEqual(cpu_time, 0.188011)
        self.assertAlmostEqual(wallclock_time, 0.259524)
        self.assertTrue(error is None)

    def test_read_runsolver_output_wallclock(self):
        cpu_time, wallclock_time, error = runsolver_wrapper\
            .read_runsolver_output("runsolver_wallclock_time_limit.txt")
        self.assertAlmostEqual(cpu_time, 0.044002)
        self.assertAlmostEqual(wallclock_time, 0.066825)
        self.assertEqual(error, "Wall clock time exceeded")

    def test_read_runsolver_output_vsize(self):
        cpu_time, wallclock_time, error = runsolver_wrapper\
            .read_runsolver_output("runsolver_vsize_exceeded.txt")
        self.assertAlmostEqual(cpu_time,  0.016)
        self.assertAlmostEqual(wallclock_time,0.039276)
        self.assertEqual(error, "VSize exceeded")

    def test_read_runsolver_output_warning(self):
        cpu_time, wallclock_time, error = runsolver_wrapper\
            .read_runsolver_output("runsolver_positive_with_warning.txt")
        self.assertAlmostEqual(cpu_time,  0.820027)
        self.assertAlmostEqual(wallclock_time, 1.00203)
        self.assertTrue(error is None)

    def test_read_run_instance_output_no_result(self):
        result_array, result_string = runsolver_wrapper.\
            read_run_instance_output("run_instance_no_result.txt")
        self.assertTrue(result_string is None)
        self.assertTrue(result_array is None)

    def test_read_run_instance_output_result(self):
        result_array, result_string = runsolver_wrapper.\
            read_run_instance_output("run_instance_result.txt")
        self.assertEqual(result_string, "Result for ParamILS: SAT, 0.35, 1, "
                                       "0.5, -1, Random file")
        self.assertListEqual(result_array, ["Result", "for", "ParamILS:",
                                            "SAT", "0.35", "1", "0.5",
                                            "-1", "Random", "file"])

    def test_get_trial_index_cv(self):
        try:
            os.remove("test_get_trial_index.pkl")
        except OSError:
            pass

        try:
            os.remove("test_get_trial_index.pkl.lock")
        except OSError:
            pass

        experiment = Experiment.Experiment(".", "test_get_trial_index", folds=5)
        params0 = {"x": "1"}
        params1 = {"x": "2"}
        params2 = {"x": "3"}
        params3 = {"x": "4"}
        params4 = {"x": "5"}

        trial_index0 = runsolver_wrapper.get_trial_index(experiment, 0, params0)
        self.assertEqual(trial_index0, 0)
        experiment.set_one_fold_running(trial_index0, 0)
        experiment.set_one_fold_complete(trial_index0, 0, 1, 1)
        self.assertEqual(trial_index0,
                         runsolver_wrapper.get_trial_index(experiment, 1, params0))
        experiment.set_one_fold_running(trial_index0, 1)
        experiment.set_one_fold_complete(trial_index0, 1, 1, 1)
        self.assertEqual(trial_index0,
                         runsolver_wrapper.get_trial_index(experiment, 2, params0))
        experiment.set_one_fold_running(trial_index0, 2)
        experiment.set_one_fold_complete(trial_index0, 2, 1, 1)
        self.assertEqual(trial_index0,
                         runsolver_wrapper.get_trial_index(experiment, 3, params0))
        experiment.set_one_fold_running(trial_index0, 3)
        experiment.set_one_fold_complete(trial_index0, 3, 1, 1)
        self.assertEqual(trial_index0,
                         runsolver_wrapper.get_trial_index(experiment, 4, params0))
        experiment.set_one_fold_running(trial_index0, 4)
        experiment.set_one_fold_complete(trial_index0, 4, 1, 1)

        trial_index1 = runsolver_wrapper.get_trial_index(experiment, 0, params1)
        self.assertEqual(trial_index1, 1)
        experiment.set_one_fold_running(trial_index1, 0)
        experiment.set_one_fold_complete(trial_index1, 0, 1, 1)
        self.assertEqual(trial_index1,
                         runsolver_wrapper.get_trial_index(experiment, 1, params1))
        experiment.set_one_fold_running(trial_index1, 1)
        experiment.set_one_fold_complete(trial_index1, 1, 1, 1)
        self.assertEqual(trial_index1,
                         runsolver_wrapper.get_trial_index(experiment, 2, params1))
        experiment.set_one_fold_running(trial_index1, 2)
        experiment.set_one_fold_complete(trial_index1, 2, 1, 1)
        self.assertEqual(trial_index1,
                         runsolver_wrapper.get_trial_index(experiment, 3, params1))
        experiment.set_one_fold_running(trial_index1, 3)
        experiment.set_one_fold_complete(trial_index1, 3, 1, 1)
        self.assertEqual(trial_index1,
                         runsolver_wrapper.get_trial_index(experiment, 4, params1))
        experiment.set_one_fold_running(trial_index1, 4)
        experiment.set_one_fold_complete(trial_index1, 4, 1, 1)

        trial_index2 = runsolver_wrapper.get_trial_index(experiment, 0, params2)
        self.assertEqual(trial_index2, 2)
        experiment.set_one_fold_running(trial_index2, 0)
        experiment.set_one_fold_complete(trial_index2, 0, 1, 1)

        trial_index3 = runsolver_wrapper.get_trial_index(experiment, 0, params3)
        self.assertEqual(trial_index3, 3)
        experiment.set_one_fold_running(trial_index3, 0)
        experiment.set_one_fold_complete(trial_index3, 0, 1, 1)

        trial_index4 = runsolver_wrapper.get_trial_index(experiment, 0, params4)
        self.assertEqual(trial_index4, 4)
        experiment.set_one_fold_running(trial_index4, 0)
        experiment.set_one_fold_complete(trial_index4, 0, 1, 1)

        self.assertEqual(trial_index2,
                         runsolver_wrapper.get_trial_index(experiment, 3, params2))
        self.assertEqual(trial_index4,
                         runsolver_wrapper.get_trial_index(experiment, 4, params4))

        # Since params1 were already evaluated, this should be a new trial_index
        trial_index_test1 = runsolver_wrapper.get_trial_index(experiment, 0, params1)
        self.assertEqual(trial_index_test1, 5)

    def test_get_trial_index_nocv(self):
        try:
            os.remove("test_get_trial_index.pkl")
        except OSError:
            pass

        try:
            os.remove("test_get_trial_index.pkl.lock")
        except OSError:
            pass

        experiment = Experiment.Experiment(".", "test_get_trial_index", folds=1)
        params0 = {"x": "1"}
        params1 = {"x": "2"}
        params2 = {"x": "3"}
        params3 = {"x": "4"}
        params4 = {"x": "5"}

        trial_index0 = runsolver_wrapper.get_trial_index(experiment, 0, params0)
        self.assertEqual(trial_index0, 0)
        experiment.set_one_fold_running(trial_index0, 0)
        experiment.set_one_fold_complete(trial_index0, 0, 1, 1)

        trial_index1 = runsolver_wrapper.get_trial_index(experiment, 0, params1)
        self.assertEqual(trial_index1, 1)
        experiment.set_one_fold_running(trial_index1, 0)
        experiment.set_one_fold_complete(trial_index1, 0, 1, 1)

        trial_index2 = runsolver_wrapper.get_trial_index(experiment, 0, params2)
        self.assertEqual(trial_index2, 2)
        experiment.set_one_fold_running(trial_index2, 0)
        experiment.set_one_fold_complete(trial_index2, 0, 1, 1)

        trial_index3 = runsolver_wrapper.get_trial_index(experiment, 0, params3)
        self.assertEqual(trial_index3, 3)
        experiment.set_one_fold_running(trial_index3, 0)
        experiment.set_one_fold_complete(trial_index3, 0, 1, 1)

        trial_index4 = runsolver_wrapper.get_trial_index(experiment, 0, params4)
        self.assertEqual(trial_index4, 4)
        experiment.set_one_fold_running(trial_index4, 0)
        experiment.set_one_fold_complete(trial_index4, 0, 1, 1)

        self.assertEqual(5,
                         runsolver_wrapper.get_trial_index(experiment, 0, params2))
