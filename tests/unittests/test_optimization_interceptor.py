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

from HPOlib import optimization_interceptor, Experiment


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

        trial_index0 = optimization_interceptor.get_trial_index(experiment, 0, params0)
        self.assertEqual(trial_index0, 0)
        experiment.set_one_fold_running(trial_index0, 0)
        experiment.set_one_fold_complete(trial_index0, 0, 1, 1)
        self.assertEqual(trial_index0,
                         optimization_interceptor.get_trial_index(experiment, 1, params0))
        experiment.set_one_fold_running(trial_index0, 1)
        experiment.set_one_fold_complete(trial_index0, 1, 1, 1)
        self.assertEqual(trial_index0,
                         optimization_interceptor.get_trial_index(experiment, 2, params0))
        experiment.set_one_fold_running(trial_index0, 2)
        experiment.set_one_fold_complete(trial_index0, 2, 1, 1)
        self.assertEqual(trial_index0,
                         optimization_interceptor.get_trial_index(experiment, 3, params0))
        experiment.set_one_fold_running(trial_index0, 3)
        experiment.set_one_fold_complete(trial_index0, 3, 1, 1)
        self.assertEqual(trial_index0,
                         optimization_interceptor.get_trial_index(experiment, 4, params0))
        experiment.set_one_fold_running(trial_index0, 4)
        experiment.set_one_fold_complete(trial_index0, 4, 1, 1)

        trial_index1 = optimization_interceptor.get_trial_index(experiment, 0, params1)
        self.assertEqual(trial_index1, 1)
        experiment.set_one_fold_running(trial_index1, 0)
        experiment.set_one_fold_complete(trial_index1, 0, 1, 1)
        self.assertEqual(trial_index1,
                         optimization_interceptor.get_trial_index(experiment, 1, params1))
        experiment.set_one_fold_running(trial_index1, 1)
        experiment.set_one_fold_complete(trial_index1, 1, 1, 1)
        self.assertEqual(trial_index1,
                         optimization_interceptor.get_trial_index(experiment, 2, params1))
        experiment.set_one_fold_running(trial_index1, 2)
        experiment.set_one_fold_complete(trial_index1, 2, 1, 1)
        self.assertEqual(trial_index1,
                         optimization_interceptor.get_trial_index(experiment, 3, params1))
        experiment.set_one_fold_running(trial_index1, 3)
        experiment.set_one_fold_complete(trial_index1, 3, 1, 1)
        self.assertEqual(trial_index1,
                         optimization_interceptor.get_trial_index(experiment, 4, params1))
        experiment.set_one_fold_running(trial_index1, 4)
        experiment.set_one_fold_complete(trial_index1, 4, 1, 1)

        trial_index2 = optimization_interceptor.get_trial_index(experiment, 0, params2)
        self.assertEqual(trial_index2, 2)
        experiment.set_one_fold_running(trial_index2, 0)
        experiment.set_one_fold_complete(trial_index2, 0, 1, 1)

        trial_index3 = optimization_interceptor.get_trial_index(experiment, 0, params3)
        self.assertEqual(trial_index3, 3)
        experiment.set_one_fold_running(trial_index3, 0)
        experiment.set_one_fold_complete(trial_index3, 0, 1, 1)

        trial_index4 = optimization_interceptor.get_trial_index(experiment, 0, params4)
        self.assertEqual(trial_index4, 4)
        experiment.set_one_fold_running(trial_index4, 0)
        experiment.set_one_fold_complete(trial_index4, 0, 1, 1)

        self.assertEqual(trial_index2,
                         optimization_interceptor.get_trial_index(experiment, 3, params2))
        self.assertEqual(trial_index4,
                         optimization_interceptor.get_trial_index(experiment, 4, params4))

        # Since params1 were already evaluated, this should be a new trial_index
        trial_index_test1 = optimization_interceptor.get_trial_index(experiment, 0, params1)
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

        trial_index0 = optimization_interceptor.get_trial_index(experiment, 0, params0)
        self.assertEqual(trial_index0, 0)
        experiment.set_one_fold_running(trial_index0, 0)
        experiment.set_one_fold_complete(trial_index0, 0, 1, 1)

        trial_index1 = optimization_interceptor.get_trial_index(experiment, 0, params1)
        self.assertEqual(trial_index1, 1)
        experiment.set_one_fold_running(trial_index1, 0)
        experiment.set_one_fold_complete(trial_index1, 0, 1, 1)

        trial_index2 = optimization_interceptor.get_trial_index(experiment, 0, params2)
        self.assertEqual(trial_index2, 2)
        experiment.set_one_fold_running(trial_index2, 0)
        experiment.set_one_fold_complete(trial_index2, 0, 1, 1)

        trial_index3 = optimization_interceptor.get_trial_index(experiment, 0, params3)
        self.assertEqual(trial_index3, 3)
        experiment.set_one_fold_running(trial_index3, 0)
        experiment.set_one_fold_complete(trial_index3, 0, 1, 1)

        trial_index4 = optimization_interceptor.get_trial_index(experiment, 0, params4)
        self.assertEqual(trial_index4, 4)
        experiment.set_one_fold_running(trial_index4, 0)
        experiment.set_one_fold_complete(trial_index4, 0, 1, 1)

        self.assertEqual(5,
                         optimization_interceptor.get_trial_index(experiment, 0, params2))


if __name__ == "__main__":
    unittest.main()
