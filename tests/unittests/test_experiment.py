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
import numpy as np
import unittest

import HPOlib.Experiment as Experiment


def _sanity_check(experiment):
    experiment._sanity_check()
    pass


class ExperimentTest(unittest.TestCase):
    # TODO: Test timing mechanism!!!
    def setUp(self):
        # Change into the test directory
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        try:
            os.remove("test_exp.pkl")
        except OSError:
            pass

    def tearDown(self):
        # Delete the test experiment
        try:
            os.remove("test_exp.pkl")
        except OSError:
            pass
        try:
            os.remove("test_exp.pkl.lock")
        except OSError:
            pass

    def test_init(self):
        # TODO: Somehow test in which folder the experiment is created
        # TODO: Remove the case that it is saved
        exp = Experiment.Experiment(".", "test_exp")
        _sanity_check(exp)
        exp.title = "test"
        exp._save_jobs()
        del exp

        # Make sure reloading works
        exp = Experiment.Experiment(".", "test_exp")
        self.assertEqual(exp.title, "test")
        del exp
        #self.fail()
        
    def test_status_getters(self):
        experiment = Experiment.Experiment(".", "test_exp", folds=2)
        # Candidate jobs
        experiment.add_job({"x": "0"})
        experiment.add_job({"x": "1"})
        # Complete jobs
        experiment.add_job({"x": "2"})
        experiment.set_one_fold_running(2, 0)
        experiment.set_one_fold_complete(2, 0, 1, 1)
        experiment.set_one_fold_running(2, 1)
        experiment.set_one_fold_complete(2, 1, 1, 1)
        experiment.add_job({"x": "3"})
        experiment.set_one_fold_running(3, 0)
        experiment.set_one_fold_complete(3, 0, 1, 1)
        experiment.set_one_fold_running(3, 1)
        experiment.set_one_fold_complete(3, 1, 1, 1)
        # Incomplete jobs
        experiment.add_job({"x": "4"})
        experiment.set_one_fold_running(4, 0)
        experiment.set_one_fold_complete(4, 0, 1, 1)
        experiment.add_job({"x": "5"})
        experiment.set_one_fold_running(5, 0)
        experiment.set_one_fold_complete(5, 0, 1, 1)
        # Running Jobs
        experiment.add_job({"x": "6"})
        experiment.set_one_fold_running(6, 0)
        experiment.add_job({"x": "7"})
        experiment.set_one_fold_running(7, 0)
        # Broken Jobs
        experiment.add_job({"x": "8"})
        experiment.set_one_fold_running(8, 0)
        experiment.set_one_fold_crashed(8, 0, 1000, 1)
        experiment.add_job({"x": "9"})
        experiment.set_one_fold_running(9, 0)
        experiment.set_one_fold_crashed(9, 0, 1000, 1)
        experiment.set_one_fold_running(9, 1)
        experiment.set_one_fold_crashed(9, 1, 1000, 1)
        self.assertEqual(len(experiment.get_candidate_jobs()), 2)
        self.assertEqual(len(experiment.get_complete_jobs()), 2)
        self.assertEqual(len(experiment.get_incomplete_jobs()), 3)
        self.assertEqual(len(experiment.get_running_jobs()), 2)
        self.assertEqual(len(experiment.get_broken_jobs()), 1)
        self.assertEqual(experiment.trials[9]['result'], 1000)
        self.assertNotEqual(experiment.trials[8]['result'], np.NaN)
        
    def test_get_arg_best(self):
        experiment = Experiment.Experiment(".", "test_exp", folds=2)
        [experiment.add_job({"x": i}) for i in range(10)]
        [experiment.set_one_fold_running(i, 0) for i in range(10)]
        [experiment.set_one_fold_complete(i, 0, 10 - i, 1) for i in range(10)]
        [experiment.set_one_fold_running(i, 1) for i in range(10)]
        [experiment.set_one_fold_complete(i, 1, i, 1) for i in range(10)]
        self.assertEqual(experiment.get_arg_best(), 0)

    def test_get_arg_best_NaNs(self):
        experiment = Experiment.Experiment(".", "test_exp", folds=2)
        [experiment.add_job({"x": i}) for i in range(10)]
        [experiment.set_one_fold_running(i, 0) for i in range(10)]
        [experiment.set_one_fold_complete(i, 0, 10 - i, 1) for i in range(10)]
        [experiment.set_one_fold_running(i, 1) for i in range(10)]
        [experiment.set_one_fold_complete(i, 1, i, 1) for i in range(1, 10)]
        self.assertEqual(experiment.get_arg_best(), 1)
        
    def test_add_job(self):
        exp = Experiment.Experiment(".", "test_exp", folds=5)
        self.assertEqual(len(exp.trials), 0)
        self.assertEqual(len(exp.instance_order), 0)

        _id = exp.add_job({"x": 1, "y": 2})
        self.assertEqual(len(exp.trials), 1)
        self.assertEqual(len(exp.instance_order), 0)
        self.assertEqual(exp.get_trial_from_id(_id)['instance_results'].shape,
                         (5,))
        self.assertEqual(exp.get_trial_from_id(_id)['instance_durations'].shape,
                         (5,))
        self.assertEqual(exp.get_trial_from_id(_id)['instance_status'].shape,
                         (5,))
        self.assertDictEqual(exp.get_trial_from_id(_id)['params'], {"x": 1, "y": 2})
        _sanity_check(exp)

    def test_set_one_fold_crashed(self):
        experiment = Experiment.Experiment(".", "test_exp", folds=1)
        experiment.add_job({"x": 0})
        experiment.set_one_fold_running(0, 0)
        experiment.set_one_fold_crashed(0, 0, 1000, 0)
        
    def test_one_fold_workflow(self):
        experiment = Experiment.Experiment(".", "test_exp", folds=5)
        trial_index = experiment.add_job({"x": 5})
        experiment.set_one_fold_running(trial_index, 0)
        self.assertEqual(len(experiment.get_broken_jobs()), 0)
        self.assertEqual(len(experiment.get_complete_jobs()), 0)
        self.assertEqual(len(experiment.get_running_jobs()), 1)
        self.assertEqual(experiment.get_trial_from_id(trial_index)
                         ['instance_status'][0], Experiment.RUNNING_STATE)

        experiment.set_one_fold_complete(trial_index, 0, 1, 1)
        self.assertEqual(len(experiment.get_complete_jobs()), 0)
        self.assertEqual(len(experiment.get_incomplete_jobs()), 1)
        self.assertEqual(experiment.get_trial_from_id(trial_index)
                         ['instance_status'][0], Experiment.COMPLETE_STATE)

        experiment.set_one_fold_running(trial_index, 1)
        experiment.set_one_fold_complete(trial_index, 1, 2, 1)
        self.assertEqual(len(experiment.get_incomplete_jobs()), 1)
        experiment.set_one_fold_running(trial_index, 2)
        experiment.set_one_fold_complete(trial_index, 2, 3, 1)
        self.assertEqual(len(experiment.get_incomplete_jobs()), 1)
        experiment.set_one_fold_running(trial_index, 3)
        experiment.set_one_fold_complete(trial_index, 3, 4, 1)
        self.assertEqual(len(experiment.get_incomplete_jobs()), 1)
        experiment.set_one_fold_running(trial_index, 4)
        experiment.set_one_fold_complete(trial_index, 4, 5, 1)
        self.assertEqual(len(experiment.trials), 1)
        self.assertTrue((experiment.get_trial_from_id(trial_index)["instance_results"] == [1, 2, 3, 4, 5]).all())
        self.assertEqual(len(experiment.get_complete_jobs()), 1)
        self.assertEqual(experiment.get_trial_from_id(trial_index)['status'],
                         Experiment.COMPLETE_STATE)

        trial_index1 = experiment.add_job({"x": 6})
        self.assertEqual(len(experiment.get_complete_jobs()), 1)
        self.assertEqual(len(experiment.get_candidate_jobs()), 1)
        experiment.set_one_fold_running(trial_index1, 3)
        self.assertTrue((experiment.get_trial_from_id(trial_index1)
                         ['instance_status'] ==  [0, 0, 0, 2, 0]).all())
        experiment.set_one_fold_complete(trial_index1, 3, 1, 1)
        self.assertTrue((experiment.get_trial_from_id(trial_index1)
                         ['instance_status'] == [0, 0, 0, 3, 0]).all())

        self.assertEqual(experiment.instance_order, [(0, 0), (0, 1), (0, 2),
                                                     (0, 3), (0, 4), (1, 3)])
        self.assertEqual(experiment.total_wallclock_time, 6)
        self.assertTrue((experiment.get_trial_from_id(trial_index)
                         ["instance_durations"] == [1, 1, 1, 1, 1]).all())

        # Check that check_cv_finished kicked in
        self.assertEqual(experiment.get_trial_from_id(trial_index)
                         ["result"], 3.0)
        self.assertAlmostEqual(experiment.get_trial_from_id(trial_index)['std'], 1.4142135623730951)
        # Check that check_cv_finished did not kick in
        self.assertNotEqual(experiment.get_trial_from_id(trial_index1)
                         ["result"], experiment.get_trial_from_id(trial_index1)
                         ["result"])
        self.assertNotEqual(experiment.get_trial_from_id(trial_index1)['std'],
                            experiment.get_trial_from_id(trial_index1)['std'])
        self.assertEqual(len(experiment.trials), 2)
        _sanity_check(experiment)
        
    def test_remove_all_but_first_runs(self):
        experiment = Experiment.Experiment(".", "test_exp", folds=5)
        for i in range(5):
            experiment.add_job({"x": i})
            experiment.set_one_fold_running(i, i)
            experiment.set_one_fold_complete(i, i, 1, 1)
        experiment.set_one_fold_running(2, 3)
        experiment.set_one_fold_complete(2, 3, 1, 1)

        self.assertEqual(len(experiment.get_incomplete_jobs()), 5)
        self.assertEqual(len(experiment.instance_order), 6)

        experiment.remove_all_but_first_runs(3)
        self.assertEqual(len(experiment.get_incomplete_jobs()), 3)
        self.assertEqual(len(experiment.instance_order), 3)
        self.assertTrue((experiment.get_trial_from_id(2)["instance_status"] ==
                         [0, 0, 3, 0, 0]).all())

        _sanity_check(experiment)


if __name__ == "__main__": 
    unittest.main()