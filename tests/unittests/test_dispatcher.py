import os
import unittest

import HPOlib.dispatcher.dispatcher as dispatcher
import HPOlib.Experiment as Experiment


class DispatcherTest(unittest.TestCase):
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

        trial_index0 = dispatcher.get_trial_index(experiment, 0, params0)
        self.assertEqual(trial_index0, 0)
        experiment.set_one_fold_running(trial_index0, 0)
        experiment.set_one_fold_complete(trial_index0, 0, 1, 1)
        self.assertEqual(trial_index0,
                         dispatcher.get_trial_index(experiment, 1, params0))
        experiment.set_one_fold_running(trial_index0, 1)
        experiment.set_one_fold_complete(trial_index0, 1, 1, 1)
        self.assertEqual(trial_index0,
                         dispatcher.get_trial_index(experiment, 2, params0))
        experiment.set_one_fold_running(trial_index0, 2)
        experiment.set_one_fold_complete(trial_index0, 2, 1, 1)
        self.assertEqual(trial_index0,
                         dispatcher.get_trial_index(experiment, 3, params0))
        experiment.set_one_fold_running(trial_index0, 3)
        experiment.set_one_fold_complete(trial_index0, 3, 1, 1)
        self.assertEqual(trial_index0,
                         dispatcher.get_trial_index(experiment, 4, params0))
        experiment.set_one_fold_running(trial_index0, 4)
        experiment.set_one_fold_complete(trial_index0, 4, 1, 1)

        trial_index1 = dispatcher.get_trial_index(experiment, 0, params1)
        self.assertEqual(trial_index1, 1)
        experiment.set_one_fold_running(trial_index1, 0)
        experiment.set_one_fold_complete(trial_index1, 0, 1, 1)
        self.assertEqual(trial_index1,
                         dispatcher.get_trial_index(experiment, 1, params1))
        experiment.set_one_fold_running(trial_index1, 1)
        experiment.set_one_fold_complete(trial_index1, 1, 1, 1)
        self.assertEqual(trial_index1,
                         dispatcher.get_trial_index(experiment, 2, params1))
        experiment.set_one_fold_running(trial_index1, 2)
        experiment.set_one_fold_complete(trial_index1, 2, 1, 1)
        self.assertEqual(trial_index1,
                         dispatcher.get_trial_index(experiment, 3, params1))
        experiment.set_one_fold_running(trial_index1, 3)
        experiment.set_one_fold_complete(trial_index1, 3, 1, 1)
        self.assertEqual(trial_index1,
                         dispatcher.get_trial_index(experiment, 4, params1))
        experiment.set_one_fold_running(trial_index1, 4)
        experiment.set_one_fold_complete(trial_index1, 4, 1, 1)

        trial_index2 = dispatcher.get_trial_index(experiment, 0, params2)
        self.assertEqual(trial_index2, 2)
        experiment.set_one_fold_running(trial_index2, 0)
        experiment.set_one_fold_complete(trial_index2, 0, 1, 1)

        trial_index3 = dispatcher.get_trial_index(experiment, 0, params3)
        self.assertEqual(trial_index3, 3)
        experiment.set_one_fold_running(trial_index3, 0)
        experiment.set_one_fold_complete(trial_index3, 0, 1, 1)

        trial_index4 = dispatcher.get_trial_index(experiment, 0, params4)
        self.assertEqual(trial_index4, 4)
        experiment.set_one_fold_running(trial_index4, 0)
        experiment.set_one_fold_complete(trial_index4, 0, 1, 1)

        self.assertEqual(trial_index2,
                         dispatcher.get_trial_index(experiment, 3, params2))
        self.assertEqual(trial_index4,
                         dispatcher.get_trial_index(experiment, 4, params4))

        # Since params1 were already evaluated, this should be a new trial_index
        trial_index_test1 = dispatcher.get_trial_index(experiment, 0, params1)
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

        trial_index0 = dispatcher.get_trial_index(experiment, 0, params0)
        self.assertEqual(trial_index0, 0)
        experiment.set_one_fold_running(trial_index0, 0)
        experiment.set_one_fold_complete(trial_index0, 0, 1, 1)

        trial_index1 = dispatcher.get_trial_index(experiment, 0, params1)
        self.assertEqual(trial_index1, 1)
        experiment.set_one_fold_running(trial_index1, 0)
        experiment.set_one_fold_complete(trial_index1, 0, 1, 1)

        trial_index2 = dispatcher.get_trial_index(experiment, 0, params2)
        self.assertEqual(trial_index2, 2)
        experiment.set_one_fold_running(trial_index2, 0)
        experiment.set_one_fold_complete(trial_index2, 0, 1, 1)

        trial_index3 = dispatcher.get_trial_index(experiment, 0, params3)
        self.assertEqual(trial_index3, 3)
        experiment.set_one_fold_running(trial_index3, 0)
        experiment.set_one_fold_complete(trial_index3, 0, 1, 1)

        trial_index4 = dispatcher.get_trial_index(experiment, 0, params4)
        self.assertEqual(trial_index4, 4)
        experiment.set_one_fold_running(trial_index4, 0)
        experiment.set_one_fold_complete(trial_index4, 0, 1, 1)

        self.assertEqual(5,
                         dispatcher.get_trial_index(experiment, 0, params2))
