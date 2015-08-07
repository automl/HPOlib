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

from collections import defaultdict
import cPickle
import logging
import os
import sys
import tempfile
import warnings

import numpy as np

import HPOlib.Locker as Locker
import HPOlib.wrapping_util as wrapping_util


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


logger = logging.getLogger("HPOlib.experiment")

# Do not forget to increment this if you add a new field either to Experiment
#  or Trial
VERSION = 1

CANDIDATE_STATE = 0
INCOMPLETE_STATE = 1
RUNNING_STATE = 2
COMPLETE_STATE = 3
BROKEN_STATE = -1



def load_experiment_file():
    optimizer = wrapping_util.get_optimizer()
    experiment = Experiment(".", optimizer)
    return experiment


class Experiment:
    def __init__(self, expt_dir, expt_name, max_wallclock_time=
                 sys.float_info.max, title=None, folds=1):
        self.expt_dir = expt_dir

        if folds < 1:
            folds = 1

        self.jobs_pkl = os.path.abspath(
            os.path.join(expt_dir, expt_name + ".pkl"))
        self.locker = Locker.Locker()

        # Only one process at a time is allowed to have access to this.
        #logger.info("Waiting to lock experiments file " +
        #                 self.jobs_pkl + "...")
        self.locker.lock_wait(self.jobs_pkl)
        #logger.info("...acquired\n")

        # Does this exist already?
        if not os.path.exists(self.jobs_pkl):

            # Set up the experiments file for the first time
            # General information
            # TODO: Unfortunately, this is also the optimizer name
            self.experiment_name = expt_name
            self.title = title
            self.optimizer = None
            self.folds = folds
            self.instance_order = []

            self.trials = []

            # Time information
            # Wallclock_time used for the functions (should be the sum of all
            # instance_durations)
            self.total_wallclock_time = 0
            # The maximal allowed wallclock time
            self.max_wallclock_time = max_wallclock_time
            # Time when wrapping.py kicks of the optimizer
            self.starttime = []
            # Time when the focus is passed back to the optimizer
            self.endtime = []
            # Is triggered everytime cv.py is called, is used to calculate the
            # optimizer time
            self.cv_starttime = []
            # Is triggered when cv.py leaves, used to calculate the optimizer
            # time They are alternatively called when runsolver_wrapper is
            # called by SMAC
            self.cv_endtime = []
            # Dummy field, this will be calculated by wrapping.py after
            # everything is finished
            self.optimizer_time = []
            # A field which denotes the version of the Experiment format,
            # in new versions, there might be new fields which have to be
            # pickled and loaded and so on...
            self.version = VERSION
            # Determine whether the experiment is open for reading and writing

            # Save this out.
            # self._save_jobs()

        else:
            # Load in from the pickle.
            self._load_jobs()

        self._is_closed = False

    def _create_trial(self):
        trial = dict()
        # Status of the trial object
        trial['status'] = 0
        trial['test_status'] = 0
        trial['params'] = dict()
        # Stores the validation error
        trial['result'] = np.NaN
        trial['test_result'] = np.NaN
        # Validation error for every instance
        trial['instance_results'] = np.ones((self.folds)) * np.NaN
        trial['test_instance_results'] = np.ones((1,)) * np.NaN
        # Status for every instance
        trial['instance_status'] = np.zeros((self.folds), dtype=int)
        trial['test_instance_status'] = np.zeros((1,), dtype=int)
        # Contains the standard deviation in case of cross validation
        trial['std'] = np.NaN
        trial['test_std'] = np.NaN
        # Accumulated duration over all instances
        trial['duration'] = np.NaN
        trial['test_duration'] = np.NaN
        # Stores the duration for every instance
        trial['instance_durations'] = np.ones((self.folds)) * np.NaN
        trial['test_instance_durations'] = np.ones((1,)) * np.NaN
        # Store additional data in form of strings for every fold, this canv
        # e.g. be a machine learning model which is saved to the disk
        trial['additional_data'] = defaultdict(str)
        trial['test_additional_data'] = defaultdict(str)
        return trial

    def close(self):
        if self.is_closed():
            pass
        elif self.locker.unlock(self.jobs_pkl):
            self._is_closed = True
        else:
            raise Exception("Could not release lock on job grid.\n")

    def is_closed(self):
        return self._is_closed

    def __del__(self):
        self.close()

    def result_array(self):
        result = np.array([trial['result'] for trial in self.trials])
        return result

    def test_result_array(self):
        test_result = np.array([trial['test_result'] for trial in self.trials])
        return test_result

    def instance_results_array(self):
        instance_results = np.array([trial['instance_results'] for trial in
                                     self.trials])
        return instance_results

    def test_instance_results_array(self):
        test_instance_results = np.array([trial['test_instance_results'] for
                                          trial in self.trials])
        return test_instance_results

    def status_array(self):
        status = np.array([trial['status'] for trial in self.trials])
        return status

    def test_status_array(self):
        test_status = np.array([trial['test_status'] for trial in self.trials])
        return test_status

    # Return the ID of all candidate jobs
    def get_candidate_jobs(self):
        return self._get_jobs_by_status(CANDIDATE_STATE, False)

    def get_candidate_test_jobs(self):
        return self._get_jobs_by_status(CANDIDATE_STATE, True)

    # Return the ID of all running jobs
    def get_running_jobs(self):
        return self._get_jobs_by_status(RUNNING_STATE, False)

    def get_running_test_jobs(self):
        return self._get_jobs_by_status(RUNNING_STATE, True)

    # Return the ID of all incomplete jobs
    def get_incomplete_jobs(self):
        return self._get_jobs_by_status(INCOMPLETE_STATE, False)

    def get_incomplete_test_jobs(self):
        return self._get_jobs_by_status(INCOMPLETE_STATE, True)

    # Return the ID of all complete jobs
    def get_complete_jobs(self):
        return self._get_jobs_by_status(COMPLETE_STATE, False)

    def get_complete_test_jobs(self):
        return self._get_jobs_by_status(COMPLETE_STATE, True)

    # Return the ID of all broken jobs
    def get_broken_jobs(self):
        return self._get_jobs_by_status(BROKEN_STATE, False)

    def get_broken_test_jobs(self):
        return self._get_jobs_by_status(BROKEN_STATE, True)

    # The basic functionality to return all jobs with a given state
    def _get_jobs_by_status(self, status, test=False):
        if test:
            return np.nonzero(self.test_status_array() == status)[0]
        else:
            return np.nonzero(self.status_array() == status)[0]

    def get_arg_best(self, consider_incomplete=False):
        """Get the job id of the best result.

        If there is no result for a configuration, the behaviour depends on
        the argument ``consider_incomplete``. If it is set to True, the mean
        of all non-NaN values is considered.

        Parameters
        ----------
        consider_incomplete : bool, default=False
            Consider the nanmean of incomplete trials if True.

        Returns
        -------
        int
            ID of the best hyperparameter configuration found so far.

        Raises
        ------
        ValueError
            If no non-NaN value is found.
        """
        best_idx = -1
        best_value = sys.maxint
        for i, trial in enumerate(self.trials):
            tmp_res = np.NaN
            if np.isfinite(trial['result']):
                tmp_res = trial['result']
            elif consider_incomplete and np.isfinite(trial[\
                    'instance_results']).any():
                tmp_res = wrapping_util.nan_mean(trial['instance_results'])
            else:
                continue
            if tmp_res < best_value:
                best_idx = i
                best_value = tmp_res
        if best_idx == -1:
            raise ValueError("No best value found.")
        return best_idx

    def get_best(self):
        """Get the result of the ID returned by get_arg_best.

        Returns
        -------
        float
            Best validation result found so far.

        Raises
        ------
        ValueError
            If no non-NaN value is found.
        """
        best_idx = self.get_arg_best(consider_incomplete=False)
        return self.trials[best_idx]['result']

    def get_trial_from_id(self, _id):
        try:
            return self.trials[_id]
        except IndexError as e:
            logger.critical("IndexError in get_trial_from_id. len(trials): "
                            "%d, accessed index: %d" % (len(self.trials), _id))
            raise e

    def add_job(self, params):
        """Create a trials dictionary for a hyperparameter configuration.

        Parameters
        ----------
        configuration : dict
            A dictionary of hyperparameters.

        Returns
        -------
        int
            The trial ID of the created trials dictionary
        """
        trial = self._create_trial()
        trial['params'] = params
        self.trials.append(trial)
        self._sanity_check()
        return len(self.trials) - 1

    def clean_test_outputs(self, _id):
        """Revert all information for the run with `_id` to the default so
        the evaluation can be carried out again.

        Parameters
        ----------
        _id : int
            The ID of the trial dictionary

        Returns
        -------
        """
        trial = self.get_trial_from_id(_id)

        trial['test_status'] = CANDIDATE_STATE
        trial['test_result'] = np.NaN
        trial['test_std'] = np.NaN
        trial['test_duration'] = np.NaN

        duration = np.nansum(trial['test_instance_durations'])
        self.total_wallclock_time -= duration

        for fold in range(len(trial['test_instance_status'])):
            trial['test_instance_status'][fold] = CANDIDATE_STATE
            trial['test_instance_durations'][fold] = np.NaN
            trial['test_instance_results'][fold] = np.NaN
            trial['test_additional_data'][fold] = ""


    def set_one_fold_running(self, _id, fold):
        """Change the status of one fold to running.

        Parameters
        ----------
        _id : int
            The ID of the trial dictionary
        fold : int
            The fold is set running
        """
        trial = self.get_trial_from_id(_id)
        assert(self.get_trial_from_id(_id)['instance_status'][fold] ==
               CANDIDATE_STATE)
        trial['status'] = RUNNING_STATE
        trial['instance_status'][fold] = RUNNING_STATE
        self.instance_order.append((_id, fold))
        self._sanity_check()

    def set_one_test_fold_running(self, _id, fold):
        """Change the status of one test fold to running.

        Parameters
        ----------
        _id : int
            The ID of the trial dictionary
        fold : int
            The test fold is set running
        """
        if fold != 0:
            raise ValueError("Currently, only one test instance is allowed.")
        trial = self.get_trial_from_id(_id)
        assert(trial['test_instance_status'][fold] == CANDIDATE_STATE)
        trial['test_status'] = RUNNING_STATE
        trial['test_instance_status'][fold] = RUNNING_STATE
        self._sanity_check()

    def set_one_fold_crashed(self, _id, fold, result, duration,
                             additional_data=None):
        """Change the status of one fold to crashed.

        Parameters
        ----------
        _id : int
            The ID of the trial dictionary
        fold : int
            The fold is set crashed
        result : float
            The result of the algorithm run
        duration : float
            Number of seconds the algorithm run until it crashed
        additional_data : str
            A string with additional data from the algorithm run which crashed.
        """
        trial = self.get_trial_from_id(_id)
        assert(trial['instance_status'][fold] == RUNNING_STATE)
        trial['instance_status'][fold] = BROKEN_STATE
        trial['instance_durations'][fold] = duration
        trial['instance_results'][fold] = result
        trial['additional_data'][fold] = additional_data
        if (trial['instance_status'] != RUNNING_STATE).all():
            trial['status'] = INCOMPLETE_STATE
        self._check_cv_finished(_id)
        self.total_wallclock_time += duration
        self._sanity_check()

    def set_one_test_fold_crashed(self, _id, fold, result, duration,
                                  additional_data=None):
        """Change the status of one test fold to crashed.

        Parameters
        ----------
        _id : int
            The ID of the trial dictionary
        fold : int
            The test fold is set crashed
        result : float
            The result of the algorithm run
        duration : float
            Number of seconds the algorithm run until it crashed
        additional_data : str
            A string with additional data from the algorithm run which crashed.
        """
        if fold != 0:
            raise ValueError("Currently, only one test instance is allowed.")
        trial = self.get_trial_from_id(_id)
        assert (trial['test_instance_status'][fold] == RUNNING_STATE)
        trial['test_instance_status'][fold] = BROKEN_STATE
        trial['test_instance_durations'][fold] = duration
        trial['test_instance_results'][fold] = result
        trial['test_additional_data'][fold] = additional_data
        if (trial['test_instance_status'] != RUNNING_STATE).all():
            trial['test_status'] = INCOMPLETE_STATE
        self._check_test_finished(_id)
        self.total_wallclock_time += duration
        self._sanity_check()

    def set_one_fold_complete(self, _id, fold, result, duration,
                              additional_data=None):
        """Change the status of one test fold to complete.

        Parameters
        ----------
        _id : int
            The ID of the trial dictionary
        fold : int
            The fold is set complete
        result : float
            The result of the algorithm run
        duration : float
            Number of seconds the algorithm run needed
        additional_data : str
            A string with additional data from the algorithm run
        """

        trial = self.get_trial_from_id(_id)
        assert(trial['instance_status'][fold] == RUNNING_STATE)
        trial['instance_results'][fold] = result
        trial['instance_status'][fold] = COMPLETE_STATE
        trial['instance_durations'][fold] = duration
        trial['additional_data'][fold] = additional_data
        # Set to incomplete if no job is running
        if (trial['instance_status'] != RUNNING_STATE).all():
            trial['status'] = INCOMPLETE_STATE
        # Check if all runs are finished
        self._check_cv_finished(_id)
        self.total_wallclock_time += duration
        self._sanity_check()

    def set_one_test_fold_complete(self, _id, fold, result, duration,
                                   additional_data=None):
        """Change the status of one test fold to complete.

        Parameters
        ----------
        _id : int
            The ID of the trial dictionary
        fold : int
            The test fold is set complete
        result : float
            The result of the algorithm run
        duration : float
            Number of seconds the algorithm run needed
        additional_data : str
            A string with additional data from the algorithm run
        """
        if fold != 0:
            raise ValueError("Currently, only one test instance is allowed.")
        trial = self.get_trial_from_id(_id)
        assert (trial['test_instance_status'][fold] == RUNNING_STATE)
        trial['test_instance_results'][fold] = result
        trial['test_instance_status'][fold] = COMPLETE_STATE
        trial['test_instance_durations'][fold] = duration
        trial['test_additional_data'][fold] = additional_data
        # Set to incomplete if no job is running
        if (trial['test_instance_status'] != RUNNING_STATE).all():
            trial['test_status'] = INCOMPLETE_STATE
        # Check if all runs are finished
        self._check_test_finished(_id)
        self.total_wallclock_time += duration
        self._sanity_check()

    def start_cv(self, time):
        """Set the timer for the start of a new cross-validation run.

        Parameters
        ----------
        time : float
            Start time of the new cross-validation run.
        """
        self.cv_starttime.append(time)

    def end_cv(self, time):
        """Set the timer for the end of a running cross-validation run.

        Parameters
        ----------
        time : float
            End time of the running cross-validation run.
        """
        self.cv_endtime.append(time)

    def _check_cv_finished(self, _id):
        trial = self.get_trial_from_id(_id)
        if np.isfinite(trial["instance_results"]).all():
            if np.sum(trial['instance_status'] == -1) == self.folds:
                trial['status'] = BROKEN_STATE
            else:
                trial['status'] = COMPLETE_STATE
            trial['result'] = np.sum(trial['instance_results']) / self.folds
            trial['std'] = np.std(trial['instance_results'])
            trial['duration'] = np.sum(trial['instance_durations'])
            return True
        else:
            return False

    def _check_test_finished(self, _id):
        trial = self.get_trial_from_id(_id)
        if np.isfinite(trial["test_instance_results"]).all():
            if np.sum(trial['test_instance_status'] == BROKEN_STATE) == \
                    len(trial['test_instance_status']):
                trial['test_status'] = BROKEN_STATE
            else:
                trial['test_status'] = COMPLETE_STATE
            trial['test_result'] = np.sum(trial['test_instance_results']) / \
                                   len(trial['test_instance_results'])
            trial['test_std'] = np.std(trial['test_instance_results'])
            trial['test_duration'] = np.sum(trial['test_instance_durations'])
            return True
        else:
            return False

    def remove_all_but_first_runs(self, restored_runs):
        """Delete all but the first *restored_runs* instances.

        Useful to delete all unnecessary entries after a crash in order to
        restart.

        Parameters
        ----------
        int : restored runs
            The number of instance runs to restore. In contrast to most other
            arguments, this argument is 1-based.
        """
        logger.info("Restored runs %d", restored_runs)
        logger.info("%s %s", self.instance_order, len(self.instance_order))
        if len(self.instance_order) == restored_runs:
            pass
        else:
            for _id, instance in self.instance_order[-1:restored_runs - 1:-1]:
                logger.info("Deleting %d %d", _id, instance)

                trial = self.get_trial_from_id(_id)
                if np.isfinite(trial['instance_durations'][instance]):
                    self.total_wallclock_time -= \
                        trial['instance_durations'][instance]

                trial['instance_durations'][instance] = np.NaN
                trial['instance_results'][instance] = np.NaN
                trial['instance_status'][instance] = 0
                self.instance_order.pop()
                
                trial['duration'] = np.NaN
                
                if not np.isfinite(trial['instance_results']).any():
                    del self.trials[_id]

            # now delete all unnecessary entries in instance_order
            del self.instance_order[restored_runs:]

        # now remove all timing stuff from cv_starttime and cv_endtime
        if restored_runs / self.folds == len(self.cv_starttime) - 1:
            del self.cv_starttime[-1]
        elif restored_runs / self.folds == len(self.cv_starttime):
            pass
        # TODO: this is a very general assumption, there should be a more
        # constraining one
        elif len(self.cv_starttime) >= len(self.instance_order):
            # Intensifying optimizer, delete all except the first few entries
            del self.cv_starttime[restored_runs:]
        else:
            raise Exception("Illegal state in experiment pickle with " +
                            "restored_runs %d, length of cv_starttime " +
                            "being %d, length of instance order %d and " +
                            "number of folds %d" % \
                           (restored_runs, len(self.cv_starttime),
                            len(self.instance_order), self.folds))
        if len(self.cv_endtime) > len(self.cv_starttime):
            del self.cv_endtime[len(self.cv_starttime):]
        
        assert(len(self.instance_order) == restored_runs),\
            (len(self.instance_order), restored_runs)
        #assert(np.sum(np.isfinite(self.instance_results)) == restored_runs),\
        #    (np.sum(np.isfinite(self.instance_results)), restored_runs)
        assert(len(self.cv_starttime) == len(self.cv_endtime)),\
            (len(self.cv_starttime), len(self.cv_endtime))
        
        self._sanity_check()

    def _trial_sanity_check(self, trial):
        for key in ['instance_results', 'instance_status',
                    'instance_durations']:
            if len(trial[key]) != self.folds:
                raise ValueError("Length of array %s (%d) is not equal to the "
                                 "number of folds (%d)." %
                                 (key, len(trial[key], trial.folds)))

        for key in ['test_instance_results', 'test_instance_status',
                    'test_instance_durations']:
            if len(trial[key]) != 1:
                raise ValueError("Length of array %s (%d) is not equal to the "
                                 "number of test folds (%d)." %
                                 (key, len(trial[key]), 1))

        for i in range(len(trial['instance_results'])):
            assert ((np.isfinite(trial['instance_results'][i]) and
                    trial['instance_status'][i] in (COMPLETE_STATE, BROKEN_STATE)) or
                    (not np.isfinite(trial['instance_results'][i]) and
                    trial['instance_status'][i] not in (COMPLETE_STATE, BROKEN_STATE))), \
                   (trial['instance_results'][i], trial['instance_status'][i])

        for i in range(len(trial['test_instance_results'])):
            assert ((np.isfinite(trial['test_instance_results'][i]) and
                     trial['test_instance_status'][i] in (COMPLETE_STATE, BROKEN_STATE)) or
                    (not np.isfinite(trial['test_instance_results'][i]) and
                     trial['test_instance_status'][i] not in (COMPLETE_STATE, BROKEN_STATE))), \
                (trial['test_instance_results'][i], trial['test_instance_status'][i])

    def _sanity_check(self):
        total_wallclock_time = 0
        for trial in self.trials:
            self._trial_sanity_check(trial)

            # Backwards compability with numpy 1.6
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wallclock_time = np.nansum(trial['instance_durations'])
                test_wallclock_time = np.nansum(trial['test_instance_durations'])
                total_wallclock_time += wallclock_time if \
                    np.isfinite(wallclock_time) else 0
                total_wallclock_time += test_wallclock_time if \
                    np.isfinite(test_wallclock_time) else 0

        if not wrapping_util.float_eq(total_wallclock_time,
                                       self.total_wallclock_time):
            raise ValueError("Found an error in the time measurement. The "
                             "values %f and %f should be equal, but aren't" %
                             (total_wallclock_time, self.total_wallclock_time))

    # Automatically loads this object from a pickle file
    def _load_jobs(self):
        fh = open(self.jobs_pkl, 'r')
        jobs = cPickle.load(fh)
        fh.close()

        self.experiment_name = jobs['experiment_name']
        self.title           = jobs['title']
        self.folds           = jobs['folds']
        self.total_wallclock_time    = jobs['total_wallclock_time']
        self.max_wallclock_time      = jobs['max_wallclock_time']
        self.starttime               = jobs['starttime']
        self.endtime                 = jobs['endtime']
        self.cv_starttime            = jobs['cv_starttime']
        self.cv_endtime              = jobs['cv_endtime']
        self.optimizer               = jobs['optimizer']
        self.optimizer_time          = jobs['optimizer_time']
        self.instance_order          = jobs['instance_order']
        self.trials                  = jobs['trials']

    def _save_jobs(self):
        # Write everything to a temporary file first.
        self._sanity_check()
        fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
        cPickle.dump({'experiment_name': self.experiment_name,
                       'title'          : self.title,
                       'folds'          : self.folds,
                       'total_wallclock_time' : self.total_wallclock_time,
                       'max_wallclock_time'   : self.max_wallclock_time,
                       'starttime'            : self.starttime,
                       'endtime'              : self.endtime,
                       'cv_starttime'         : self.cv_starttime,
                       'cv_endtime'           : self.cv_endtime,
                       'optimizer'            : self.optimizer,
                       'optimizer_time'       : self.optimizer_time,
                       'instance_order'       : self.instance_order,
                       'trials'               : self.trials}, fh)
        fh.close()
        cmd = 'mv "%s" "%s"' % (fh.name, self.jobs_pkl)
        os.system(cmd)  # TODO: Replace with subprocess modules
