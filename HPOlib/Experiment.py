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

            # Save this out.
            # self._save_jobs()

        else:
            # Load in from the pickle.
            self._load_jobs()

    def create_trial(self):
        trial = dict()
        # Status of the trial object
        trial['status'] = 0
        trial['params'] = dict()
        # Stores the validation error
        trial['result'] = np.NaN
        trial['test_error'] = np.NaN
        # Validation error for every instance
        trial['instance_results'] = np.ones((self.folds)) * np.NaN
        # Status for every instance
        trial['instance_status'] = np.zeros((self.folds), dtype=int)
        # Contains the standard deviation in case of cross validation
        trial['std'] = np.NaN
        # Accumulated duration over all instances
        trial['duration'] = np.NaN
        # Stores the duration for every instance
        trial['instance_durations'] = np.ones((self.folds)) * np.NaN
        # Store additional data in form of strings for every fold, this can
        # e.g. be a machine learning model which is saved to the disk
        trial['additional_data'] = defaultdict(str)
        return trial

    def __del__(self):
        # self._save_jobs()
        if self.locker.unlock(self.jobs_pkl):
            pass
            #    sys.stderr.write("Released lock on job grid.\n")
        else:
            raise Exception("Could not release lock on job grid.\n")

    def result_array(self):
        result = np.array([trial['result'] for trial in self.trials])
        return result

    def instance_results_array(self):
        instance_results = np.array([trial['instance_results'] for trial in
                                     self.trials])
        return instance_results

    def status_array(self):
        status = np.array([trial['status'] for trial in self.trials])
        return status

    def result_array(self):
        results = np.array([trial['result'] for trial in self.trials])
        return results

    # Return the ID of all candidate jobs
    def get_candidate_jobs(self):
        return np.nonzero(self.status_array() == CANDIDATE_STATE)[0]

    # Return the ID of all running jobs
    def get_running_jobs(self):
        return np.nonzero(self.status_array() == RUNNING_STATE)[0]

    # Return the ID of all incomplete jobs
    def get_incomplete_jobs(self):
        return np.nonzero(self.status_array() == INCOMPLETE_STATE)[0]

    # Return the ID of all complete jobs
    def get_complete_jobs(self):
        return np.nonzero(self.status_array() == COMPLETE_STATE)[0]

    # Return the ID of all broken jobs
    def get_broken_jobs(self):
        return np.nonzero(self.status_array() == BROKEN_STATE)[0]

    # Get the job id of the best value so far, if there is no result
    # available, this method consults the instance_results. If there are more
    #  than one trials with the same response value, the first trial is
    # considered to be the best. If no trial with a better response value
    # than sys.maxint is found, a ValueError is raised.
    # TODO: add a method that incomplete jobs are not considered
    def get_arg_best(self):
        best_idx = -1
        best_value = sys.maxint
        for i, trial in enumerate(self.trials):
            tmp_res = np.NaN
            if np.isfinite(trial['result']):
                tmp_res = trial['result']
            elif np.isfinite(trial['instance_results']).any():
                tmp_res = wrapping_util.nan_mean(trial['instance_results'])
                # np.nanmean is not available in older numpy versions
                # tmp_res = scipy.nanmean(trial['instance_results'])
            else:
                continue
            if tmp_res < best_value:
                best_idx = i
                best_value = tmp_res
        if best_idx == -1:
            raise ValueError("No best value found.")
        return best_idx

    # Get the best value so far, for more documentation see get_arg_best
    def get_best(self):
        best_idx = self.get_arg_best()
        return self.trials[best_idx]

    def get_trial_from_id(self, _id):
        try:
            return self.trials[_id]
        except IndexError as e:
            logger.critical("IndexError in get_trial_from_id. len(trials): "
                            "%d, accessed index: %d" % (len(self.trials), _id))
            raise e


    # Add a job to the list of all jobs
    def add_job(self, params):
        trial = self.create_trial()
        trial['params'] = params
        self.trials.append(trial)
        # Save this out.
        self._sanity_check()
        # self._save_jobs()
        return len(self.trials) - 1

    # Set the status of a job to be running
    def set_one_fold_running(self, _id, fold):
        assert(self.get_trial_from_id(_id)['instance_status'][fold] ==
               CANDIDATE_STATE)
        self.get_trial_from_id(_id)['status'] = RUNNING_STATE
        self.get_trial_from_id(_id)['instance_status'][fold] = RUNNING_STATE
        self.instance_order.append((_id, fold))
        self._sanity_check()
        # self._save_jobs()

    # Set the status of a job to be crashed
    def set_one_fold_crashed(self, _id, fold, result, duration,
                             additional_data=None):
        assert(self.get_trial_from_id(_id)['instance_status'][fold] ==
               RUNNING_STATE)
        self.trials[_id]['instance_status'][fold] = BROKEN_STATE
        self.trials[_id]['instance_durations'][fold] = duration
        self.trials[_id]['instance_results'][fold] = result
        self.trials[_id]['additional_data'][fold] = additional_data
        if (self.trials[_id]['instance_status'] != RUNNING_STATE).all():
            self.trials[_id]['status'] = INCOMPLETE_STATE
        self.check_cv_finished(_id)
        self.total_wallclock_time += duration
        self._sanity_check()
        # self._save_jobs()

    # Set the results of one fold of crossvalidation (SMAC)
    def set_one_fold_complete(self, _id, fold, result, duration,
                              additional_data=None):
        assert(self.get_trial_from_id(_id)['instance_status'][fold] ==
               RUNNING_STATE)
        self.get_trial_from_id(_id)['instance_results'][fold] = result
        self.get_trial_from_id(_id)['instance_status'][fold] = COMPLETE_STATE
        self.get_trial_from_id(_id)['instance_durations'][fold] = duration
        self.get_trial_from_id(_id)['additional_data'][fold] = additional_data
        # Set to incomplete if no job is running
        if (self.trials[_id]['instance_status'] != RUNNING_STATE).all():
            self.trials[_id]['status'] = INCOMPLETE_STATE
        # Check if all runs are finished
        self.check_cv_finished(_id)
        self.total_wallclock_time += duration
        self._sanity_check()
        # self._save_jobs()
        
    # Set the timer for the start of a new cross-validation
    def start_cv(self, time):
        self.cv_starttime.append(time)
        # self._save_jobs()
    
    # Set the timer for the end of a cross validation
    def end_cv(self, time):
        self.cv_endtime.append(time)
        # self._save_jobs()
    
    # Check if one set of cross validations is finished
    def check_cv_finished(self, _id):
        if np.isfinite(self.get_trial_from_id(_id)["instance_results"]).all():
            if np.sum(self.get_trial_from_id(_id)['instance_status'] == -1) == self.folds:
                self.get_trial_from_id(_id)['status'] = BROKEN_STATE
            else:
                self.get_trial_from_id(_id)['status'] = COMPLETE_STATE
            self.get_trial_from_id(_id)['result'] = \
                np.sum(self.get_trial_from_id(_id)['instance_results'])\
                / self.folds
            self.get_trial_from_id(_id)['std'] =\
                np.std(self.get_trial_from_id(_id)['instance_results'])
            self.get_trial_from_id(_id)['duration'] =\
                np.sum(self.get_trial_from_id(_id)['instance_durations'])
            return True
        else:
            return False
    
    # Deletes all instance runs except the first ones which are specified by the
    # parameters. Useful to delete all unnecessary entries after a crash in order
    # to restart
    def remove_all_but_first_runs(self, restored_runs):
        logger.info("Restored runs %d", restored_runs)
        logger.info("%s %s" ,self.instance_order, len(self.instance_order))
        if len(self.instance_order) == restored_runs:
            pass
        else:
            for _id, instance in self.instance_order[-1:restored_runs - 1:-1]:
                logger.info("Deleting %d %d", _id, instance)
                if np.isfinite(self.trials[_id]['instance_durations'][instance]):
                    self.total_wallclock_time -= \
                        self.trials[_id]['instance_durations'][instance]

                self.trials[_id]['instance_durations'][instance] = np.NaN
                self.trials[_id]['instance_results'][instance] = np.NaN
                self.trials[_id]['instance_status'][instance] = 0
                self.instance_order.pop()
                
                self.trials[_id]['duration'] = np.NaN
                
                if not np.isfinite(self.trials[_id]['instance_results']).any():
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
        # self._save_jobs()

    def _trial_sanity_check(self, trial):
        assert(len(trial['instance_results']) == len(trial['instance_status'])
               == len(trial['instance_durations']))
        for i in range(len(trial['instance_results'])):
            assert ((np.isfinite(trial['instance_results'][i]) and
                    trial['instance_status'][i] in (COMPLETE_STATE, BROKEN_STATE)) or
                    (not np.isfinite(trial['instance_results'][i]) and
                    trial['instance_status'][i] not in (COMPLETE_STATE, BROKEN_STATE))), \
                   (trial['instance_results'][i], trial['instance_status'][i])

    def _sanity_check(self):
        total_wallclock_time = 0
        finite_instance_results = 0
        for trial in self.trials:
            self._trial_sanity_check(trial)

            # Backwards compability with numpy 1.6
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wallclock_time = np.nansum(trial['instance_durations'])
                total_wallclock_time += wallclock_time if np.isfinite(wallclock_time) else 0
        assert (wrapping_util.float_eq(total_wallclock_time,
                                       self.total_wallclock_time)), \
            (total_wallclock_time, self.total_wallclock_time)

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
