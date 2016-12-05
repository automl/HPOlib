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

import glob
import logging
import os
import re
import subprocess
import sys
import time
from HPOlib.optimizer_algorithm import OptimizerAlgorithm
import numpy as np
import HPOlib.wrapping_util as wrapping_uti


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

version_info = ["Algorithm Execution & Abstraction Toolkit ==> v2.08.00-master-766 (85fc099c674a)",
                "Random Forest Library ==> v1.05.01-master-106 (7fba58fe4271)",
                "SMAC ==> v2.08.00-master-731 (0e43c26c3d1f)"
               ]


class SMAC(OptimizerAlgorithm):

    def __init__(self):
        self.optimizer_name = 'SMAC'
        self.optimizer_dir = os.path.abspath("./smac_2_08_00-master")
        self.logger = logging.getLogger("HPOlib.smac_2_08_00-master")
        self.logger.info("optimizer_name:%s" % self.optimizer_name)
        self.logger.info("optimizer_dir:%s" % self.optimizer_dir)

    def get_algo_exec(self):
        return '"python ' + os.path.join(os.path.dirname(__file__),
                                         'SMAC_to_HPOlib.py') + '"'

    def check_dependencies(self):
        process = subprocess.Popen("which java", stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, shell=True, executable="/bin/bash")
        stdoutdata, stderrdata = process.communicate()

        if stdoutdata is not None and "java" in stdoutdata:
            pass
        else:
            raise Exception("Java cannot not be found. "
                            "Are you sure that it's installed?\n"
                            "Your $PATH is: " + os.environ['PATH'])

        # Check Java Version
        version_str = 'java version "1.7.0_65"'
        output = subprocess.check_output(["java", "-version"],
                                         stderr=subprocess.STDOUT)
        if version_str not in output:
            self.logger.critical("Java version (%s) does not contain %s,"
                            "you continue at you own risk" % (output, version_str))

    def _get_state_run(self, optimizer_dir):
        rungroups = glob.glob(optimizer_dir + "/" + "scenario-SMAC*")
        if len(rungroups) == 0:
            raise Exception("Could not find a rungroup in %s" % optimizer_dir)
        if len(rungroups) == 1:
            rungroup = rungroups[0]
        else:
            self.logger.warning("Found multiple rungroups, take the newest one.")
            creation_times = []
            for i, filename in enumerate(rungroups):
                creation_times.append(float(os.path.getctime(filename)))
            newest = np.argmax(creation_times)
            rungroup = rungroups[newest]
            self.logger.info(creation_times, newest, rungroup)
        state_runs = glob.glob(rungroup + "/state-run*")

        if len(state_runs) != 1:
            raise Exception("wrapping.py can only restore runs with only one" +
                            " state-run. Please delete all others you don't want" +
                            "to use.")
        return state_runs[0]

    def build_call(self, config, options, optimizer_dir):
        import HPOlib
        algo_exec_dir = os.path.dirname(HPOlib.__file__)

        call = config.get('SMAC', 'path_to_optimizer') + "/smac"

        # Set all general parallel stuff here
        call = " ".join([call, '--numRun', str(options.seed),
                         '--cli-log-all-calls true',
                         '--cutoffTime', config.get('SMAC', 'cutoff_time'),
                         # The instance file does interfere with state restoration, it will only
                         # be loaded if no state is restored (look further down in the code
                         # '--instanceFile', config.get('SMAC', 'instanceFile'),
                         '--intraInstanceObj', config.get('SMAC', 'intra_instance_obj'),
                         '--runObj', config.get('SMAC', 'run_obj'),
                         # '--testInstanceFile', config.get('SMAC', 'testInstanceFile'),
                         '--algoExec',  self.get_algo_exec(),
                         '--numIterations', config.get('SMAC', 'num_iterations'),
                         '--totalNumRunsLimit', config.get('SMAC', 'total_num_runs_limit'),
                         '--outputDirectory', optimizer_dir,
                         '--numConcurrentAlgoExecs', config.get('SMAC', 'num_concurrent_algo_execs'),
                         # '--runGroupName', config.get('SMAC', 'runGroupName'),
                         '--maxIncumbentRuns', config.get('SMAC', 'max_incumbent_runs'),
                         '--retryTargetAlgorithmRunCount',
                         config.get('SMAC', 'retry_target_algorithm_run_count'),
                         '--intensification-percentage',
                         config.get('SMAC', 'intensification_percentage'),
                         '--initial-incumbent', config.get('SMAC', 'initial_incumbent'),
                         '--rf-split-min', config.get('SMAC', 'rf_split_min'),
                         '--validation', config.get('SMAC', 'validation'),
                         '--runtime-limit', config.get('SMAC', 'runtime_limit'),
                         '--exec-mode', config.get('SMAC', 'exec_mode'),
                         '--rf-num-trees', config.get('SMAC', 'rf_num_trees'),
                         '--continous-neighbours', config.get('SMAC', 'continous_neighbours')])

        if config.getboolean('SMAC', 'save_runs_every_iteration'):
            call = " ".join([call, '--save-runs-every-iteration true'])
        else:
            call = " ".join([call, '--save-runs-every-iteration false'])

        if config.getboolean('SMAC', 'deterministic'):
            call = " ".join([call, '--deterministic true'])

        if config.getboolean('SMAC', 'adaptive_capping') and \
                config.get('SMAC', 'run_obj') == "RUNTIME":
            call = " ".join([call, '--adaptiveCapping true'])

        if config.getboolean('SMAC', 'rf_full_tree_bootstrap'):
            call = " ".join([call, '--rf-full-tree-bootstrap true'])

        # This options are set separately, because they depend on the optimizer directory and might cause trouble when
        # using a shared model
        if config.get('SMAC', 'shared_model') != 'False':
            call = " ".join([call, "--shared-model-mode true",
                             "--shared-model-mode-frequency",
                             config.get("SMAC", "shared_model_mode_frequency"),
                             '-p', os.path.join(optimizer_dir, os.path.basename(config.get('SMAC', 'p'))),
                             '--scenario-file', os.path.join(optimizer_dir, 'scenario.txt')])
        else:
            call = " ".join([call, '-p', os.path.join(optimizer_dir, os.path.basename(config.get('SMAC', 'p'))),
                             '--execDir', optimizer_dir,
                             '--scenario-file', os.path.join(optimizer_dir, 'scenario.txt')])

        if options.restore:
            raise NotImplementedError("Restoring has not been tested for this SMAC version")
            state_run = _get_state_run(optimizer_dir)

            restore_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                        os.getcwd(), state_run)
            call = " ".join([call, "--restore-scenario", restore_path])
        else:
            call = " ".join([call, '--instanceFile',
                             os.path.join(optimizer_dir, 'train.txt'),
                             '--testInstanceFile',
                             os.path.join(optimizer_dir, 'test.txt')])
        return call

    def restore(self, config, optimizer_dir, **kwargs):
        """
        Returns the number of restored runs.
        """

        ############################################################################
        # Run SMAC in a manner that it restores the files but then exits
        fh = open(optimizer_dir + "smac_restart.out", "w")
        smac_cmd = re.sub('python ' + os.path.dirname(os.path.realpath(__file__)) +
                          "/" + config.get('SMAC', 'algo_exec'), 'pwd',
                          kwargs['cmd'])
        smac_cmd = re.sub('--outputDirectory ' + optimizer_dir, '--outputDirectory '
                          + optimizer_dir + "restart_rungroups", smac_cmd)
        self.logger.info(smac_cmd)
        process = subprocess.Popen(smac_cmd, stdout=fh, stderr=fh, shell=True,
                                   executable="/bin/bash")
        self.logger.info("----------------------RUNNING--------------------------------")
        ret = process.wait()
        fh.close()
        self.logger.info("Finished with return code: " + str(ret))
        # os.remove("smac_restart.out")

        # read smac.out and look how many states are restored
        fh = open(optimizer_dir + "smac_restart.out")
        prog = re.compile(r"(Restored) ([0-9]{1,100}) (runs)")
        restored_runs = 0
        for line in fh.readlines():
            match = prog.search(line)
            if match:
                restored_runs = int(match.group(2))

        # Find out all rungroups and state-runs
        ############################################################################
        state_run = self._get_state_run(optimizer_dir)

        state_run_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      os.getcwd(), state_run)
        state_runs = glob.glob(state_run_path + "/runs_and_results-it*.csv")
        state_run_iterations = []
        for state_run in state_runs:
            match = re.search(r"(runs_and_results-it)([0-9]{1,100})(.csv)",
                              state_run)
            if match:
                state_run_iterations.append(float(match.group(2)))
        run_and_results_fn = state_runs[np.argmax(state_run_iterations)]

        runs_and_results = open(run_and_results_fn)
        lines = runs_and_results.readlines()
        state_run_iters = len(lines) - 1
        runs_and_results.close()

        fh.close()

        # TODO: Wait for a fix in SMAC
        # In SMAC, right now the number of restored iterations is at least one too high
        assert state_run_iters == restored_runs - 1, (state_run_iters, restored_runs)
        restored_runs = state_run_iters

        return restored_runs

    # setup directory where experiment will run
    def custom_setup(self, config, options, experiment_dir, optimizer_dir):

        optimizer_str = os.path.splitext(os.path.basename(__file__))[0]
        # Find experiment directory
        if options.restore:
            if not os.path.exists(options.restore):
                raise Exception("The restore directory does not exist")
            optimizer_dir = options.restore
        elif config.get('SMAC', 'shared_model') != 'False':
            optimizer_dir = os.path.join(experiment_dir, optimizer_str + "_sharedModel_" +
                                         config.get('SMAC', 'shared_model'))

        # Set up experiment directory
        # if not os.path.exists(optimizer_dir):
        try:
            os.mkdir(optimizer_dir)
            # TODO: This can cause huge problems when the files are located
            # somewhere else?
            space = config.get('SMAC', "p")
            abs_space = os.path.abspath(space)
            parent_space = os.path.join(experiment_dir, optimizer_str, space)
            if os.path.exists(abs_space):
                space = abs_space
            elif os.path.exists(parent_space):
                space = parent_space
            else:
                raise Exception("SMAC search space not found. Searched at %s and "
                                "%s" % (abs_space, parent_space))

            # if not os.path.exists(os.path.join(optimizer_dir, os.path.basename(space))):
            os.symlink(os.path.join(experiment_dir, optimizer_str, space),
                       os.path.join(optimizer_dir, os.path.basename(space)))

            # Copy the smac search space and create the instance information
            fh = open(os.path.join(optimizer_dir, 'train.txt'), "w")
            for i in range(config.getint('HPOLIB', 'number_cv_folds')):
                fh.write(str(i) + "\n")
            fh.close()

            fh = open(os.path.join(optimizer_dir, 'test.txt'), "w")
            for i in range(config.getint('HPOLIB', 'number_cv_folds')):
                fh.write(str(i) + "\n")
            fh.close()

            fh = open(os.path.join(optimizer_dir, "scenario.txt"), "w")
            fh.close()
        except OSError:
            space = config.get('SMAC', "p")
            abs_space = os.path.abspath(space)
            parent_space = os.path.join(experiment_dir, optimizer_str, space)
            ct = 0
            all_found = False
            while ct < config.getint('SMAC', 'wait_for_shared_model') and not all_found:
                time.sleep(1)
                ct += 1
                # So far we have not not found anything
                all_found = None
                if not os.path.isdir(optimizer_dir):
                    all_found = optimizer_dir
                    continue

                if not os.path.exists(os.path.join(optimizer_dir, os.path.basename(space))) and \
                        not os.path.exists(parent_space):
                    all_found = parent_space
                    continue

                if not os.path.exists(os.path.join(optimizer_dir, 'train.txt')):
                    all_found = os.path.join(optimizer_dir, 'train.txt')
                    continue
                if not os.path.exists(os.path.join(optimizer_dir, 'test.txt')):
                    all_found = os.path.join(optimizer_dir, 'test.txt')
                    continue
                if not os.path.exists(os.path.join(optimizer_dir, "scenario.txt")):
                    all_found = os.path.join(optimizer_dir, "scenario.txt")
                    continue
            if all_found is not None:
                self.logger.critical("Could not find all necessary files..abort. " +
                                "Experiment directory %s is somehow created, but not complete\n" % optimizer_dir +
                                "Missing: %s" % all_found)
                sys.exit(1)

        return optimizer_dir

    def manipulate_config(self, config):
        if not config.has_option('SMAC', 'cutoff_time'):
            print config.get('HPOLIB', 'runsolver_time_limit')
            if config.get('HPOLIB', 'runsolver_time_limit'):
                config.set('SMAC', 'cutoff_time',
                       str(config.getint('HPOLIB', 'runsolver_time_limit') + 100))
            else:
                # SMACs maxint
                config.set('SMAC', 'cutoff_time', "2147483647")
        if not config.has_option('SMAC', 'total_num_runs_limit'):
            config.set('SMAC', 'total_num_runs_limit',
                       str(config.getint('HPOLIB', 'number_of_jobs') *
                           config.getint('HPOLIB', 'number_cv_folds')))
        if not config.has_option('SMAC', 'num_concurrent_algo_execs'):
            config.set('SMAC', 'num_concurrent_algo_execs',
                       config.get('HPOLIB', 'number_of_concurrent_jobs'))

        path_to_optimizer = config.get('SMAC', 'path_to_optimizer')
        if not os.path.isabs(path_to_optimizer):
            path_to_optimizer = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_to_optimizer)

        path_to_optimizer = os.path.normpath(path_to_optimizer)
        if not os.path.exists(path_to_optimizer):
            self.logger.critical("Path to optimizer not found: %s" % path_to_optimizer)
            sys.exit(1)

        config.set('SMAC', 'path_to_optimizer', path_to_optimizer)
        config.set('SMAC', 'exec_mode', 'SMAC')

        shared_model = config.get('SMAC', 'shared_model')
        wait_time = config.getint('SMAC', 'wait_for_shared_model')
        if shared_model != 'False':
            config.getint('SMAC', 'shared_model')
            if not os.path.isdir(shared_model):
                config.set('SMAC', 'shared_model_scenario_file', os.path.join(shared_model, 'scenario.txt'))

            if config.get('HPOLIB', 'temporary_output_directory') != '':
                self.logger.critical('Using a temp_out_dir and a shared model is not possible')
                sys.exit(1)
        return config
