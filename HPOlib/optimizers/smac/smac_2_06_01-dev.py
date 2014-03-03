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

import numpy as np

import HPOlib.wrapping_util as wrapping_util


logger = logging.getLogger("HPOlib.smac_2_06_01-dev")


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

version_info = ["Automatic Configurator Library ==> v2.06.01-development-643 (a1f71813a262)",
                "Random Forest Library ==> v1.05.01-development-95 (4a8077e95b21)",
                "SMAC ==> v2.06.01-development-620 (9380d2c6bab9)"]

#optimizer_str = "smac_2_06_01-dev"


def _get_state_run(optimizer_dir):
    rungroups = glob.glob(optimizer_dir + "scenario-SMAC*")
    if len(rungroups) == 1:
        rungroup = rungroups[0]
    else:
        logger.warning("Found multiple rungroups, take the newest one.")
        creation_times = []
        for i, filename in enumerate(rungroups):
            creation_times.append(float(os.path.getctime(filename)))
        newest = np.argmax(creation_times)
        rungroup = rungroups[newest]
        logger.info(creation_times, newest, rungroup)
    state_runs = glob.glob(rungroup + "/state-run*")

    if len(state_runs) != 1:
        raise Exception("wrapping.py can only restore runs with only one" +
                        " state-run. Please delete all others you don't want" +
                        "to use.")
    return state_runs[0]


def build_smac_call(config, options, optimizer_dir):
    import HPOlib
    algo_exec_dir = os.path.dirname(HPOlib.__file__)

    call = config.get('SMAC', 'path_to_optimizer') + "/smac"
    call = " ".join([call, '--numRun', str(options.seed),
                    '--scenario-file', os.path.join(optimizer_dir, 'scenario.txt'),
                    '--cutoffTime', config.get('SMAC', 'cutoffTime'),
                    # The instance file does interfere with state restoration, it will only
                    # be loaded if no state is restored (look further down in the code
                    # '--instanceFile', config.get('SMAC', 'instanceFile'),
                    '--intraInstanceObj', config.get('SMAC', 'intraInstanceObj'),
                    '--runObj', config.get('SMAC', 'runObj'),
                    # '--testInstanceFile', config.get('SMAC', 'testInstanceFile'),
                    '--algoExec',  '"python', os.path.join(algo_exec_dir, config.get('SMAC', 'algoExec')) + '"',
                    '--execDir', optimizer_dir,
                    '-p', config.get('SMAC', 'p'),
                    # The experiment dir MUST not be specified when restarting, it is set
                    # further down in the code
                    # '--experimentDir', optimizer_dir,
                    '--numIterations', config.get('SMAC', 'numIterations'),
                    '--totalNumRunsLimit', config.get('SMAC', 'totalNumRunsLimit'),
                    '--outputDirectory', optimizer_dir,
                    '--numConcurrentAlgoExecs', config.get('SMAC', 'numConcurrentAlgoExecs'),
                    # '--runGroupName', config.get('SMAC', 'runGroupName'),
                    '--maxIncumbentRuns', config.get('SMAC', 'maxIncumbentRuns'),
                    '--retryTargetAlgorithmRunCount',
                    config.get('SMAC', 'retryTargetAlgorithmRunCount'),
                    '--save-runs-every-iteration true',
                    '--intensification-percentage',
                    config.get('SMAC', 'intensification_percentage'),
                    '--rf-split-min', config.get('SMAC', 'rf_split_min')])

    if config.getboolean('SMAC', 'deterministic'):
        call = " ".join([call, '--deterministic true'])

    if config.getboolean('SMAC', 'adaptiveCapping') and \
            config.get('SMAC', 'runObj') == "RUNTIME":
        call = " ".join([call, '--adaptiveCapping true'])
    
    if config.getboolean('SMAC', 'rf_full_tree_bootstrap'):
        call = " ".join([call, '--rf-full-tree-bootstrap true'])
    
    if options.restore:
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


def restore(config, optimizer_dir, **kwargs):
    """
    Returns the number of restored runs.
    """

    ############################################################################
    # Run SMAC in a manner that it restores the files but then exits
    fh = open(optimizer_dir + "smac_restart.out", "w")
    smac_cmd = re.sub('python ' + os.path.dirname(os.path.realpath(__file__)) +
                      "/" + config.get('SMAC', 'algoExec'), 'pwd',
                      kwargs['cmd'])
    smac_cmd = re.sub('--outputDirectory ' + optimizer_dir, '--outputDirectory '
                      + optimizer_dir + "restart_rungroups", smac_cmd)
    logger.info(smac_cmd)
    process = subprocess.Popen(smac_cmd, stdout=fh, stderr=fh, shell=True,
                               executable="/bin/bash")
    logger.info("----------------------RUNNING--------------------------------")
    ret = process.wait()
    fh.close()
    logger.info("Finished with return code: " + str(ret))
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
    state_run = _get_state_run(optimizer_dir)

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


#noinspection PyUnusedLocal
def main(config, options, experiment_dir, **kwargs):
    # config:           Loaded .cfg file
    # options:          Options containing seed, restore_dir, 
    # experiment_dir:   Experiment directory/Benchmark_directory
    # **kwargs:         Nothing so far
    time_string = wrapping_util.get_time_string()

    optimizer_str = os.path.splitext(os.path.basename(__file__))[0]

    # Find experiment directory
    if options.restore:
        if not os.path.exists(options.restore):
            raise Exception("The restore directory does not exist")
        optimizer_dir = options.restore
    else:
        optimizer_dir = os.path.join(experiment_dir, optimizer_str + "_" +
                                     str(options.seed) + "_" + time_string)
    # Build call
    cmd = build_smac_call(config, options, optimizer_dir)

    # Set up experiment directory
    if not os.path.exists(optimizer_dir):
        os.mkdir(optimizer_dir)
        # TODO: This can cause huge problems when the files are located
        # somewhere else?
        params = os.path.split(config.get('SMAC', "p"))[1]
        
        # Copy the smac search space and create the instance information
        fh = open(os.path.join(optimizer_dir, 'train.txt'), "w")
        for i in range(config.getint('HPOLIB', 'numberCV')):
            fh.write(str(i) + "\n")
        fh.close()
        
        fh = open(os.path.join(optimizer_dir, 'test.txt'), "w")
        for i in range(config.getint('HPOLIB', 'numberCV')):
            fh.write(str(i) + "\n")
        fh.close()
        
        fh = open(os.path.join(optimizer_dir, "scenario.txt"), "w")
        fh.close()
                        
        if not os.path.exists(os.path.join(optimizer_dir, params)):
            os.symlink(os.path.join(experiment_dir, optimizer_str, params),
                       os.path.join(optimizer_dir, params))
    logger.info("### INFORMATION ################################################################")
    logger.info("# You're running %40s                      #" % config.get('SMAC', 'path_to_optimizer'))
    for v in version_info:
        logger.info("# %76s #" % v)
    logger.info("# A newer version might be available, but not yet built in.                    #")
    logger.info("# Please use this version only to reproduce our results on automl.org          #")
    logger.info("################################################################################")
    return cmd, optimizer_dir