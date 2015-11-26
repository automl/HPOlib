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

import logging
import os
import subprocess
import sys
import time
import shutil

import HPOlib.wrapping_util as wrapping_util


logger = logging.getLogger("HPOlib.irace_1_06-dev")


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

version_info = ["R ==> 3.0.2",
                "irace ==> 1.06.997"
                ]


def get_algo_exec():
    return '"python ' + os.path.join(os.path.dirname(__file__),
                                     'irace_to_HPOlib.py') + '"'


def check_dependencies():
    process = subprocess.Popen("which R", stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, shell=True,
                               executable="/bin/bash")
    stdoutdata, stderrdata = process.communicate()

    if stdoutdata is not None and "R" in stdoutdata:
        pass
    else:
        raise Exception("R cannot be found"
                        "Are you sure that it's installed?\n"
                        "Your $PATH is: " + os.environ['PATH'])


def build_irace_call(config, options, optimizer_dir):
    call = "".join([config.get('irace', 'path_to_optimizer'), "/irace"])
    # call = "irace"
    call = " ".join([call,
                     '--param-file', config.get("irace", "params"),
                     '--max-experiments', config.get("HPOLIB", "number_of_jobs"),
                     '--digits', config.get("irace", "digits"),
                     '--debug-level', config.get("irace", "debug-level"),
                     '--iterations', config.get("irace", "iterations"),
                     '--experiment-per-iteration', config.get("irace", "experiment-per-iteration"),
                     '--sample-instances', config.get("irace", "sample-instances"),
                     '--test-type', config.get("irace", "test-type"),
                     '--first-test', config.get("irace", "first-test"),
                     '--each-test', config.get("irace", "each-test"),
                     '--min-survival', config.get("irace", "min-survival"),
                     '--num-candidates', config.get("irace", "num-candidates"),
                     '--mu', config.get("irace", "mu"),
                     '--confidence', config.get("irace", "confidence"),
                     '--seed', config.get("irace", "seed"),
                     '--soft-restart', config.get("irace", "softRestart"),
                     ])
    logger.info("call:%s", call)
    # # Set all general parallel stuff here
    # call = " ".join([call, '--numRun', str(options.seed),
    #                 '--cli-log-all-calls true',
    #                 '--cutoffTime', config.get('SMAC', 'cutoff_time'),
    #                 '--intraInstanceObj', config.get('SMAC', 'intra_instance_obj'),
    #                 '--runObj', config.get('SMAC', 'run_obj'),
    #                 '--algoExec', get_algo_exec(),
    #                 '--numIterations', config.get('SMAC', 'num_iterations'),
    #                 '--totalNumRunsLimit', config.get('SMAC', 'total_num_runs_limit'),
    #                 '--outputDirectory', optimizer_dir,
    #                 '--numConcurrentAlgoExecs', config.get('SMAC', 'num_concurrent_algo_execs'),
    #                 '--maxIncumbentRuns', config.get('SMAC', 'max_incumbent_runs'),
    #                 '--retryTargetAlgorithmRunCount',
    #                 config.get('SMAC', 'retry_target_algorithm_run_count'),
    #                 '--intensification-percentage',
    #                 config.get('SMAC', 'intensification_percentage'),
    #                 '--initial-incumbent', config.get('SMAC', 'initial_incumbent'),
    #                 '--rf-split-min', config.get('SMAC', 'rf_split_min'),
    #                 '--validation', config.get('SMAC', 'validation'),
    #                 '--runtime-limit', config.get('SMAC', 'runtime_limit'),
    #                 '--exec-mode', config.get('SMAC', 'exec_mode')])
    #
    # if config.getboolean('SMAC', 'save_runs_every_iteration'):
    #     call = " ".join([call, '--save-runs-every-iteration true'])
    # else:
    #     call = " ".join([call, '--save-runs-every-iteration false'])
    #
    # if config.getboolean('SMAC', 'deterministic'):
    #     call = " ".join([call, '--deterministic true'])
    #
    # if config.getboolean('SMAC', 'adaptive_capping') and \
    #         config.get('SMAC', 'run_obj') == "RUNTIME":
    #     call = " ".join([call, '--adaptiveCapping true'])
    #
    # if config.getboolean('SMAC', 'rf_full_tree_bootstrap'):
    #     call = " ".join([call, '--rf-full-tree-bootstrap true'])
    #
    # # This options are set separately, because they depend on the optimizer
    # # directory and might cause trouble when using a shared model
    # if config.get('SMAC', 'shared_model') != 'False':
    #     call = " ".join([call, "--shared-model-mode true",
    #                      "--shared-model-mode-frequency",
    #                      config.get("SMAC", "shared_model_mode_frequency"),
    #                      '-p', os.path.join(optimizer_dir, os.path.basename(config.get('SMAC', 'p'))),
    #                      '--scenario-file', os.path.join(optimizer_dir, 'scenario.txt')])
    # else:
    #     call = " ".join([call, '-p', os.path.join(optimizer_dir, os.path.basename(config.get('SMAC', 'p'))),
    #                      '--execDir', optimizer_dir,
    #                      '--scenario-file', os.path.join(optimizer_dir, 'scenario.txt')])
    #
    # call = " ".join([call, '--instanceFile',
    #                  os.path.join(optimizer_dir, 'train.txt'),
    #                  '--testInstanceFile',
    #                  os.path.join(optimizer_dir, 'test.txt')])
    return call


# noinspection PyUnusedLocal
def main(config, options, experiment_dir, experiment_directory_prefix, **kwargs):
    # config:           Loaded .cfg file
    # options:          Options containing seed, restore_dir,
    # experiment_dir:   Experiment directory/Benchmark_directory
    # **kwargs:         Nothing so far
    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # os.environ["PATH"] = config.get("irace", "path_to_optimizer") + ":" + os.environ["PATH"]
    # logger.info("path variable:%s", os.environ["PATH"])
    logger.info("abs path:%s", config.sections())
    time_string = wrapping_util.get_time_string()
    # sys.exit()
    optimizer_str = os.path.splitext(os.path.basename(__file__))[0]
    # logger.info("Optimizer_str: %s" % optimizer_str)

    # Find experiment directory
    optimizer_dir = os.path.join(experiment_dir,
                                 experiment_directory_prefix +
                                 optimizer_str + "_" +
                                 str(options.seed) + "_" + time_string)

    # Build call
    cmd = build_irace_call(config, options, optimizer_dir)
    # logger.info("cmd:%s", cmd)
    # logger.info("opt_dir:%s", optimizer_dir)
    # Set up experiment directory
    os.mkdir(optimizer_dir)
    os.mkdir(os.path.join(optimizer_dir, "Instances"))
    instances_dir = os.path.join(optimizer_dir, "Instances", "1")
    # logger.info("instances_dir:%s", instances_dir)
    open(instances_dir, 'a').close()

    # copy tune-conf and hook-run to experiment dir
    # logger.info("cwd:%s", os.getcwd())
    shutil.copy("../../optimizers/irace/tune-conf", optimizer_dir)
    shutil.copy("../../optimizers/irace/hook-run", optimizer_dir)

    logger.info("### INFORMATION ################################################################")
    logger.info("# You're running %35s                  #" % config.get('irace', 'path_to_optimizer'))
    for v in version_info:
        logger.info("# %76s #" % v)
    logger.info("# This is an updated version.                                                  #")
    logger.info("################################################################################")
    return cmd, optimizer_dir
