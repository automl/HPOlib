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
    call = os.path.join(config.get('irace', 'path_to_optimizer'), "bin", "irace")
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
    return call


# noinspection PyUnusedLocal
def main(config, options, experiment_dir, experiment_directory_prefix, **kwargs):
    # config:           Loaded .cfg file
    # options:          Options containing seed, restore_dir,
    # experiment_dir:   Experiment directory/Benchmark_directory
    # **kwargs:         Nothing so far

    r_lib = os.path.dirname(os.path.abspath(config.get("irace", "path_to_optimizer")))
    if "R_LIBS" in os.environ:
        os.environ["R_LIBS"] = r_lib + ":" + os.environ["R_LIBS"]
    else:
        os.environ["R_LIBS"] = r_lib

    logger.info("R_LIBS: %s" % str(os.environ["R_LIBS"]))
    time_string = wrapping_util.get_time_string()
    optimizer_str = os.path.splitext(os.path.basename(__file__))[0]

    # Find experiment directory
    optimizer_dir = os.path.join(experiment_dir,
                                 experiment_directory_prefix +
                                 optimizer_str + "_" +
                                 str(options.seed) + "_" + time_string)

    # Build call
    cmd = build_irace_call(config, options, optimizer_dir)

    # Set up experiment directory
    os.mkdir(optimizer_dir)
    os.mkdir(os.path.join(optimizer_dir, "Instances"))
    instances_dir = os.path.join(optimizer_dir, "Instances", "1")

    open(instances_dir, 'a').close()

    # copy tune-conf and hook-run to experiment dir
    shutil.copy("../../optimizers/irace/tune-conf", optimizer_dir)
    shutil.copy("../../optimizers/irace/hook-run", optimizer_dir)

    logger.info("### INFORMATION ################################################################")
    logger.info("# You're running %35s                  #" % config.get('irace', 'path_to_optimizer'))
    for v in version_info:
        logger.info("# %76s #" % v)
    logger.info("# This is an updated version.                                                  #")
    logger.info("################################################################################")
    return cmd, optimizer_dir
