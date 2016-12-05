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
import sys
import HPOlib.wrapping_util as wrapping_util
from HPOlib.optimizer_algorithm import OptimizerAlgorithm


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

version_info = ["R ==> 3.0.2",
                "irace ==> 1.07.1202"
                ]


class IRACE(OptimizerAlgorithm):

    def __init__(self):
        self.optimizer_name = 'irace'
        self.optimizer_dir = os.path.abspath("./irace_1_07")
        self.logger = logging.getLogger("HPOlib.irace_1_07")
        self.logger.info("optimizer_name:%s" % self.optimizer_name)
        self.logger.info("optimizer_dir:%s" % self.optimizer_dir)

    def check_dependencies(self):
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

    def build_call(self, config, options, optimizer_dir):
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
                         '--seed', str(options.seed),
                         '--soft-restart', config.get("irace", "softRestart"),
                         ])
        self.logger.info("call:%s", call)
        return call

    # setup directory where experiment will run
    def custom_setup(self, config, options, experiment_dir, optimizer_dir):

        # setup path of irace optimizer
        r_lib = os.path.dirname(os.path.abspath(config.get("irace", "path_to_optimizer")))
        if "R_LIBS" in os.environ:
            os.environ["R_LIBS"] = r_lib + ":" + os.environ["R_LIBS"]
        else:
            os.environ["R_LIBS"] = r_lib

        self.logger.info("R_LIBS: %s" % str(os.environ["R_LIBS"]))

        # Set up experiment directory
        os.mkdir(optimizer_dir)
        os.mkdir(os.path.join(optimizer_dir, "Instances"))
        instances_dir = os.path.join(optimizer_dir, "Instances", "1")

        open(instances_dir, 'a').close()

        # copy tune-conf and hook-run to experiment dir
        shutil.copy(os.path.join(os.path.dirname(os.path.realpath(__file__)), "tune-conf"), optimizer_dir)
        shutil.copy(os.path.join(os.path.dirname(os.path.realpath(__file__)), "hook-run"), optimizer_dir)

        return optimizer_dir

    def manipulate_config(self, config):
        if not config.has_section('irace'):
            config.add_section('irace')

        # optional cases
        if not config.has_option('irace', 'params'):
            raise Exception("irace:params not specified in .cfg")

        path_to_optimizer = config.get('irace', 'path_to_optimizer')
        if not os.path.isabs(path_to_optimizer):
            path_to_optimizer = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_to_optimizer)

        path_to_optimizer = os.path.normpath(path_to_optimizer)
        if not os.path.exists(path_to_optimizer):
            self.logger.critical("Path to optimizer not found: %s" % path_to_optimizer)
            sys.exit(1)

        config.set('irace', 'path_to_optimizer', path_to_optimizer)
        self.logger.info("path_to_opt:%s" % path_to_optimizer)

        return config
