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
import sys

import HPOlib.wrapping_util
from HPOlib.optimizer_algorithm import OptimizerAlgorithm


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


path_to_optimizer = "optimizers/spearmint_march2014_mod/"
version_info = ("# %76s #\n" % "git version march 2014")


class SPEARMINT(OptimizerAlgorithm):

    def __init__(self):
        self.optimizer_name = 'SPEARMINT'
        self.optimizer_dir = os.path.abspath("./spearmint_gitfork_mod")
        self.logger = logging.getLogger("HPOlib.spearmint_april2013_mod")
        self.logger.info("optimizer_name:%s" % self.optimizer_name)
        self.logger.info("optimizer_dir:%s" % self.optimizer_dir)

    # noinspection PyUnresolvedReferences
    def check_dependencies(self):
        try:
            import google.protobuf
            try:
                from google.protobuf.internal import enum_type_wrapper
            except ImportError:
                raise ImportError("Installed google.protobuf version is too old, "
                                  "you need at least 2.5.0")
        except ImportError:
            raise ImportError("Google protobuf cannot be imported. Are you sure "
                              "it's  installed?")
        try:
            import numpy
        except ImportError:
            raise ImportError("Numpy cannot be imported. Are you sure that it's"
                              " installed?")
        try:
            import scipy
        except ImportError:
            raise ImportError("Scipy cannot be imported. Are you sure that it's"
                              " installed?")

    def build_call(self, config, options, optimizer_dir):
        if not 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] = \
                os.path.join(config.get('SPEARMINT', 'path_to_optimizer'),
                             'spearmint') + os.pathsep
        else:
            os.environ['PYTHONPATH'] = \
                os.path.join(config.get('SPEARMINT', 'path_to_optimizer'),
                             'spearmint') + os.pathsep + os.environ['PYTHONPATH']
        print os.environ['PYTHONPATH']
        call = 'python ' + \
               os.path.join(config.get('SPEARMINT', 'path_to_optimizer'),
                            'spearmint', 'spearmint', 'main.py')
        call = ' '.join([call, os.path.join(optimizer_dir,
                                            config.get('SPEARMINT', 'config')),
                         '--driver=local',
                         '--max-concurrent',
                         config.get('HPOLIB', 'number_of_concurrent_jobs'),
                         '--max-finished-jobs',
                         config.get('SPEARMINT', 'max_finished_jobs'),
                         '--polling-time',
                         config.get('SPEARMINT', 'spearmint_polling_time'),
                         '--grid-size', config.get('SPEARMINT', 'grid_size'),
                         '--method', config.get('SPEARMINT', 'method'),
                         '--method-args=' + config.get('SPEARMINT', 'method_args'),
                         '--grid-seed', str(options.seed)])
        if config.get('SPEARMINT', 'method') != "GPEIChooser" and \
                config.get('SPEARMINT', 'method') != "GPEIOptChooser":
            self.logger.warning('WARNING: This chooser might not work yet\n')
            call = ' '.join([call, config.get("SPEARMINT", 'method_args')])
        return call

    # noinspection PyUnusedLocal
    def restore(self, config, optimizer_dir, **kwargs):
        raise NotImplementedError("Restoring is not possible for this optimizer")

    # setup directory where experiment will run
    def custom_setup(self, config, options, experiment_dir, optimizer_dir):

        optimizer_str = os.path.splitext(os.path.basename(__file__))[0]
        # Find experiment directory
        if options.restore:
            if not os.path.exists(options.restore):
                raise Exception("The restore directory does not exist")
            optimizer_dir = options.restore
        # Alter the python path
        if not 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] = os.path.dirname(__file__)
        else:
            os.environ['PYTHONPATH'] = os.path.dirname(__file__) + os.pathsep + \
                                       os.environ['PYTHONPATH']

        # Set up experiment directory
        if not os.path.exists(optimizer_dir):
            os.mkdir(optimizer_dir)
            # Make a link to the Protocol-Buffer config file
            configpb = config.get('SPEARMINT', 'config')
            if not os.path.exists(os.path.join(optimizer_dir, configpb)):
                os.symlink(os.path.join(experiment_dir, optimizer_str, configpb),
                           os.path.join(optimizer_dir, configpb))

        return optimizer_dir

    def manipulate_config(self, config):
        # special cases
        if not config.has_option('SPEARMINT', 'method'):
            raise Exception("SPEARMINT:method not specified in .cfg")
        if not config.has_option('SPEARMINT', 'method_args'):
            raise Exception("SPEARMINT:method-args not specified in .cfg")

        # GENERAL
        if not config.has_option('SPEARMINT', 'max_finished_jobs'):
            config.set('SPEARMINT', 'max_finished_jobs',
                       config.get('HPOLIB', 'number_of_jobs'))

        path_to_optimizer = config.get('SPEARMINT', 'path_to_optimizer')
        if not os.path.isabs(path_to_optimizer):
            path_to_optimizer = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_to_optimizer)

        path_to_optimizer = os.path.normpath(path_to_optimizer)
        if not os.path.exists(path_to_optimizer):
            self.logger.critical("Path to optimizer not found: %s" % path_to_optimizer)
            sys.exit(1)

        config.set('SPEARMINT', 'path_to_optimizer', path_to_optimizer)

        return config

# # noinspection PyUnusedLocal
# def main(config, options, experiment_dir, experiment_directory_prefix,
#          **kwargs):
#     # config:           Loaded .cfg file
#     # options:          Options containing seed, restore_dir,
#     # experiment_dir:   Experiment directory/Benchmark_directory
#     # **kwargs:         Nothing so far
#
#     time_string = HPOlib.wrapping_util.get_time_string()
#     optimizer_str = os.path.splitext(os.path.basename(__file__))[0]
#
#     optimizer_dir = os.path.join(experiment_dir,
#                                  experiment_directory_prefix +
#                                  optimizer_str + "_" +
#                                  str(options.seed) + "_" + time_string)
#
#     # setup directory where experiment will run
#     optimizer_dir = custom_setup(config, options, experiment_dir, experiment_directory_prefix, optimizer_dir)
#
#     # Build call
#     cmd = build_call(config, options, optimizer_dir)
#
#     self.logger.info("""
#     ### INFORMATION ############################################################
#     # You're running: %40s                 #
#     # Version:        %40s                 #
#     # A newer version might be available, but not yet built in.                #
#     # Please use this version only to reproduce our results on automl.org      #
#     ############################################################################
#     """ % (path_to_optimizer, version_info))
#
#     return cmd, optimizer_dir
