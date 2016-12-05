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

import cPickle
import logging
import os
import sys

import HPOlib.wrapping_util as wrapping_util
from HPOlib.optimizer_algorithm import OptimizerAlgorithm


version_info = ("# %76s #" % "https://github.com/hyperopt/hyperopt/tree/486aebec8a4170e4781d99bbd6cca09123b12717")
__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


class TPE(OptimizerAlgorithm):

    def __init__(self):
        self.optimizer_name = 'TPE'
        self.optimizer_dir = os.path.abspath("./hyperopt_august2013_mod")
        self.logger = logging.getLogger("HPOlib.optimizers.tpe.hyperopt_august2013_mod")
        self.logger.info("optimizer_name:%s" % self.optimizer_name)
        self.logger.info("optimizer_dir:%s" % self.optimizer_dir)

    # noinspection PyUnresolvedReferences
    def check_dependencies(self):
        try:
            import nose
            self.logger.debug("\tNose: %s\n" % str(nose.__version__))
        except ImportError:
            raise ImportError("Nose cannot be imported. Are you sure it's "
                              "installed?")
        try:
            import networkx
            self.logger.debug("\tnetworkx: %s\n" % str(networkx.__version__))
        except ImportError:
            raise ImportError("Networkx cannot be imported. Are you sure it's "
                              "installed?")
        try:
            import pymongo
            self.logger.debug("\tpymongo: %s\n" % str(pymongo.version))
            from bson.objectid import ObjectId
        except ImportError:
            raise ImportError("Pymongo cannot be imported. Are you sure it's"
                              " installed?")
        try:
            import numpy
            self.logger.debug("\tnumpy: %s" % str(numpy.__version__))
        except ImportError:
            raise ImportError("Numpy cannot be imported. Are you sure that it's"
                              " installed?")
        try:
            import scipy
            self.logger.debug("\tscipy: %s" % str(scipy.__version__))
        except ImportError:
            raise ImportError("Scipy cannot be imported. Are you sure that it's"
                              " installed?")

    def build_call(self, config, options, optimizer_dir):
        # For TPE we have to cd to the exp_dir
        call = "python " + os.path.dirname(os.path.realpath(__file__)) + \
               "/tpecall.py"
        call = ' '.join([call, '-p', os.path.join(optimizer_dir, os.path.basename(config.get('TPE', 'space'))),
                         "-m", config.get('TPE', 'number_evals'),
                         "-s", str(options.seed),
                         "--cwd", optimizer_dir])
        if options.restore:
            call = ' '.join([call, '-r'])
        return call

    # noinspection PyUnusedLocal
    def restore(self, config, optimizer_dir, **kwargs):
        """
        Returns the number of restored runs. This is the number of different configs
        tested multiplied by the number of crossvalidation folds.
        """
        restore_file = os.path.join(optimizer_dir, 'state.pkl')
        if not os.path.exists(restore_file):
            print "Oups, this should have been checked before"
            raise Exception("%s does not exist" % (restore_file,))

        # Special settings for restoring
        with open(restore_file) as fh:
            state = cPickle.load(fh)
        complete_runs = 0
        # noinspection PyProtectedMember
        tpe_trials = state['trials']._trials
        for trial in tpe_trials:
            # Assumes that all not valid states states are marked crashed
            if trial['state'] == 2:
                complete_runs += 1
        restored_runs = complete_runs * config.getint('HPOLIB', 'number_cv_folds')
        return restored_runs

    # setup directory where experiment will run
    def custom_setup(self, config, options, experiment_dir, optimizer_dir):
        # Add path_to_optimizer to PYTHONPATH and to sys.path
        # Only for HYPEROPT
        if not 'PYTHONPATH' in os.environ:
            os.environ['PYTHONPATH'] = config.get('TPE', 'path_to_optimizer')
        else:
            os.environ['PYTHONPATH'] = config.get('TPE', 'path_to_optimizer') + os.pathsep + os.environ['PYTHONPATH']
        sys.path.append(config.get('TPE', 'path_to_optimizer'))

        optimizer_str = os.path.splitext(os.path.basename(__file__))[0]

        if options.restore:
            if not os.path.exists(options.restore):
                raise Exception("The restore directory does not exist")
            optimizer_dir = options.restore

        # Set up experiment directory
        if not os.path.exists(optimizer_dir):
            os.mkdir(optimizer_dir)
            space = config.get('TPE', 'space')
            abs_space = os.path.abspath(space)
            parent_space = os.path.join(experiment_dir, optimizer_str, space)
            if os.path.exists(abs_space):
                space = abs_space
            elif os.path.exists(parent_space):
                space = parent_space
            else:
                raise Exception("TPE search space not found. Searched at %s and "
                                "%s" % (abs_space, parent_space))
            # Copy the hyperopt search space
            if not os.path.exists(os.path.join(optimizer_dir, os.path.basename(space))):
                os.symlink(os.path.join(experiment_dir, optimizer_str, space),
                           os.path.join(optimizer_dir, os.path.basename(space)))

        import hyperopt
        path_to_loaded_optimizer = os.path.abspath(os.path.dirname(os.path.dirname(hyperopt.__file__)))

        return optimizer_dir

    def manipulate_config(self, config):
        if not config.has_section('TPE'):
            config.add_section('TPE')

        # optional cases
        if not config.has_option('TPE', 'space'):
            raise Exception("TPE:space not specified in .cfg")

        number_of_jobs = config.getint('HPOLIB', 'number_of_jobs')
        if not config.has_option('TPE', 'number_evals'):
            config.set('TPE', 'number_evals', config.get('HPOLIB', 'number_of_jobs'))
        elif config.getint('TPE', 'number_evals') != number_of_jobs:
            self.logger.warning("Found a total_num_runs_limit (%d) which differs from "
                           "the one read from the config (%d). This can e.g. "
                           "happen when restoring a TPE run" %
                           (config.getint('TPE', 'number_evals'),
                            number_of_jobs))
            config.set('TPE', 'number_evals', str(number_of_jobs))

        path_to_optimizer = config.get('TPE', 'path_to_optimizer')
        if not os.path.isabs(path_to_optimizer):
            path_to_optimizer = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_to_optimizer)

        path_to_optimizer = os.path.normpath(path_to_optimizer)
        if not os.path.exists(path_to_optimizer):
            self.logger.critical("Path to optimizer not found: %s" % path_to_optimizer)
            sys.exit(1)

        config.set('TPE', 'path_to_optimizer', path_to_optimizer)

        return config
