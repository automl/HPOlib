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

import numpy as np

import HPOlib.wrapping_util


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


logger = logging.getLogger("HPOlib.spearmint_april2013_mod")


path_to_optimizer = "optimizers/spearmint_april2013_mod/"
version_info = ("# %76s #\n" %
                "https://github.com/JasperSnoek/spearmint/tree/613350f2f617de3af5f101b1dc5eccf60867f67e")


def check_dependencies():
    try:
        import google.protobuf
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


def build_spearmint_call(config, options, optimizer_dir):
    print
    call = 'python ' + os.path.join(config.get('SPEARMINT', 'path_to_optimizer'), 'spearmint_sync.py')
    call = ' '.join([call, optimizer_dir,
                    '--config', os.path.join(optimizer_dir, os.path.basename(config.get('SPEARMINT', 'config'))),
                    '--max-concurrent', config.get('HPOLIB', 'number_of_concurrent_jobs'),
                    '--max-finished-jobs', config.get('SPEARMINT', 'max_finished_jobs'),
                    '--polling-time', config.get('SPEARMINT', 'spearmint_polling_time'),
                    '--grid-size', config.get('SPEARMINT', 'grid_size'),
                    '--method',  config.get('SPEARMINT', 'method'),
                    '--method-args=' + config.get('SPEARMINT', 'method_args'),
                    '--grid-seed', str(options.seed)])
    if config.get('SPEARMINT', 'method') != "GPEIChooser" and \
            config.get('SPEARMINT', 'method') != "GPEIOptChooser":
        logger.warning('WARNING: This chooser might not work yet\n')
        call = ' '.join([call, config.get("SPEARMINT", 'method_args')])
    return call


#noinspection PyUnusedLocal
def restore(config, optimizer_dir, **kwargs):
    """
    Returns the number of restored runs. This is the number of different configs
    tested multiplied by the number of crossvalidation folds.
    """
    restore_file = os.path.join(optimizer_dir, "expt-grid.pkl")
    if not os.path.exists(restore_file):
        logger.error("Oups, this should have been checked before")
        raise Exception("%s does not exist" % (restore_file,))
    sys.path.append(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), path_to_optimizer))
    # We need the Grid because otherwise we cannot load the pickle file
    import ExperimentGrid
    # Assumes that all not valid states are marked as crashed
    fh = open(restore_file)
    exp_grid = cPickle.load(fh)
    fh.close()
    complete_runs = np.sum(exp_grid['status'] == 3)
    restored_runs = complete_runs * config.getint('HPOLIB', 'number_cv_folds')
    try:
        os.remove(os.path.join(optimizer_dir, "expt-grid.pkl.lock"))
    except OSError:
        pass
    return restored_runs


#noinspection PyUnusedLocal
def main(config, options, experiment_dir, **kwargs):
    # config:           Loaded .cfg file
    # options:          Options containing seed, restore_dir,
    # experiment_dir:   Experiment directory/Benchmark_directory
    # **kwargs:         Nothing so far

    time_string = HPOlib.wrapping_util.get_time_string()
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
    cmd = build_spearmint_call(config, options, optimizer_dir)

    # Set up experiment directory
    if not os.path.exists(optimizer_dir):
        os.mkdir(optimizer_dir)
        # Make a link to the Protocol-Buffer config file
        space = config.get('SPEARMINT', 'config')
        abs_space = os.path.abspath(space)
        parent_space = os.path.join(experiment_dir, optimizer_str, space)
        if os.path.exists(abs_space):
            space = abs_space
        elif os.path.exists(parent_space):
            space = parent_space
        else:
            raise Exception("Spearmint search space not found. Searched at %s and "
                            "%s" % (abs_space, parent_space))
        # Copy the hyperopt search space
        if not os.path.exists(os.path.join(optimizer_dir, os.path.basename(space))):
            os.symlink(os.path.join(experiment_dir, optimizer_str, space),
                       os.path.join(optimizer_dir, os.path.basename(space)))
    logger.info("### INFORMATION ################################################################")
    logger.info("# You're running %40s                      #" % path_to_optimizer)
    logger.info("%s" % version_info)
    logger.info("# A newer version might be available, but not yet built in.                    #")
    logger.info("# Please use this version only to reproduce our results on automl.org          #")
    logger.info("################################################################################")
    return cmd, optimizer_dir