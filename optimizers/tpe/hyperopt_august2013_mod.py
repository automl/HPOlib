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

logger = logging.getLogger("HPOlib.optimizers.tpe.hyperopt_august2013_mod")

version_info = ("# %76s #" % "https://github.com/hyperopt/hyperopt/tree/486aebec8a4170e4781d99bbd6cca09123b12717")
__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


# noinspection PyUnresolvedReferences
def check_dependencies():
    try:
        import nose
        logger.debug("\tNose: %s\n" % str(nose.__version__))
    except ImportError:
        raise ImportError("Nose cannot be imported. Are you sure it's "
                          "installed?")
    try:
        import networkx
        logger.debug("\tnetworkx: %s\n" % str(networkx.__version__))
    except ImportError:
        raise ImportError("Networkx cannot be imported. Are you sure it's "
                          "installed?")
    try:
        import pymongo
        logger.debug("\tpymongo: %s\n" % str(pymongo.version))
        from bson.objectid import ObjectId
    except ImportError:
        raise ImportError("Pymongo cannot be imported. Are you sure it's"
                          " installed?")
    try:
        import numpy
        logger.debug("\tnumpy: %s" % str(numpy.__version__))
    except ImportError:
        raise ImportError("Numpy cannot be imported. Are you sure that it's"
                          " installed?")
    try:
        import scipy
        logger.debug("\tscipy: %s" % str(scipy.__version__))
    except ImportError:
        raise ImportError("Scipy cannot be imported. Are you sure that it's"
                          " installed?")


def build_tpe_call(config, options, optimizer_dir):
    # For TPE we have to cd to the exp_dir
    call = "python " + os.path.dirname(os.path.realpath(__file__)) + \
           "/tpecall.py"
    call = ' '.join([call, '-p', config.get('TPE', 'space'),
                     "-m", config.get('TPE', 'number_evals'),
                     "-s", str(options.seed),
                     "--cwd", optimizer_dir])
    if options.restore:
        call = ' '.join([call, '-r'])
    return call


#noinspection PyUnusedLocal
def restore(config, optimizer_dir, **kwargs):
    """
    Returns the number of restored runs. This is the number of different configs
    tested multiplied by the number of crossvalidation folds.
    """
    restore_file = os.path.join(optimizer_dir, 'state.pkl')
    if not os.path.exists(restore_file):
        print "Oups, this should have been checked before"
        raise Exception("%s does not exist" % (restore_file,))

    # Special settings for restoring
    fh = open(restore_file)
    state = cPickle.load(fh)
    fh.close()
    complete_runs = 0
    #noinspection PyProtectedMember
    tpe_trials = state['trials']._trials
    for trial in tpe_trials:
        # Assumes that all not valid states states are marked crashed
        if trial['state'] == 2:
            complete_runs += 1
    restored_runs = complete_runs * config.getint('HPOLIB', 'number_cv_folds')
    return restored_runs


#noinspection PyUnusedLocal
def main(config, options, experiment_dir, **kwargs):
    # config:           Loaded .cfg file
    # options:          Options containing seed, restore_dir, 
    # experiment_dir:   Experiment directory/Benchmark_directory
    # **kwargs:         Nothing so far
    time_string = wrapping_util.get_time_string()
    cmd = ""

    # Add path_to_optimizer to PYTHONPATH and to sys.path
    # Only for HYPEROPT
    if not 'PYTHONPATH' in os.environ:
        os.environ['PYTHONPATH'] = config.get('TPE', 'path_to_optimizer')
    else:
        os.environ['PYTHONPATH'] = config.get('TPE', 'path_to_optimizer') + os.pathsep + os.environ['PYTHONPATH']
    sys.path.append(config.get('TPE', 'path_to_optimizer'))

    optimizer_str = os.path.splitext(os.path.basename(__file__))[0]

# TODO: Check whether we might need this again
#    SYSTEM_WIDE = 0
#    AUGUST_2013_MOD = 1
#    try:
#        import hyperopt
#        version = SYSTEM_WIDE
#    except ImportError:
#        try:
#            cmd += "export PYTHONPATH=$PYTHONPATH:" + os.path.dirname(os.path.abspath(__file__)) + \
#                "/optimizers/hyperopt_august2013_mod\n"
#            import optimizers.hyperopt_august2013_mod.hyperopt as hyperopt
#        except ImportError, e:
#            import HPOlib.optimizers.hyperopt_august2013_mod.hyperopt as hyperopt
#        version = AUGUST_2013_MOD

    # Find experiment directory
    if options.restore:
        if not os.path.exists(options.restore):
            raise Exception("The restore directory does not exist")
        optimizer_dir = options.restore
    else:
        optimizer_dir = os.path.join(experiment_dir, optimizer_str + "_" +
                                     str(options.seed) + "_" +
                                     time_string)

    # Build call
    cmd += build_tpe_call(config, options, optimizer_dir)

    # Set up experiment directory
    if not os.path.exists(optimizer_dir):
        os.mkdir(optimizer_dir)
        space = config.get('TPE', 'space')
        # Copy the hyperopt search space
        if not os.path.exists(os.path.join(optimizer_dir, space)):
            os.symlink(os.path.join(experiment_dir, optimizer_str, space),
                       os.path.join(optimizer_dir, space))

    import hyperopt
    path_to_loaded_optimizer = os.path.abspath(os.path.dirname(os.path.dirname(hyperopt.__file__)))

    logger.info("### INFORMATION ################################################################")
    logger.info("# You are running:                                                             #")
    logger.info("# %76s #" % path_to_loaded_optimizer)
    if not os.path.samefile(path_to_loaded_optimizer, config.get('TPE', 'path_to_optimizer')):
        logger.warning("# BUT hyperopt_august2013_modDefault.cfg says:")
        logger.warning("# %76s #" % config.get('TPE', 'path_to_optimizer'))
        logger.warning("# Found a global hyperopt version. This installation will be used!             #")
    else:
        logger.info("# To reproduce our results you need version 0.0.3.dev, which can be found here:#")
        logger.info("%s" % version_info)
        logger.info("# A newer version might be available, but not yet built in.                    #")
    logger.info("################################################################################")
    return cmd, optimizer_dir
