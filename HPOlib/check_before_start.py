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
import subprocess
import sys

from HPOlib.wrapping_util import get_configuration


logger = logging.getLogger("HPOlib.check_before_start")


"""This script checks whether all dependencies are installed"""


def _check_runsolver():
    # check whether runsolver is in path
    process = subprocess.Popen("which runsolver", stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, shell=True,
                               executable="/bin/bash")
    stdoutdata, stderrdata = process.communicate()
    if stdoutdata is not None and "runsolver" in stdoutdata:
        pass
    else:
        raise Exception("Runsolver cannot not be found. Are you sure that it's installed?\n"
                        "Your $PATH is: " + os.environ['PATH'])


# noinspection PyUnresolvedReferences
def _check_modules():
    try:
        import numpy
        if numpy.__version__ < "1.6.0":
            logger.warning("WARNING: You are using a numpy %s < 1.6.0. This "
                           "might not work" % numpy.__version__)
    except:
        raise ImportError("Numpy cannot be imported. Are you sure that it's installed?")

    try:
        import scipy
        if scipy.__version__ < "0.12.0":
            logger.warning("WARNING: You are using a scipy %s < 0.12.0. "
                           "This might not work" % scipy.__version__)
    except:
        raise ImportError("Scipy cannot be imported. Are you sure that it's installed?")

    import networkx
    import google.protobuf

    try:
        import theano
    except ImportError:
        logger.warning("Theano not found. You might need this to run some "
                       "more complex benchmarks!")

    try:
        import pymongo
        from bson.objectid import ObjectId
    except ImportError:
        raise ImportError("Pymongo cannot be imported. Are you sure it's installed?")

    if 'cuda' not in os.environ['PATH']:
        logger.warning("CUDA not in $PATH")


def _check_config(experiment_dir):
    # check whether config file exists
    config_file = os.path.join(experiment_dir, "config.cfg")
    if not os.path.exists(config_file):
        raise Exception("There is no config.cfg in %s" % experiment_dir)


def _check_function(experiment_dir):
    # Check whether path function exists
    config_file = os.path.join(experiment_dir, "config.cfg")
    config = get_configuration(experiment_dir, None, None)

    path = config.get("HPOLIB", "function")

    if not os.path.isabs(path):
        path = os.path.join(experiment_dir, path)

    if not os.path.exists(path):
        raise ValueError("Function: %s does not exist" % path)
    return


def check_optimizer(optimizer):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "optimizers", optimizer)
    if os.path.isdir(path):
        # User told us, e.g. "tpe"
        # Optimizer is in our optimizer directory
        # Now check how many versions are present
        parser = glob.glob(os.path.join(path, '*_parser.py'))
        if len(parser) > 1:
            logger.critical("Sorry I don't know which optimizer to use: %s" % parser)
            sys.exit(1)
        version = parser[0][:-10]
    elif len(glob.glob(path + '*_parser.py')) == 1:
        parser = glob.glob(path + '*_parser.py')
        version = parser[0][:-10]
    elif len(glob.glob(path + '*_parser.py')) > 1:
        # Note this is a different case
        # User told us e.g. "tpe/hyperopt_august" but this was not specific enough
        logger.critical("Sorry I don't know which optimizer to use: %s" % glob.glob(path + '*_parser.py'))
        sys.exit(1)
    else:
        logger.critical("We cannot find: %s" % path)
        sys.exit(1)

    # Now check the other stuff
    if not os.path.exists(version + "Default.cfg"):
        logger.critical("Sorry I cannot find the default config for your optimizer: %sDefault.cfg" % version)
        sys.exit(1)
    if not os.path.exists(version + ".py"):
        logger.critical("Sorry I cannot find the script to call your optimizer: %s.py" % version)
        sys.exit(1)
    else:
        return version


def check_zeroth(experiment_dir):
    logger.info("Check config.cfg..",)
    _check_config(experiment_dir)
    logger.info("..passed")


# noinspection PyUnusedLocal
def check_first(experiment_dir):
    """ Do some checks before optimizer is loaded """
    main()


def check_second(experiment_dir):
    """ Do remaining tests """
    logger.info("Check algorithm..",)
    _check_function(experiment_dir)
    logger.info("..passed")


def main():
    logger.info("Check dependencies:")
    logger.info("Runsolver..")
    _check_runsolver()
    logger.info("..passed")
    logger.info("Check python_modules..")
    _check_modules()
    logger.info("..passed")


if __name__ == "__main__":
    main()