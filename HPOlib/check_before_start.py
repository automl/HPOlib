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
import imp
import logging
import os
import subprocess
import sys
import inspect

logger = logging.getLogger("HPOlib.check_before_start")


def _check_runsolver():
    # check whether runsolver is in path
    process = subprocess.Popen("which runsolver", stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, shell=True,
                               executable="/bin/bash")
    stdoutdata, _stderrdata = process.communicate()

    if stdoutdata is not None and "runsolver" in stdoutdata:
        pass
    else:
        raise Exception("Runsolver cannot not be found. "
                        "Are you sure that it's installed?\n"
                        "Your $PATH is: " + os.environ['PATH'])


def _check_modules():
    """Checks whether all dependencies are installed"""

    # [issue #110] bug in line: if numpy.__version__ < "1.6.0":
    # e.g. "1.11.0" < "1.6.0" (False)
    # --- FIX
    # Just replace "X.XX.X" --> distutils.version.LooseVersion("X.XX.X")
    from distutils.version import LooseVersion

    try:
        import numpy
        if LooseVersion(numpy.__version__) < LooseVersion("1.6.0"):
            logger.warning("WARNING: You are using a numpy %s < 1.6.0. This "
                           "might not work", numpy.__version__)
    except:
        raise ImportError("Numpy cannot be imported. Are you sure that it's installed?")

    try:
        import scipy
        if LooseVersion(scipy.__version__) < LooseVersion("0.12.0"):
            logger.warning("WARNING: You are using a scipy %s < 0.12.0. "
                           "This might not work", scipy.__version__)
    except:
        raise ImportError("Scipy cannot be imported. Are you sure that it's installed?")


def _check_config(experiment_dir):
    # check whether config file exists
    config_file = os.path.join(experiment_dir, "config.cfg")
    if not os.path.exists(config_file):
        logger.warn("There is no config.cfg in %s, all options need to be "
                    "provided as CLI arguments" % experiment_dir)


def check_optimizer(optimizer):
    # *parser.py files were integrated into optimizer class files but parser variable is still used to avoid too many
    # variable changes
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "optimizers", optimizer)
    if os.path.isdir(path):
        # User told us, e.g. "tpe"
        # Optimizer is in our optimizer directory
        # Now check how many versions are present
        parser = glob.glob(path + '*.py')
        logger.info("parser %s", parser)
        if len(parser) > 1:
            logger.critical("Sorry I don't know which optimizer to use: %s",
                            parser)
            sys.exit(1)
        version = parser[0][:-3]
    elif len(glob.glob(path + '.py')) == 1:
        parser = glob.glob(path + '.py')
        version = parser[0][:-3]
    elif len(glob.glob(path + '.py')) > 1:
        # Note this is a different case
        # User told us e.g. "tpe/hyperopt_august" but this was not specific enough
        logger.critical("Sorry I don't know which optimizer to use: %s",
                        glob.glob(path + '*.py'))
        sys.exit(1)
    else:
        logger.critical("We cannot find: %s", path)
        sys.exit(1)

    # Now check the other stuff
    if not os.path.exists(version + "Default.cfg"):
        logger.critical("Sorry I cannot find the default config for your "
                        "optimizer: %sDefault.cfg", version)
        sys.exit(1)
    if not os.path.exists(version + ".py"):
        logger.critical("Sorry I cannot find the script to call your "
                        "optimizer: %s.py", version)
        sys.exit(1)

    # dynamically load optimizer class which is abstracted by OptimizerAlgorithm
    imp.load_source("opt", version + ".py")
    import opt
    opt_class = [m[1] for m in inspect.getmembers(opt, inspect.isclass) if m[0] != "OptimizerAlgorithm"]
    obj = getattr(opt, opt_class[0].__name__)
    opt_obj = obj()
    opt_obj.check_dependencies()
    return version, opt_obj


# noinspection PyUnusedLocal
def check_first(experiment_dir):
    """ Do some checks before optimizer is loaded """
    logger.info("Check config.cfg..",)
    _check_config(experiment_dir)
    logger.info("..passed")
    logger.info("Check dependencies:")
    logger.info("Runsolver..")
    _check_runsolver()
    logger.info("..passed")
    logger.info("Check python_modules..")
    _check_modules()
    logger.info("..passed")