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

#!/usr/bin/env python

import os
import subprocess
import sys
import imp
from config_parser.parse import parse_config

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


def _check_modules():
    try:
        import numpy
        if numpy.__version__ < "1.6.0":
            print "WARNING: You are using a numpy %s < 1.6.0. This might not work"\
                % numpy.__version__
    except:
        raise ImportError("Numpy cannot be imported. Are you sure that it's installed?")

    try:
        import scipy
        if scipy.__version__ < "0.12.0":
            print "WARNING: You are using a scipy %s < 0.12.0. This might not work"\
                % scipy.__version__
    except:
        raise ImportError("Scipy cannot be imported. Are you sure that it's installed?")
    #import bson
    import networkx
    import google.protobuf

    try:
        import theano
    except ImportError:
        print "Theano not found"

    if 'cuda' not in os.environ['PATH']:
        print "\tCUDA not in $PATH"
    # if 'cuda' not in os.environ['LD_LIBRARY_PATH']:
    #     print "CUDA not in $LD_LIBRARY_PATH"


def _check_config(experiment_dir):
    # check whether config file exists
    config_file = os.path.join(experiment_dir, "config.cfg")
    if not os.path.exists(config_file):
        raise Exception("There is no config.cfg in %s" % experiment_dir)


def _check_function(experiment_dir, optimizer_dir):
    # Check whether function exists
    config_file = os.path.join(experiment_dir, "config.cfg")
    config = parse_config(config_file, allow_no_value=True)

    fn = None
    if os.path.isabs(config.get("DEFAULT", "function")):
        fn_path = config.get("DEFAULT", "function")
        fn_name, ext = os.path.splitext(os.path.basename(fn_path))
        try:
            fn = imp.load_source(fn_name, fn_path)
        except (ImportError, IOError):
            print "Could not find algorithm in %s" % fn_path
            import traceback
            print traceback.format_exc()
    else:
        fn = config.get("DEFAULT", "function").replace("../..", "..", 1)
        fn_path = os.path.join(optimizer_dir, fn)
        fn_path_parent = os.path.join(optimizer_dir, "..", fn)
        fn_name, ext = os.path.splitext(os.path.basename(fn_path))
        try:
            fn = imp.load_source(fn_name, fn_path)
        except (ImportError, IOError) as e:
            print e
            try:
                fn = imp.load_source(fn_name, fn_path_parent)
            except IOError as e:
                print(("Could not find\n%s\n\tin\n%s\n\tor its parent directory " +
                       "\n%s")
                      % (fn_name, fn_path, fn_path_parent))
                import traceback
                print traceback.format_exc()
                sys.exit(1)
            except ImportError as e:
                import traceback
                print traceback.format_exc()
                sys.exit(1)
    del fn


def _check_zeroth(experiment_dir):
    print "\tconfig.cfg..",
    _check_config(experiment_dir)
    print "..passed"


def _check_first(experiment_dir):
    """ Do some checks before optimizer is loaded """
    main()


def _check_second(experiment_dir, optimizer_dir):
    """ Do remaining tests """
    print "\talgorithm..",
    _check_function(experiment_dir, optimizer_dir)
    print "..passed"


def main():
    print "Checking dependencies:"
    print "\tRunsolver..",
    _check_runsolver()
    print "..passed"
    print "\tpython_modules..",
    _check_modules()
    print "..passed"


if __name__ == "__main__":
    main()