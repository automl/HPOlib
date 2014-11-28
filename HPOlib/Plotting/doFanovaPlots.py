#!/usr/bin/env python

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

from argparse import ArgumentParser
import logging
import os
import sys
import time

from HPOlib.Plotting import plot_util
from HPOlib.wrapping_util import format_traceback

logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
hpolib_logger = logging.getLogger("HPOlib")
hpolib_logger.setLevel(logging.INFO)
logger = logging.getLogger("HPOlib.doFanovaPlots")

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

def main():

    prog = "python doFanovaPlots.py <oneOrMorePickles>"
    description = "Tries to run Fanova using pyfanova"
    parser = ArgumentParser(description=description, prog=prog)

    # General Options
    parser.add_argument("--savedir", dest="savedir", required=True,
                        default="", help="Where to save plots? (directory)")
    parser.add_argument("--pcsFile", dest="pcsfile", required=True,
                        default="", help="Path to a .pcs file")
    args, unknown = parser.parse_known_args()

    # First check whether we can import pyfanova
    try:
        from pyfanova.visualizer import Visualizer
        from pyfanova.fanova_from_hpolib import FanovaFromHPOLib
    except ImportError:
        logger.error("pyfanova is not installed, try 'pip install pyfanova'")

    # Check whether pcs exists
    if not os.path.exists(args.pcsfile):
        logger.error("%s does not exist" % args.pcsfile)

    # Check whether directory exists
    if not os.path.isdir(args.savedir):
        logger.error("%s does not exist" % args.savedir)
    save_dir = os.path.realpath(args.savedir)

    # Now run pyfanova
    logger.info("Found " + str(len(unknown)) + " arguments\n")

    logger.info("Starting pyfanova .. this might take a bit")

    f = FanovaFromHPOLib(args.pcsfile, unknown)
    vis = Visualizer(f)
    vis.create_all_plots(save_dir)

    logger.info("Successfully created plots")
    return

if __name__ == "__main__":
    main()