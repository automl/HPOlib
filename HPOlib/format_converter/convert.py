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
import os
from string import upper
import sys
import tempfile

import smac_to_spearmint
import tpe_to_smac


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def smac_to_spearmint_helper(space, save=""):
    # print "Convert %s from SMAC to SPEARMINT" % space
    return smac_to_spearmint.convert_smac_to_spearmint(space)


def smac_to_tpe_helper(space, save=""):
    print "This is not yet implemented"


def spearmint_to_smac_helper(space, save=""):
    print "This is not yet implemented"


def spearmint_to_tpe_helper(space, save=""):
    print "This is not yet implemented"


def tpe_to_spearmint_helper(space, save=""):
    try:
        import hyperopt
    except ImportError:
        print "Cannot find hyperopt. To use this converter, modify $PYTHONPATH to contain a hyperopt installation"

    # First convert to smac
    tmp = tpe_to_smac.convert_tpe_to_smac_from_file(space)
    handle, tmp_file_name = tempfile.mkstemp()
    fh = open(tmp_file_name, 'w')
    fh.write(tmp)
    fh.close()

    # From smac convert to spearmint
    new_space = smac_to_spearmint.convert_smac_to_spearmint(tmp_file_name)

    os.remove(tmp_file_name)
    return new_space


def tpe_to_smac_helper(space, save=""):
    try:
        import hyperopt
    except ImportError:
        print "Cannot find hyperopt. To use this converter, modify $PYTHONPATH to contain a hyperopt installation"
    return tpe_to_smac.convert_tpe_to_smac_from_file(space)


def main():
    # python convert.py --from SMAC --to TPE -f space.any -s space.else
    prog = "python convert.py"
    description = "Automatically convert a searchspace from one format to another"

    parser = ArgumentParser(description=description, prog=prog)

    parser.add_argument("--from", dest="conv_from", choices=['SMAC', 'Smac', 'smac',
                                                             'TPE', 'Tpe', 'tpe', 'hyperopt',
                                                             'SPEARMINT', 'Spearmint', 'spearmint'],
                        default="", help="Convert from which format?", required=True)
    parser.add_argument("--to", dest="conv_to", choices=['SMAC', 'Smac', 'smac',
                                                         'TPE', 'Tpe', 'tpe', 'hyperopt',
                                                         'SPEARMINT', 'Spearmint', 'spearmint'],
                        default="", help="Convert to which format?", required=True)
    parser.add_argument("-f", "--file", dest="space",
                        default="", help="Where is the searchspace to be converted?", required=True)
    parser.add_argument("-s", "--save", dest="save",
                        default="", help="Where to save the new searchspace?")

    args, unknown = parser.parse_known_args()

    space = os.path.abspath(args.space)
    if not os.path.isfile(space):
        print "%s is not a valid path" % space
        sys.exit(1)

    # Unifying strings
    args.conv_to = upper(args.conv_to)
    args.conv_from = upper(args.conv_from)
    if args.conv_from == "HYPEROPT":
        args.conv_from = "TPE"
    if args.conv_to == "HYPEROPT":
        args.conv_to == "TPE"

    if args.conv_to == args.conv_from:
        print "Converting from %s to %s makes no sense" % (args.conv_to, args.conv_from)

    # This is like a switch statement
    options = {'SMAC': {'SPEARMINT': smac_to_spearmint_helper,
                         'TPE': smac_to_tpe_helper},
               'SPEARMINT': {'SMAC': spearmint_to_smac_helper,
                             'TPE': spearmint_to_tpe_helper},
               'TPE': {'SPEARMINT': tpe_to_spearmint_helper,
                       'SMAC': tpe_to_smac_helper}
               }
    new_space = options[args.conv_from][args.conv_to](space, args.save)
    if args.save != "":
        fh = open(args.save, 'w')
        fh.write(new_space)
        fh.close()
    else:
        print new_space

if __name__ == "__main__":
    main()