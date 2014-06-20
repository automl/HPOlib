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

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

import sys

import numpy as np
import pyparsing

import configuration_space


def build_categorical(param):
    cat_template = "%s {%s} [%s]"
    return cat_template % (param.name, ", ".join(param.domain.choices), param.domain.choices[0])


def build_continuous(param):
    float_template = "%s [%s, %s] [%s]"
    int_template = "%s [%d, %d] [%d]i"
    default = None
    if param.domain.base is not None:
        if param.domain.base == 10:
            # SMAC can naturally handle this
            float_template += "l"
            int_template += "l"
            if param.domain.lower < 0:
                raise ValueError("This is no log domain: %s" % param.name)
            if param.domain.upper < 0:
                raise ValueError("This is no log domain: %s" % param.name)
        else:
            if int(param.domain.base) != param.domain.base:
                raise NotImplementedError("We cannot yet handle non-int bases: %s (%s)" %
                                          (str(param.domain.base), param.name))
            # HPOlib has to take care of this
            param.name = "LOG%d_%s" % (int(param.domain.base), param.name)
            param.domain.lower = np.log10(param.domain.lower) / np.log10(param.domain.base)
            param.domain.upper = np.log10(param.domain.upper) / np.log10(param.domain.base)

    if param.domain.q is not None:
        param.name = "Q%d_%s" % (int(param.domain.q), param.name)

    default = (param.domain.upper + param.domain.lower)/2
    if param.domain.upper >= default <= param.domain.lower:
        raise NotImplementedError("Cannot find mean for %s" % param.name)
    if param.domain.type == "int":
        param.domain.lower = int(param.domain.lower)
        param.domain.upper = int(param.domain.upper)
        default = int(default)
        return int_template % (param.name, param.domain.lower, param.domain.upper, default)
    else:
        return float_template % (param.name, str(param.domain.lower), str(param.domain.upper), str(default))


def build_condition(name, condition):
    condition_template = "%s | %s in {%s}"
    condition = condition.split(" ")
    if condition[1] != "==":
        raise NotImplementedError("SMAC cannot handle >< conditions: %s (%s)" % (condition, name))
    return condition_template % (name, condition[0], condition[2][1:-1].replace(",", ", "))


def write_smac(searchspace):
    lines = list()
    for para_name in searchspace:
        if searchspace[para_name].domain.type in ("int", "float"):
            lines.append(build_continuous(searchspace[para_name]))
        elif searchspace[para_name].domain.type == "categorical":
            lines.append(build_categorical(searchspace[para_name]))
        else:
            raise NotImplementedError("Unknown type: %s (%s)" %(searchspace[para_name].type, para_name))
        for condition in searchspace[para_name].conditions:
            if len(condition) > 1:
                raise NotImplementedError("SMAC cannot handle OR conditions on different parents: %s (%s)" % (condition, para_name))
            condition = condition[0]
            lines.append(build_condition(para_name, condition))
    return lines

if __name__ == "__main__":
    import read_smac
    fh = open(sys.argv[1])
    orig_pcs = fh.readlines()
    sp = read_smac.read_smac(orig_pcs)
    created_pcs = write_smac(sp)
    print "============== Writing Results"
    print "#Lines: ", len(created_pcs)
    print "#LostLines: ", len(orig_pcs) - len(created_pcs)
    diff = ["%s\n" % i for i in created_pcs if i not in " ".join(orig_pcs)]
    print "Identical Lines: ", len(created_pcs) - len(diff)
    print 
    print "Up to 10 random different lines (of %d):" % len(diff)
    print "".join(diff[:10])

