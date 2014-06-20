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

from google.protobuf import text_format
import numpy as np

from spearmint_april2013_mod_spearmint_pb2 import Experiment, PYTHON

import configuration_space


def read(pb_string):
    searchspace = dict()
    if type(pb_string) == file:
        pb_string = pb_string.read()

    if not isinstance(pb_string, str):
        raise ValueError("Input is not a string or a file")

    pb_space = Experiment()
    text_format.Merge(pb_string, pb_space)
    for para in pb_space.variable:
        name = str(para.name)
        size = para.size
        if size != 1:
            raise NotImplementedError("%s has size %s, we only convert with size 1" % (name, size))
        if para.type == Experiment.ParameterSpec.ENUM:
            choices = list()
            for ch in para.options:
                choices.append(str(ch))
            searchspace[name] = configuration_space.create_categorical(name=name, choices=choices, conditions=None)
        elif para.type == Experiment.ParameterSpec.FLOAT:
            searchspace[name] = configuration_space.create_float(name=name, lower=para.min, upper=para.max,
                                                                 base=None, q=None, conditions=None)
        elif para.type == Experiment.ParameterSpec.INT:
            searchspace[name] = configuration_space.create_int(name=name, lower=para.min, upper=para.max,
                                                               base=None, q=None, conditions=None)
        else:
            raise NotImplementedError("Don't know that type: %s (%s)" % (para.type, para.name))
    return searchspace


def write(searchspace):
    exp = Experiment()
    # Assumption: Algo is written in Python called cv.py, only works for HPOlib
    exp.language = PYTHON
    exp.name = "HPOlib.cv"

    tmp_para_list = list()
    for parakey in searchspace:
        param = searchspace[parakey]
        # We use an Experiment to store the variables
        constructed_param = Experiment.ParameterSpec()
        constructed_param.size = 1
        constructed_param.name = param.name
        if param.domain.type == "categorical":
            constructed_param.type = Experiment.ParameterSpec.ENUM
            for choice in param.domain.choices:
                constructed_param.options.append(choice)
        else:
            if param.domain.type == "int":
                constructed_param.type = Experiment.ParameterSpec.INT
                constructed_param.min = param.domain.lower
                constructed_param.max = param.domain.upper
            elif param.domain.type == "float":
                constructed_param.type = Experiment.ParameterSpec.FLOAT
                constructed_param.min = param.domain.lower
                constructed_param.max = param.domain.upper
            else:
                raise NotImplementedError("Unknown type: %s (%s)" % (param.domain.type, param.name))

            # Handle LOG params
            if param.domain.base is not None:
                if int(param.domain.base) != param.domain.base:
                    raise NotImplementedError("We cannot handle non-int bases: %s (%s)" %
                                              (str(param.domain.base), param.name))
                constructed_param.name = "LOG%d_%s" % (int(param.domain.base), constructed_param.name)
                constructed_param.min = np.log10(constructed_param.min) / np.log10(param.domain.base)
                constructed_param.max = np.log10(constructed_param.max) / np.log10(param.domain.base)

            # Handle q params
            if param.domain.q is not None:
                constructed_param.name = "Q%d_%s" % (int(param.domain.q), constructed_param.name)

            # Do NOT handle conditions:
            if param.has_conditions():
                print "WARNING: We loose condition: %s" % str(param.conditions)
        tmp_para_list.append(constructed_param)
    exp.variable.extend(tmp_para_list)

    result = text_format.MessageToString(exp)
    return result


if __name__ == "__main__":
    fh = open(sys.argv[1])
    sp = read(fh)
    fh.close()
    print "==== READ Result"
    print "\n".join(["%s %s" % (i, sp[i]) for i in sp])

    print "==== WRITE Result"
    created_sp = write(sp)
    print created_sp