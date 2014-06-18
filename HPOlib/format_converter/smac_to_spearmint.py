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


"""
This method reads a SMAC search space and returns a spearmint searchspace
Default value and conditions are lost
"""

import sys
import os

from spearmint_april2013_mod_spearmint_pb2 import Experiment, PYTHON
from google.protobuf import text_format
import numpy as np


def convert_smac_to_spearmint(filename):
    fl_handle = open(filename)
    
    params = []
    for line in fl_handle:
        line = line.strip()
        if line.strip().find('#') == 0 or line == "":
            continue
        # Spearmint does not support conditionals, so ignore them
        if '|' in line:
            continue
        if '{' not in line and '[' not in line:
            # Whatever this is, we can't use it
            continue
        
        # We use an Experiment to store the variables
        param = Experiment.ParameterSpec()
        # We only allow variables of size 1
        param.size = 1
        pos = line.find("[")
        tmp_pos = line.find("{")
        if pos > -1 and tmp_pos > -1 and tmp_pos < pos:
            pos = tmp_pos
        param.name = line[0:pos].strip()
        line = line[pos:].strip()
        if '{' in line:
            # Found an ENUM
            param.type = Experiment.ParameterSpec.ENUM
            pos = line.find('}')
            for option in line[1:pos].split(','):
                param.options.append(str(option.strip().strip("'")))
        elif "]i " in line:
            # Found an INT
            param.type = Experiment.ParameterSpec.INT
            pos = line.find(']')
            lower, upper = line[1:pos].split(',')
            param.min = float(int(lower.strip("'")))
            param.max = float(int(upper.strip("'")))
        elif "]l " in line:
            # Found a LOG float
            param.type = Experiment.ParameterSpec.FLOAT
            pos = line.find(']')
            lower, upper = line[1:pos].split(',')
            param.name = "LOG10_" + param.name
            param.min = np.log10(float(lower.strip("'")))
            param.max = np.log10(float(upper.strip("'")))
        elif "]il" in line or "]li" in line:
            # Found a LOG INT
            param.type = Experiment.ParameterSpec.INT
            pos = line.find(']')
            lower, upper = line[1:pos].split(',')
            param.name = "Q1_LOG10_" + param.name
            param.min = np.log10(float(lower.strip("'")))
            param.max = np.log10(float(upper.strip("'")))
        else:
            # Must be a FLOAT
            param.type = Experiment.ParameterSpec.FLOAT
            pos = line.find(']')
            lower, upper = line[1:pos].split(',')
            param.min = float(lower.strip())
            param.max = float(upper.strip())

        params.append(param)

    exp = Experiment()
    # Assumption: Algo is written in Python called cv.py, only works for BBoM
    exp.language = PYTHON
    exp.name = "HPOlib.cv"
    exp.variable.extend(params)
    return text_format.MessageToString(exp)


