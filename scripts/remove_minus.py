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
import re
import sys

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

sys.stdout.write("NO WARRANTY THAT THIS METHOD WILL OPERATE WITHOUT ERROR "
                 "OR INTERRUPTION.")

print "Removing minus from %s" % sys.argv[1]
exp = cPickle.load(file(sys.argv[1]))

for t in exp['trials']:
    for p in t['params'].keys():
        new_key = re.sub('^-', '', p)
        t['params'][new_key] = t['params'][p]
        del t['params'][p]

fh = open(sys.argv[1], 'w')
cPickle.dump(exp, fh)
fh.close()