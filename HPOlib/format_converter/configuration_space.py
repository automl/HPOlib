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


class Hyperparameter(object):
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.name == other.name and self.domain == other.domain)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        repr_str = ["Name: %s" % self.name, ]
        repr_str.append(repr(self.domain))
        if self.has_conditions():
            repr_str.append("Conditions: %s" % (str(self.conditions)))
        else:
            repr_str.append("Conditions: None")
        return ", ".join(repr_str)

    def __str__(self):
        return self.__repr__()

    def has_conditions(self):
        return not len(self.conditions[0]) == 0


class Domain(object):
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)
    pass

    def __repr__(self):
        repr_str = ["Type: %s" % self.type, ]
        if self.type == "categorical":
            repr_str.append("Choices: %s" % str(self.choices))
        elif self.type in ("float", "int"):
            repr_str.append("Range: [%s, %s]" % (str(self.lower), str(self.upper)))
            repr_str.append("Base: %s" % self.base)
            repr_str.append("Q: %s" % self.q)

        return ", ".join(repr_str)

    def __str__(self):
        return self.__repr__()


def create_categorical(name, choices, conditions=None):
    domain = Domain()
    domain.type = "categorical"
    domain.choices = choices

    theta = Hyperparameter()
    theta.name = name
    theta.domain = domain
    if conditions is None:
        theta.conditions = [[]]
    else:
        # Hopefully this is a array of arrays
        if type(conditions) != list:
            raise ValueError("Conditions are not a list: %s" % str(conditions))
        for o_r in conditions:
            if type(o_r) != list:
                raise ValueError("This conditions are not a list: %s" % str(o_r))
        theta.conditions = conditions
    return theta


def create_float(name, lower, upper, q=None, base=None, conditions=None):
    if lower >= upper:
        raise ValueError("%f is more/equal than %f: %s" % (lower, upper, name))

    domain = Domain()
    domain.type = "float"
    domain.lower = float(lower)
    domain.upper = float(upper)
    domain.q = q
    domain.base = base

    theta = Hyperparameter()
    theta.name = name
    theta.domain = domain
    if conditions is None:
        theta.conditions = [[]]
    else:
        # Hopefully this is a array of arrays
        if type(conditions) != list:
            raise ValueError("Conditions are not a list: %s" % str(conditions))
        for o_r in conditions:
            if type(o_r) != list:
                raise ValueError("This conditions are not a list: %s" % str(o_r))
        theta.conditions = conditions
    return theta


def create_int(name, lower, upper, q=None, base=None, conditions=None):
    if lower >= upper:
        raise ValueError("%f is more/equal than %f: %s" % (lower, upper, name))
    if int(lower) != lower:
        raise ValueError("%f is no int: %s" % (lower, name))
    if int(upper) != upper:
        raise ValueError("%f is no int: %s" % (upper, name))
    domain = Domain()
    domain.type = "int"
    domain.lower = int(lower)
    domain.upper = int(upper)
    domain.q = q
    domain.base = base

    theta = Hyperparameter()
    theta.name = name
    theta.domain = domain
    if conditions is None:
        theta.conditions = [[]]
    else:
        # Hopefully this is a array of arrays
        if type(conditions) != list:
            raise ValueError("Conditions are not a list: %s" % str(conditions))
        for o_r in conditions:
            if type(o_r) != list:
                raise ValueError("This conditions are not a list: %s" % str(o_r))
        theta.conditions = conditions
    return theta
