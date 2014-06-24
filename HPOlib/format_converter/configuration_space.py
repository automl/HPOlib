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

from collections import deque, OrderedDict

import networkx as nx


class Hyperparameter(object):
    def __init__(self):
        raise ValueError("Class %s is not supposed to be instantiated" % self
                         .__class__)

    def __ne__(self, other):
        return not self.__eq__(other)
    pass

    def __str__(self):
        return self.__repr__()

    def check_conditions(self):
        if self.conditions is None:
            self.conditions = [[]]
        # Hopefully this is a array of arrays
        if type(self.conditions) != list:
            raise ValueError("Conditions are not a list: %s" % str(self.conditions))
        for o_r in self.conditions:
            if type(o_r) != list:
                raise ValueError("This conditions are not a list: %s" % str(o_r))

    def has_conditions(self):
        return len(self.conditions[0]) > 0

    def get_conditions_as_string(self):
        if self.has_conditions():
            return "Conditions: %s" % (str(self.conditions))
        else:
            return "Conditions: None"


class NumericalHyperparameter(Hyperparameter):
    pass


class FloatHyperparameter(NumericalHyperparameter):
    pass


class IntegerHyperparameter(NumericalHyperparameter):
    def check_int(self, parameter, name):
        if abs(int(parameter) - parameter) > 0.00000001 and \
                type(parameter) is not int:
            raise ValueError("For an Integer parameter, the quantization "
                 "value %s must be an Integer, too. Right now it "
                 "is a %s with value %s" %
                 (name, type(parameter), str(parameter)))
        return int(parameter)


class CategoricalHyperparameter(Hyperparameter):
    def __init__(self, name, choices, conditions=None):
        self.name = name
        self.choices = choices
        self.conditions = conditions
        self.check_conditions()

    def __repr__(self):
        repr_str = ["Name: %s" % self.name]
        repr_str.append("Type: Categorical")
        repr_str.append("Choices: %s" % str(self.choices))
        repr_str.append(self.get_conditions_as_string())
        return ", ".join(repr_str)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if len(self.choices) != len(other.choices):
                return False
            else:
                for i in range(len(self.choices)):
                    if self.choices[i] != other.choices[i]:
                        return False
            return True
        else:
            return False


class UniformFloatHyperparameter(FloatHyperparameter):
    def __init__(self, name, lower, upper, q=None, base=None, conditions=None):
        self.name = name
        self.lower = float(lower)
        self.upper = float(upper)
        self.q = float(q) if q is not None else None
        self.base = float(base) if base is not None else None
        self.conditions = conditions
        self.check_conditions()

    def __repr__(self):
        repr_str = ["Name: %s" % self.name]
        repr_str.append("Type: UniformFloat")
        repr_str.append("Range: [%s, %s]" % (str(self.lower), str(self.upper)))
        repr_str.append("Base: %s" % self.base)
        repr_str.append("Q: %s" % self.q)
        repr_str.append(self.get_conditions_as_string())
        return ", ".join(repr_str)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([abs(self.lower - other.lower) < 0.00000001,
                        abs(self.upper - other.upper) < 0.00000001,
                        self.base is None and other.base is None or
                        self.base is not None and other.base is not None and
                        abs(self.base - other.base) < 0.00000001,
                        self.q is None and other.q is None or
                        self.q is not None and other.q is not None and
                        abs(self.q - other.q) < 0.00000001])
        else:
            return False


class NormalFloatHyperparameter(FloatHyperparameter):
    def __init__(self, name, mu, sigma, q=None, base=None, conditions=None):
        self.name = name
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.q = float(q) if q is not None else None
        self.base = float(base) if base is not None else None
        self.conditions = conditions
        self.check_conditions()

    def __repr__(self):
        repr_str = ["Name: %s" % self.name]
        repr_str.append("Type: NormalFloat")
        repr_str.append("Mu: %s Sigma: %s" % (str(self.mu), str(self.sigma)))
        repr_str.append("Base: %s" % self.base)
        repr_str.append("Q: %s" % self.q)
        repr_str.append(self.get_conditions_as_string())
        return ", ".join(repr_str)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([abs(self.mu - other.mu) < 0.00000001,
                        abs(self.sigma - other.sigma) < 0.00000001,
                        self.base is None and other.base is None or
                        self.base is not None and other.base is not None and
                        abs(self.base - other.base) < 0.00000001,
                        self.q is None and other.q is None or
                        self.q is not None and other.q is not None and
                        abs(self.q - other.q) < 0.00000001])
        else:
            return False


class UniformIntegerHyperparameter(IntegerHyperparameter):
    def __init__(self, name, lower, upper, q=None, base=None, conditions=None):
        self.name = name
        self.lower = self.check_int(lower, "lower")
        self.upper = self.check_int(upper, "upper")
        if q is not None:
            self.q = self.check_int(q, "q")
        else:
            self.q = None
        self.base = float(base) if base is not None else None
        self.conditions = conditions
        self.check_conditions()

    def __repr__(self):
        repr_str = ["Name: %s" % self.name]
        repr_str.append("Type: UniformInteger")
        repr_str.append("Range: [%s, %s]" % (str(self.lower), str(self.upper)))
        repr_str.append("Base: %s" % self.base)
        repr_str.append("Q: %s" % self.q)
        repr_str.append(self.get_conditions_as_string())
        return ", ".join(repr_str)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([self.lower == other.lower,
                        self.upper == other.upper,
                        self.base is None and other.base is None or
                        self.base is not None and other.base is not None and
                        abs(self.base - other.base) < 0.00000001,
                        self.q is None and other.q is None or
                        self.q is not None and other.q is not None and
                        self.q == other.q])
        else:
            return False


class NormalIntegerHyperparameter(IntegerHyperparameter):
    def __init__(self, name, lower, upper, q=None, base=None, conditions=None):
        self.name = name
        self.lower = self.check_int(lower, "lower")
        self.upper = self.check_int(upper, "upper")
        if q is not None:
            self.q = self.check_int(q, "q")
        else:
            self.q = None
        self.base = float(base) if base is not None else None
        self.conditions = conditions
        self.check_conditions()

    def __repr__(self):
        repr_str = ["Name: %s" % self.name]
        repr_str.append("Type: NormalInteger")
        repr_str.append("Mu: %s Sigma: %s" % (str(self.lower), str(self.upper)))
        repr_str.append("Base: %s" % self.base)
        repr_str.append("Q: %s" % self.q)
        repr_str.append(self.get_conditions_as_string())
        return ", ".join(repr_str)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([abs(self.mu - other.mu) < 0.00000001,
                        abs(self.sigma - other.sigma) < 0.00000001,
                        self.base is None and other.base is None or
                        self.base is not None and other.base is not None and
                        abs(self.base - other.base) < 0.00000001,
                        self.q is None and other.q is None or
                        self.q is not None and other.q is not None and
                        self.q == other.q])
        else:
            return False


def create_dag_from_hyperparameters(hyperparameters):
    if type(hyperparameters) == dict:
        hyperparameters = hyperparameters.values()
    elif type(hyperparameters) in [list, tuple]:
        pass
    else:
        raise NotImplementedError()
    hyperparameters.sort(key=lambda theta: theta.name, reverse=True)


    DG = nx.DiGraph()
    DG.add_node('__HPOlib_configuration_space_root__')
    to_visit = deque(hyperparameters)
    visited = dict()

    while len(to_visit) > 0:
        # TODO: there is no cycle detection in here!
        hyperparameter = to_visit.popleft()
        name = hyperparameter.name

        if hyperparameter.conditions != [[]]:
            # Extend to handle or-conditions
            if len(hyperparameter.conditions) > 1:
                print hyperparameter.name, hyperparameter.conditions
                raise NotImplementedError()

            # extract all conditions and check if all parent nodes are
            # already inserted into the Graph
            depends_on = []

            for condition in hyperparameter.conditions[0]:
                depends_on_name = condition.split()[0]
                operator = condition.split()[1]
                value = condition.split()[2]
                if depends_on_name not in visited:
                    to_visit.append(hyperparameter)
                    continue
                depends_on.append((depends_on_name, operator, value))
            if len(depends_on) != len(hyperparameter.conditions[0]):
                continue
        else:
            depends_on = []

        visited[name] = hyperparameter
        DG.add_node(name, hyperparameter=hyperparameter)

        if len(depends_on) == 0:
            DG.add_edge('__HPOlib_configuration_space_root__', name)
        elif len(depends_on) == 1:
            d_name, op, value = depends_on[0]
            DG.add_edge(d_name, name, condition=(d_name, op, value))
        else:
            # Add links to all direct parents, these can be less than the
            # number of values in depends_on. The direct parent depends on
            # all other values in depends_on except itself
            parents = []
            for i, d in enumerate(depends_on):
                d_name, op, value = d
                parents.append(nx.ancestors(DG, d_name))

            candidates = []
            for i, d in enumerate(depends_on):
                d_name, op, value = d
                for j, d2 in enumerate(depends_on):
                    if i == j:
                        continue
                    if d_name in parents[j]:
                        candidates.append(j)
                    else:
                        continue

            if len(candidates) != 1:
                print candidates
                raise ValueError()

            parent = depends_on[candidates[0]][0]
            condition = depends_on[candidates[0]]

            DG.add_edge(parent, name, condition=condition)

    if not nx.is_directed_acyclic_graph(DG):
        cycles = list(nx.simple_cycles(DG))
        raise ValueError("Hyperparameter configurations contain a cycle %s" %
            str(cycles))

    return DG


def get_dag(dag):
    sorted_dag = dag.copy()

    # The children of a node are traversed in a random order. Therefore,
    # copy the graph, sort the children and then traverse it
    for adj in sorted_dag.adj:
        sorted_dag.adj[adj] = OrderedDict(
            sorted(sorted_dag.adj[adj].items(), key=lambda item: item[0]))

    nodes = nx.dfs_postorder_nodes\
        (sorted_dag, source='__HPOlib_configuration_space_root__')
    nodes = [node for node in nodes if node != '__HPOlib_configuration_space_root__']
    return nodes

