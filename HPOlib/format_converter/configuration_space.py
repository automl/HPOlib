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
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            print "ksdhfksdhf"
            return (self.name == other.name and self.domain == other.domain)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        repr_str = "Name: %s\n" % self.name
        repr_str += repr(self.domain)
        if self.has_conditions():
            repr_str += "Conditions: %s\n" % (str(self.conditions))
        else:
            repr_str += "Conditions: None\n"
        return repr_str + repr(self.domain)

    def __str__(self):
        return self.__repr__()

    def has_conditions(self):
        return len(self.conditions[0]) == 0


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
        repr_str = "Type: %s\n" % self.type
        if self.type == "categorical":
            repr_str += "Choices: %s\n" % str(self.choices)
        elif self.type in ("float", "int"):
            repr_str += "Range: [%s, %s]\n" % (str(self.lower), str(self.upper))
            repr_str += "Base: %s\n" % self.base
            repr_str += "Q: %s\n" % self.q

        return repr_str

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

