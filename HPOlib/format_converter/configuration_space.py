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
import re

import networkx as nx
import numpy as np


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
            raise ValueError("Conditions are not a list: %s" %
                             str(self.conditions))
        for o_r in self.conditions:
            if type(o_r) != list:
                raise ValueError("These conditions are not a list: %s" %
                                 str(o_r))

    def has_conditions(self):
        return len(self.conditions[0]) > 0

    def append_conditions(self, conditions):
        for condition in conditions:
            self.append_condition(condition)

    def append_condition(self, condition):
        if type(condition) != list:
            raise ValueError("Condition Argument is not a list: %s" %
                             str(condition))

        added = False
        if not condition:
            return

        if self.conditions == [[]]:
            self.conditions = [condition]
            added = True
        # Check if this is a single condition of form "a == 2" and there
        # exists "a == 1" so they can be condensed into "a in {1,2}"
        elif len(condition) == 1:
            depends_on = condition[0].split()[0]
            condition_to_add = condition[0].split()[2]
            for i, sc in enumerate(self.conditions):
                if len(sc) == 1 and sc[0].split()[0] == depends_on:
                    # Conditions can be condensed
                    condition_values = sc[0].split()[2]
                    if "," in condition_values:
                        condition_values = condition_values[1:-1].split(",")
                        if condition_to_add in condition_values:
                            added = True
                            break
                        else:
                            condition_values.append(condition_to_add)
                            self.conditions[i][0] = "%s in %s" % (depends_on,
                                        "{" + ",".join(condition_values) + "}")
                    else:
                        if condition_to_add == condition_values:
                            added = True
                            break
                        else:
                            condition_values = [condition_values,
                                                condition_to_add]
                            self.conditions[i][0] = "%s in %s" % (depends_on,
                                "{" + ",".join(condition_values) + "}")
                    added = True
                    break

        if not added:
            self.conditions.append(condition)

    def get_conditions_as_string(self):
        if self.has_conditions():
            return "Conditions: %s" % (str(self.conditions))
        else:
            return "Conditions: None"


class Constant(Hyperparameter):
    def __init__(self, name, value, conditions=None):
        self.name = name
        self.value = value
        self.conditions = conditions
        self.check_conditions()

    def __repr__(self):
        repr_str = ["Name: %s" % self.name,
                    "Type: Constant",
                    "Value: %s" % self.value,
                    self.get_conditions_as_string()]
        return ", ".join(repr_str)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.name != other.name:
                return False
            if self.value != other.value:
                return False
            return True
        else:
            return False


class NumericalHyperparameter(Hyperparameter):
    pass


class FloatHyperparameter(NumericalHyperparameter):
    pass


class IntegerHyperparameter(NumericalHyperparameter):
    def check_int(self, parameter, name):
        if abs(np.round(parameter, 5) - parameter) > 0.00001 and \
                type(parameter) is not int:
            raise ValueError("For the Integer parameter %s, the value must be "
                             "an Integer, too. Right now it is a %s with value"
                             " %s" % (name, type(parameter), str(parameter)))
        return int(parameter)


class CategoricalHyperparameter(Hyperparameter):
    def __init__(self, name, choices, conditions=None):
        self.name = name
        self.choices = choices
        self.conditions = conditions
        self.check_conditions()

    def __repr__(self):
        repr_str = ["Name: %s" % self.name,
                    "Type: Categorical",
                    "Choices: %s" % str(self.choices),
                    self.get_conditions_as_string()]
        return ", ".join(repr_str)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.name != other.name:
                return False
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
        self.check_name()

    def __repr__(self):
        repr_str = ["Name: %s" % self.name,
                    "Type: UniformFloat",
                    "Range: [%s, %s]" % (str(self.lower), str(self.upper)),
                    "Base: %s" % self.base,
                    "Q: %s" % self.q,
                    self.get_conditions_as_string()]
        return ", ".join(repr_str)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([self.name == other.name,
                        abs(self.lower - other.lower) < 0.00000001,
                        abs(self.upper - other.upper) < 0.00000001,
                        self.base is None and other.base is None or
                        self.base is not None and other.base is not None and
                        abs(self.base - other.base) < 0.00000001,
                        self.q is None and other.q is None or
                        self.q is not None and other.q is not None and
                        abs(self.q - other.q) < 0.00000001])
        else:
            return False

    def check_name(self):
        # TODO: replace float by decimal!
        perform_power = False
        if "LOG10_" in self.name:
            pos = self.name.find("LOG10_")
            self.name = self.name[0:pos] + self.name[pos + 6:]
            self.base = 10
            perform_power = True
        elif "LOG2_" in self.name:
            pos = self.name.find("LOG2_")
            self.name = self.name[0:pos] + self.name[pos + 5:]
            self.base = 2
            perform_power = True
        elif "LOG_" in self.name:
            pos = self.name.find("LOG_")
            self.name = self.name[0:pos] + self.name[pos + 4:]
            self.base = np.e
            perform_power = True
        if perform_power and self.base is not None:
            self.lower = np.round(np.power(self.base, self.lower), 5)
            self.upper = np.round(np.power(self.base, self.upper), 5)
        #Check for Q value, returns round(x/q)*q
        m = re.search(r'Q[0-999\.]{1,10}_', self.name)
        if m is not None:
            pos = self.name.find(m.group(0))
            self.name = self.name[0:pos] + self.name[pos + len(m.group(0)):]
            self.q = float(m.group(0)[1:-1])

    def to_integer(self):
        return UniformIntegerHyperparameter(self.name, self.lower,
                                            self.upper, self.q, self.base,
                                            self.conditions)


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
        repr_str = ["Name: %s" % self.name,
                    "Type: NormalFloat",
                    "Mu: %s Sigma: %s" % (str(self.mu), str(self.sigma)),
                    "Base: %s" % self.base,
                    "Q: %s" % self.q,
                    self.get_conditions_as_string()]
        return ", ".join(repr_str)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([self.name == other.name,
                        abs(self.mu - other.mu) < 0.00000001,
                        abs(self.sigma - other.sigma) < 0.00000001,
                        self.base is None and other.base is None or
                        self.base is not None and other.base is not None and
                        abs(self.base - other.base) < 0.00000001,
                        self.q is None and other.q is None or
                        self.q is not None and other.q is not None and
                        abs(self.q - other.q) < 0.00000001])
        else:
            return False

    def to_uniform(self, z=3):
        if self.base is not None:
            return UniformFloatHyperparameter(self.name,
                np.power(self.base, self.mu - (z * self.sigma)),
                np.power(self.base, self.mu + (z * self.sigma)),
                q=self.q, base=self.base, conditions=self.conditions)
        else:
            return UniformFloatHyperparameter(self.name,
                                              self.mu - (z * self.sigma),
                                              self.mu + (z * self.sigma),
                                              q=self.q,
                                              base=self.base,
                                              conditions=self.conditions)

    def to_integer(self):
        raise NotImplementedError()


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
        self.check_name()

    def __repr__(self):
        repr_str = ["Name: %s" % self.name,
                    "Type: UniformInteger",
                    "Range: [%s, %s]" % (str(self.lower), str(self.upper)),
                    "Base: %s" % self.base,
                    "Q: %s" % self.q,
                    self.get_conditions_as_string()]
        return ", ".join(repr_str)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([self.name == other.name,
                        self.lower == other.lower,
                        self.upper == other.upper,
                        self.base is None and other.base is None or
                        self.base is not None and other.base is not None and
                        abs(self.base - other.base) < 0.00000001,
                        self.q is None and other.q is None or
                        self.q is not None and other.q is not None and
                        self.q == other.q])
        else:
            return False

    def check_name(self):
        # TODO: replace float by decimal!
        perform_power = False
        if "LOG10_" in self.name:
            pos = self.name.find("LOG10_")
            self.name = self.name[0:pos] + self.name[pos + 6:]
            self.base = 10
            perform_power = True
        elif "LOG2_" in self.name:
            pos = self.name.find("LOG2_")
            self.name = self.name[0:pos] + self.name[pos + 5:]
            self.base = 2
            perform_power = True
        elif "LOG_" in self.name:
            pos = self.name.find("LOG_")
            self.name = self.name[0:pos] + self.name[pos + 4:]
            self.base = np.e
            perform_power = True
        if perform_power and self.base is not None:
            self.lower = np.power(self.base, self.lower)
            self.upper = np.power(self.base, self.upper)
        #Check for Q value, returns round(x/q)*q
        m = re.search(r'Q[0-999\.]{1,10}_', self.name)
        if m is not None:
            pos = self.name.find(m.group(0))
            self.name = self.name[0:pos] + self.name[pos + len(m.group(0)):]
            self.q = int(m.group(0)[1:-1])


class NormalIntegerHyperparameter(IntegerHyperparameter):
    def __init__(self, name, lower, upper, q=None, base=None, conditions=None):
        self.name = name
        self.mu = self.check_int(lower, "mu")
        self.sigma = self.check_int(upper, "sigma")
        if q is not None:
            self.q = self.check_int(q, "q")
        else:
            self.q = None
        self.base = float(base) if base is not None else None
        self.conditions = conditions
        self.check_conditions()

    def __repr__(self):
        repr_str = ["Name: %s" % self.name,
                    "Type: NormalInteger",
                    "Mu: %s Sigma: %s" % (str(self.mu), str(self.sigma)),
                    "Base: %s" % self.base,
                    "Q: %s" % self.q,
                    self.get_conditions_as_string()]
        return ", ".join(repr_str)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all([self.name == other.name,
                        abs(self.mu - other.mu) < 0.00000001,
                        abs(self.sigma - other.sigma) < 0.00000001,
                        self.base is None and other.base is None or
                        self.base is not None and other.base is not None and
                        abs(self.base - other.base) < 0.00000001,
                        self.q is None and other.q is None or
                        self.q is not None and other.q is not None and
                        self.q == other.q])
        else:
            return False

    def to_uniform(self, z=3):
        if self.base is not None:
            return UniformFloatHyperparameter(self.name,
                np.power(self.base, self.mu-(z*self.sigma)),
                np.power(self.base, self.mu+(z*self.sigma)),
                q=self.q, base=self.base, conditions=self.conditions)
        else:
            return UniformFloatHyperparameter(self.name,
                self.mu-(z*self.sigma), self.mu+(z*self.sigma),
                q=self.q, base=self.base, conditions=self.conditions)


def create_dag_from_hyperparameters(hyperparameters):
    if type(hyperparameters) in [dict, OrderedDict]:
        hyperparameters = hyperparameters.values()
    elif type(hyperparameters) in [list, tuple]:
        pass
    else:
        raise ValueError("Type %s not supported (%s)" %
                         (type(hyperparameters), str(hyperparameters)))
    hyperparameters.sort(key=lambda theta: theta.name, reverse=True)

    dg = nx.DiGraph()
    dg.add_node('__HPOlib_configuration_space_root__')
    to_visit = deque(hyperparameters)
    visited = dict()

    while len(to_visit) > 0:
        # TODO: there is no cycle detection in here!
        hyperparameter = to_visit.popleft()
        name = hyperparameter.name

        if hyperparameter.conditions != [[]]:
            # Extend to handle or-conditions
            #if len(hyperparameter.conditions) > 1:
            #    print hyperparameter.name, hyperparameter.conditions
            #    raise NotImplementedError()

            # extract all conditions and check if all parent nodes are
            # already inserted into the Graph
            depends_on = []
            do_continue = False
            for i, conditions in enumerate(hyperparameter.conditions):
                depends_on.append([])
                for condition in conditions:
                    depends_on_name = condition.split()[0]
                    operator = condition.split()[1]
                    value = condition.split()[2]
                    if depends_on_name not in visited:
                        to_visit.append(hyperparameter)
                        continue
                    depends_on[-1].append((depends_on_name, operator, value))
                if len(depends_on[-1]) != len(conditions):
                    do_continue = True
                    continue
            if do_continue:
                continue
        else:
            depends_on = []

        visited[name] = hyperparameter
        dg.add_node(name, hyperparameter=hyperparameter)

        if len(depends_on) == 0:
            dg.add_edge('__HPOlib_configuration_space_root__', name)
        else:
            for dependency_path in depends_on:
                if len(dependency_path) == 1:
                    d_name, op, value = dependency_path[0]
                    dg.add_edge(d_name, name, condition=(d_name, op, value))
                else:
                    # Add links to all direct parents, these can be less than
                    # the number of values in depends_on. The direct parent
                    # depends on all other values in depends_on except itself
                    parents = []
                    for i, d in enumerate(dependency_path):
                        d_name, op, value = d
                        parents.append(nx.ancestors(dg, d_name))

                    candidates = []
                    for i, d in enumerate(dependency_path):
                        d_name, op, value = d
                        for j, d2 in enumerate(dependency_path):
                            if i == j:
                                continue
                            if d_name in parents[j]:
                                candidates.append(j)
                            else:
                                continue

                    if len(candidates) != 1:
                        print hyperparameter, depends_on
                        print parents
                        print candidates
                        raise ValueError()

                    parent = dependency_path[candidates[0]][0]
                    condition = dependency_path[candidates[0]]

                    dg.add_edge(parent, name, condition=condition)

    if not nx.is_directed_acyclic_graph(dg):
        cycles = list(nx.simple_cycles(dg))
        raise ValueError("Hyperparameter configurations contain a cycle %s" %
                         str(cycles))
    """
    nx.write_dot(DG, "hyperparameters.dot")
    import matplotlib.pyplot as plt
    plt.title("draw_networkx")
    pos = nx.graphviz_layout(DG, prog='dot')
    nx.draw(DG, pos, with_labels=True)
    plt.savefig('nx_test.png')
    """
    return dg


def get_dag(dag):
    sorted_dag = dag.copy()

    # The children of a node are traversed in a random order. Therefore,
    # copy the graph, sort the children and then traverse it
    for adj in sorted_dag.adj:
        sorted_dag.adj[adj] = OrderedDict(
            sorted(sorted_dag.adj[adj].items(), key=lambda item: item[0]))

    nodes = nx.dfs_postorder_nodes(sorted_dag,
                                   source='__HPOlib_configuration_space_root__')
    nodes = [node for node in nodes if node !=
             '__HPOlib_configuration_space_root__']
    return nodes

