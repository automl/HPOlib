import copy
import math
import StringIO

import numpy as np  # Because the python docs don't recommend log(x, y)
                    # https://docs.python.org/2/library/math.html#math.log10

import HPOlib.format_converter.configuration_space as configuration_space_module

################################################################################
# Read functionality
def read(pyll_string):
    space = eval(pyll_string)

def read_literal(expr):
    raise NotImplementedError()

def read_container(expr):
    raise NotImplementedError()

def read_switch(expr):
    raise NotImplementedError()

def read_uniform(expr, label):
    assert len(expr.inputs()) == 2
    assert all([input.name == "literal" for input in expr.inputs()]), \
        expr.inputs()
    lower = expr.inputs()[0]._obj
    upper = expr.inputs()[1]._obj
    return configuration_space_module.UniformFloatHyperparameter(
        label, lower, upper)

def read_loguniform(expr, label):
    assert len(expr.inputs()) == 2
    assert all([input.name == "literal" for input in expr.inputs()]), \
        expr.inputs()
    lower = expr.inputs()[0]._obj
    upper = expr.inputs()[1]._obj
    return configuration_space_module.UniformFloatHyperparameter(
        label, lower, upper, base=np.e)

def read_quniform(expr, label):
    assert len(expr.inputs()) == 3
    assert all([input.name == "literal" for input in expr.inputs()]), \
        expr.inputs()
    lower = expr.inputs()[0]._obj
    upper = expr.inputs()[1]._obj
    q = expr.inputs()[2]._obj
    return configuration_space_module.UniformFloatHyperparameter(
        label, lower, upper, q=q)

def read_qloguniform(expr, label):
    assert len(expr.inputs()) == 3
    assert all([input.name == "literal" for input in expr.inputs()]), \
        expr.inputs()
    lower = expr.inputs()[0]._obj
    upper = expr.inputs()[1]._obj
    q = expr.inputs()[2]._obj
    return configuration_space_module.UniformFloatHyperparameter(
        label, lower, upper, q=q, base=np.e)

def read_normal(expr, label):
    assert len(expr.inputs()) == 2
    assert all([input.name == "literal" for input in expr.inputs()]), \
        expr.inputs()
    mu = expr.inputs()[0]._obj
    sigma = expr.inputs()[1]._obj
    return configuration_space_module.NormalFloatHyperparameter(
        label, mu, sigma)

def read_lognormal(expr, label):
    assert len(expr.inputs()) == 2
    assert all([input.name == "literal" for input in expr.inputs()]), \
        expr.inputs()
    mu = expr.inputs()[0]._obj
    sigma = expr.inputs()[1]._obj
    return configuration_space_module.NormalFloatHyperparameter(
        label, mu, sigma, base=np.e)

def read_qnormal(expr, label):
    assert len(expr.inputs()) == 3
    assert all([input.name == "literal" for input in expr.inputs()]), \
        expr.inputs()
    mu = expr.inputs()[0]._obj
    sigma = expr.inputs()[1]._obj
    q = expr.inputs()[2]._obj
    return configuration_space_module.NormalFloatHyperparameter(
        label, mu, sigma, q=q)

def read_qlognormal(expr, label):
    assert len(expr.inputs()) == 3
    assert all([input.name == "literal" for input in expr.inputs()]), \
        expr.inputs()
    mu = expr.inputs()[0]._obj
    sigma = expr.inputs()[1]._obj
    q = expr.inputs()[2]._obj
    return configuration_space_module.NormalFloatHyperparameter(
        label, mu, sigma, q=q, base=np.e)

################################################################################
# Write functionality
def write(configuration_space):
    pyll_writer = PyllWriter()
    return pyll_writer.write(configuration_space)


class PyllWriter(object):
    def __init__(self):
        self.hyperparameters = {}

    def reset_hyperparameter_countr(self):
        self.hyperparameters = {}

    def write(self, configuration_space):
        configuration_space = copy.deepcopy(configuration_space)

        # Name conversions must happen here because the hyperparameter are
        # later on referenced by this name
        for hyperparameter in configuration_space.values():
            if isinstance(hyperparameter, configuration_space_module
                    .NumericalHyperparameter):
                # TODO implement general base
                # TODO implement different distributions
                # TODO implement different Qs
                if hyperparameter.base is None:
                    continue
                if hyperparameter.base == 2:
                    hyperparameter.name = 'LOG2_' + hyperparameter.name
                    hyperparameter.lower = \
                        np.log2(hyperparameter.lower)
                    hyperparameter.upper = \
                        np.log2(hyperparameter.upper)
                    hyperparameter.base = None
                elif hyperparameter.base == 10:
                    hyperparameter.name = 'LOG10_' + hyperparameter.name
                    hyperparameter.lower = \
                        np.log10(hyperparameter.lower)
                    hyperparameter.upper = \
                        np.log10(hyperparameter.upper)
                    hyperparameter.base = None
                else:
                    print hyperparameter
                    raise NotImplementedError()

        configuration_dag = configuration_space_module\
            .create_dag_from_hyperparameters(configuration_space)

        configuration_string = StringIO.StringIO()
        configuration_string.write('from hyperopt import hp\n')
        configuration_string.write('import hyperopt.pyll as pyll')
        configuration_string.write('\n\n')

        strings, hyperparameter_names = self.traverse_dag_depth_first(
            configuration_dag)
        for string in strings:
            configuration_string.write(string)
            configuration_string.write("\n")

        configuration_string.write('\nspace = {')
        configuration_string.write(', '
            .join(['"%s": param_%s' % (name, self.hyperparameters[name])
                   for name in hyperparameter_names]))
        configuration_string.write('}\n')
        configuration_string.seek(0)
        return configuration_string.getvalue()

    def traverse_dag_depth_first(self, dag):
        hyperparameter_names = []
        strings = []

        for name in configuration_space_module.get_dag(dag):
            hyperparameter = dag.node[name]['hyperparameter']
            if hyperparameter.conditions == [[]]:
                hyperparameter_names.append(name)
            children = dag[name]
            _, string = self.write_hyperparameter(hyperparameter, children)
            strings.append(string)

        return strings, hyperparameter_names

    def write_hyperparameter(self, hyperparameter, children):
        # Which string generator to call
        if isinstance(hyperparameter, configuration_space_module.NumericalHyperparameter):
            generator_name = 'write_'
            if hyperparameter.q is not None:
                generator_name += 'q'
            if hyperparameter.base is None:
                pass
            elif hyperparameter.base == math.e:
                generator_name += 'log'
            else:
                raise NotImplementedError()

            if len(children) > 0:
                raise NotImplementedError()

            generator_name += 'uniform'

            if isinstance(hyperparameter, configuration_space_module.IntegerHyperparameter):
                generator_name += "_int"

            name, string = getattr(self, generator_name)(hyperparameter)

        elif isinstance(hyperparameter, configuration_space_module.CategoricalHyperparameter):
                name, string = self.write_choice(hyperparameter, children)

        else:
            raise NotImplementedError()

        return name, string

    def write_choice(self, parameter, children):
        name = parameter.name
        choices = dict()
        for choice in parameter.choices:
            choices[choice] = dict()

        for key in children:
            child = children[key]
            operator = child['condition'][1]
            if operator == "==":
                value = child['condition'][2]
                choices[value][key] = child
            elif operator == "in":
                values = child['condition'][2].replace("{", "").replace("}", "")
                values = values.split(",")
                for value in values:
                    choices[value][key] = child

        index = "param_%d" % len(self.hyperparameters)
        self.hyperparameters[name] = len(self.hyperparameters)
        return_string = '%s = hp.choice("%s", [\n' % (index, name)
        for choice in sorted(choices):
            return_string += '    {'
            return_string += '"%s": "%s", ' % (name, choice)
            for key in sorted(choices[choice]):
                return_string += '"%s": param_%s, ' % \
                                 (key, self.hyperparameters[key])
            return_string += '},\n'
        return_string += '    ])'

        return name, return_string

    def write_uniform(self, parameter):
        name = parameter.name
        lower = float(parameter.lower)
        upper = float(parameter.upper)
        index = "param_%d" % len(self.hyperparameters)
        self.hyperparameters[name] = len(self.hyperparameters)
        return name, '%s = hp.uniform("%s", %s, %s)' % \
            (index, name, lower, upper)

    def write_uniform_int(self, parameter):
        name = parameter.name
        lower = float(parameter.lower)
        upper = float(parameter.upper)
        q = 1.0
        index = "param_%d" % len(self.hyperparameters)
        self.hyperparameters[name] = len(self.hyperparameters)
        return name, '%s = pyll.scope.int(hp.quniform("%s", %s, %s, %s))' \
            % (index, name, lower, upper, q)

    def write_quniform(self, parameter):
        name = parameter.name
        lower = float(parameter.lower)
        upper = float(parameter.upper)
        q = parameter.q
        index = "param_%d" % len(self.hyperparameters)
        self.hyperparameters[name] = len(self.hyperparameters)
        return name, '%s = hp.quniform("%s", %s, %s, %s)' % \
            (index, name, lower, upper, q)

    def write_quniform_int(self, parameter):
        name = parameter.name
        lower = float(parameter.lower)
        upper = float(parameter.upper)
        q = float(parameter.q)
        index = "param_%d" % len(self.hyperparameters)
        self.hyperparameters[name] = len(self.hyperparameters)
        return name, '%s = pyll.scope.int(hp.quniform("%s", %s, %s, %s))' \
            % (index, name, lower, upper, q)

    def write_loguniform(self, parameter):
        name = parameter.name
        lower = float(parameter.lower)
        upper = float(parameter.upper)
        index = "param_%d" % len(self.hyperparameters)
        self.hyperparameters[name] = len(self.hyperparameters)
        return name, '%s = hp.loguniform("%s", %s, %s)' % \
            (index, name, lower, upper)

    def write_loguniform_int(self, parameter):
        name = parameter.name
        lower = float(parameter.lower)
        upper = float(parameter.upper)
        q = 1.0
        index = "param_%d" % len(self.hyperparameters)
        self.hyperparameters[name] = len(self.hyperparameters)
        return name, '%s = pyll.scope.int(hp.qloguniform("%s", %s, %s, %s))' % \
            (index, name, lower, upper, q)

    def write_qloguniform(self, parameter):
        name = parameter.name
        lower = float(parameter.lower)
        upper = float(parameter.upper)
        q = parameter.q
        index = "param_%d" % len(self.hyperparameters)
        self.hyperparameters[name] = len(self.hyperparameters)
        return name, '%s = hp.qloguniform("%s", %s, %s, %s)' % \
            (index, name, lower, upper, q)

    def write_qloguniform_int(self, parameter):
        name = parameter.name
        lower = float(parameter.lower)
        upper = float(parameter.upper)
        q = float(parameter.q)
        index = "param_%d" % len(self.hyperparameters)
        self.hyperparameters[name] = len(self.hyperparameters)
        return name, '%s = pyll.scope.int(hp.qloguniform("%s", %s, %s, %s))' % \
            (index, name, lower, upper, q)


"""
def write_normal(parameter):
    name = parameter.name
    mu = parameter.domain.mu
    sigma = parameter.domain.sigma
    return '%s = hp.normal("%s", %s, %s)' % (name, name, mu, sigma)

def write_qnormal(parameter):
    name = parameter.name
    mu = parameter.domain.mu
    sigma = parameter.domain.sigma
    q = parameter.domain.rounding
    return '%s = hp.qnormal("%s", %s, %s, %s)' % (name, name, mu, sigma, q)

def write_lognormal(parameter):
    name = parameter.name
    mu = parameter.domain.mu
    sigma = parameter.domain.sigma
    return '%s = hp.lognormal("%s", %s, %s)' % (name, name, mu, sigma)

def write_qlognormal(parameter):
    name = parameter.name
    mu = parameter.domain.mu
    sigma = parameter.domain.sigma
    q = parameter.domain.rounding
    return '%s = hp.qlognormal("%s", %s, %s)' % (name, name, mu, sigma, q)

def write_switch(parameter):
    raise NotImplementedError()
"""