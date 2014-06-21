import copy
import math
import StringIO

import numpy as np  # Because the python docs don't recommend log(x, y)
                    # https://docs.python.org/2/library/math.html#math.log10

import HPOlib.format_converter.configuration_space as configuration_space_module


def read(pyll_string):
    raise NotImplementedError("We cannot read hyperopt spaces. This feature is still missing")


def write(configuration_space):
    configuration_space = copy.deepcopy(configuration_space)

    # Name conversions must happen here because the hyperparameter are
    # later on referenced by this name
    for hyperparameter in configuration_space.values():
        if hyperparameter.domain.type in ('float', 'int'):
            if hyperparameter.domain.base == 2:
                hyperparameter.name = 'LOG2_' + hyperparameter.name
                hyperparameter.domain.lower = \
                    np.log2(hyperparameter.domain.lower)
                hyperparameter.domain.upper = \
                    np.log2(hyperparameter.domain.upper)
                hyperparameter.domain.base = None
            elif hyperparameter.domain.base == 10:
                hyperparameter.name = 'LOG10_' + hyperparameter.name
                hyperparameter.domain.lower = \
                    np.log10(hyperparameter.domain.lower)
                hyperparameter.domain.upper = \
                    np.log10(hyperparameter.domain.upper)
                hyperparameter.domain.base = None

    configuration_dag = configuration_space_module\
        .create_dag_from_hyperparameters(configuration_space)

    configuration_string = StringIO.StringIO()
    configuration_string.write('from hyperopt import hp\n')
    configuration_string.write('import hyperopt.pyll as pyll')
    configuration_string.write('\n\n')

    strings, hyperparameter_names = traverse_dag_depth_first(configuration_dag)
    for string in strings:
        configuration_string.write(string)
        configuration_string.write("\n")

    configuration_string.write('\nspace = {')
    configuration_string.write(', '
        .join(['"%s": %s' % (name, name) for name in hyperparameter_names]))
    configuration_string.write('}\n')
    configuration_string.seek(0)
    return configuration_string.getvalue()


def traverse_dag_depth_first(dag):
    hyperparameter_names = []
    strings = []

    for name in configuration_space_module.get_dag(dag):
        hyperparameter = dag.node[name]['hyperparameter']
        if hyperparameter.conditions == [[]]:
            hyperparameter_names.append(name)
        children = dag[name]
        _, string = write_hyperparameter(hyperparameter, children)
        strings.append(string)

    return strings, hyperparameter_names


def write_hyperparameter(hyperparameter, children):
    # Which string generator to call
    if hyperparameter.domain.type in ('float', 'int'):
        generator_name = 'write_'
        if hyperparameter.domain.q is not None:
            generator_name += 'q'
        if hyperparameter.domain.base is None:
            pass
        elif hyperparameter.domain.base == math.e:
            generator_name += 'log'
        else:
            raise NotImplementedError()

        if len(children) > 0:
            raise NotImplementedError()

        generator_name += 'uniform'

        if hyperparameter.domain.type == "int":
            generator_name += "_int"

        name, string = globals()[generator_name](hyperparameter)

    elif hyperparameter.domain.type == 'categorical':
            name, string = write_choice(hyperparameter, children)

    else:
        raise NotImplementedError()

    return name, string


def write_choice(parameter, children):
    name = parameter.name
    choices = dict()
    for choice in parameter.domain.choices:
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

    return_string = '%s = hp.choice("%s", [\n' % (name, name)
    for choice in sorted(choices):
        return_string += '    {'
        return_string += '"%s": "%s", ' % (name, choice)
        for key in sorted(choices[choice]):
            return_string += '"%s": %s, ' % (key, key)
        return_string += '},\n'
    return_string += '    ])'

    return name, return_string


def write_uniform(parameter):
    name = parameter.name
    lower = float(parameter.domain.lower)
    upper = float(parameter.domain.upper)
    return name, '%s = hp.uniform("%s", %s, %s)' % (name, name, lower, upper)


def write_uniform_int(parameter):
    name = parameter.name
    lower = float(parameter.domain.lower)
    upper = float(parameter.domain.upper)
    q = 1.0
    return name, '%s = pyll.scope.int(hp.uniform("%s", %s, %s, %s))' \
        % (name, name, lower, upper, q)


def write_quniform(parameter):
    name = parameter.name
    lower = float(parameter.domain.lower)
    upper = float(parameter.domain.upper)
    q = parameter.domain.q
    return name, '%s = hp.quniform("%s", %s, %s, %s)' % \
        (name, name, lower, upper, q)


def write_quniform_int(parameter):
    name = parameter.name
    lower = float(parameter.domain.lower)
    upper = float(parameter.domain.upper)
    q = float(parameter.domain.q)
    return name, '%s = pyll.scope.int(hp.quniform("%s", %s, %s, %s))' \
        % (name, name, lower, upper, q)


def write_loguniform(parameter):
    name = parameter.name
    lower = float(parameter.domain.lower)
    upper = float(parameter.domain.upper)
    return name, '%s = hp.loguniform("%s", %s, %s)' % (name, name, lower, upper)


def write_loguniform_int(parameter):
    name = parameter.name
    lower = float(parameter.domain.lower)
    upper = float(parameter.domain.upper)
    q = 1.0
    return name, '%s = pyll.scope.int(hp.qloguniform("%s", %s, %s, %s))' % \
        (name, name, lower, upper, q)


def write_qloguniform(parameter):
    name = parameter.name
    lower = float(parameter.domain.lower)
    upper = float(parameter.domain.upper)
    q = parameter.domain.q
    return name, '%s = hp.qloguniform("%s", %s, %s, %s)' % \
        (name, name, lower, upper, q)


def write_qloguniform_int(parameter):
    name = parameter.name
    lower = float(parameter.domain.lower)
    upper = float(parameter.domain.upper)
    q = float(parameter.domain.q)
    return name, '%s = pyll.scope.int(hp.qloguniform("%s", %s, %s, %s))' % \
        (name, name, lower, upper, q)


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