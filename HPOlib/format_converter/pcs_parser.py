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

from collections import OrderedDict
import StringIO
import sys

import numpy as np
import pyparsing

import configuration_space


# Build pyparsing expressions for params
pp_param_name = pyparsing.Word(pyparsing.alphanums + "_" + "-" + "@" + "." + ":" + ";" + "\\" + "/" + "?" + "!" +
                               "$" + "%" + "&" + "*" + "+" + "<" + ">")
pp_digits = "0123456789"
pp_plusorminus = pyparsing.Literal('+') | pyparsing.Literal('-')
pp_int = pyparsing.Combine(pyparsing.Optional(pp_plusorminus) + pyparsing.Word(pp_digits))
pp_float = pyparsing.Combine(pyparsing.Optional(pp_plusorminus) + pyparsing.Optional(pp_int) + "." + pp_int)
pp_eorE = pyparsing.Literal('e') | pyparsing.Literal('E')
pp_e_notation = pyparsing.Combine(pp_float + pp_eorE + pp_int)
pp_number = pp_e_notation | pp_float | pp_int
pp_numberorname = pp_number | pp_param_name
pp_il = pyparsing.Word("il")
pp_choices = pp_param_name + pyparsing.Optional(pyparsing.OneOrMore("," + pp_param_name))

pp_cont_param = pp_param_name + "[" + pp_number + "," + pp_number + "]" + \
    "[" + pp_number + "]" + pyparsing.Optional(pp_il)
pp_cat_param = pp_param_name + "{" + pp_choices + "}" + "[" + pp_param_name + "]"
pp_condition = pp_param_name + "|" + pp_param_name + "in" + "{" + pp_choices + "}"
pp_forbidden_clause = "{" + pp_param_name + "=" + pp_numberorname + \
    pyparsing.Optional(pyparsing.OneOrMore("," + pp_param_name + "=" + pp_numberorname)) + "}"


def build_categorical(param):
    cat_template = "%s {%s} [%s]"
    return cat_template % (param.name, ", ".join(param.choices), param.choices[0])


def build_constant(param):
    constant_template = "%s {%s} [%s]"
    return constant_template % (param.name, param.value, param.value)


def build_continuous(param):
    if type(param) in (configuration_space.NormalIntegerHyperparameter,
                       configuration_space.NormalFloatHyperparameter):
        param = param.to_uniform()

    float_template = "%s [%s, %s] [%s]"
    int_template = "%s [%d, %d] [%d]i"
    if param.base is not None:
        if param.base == 10:
            # SMAC can naturally handle this
            float_template += "l"
            int_template += "l"
            default = np.power(10, (np.log10(param.upper) +
                                    np.log10(param.lower))/2)
            if param.lower < 0:
                raise ValueError("This is no log domain: %s" % param.name)
            if param.upper < 0:
                raise ValueError("This is no log domain: %s" % param.name)
        elif abs(param.base - np.e) < 0.0000001:
            param.name = "LOG_%s" % param.name
            param.lower = np.log(param.lower)
            param.upper = np.log(param.upper)
            default = (param.lower + param.upper) / 2
        else:
            if int(param.base) != param.base:
                raise NotImplementedError("We cannot handle non-int bases: %s (%s)" %
                                          (str(param.base), param.name))
            # HPOlib has to take care of this
            print param
            print np.log10(param.lower), np.log10(param.base)
            param.name = "LOG%d_%s" % (int(param.base), param.name)
            param.lower = np.log10(param.lower) / np.log10(param.base)
            param.upper = np.log10(param.upper) / np.log10(param.base)
            default = (param.upper + param.lower) / 2
    else:
        default = (param.upper + param.lower)/2

    if param.q is not None:
        param.name = "Q%d_%s" % (int(param.q), param.name)

    if param.upper >= default <= param.lower:
        raise NotImplementedError("Cannot find mean for %s" % param.name)
    if isinstance(param, configuration_space.IntegerHyperparameter):
        param.lower = int(param.lower)
        param.upper = int(param.upper)
        default = int(default)
        return int_template % (param.name, param.lower, param.upper, default)
    else:
        return float_template % (param.name, str(param.lower), str(param.upper), str(default))


def build_condition(name, condition):
    condition_template = "%s | %s in {%s}"
    condition = condition.split(" ")
    if condition[1] != "==" and condition[1] != "in":
        raise NotImplementedError("SMAC cannot handle >< conditions: %s (%s)" % (condition, name))
    if condition[1] == "in":
        return condition_template % (name, condition[0], condition[2][1:-1].replace(",", ", "))
    if condition[1] == "==":
        return condition_template % (name, condition[0], condition[2])


def read(pcs_string, debug=False):
    searchspace = OrderedDict()
    conditions = list()
    # some statistics
    ct = 0
    cont_ct = 0
    cat_ct = 0
    line_ct = 0
    for line in pcs_string:
        line_ct += 1
        if str.startswith(line, '#'):
            # It's a comment
            continue
        if "#" in line:
            # It contains a comment
            pos = line.find("#")
            line = line[:pos]

        # Remove quotes and whitespaces at beginning and end
        line = line.replace('"', "").replace("'", "")
        line = line.strip()

        if "|" in line:
            # It's a condition
            try:
                c = pp_condition.parseString(line)
                conditions.append(c)
            except pyparsing.ParseException:
                raise NotImplementedError("Could not parse condition: %s" % line)
            continue
        if "}" not in line and "]" not in line:
            print "Skipping: %s" % line
            continue
        if len(line.strip()) == 0:
            continue

        ct += 1
        param = None
        # print "Parsing: " + line

        create = {"int": configuration_space.UniformIntegerHyperparameter,
                  "float": configuration_space.UniformFloatHyperparameter,
                  "categorical": configuration_space.CategoricalHyperparameter}

        try:
            param_list = pp_cont_param.parseString(line)
            il = param_list[9:]
            if len(il) > 0:
                il = il[0]
            param_list = param_list[:9]
            name = param_list[0]
            lower = float(param_list[2])
            upper = float(param_list[4])
            paramtype = "int" if "i" in il else "float"
            base = 10 if "l" in il else None
            param = create[paramtype](name=name, lower=lower, upper=upper, q=None, base=base, conditions=None)
            cont_ct += 1
        except pyparsing.ParseException:
            pass

        try:
            param_list = pp_cat_param.parseString(line)
            name = param_list[0]
            choices = [c for c in param_list[2:-4:2]]
            param = create["categorical"](name=name, choices=choices, conditions=None)
            cat_ct += 1
        except pyparsing.ParseException:
            pass

        try:
            # noinspection PyUnusedLocal
            param_list = pp_forbidden_clause.parseString(line)
            # TODO: remove this hotfix!
            continue
            raise NotImplementedError("We cannot handle forbidden clauses: %s" % line)
        except pyparsing.ParseException:
            pass

        if param is None:
            raise NotImplementedError("Could not parse: %s" % line)

        searchspace[param.name] = param

    def normalize_param_name(name):
        if "LOG10_" in child:
            pos = name.find("LOG10_")
            name = name[0:pos] + name[pos + 6:]
        elif "LOG2_" in child:
            pos = name.find("LOG2_")
            name = name[0:pos] + name[pos + 5:]
        elif "LOG_" in child:
            pos = name.find("LOG_")
            name = name[0:pos] + name[pos + 4:]
        import re
        m = re.search(r'Q[0-999\.]{1,10}_', name)
        if m is not None:
            pos = name.find(m.group(0))
            name = name[0:pos] + name[pos + len(m.group(0)):]
        return name


    #Now handle conditions
    for cond in conditions:
        child = cond[0]
        parent = cond[2]
        restrictions = cond[5:-1:2]
        # TODO remove this hotfix
        child = normalize_param_name(child)
        parent = normalize_param_name(parent)

        if child not in searchspace:
            raise ValueError("%s is not defined, values in search space are %s"
                             % (child, str(searchspace)))
        if parent not in searchspace:
            raise ValueError("%s is not defined" % parent)

        if len(restrictions) == 1:
            cond_str = "%s == %s" % (parent, restrictions[0])
        else:
            cond_str = "%s in {%s}" % (parent, ",".join(restrictions))

        if len(searchspace[child].conditions) == 0:
            searchspace[child].conditions.append([cond_str, ])
        else:
            searchspace[child].conditions[0].extend([cond_str, ])

    if debug:
        print
        print "============== Reading Results"
        print "First 10 lines:"
        sp_list = ["%s: %s" % (j, str(searchspace[j])) for j in searchspace]
        print "\n".join(sp_list[:10])
        print
        print "#Invalid lines: %d ( of %d )" % (line_ct - len(conditions) - ct, line_ct)
        print "#Parameter: %d" % len(searchspace)
        print "#Conditions: %d" % len(conditions)
        print "#Conditioned params: %d" % sum([1 if len(searchspace[j].conditions[0]) > 0 else 0 for j in searchspace])
        print "#Categorical: %d" % cat_ct
        print "#Continuous: %d" % cont_ct
    return searchspace


def write(searchspace):
    param_lines = StringIO.StringIO()
    condition_lines = StringIO.StringIO()
    for para_name in searchspace:
        # First build params
        if param_lines.tell() > 0:
            param_lines.write("\n")
        if isinstance(searchspace[para_name],
                      configuration_space.NumericalHyperparameter):
            param_lines.write(build_continuous(searchspace[para_name]))
        elif isinstance(searchspace[para_name],
                        configuration_space.CategoricalHyperparameter):
            param_lines.write(build_categorical(searchspace[para_name]))
        elif isinstance(searchspace[para_name], configuration_space.Constant):
            param_lines.write(build_constant(searchspace[para_name]))
        else:
            raise NotImplementedError("Unknown type: %s (%s)" % (
                type(searchspace[para_name]), para_name))

        # Now handle conditions
        if condition_lines.tell() > 0:
            condition_lines.write("\n")
        if len(searchspace[para_name].conditions) > 1:
            raise NotImplementedError("SMAC cannot handle OR conditions on different parents: %s (%s)" %
                                      (str(searchspace[para_name].conditions), para_name))
        for condition in searchspace[para_name].conditions[0]:
            condition_lines.write(build_condition(para_name, condition))

    if condition_lines.tell() > 0:
        condition_lines.seek(0)
        param_lines.write("\n\n")
        for line in condition_lines:
            param_lines.write(line)
    param_lines.seek(0)
    return param_lines.getvalue()


if __name__ == "__main__":
    fh = open(sys.argv[1])
    orig_pcs = fh.readlines()
    sp = read(orig_pcs, debug=True)
    created_pcs = write(sp).split("\n")
    print "============== Writing Results"
    print "#Lines: ", len(created_pcs)
    print "#LostLines: ", len(orig_pcs) - len(created_pcs)
    diff = ["%s\n" % i for i in created_pcs if i not in " ".join(orig_pcs)]
    print "Identical Lines: ", len(created_pcs) - len(diff)
    print
    print "Up to 10 random different lines (of %d):" % len(diff)
    print "".join(diff[:10])