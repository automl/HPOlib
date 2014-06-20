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

import pyparsing

import configuration_space

# Build pyparsing expressions for params
pp_param_name = pyparsing.Word(pyparsing.alphanums + "_" + "@" + ":" + "-" + "." + "\\" + "/")
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


def read_smac(pcs_string):
    searchspace = dict()
    conditions = list()
    # some statistics
    ct = 0
    cont_ct = 0
    cat_ct = 0
    for line in pcs_string:
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

        create = {"int": configuration_space.create_int,
                  "float": configuration_space.create_float,
                  "categorical": configuration_space.create_categorical}

        try:
            param_list = pp_cont_param.parseString(line)
            il = param_list[9:]
            param_list = param_list[:9]
            name = param_list[0]
            lower = float(param_list[2])
            upper = float(param_list[4])
            type = "int" if "i" in il else "float"
            base = 10 if "l" in il else None
            param = create[type](name=name, lower=lower, upper=upper, q=None, base=base, conditions=None)
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
            param_list = pp_forbidden_clause.parseString(line)
            raise NotImplementedError("We cannot handle forbidden clauses: %s" % line)
        except pyparsing.ParseException:
            pass


        if param is None:
            raise NotImplementedError("Could not parse: %s" % line)

        searchspace[param.name] = param

    #Now handle conditions
    for cond in conditions:
        child = cond[0]
        parent = cond[2]
        restrictions = cond[5:-1:2]
        #print child, parent, restrictions
        if child not in searchspace:
            raise ValueError("%s is not defined" % child)
        if parent not in searchspace:
            raise ValueError("%s is not defined" % parent)
        cond_str = "%s == [%s]" % (parent, ",".join(restrictions))
        searchspace[child].conditions.append([cond_str,])

    print
    print "============== Reading Results"
    print "First 10 lines:"
    sp_list = ["%s: %s" % (i, str(searchspace[i])) for i in searchspace]
    print "\n".join(sp_list[:10])
    print
    print "#Invalid lines: %d ( of %d )" % (len(pcs_string) - len(conditions) - ct, len(pcs_string))
    print "#Parameter: %d" % len(searchspace)
    print "#Conditions: %d" % len(conditions)
    print "#Conditioned params: %d" % sum([1 if len(searchspace[i].conditions) > 0 else 0 for i in searchspace])
    print "#Categorical: %d" % cat_ct
    print "#Continuous: %d" % cont_ct
    return searchspace

if __name__ == "__main__":
    fh = open(sys.argv[1])
    read_smac(fh.readlines())