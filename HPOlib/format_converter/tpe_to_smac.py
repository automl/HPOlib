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
Routines for interfacing with SMAC


http://www.cs.ubc.ca/labs/beta/Projects/SMAC/

"""

from hyperopt.pyll import base as base

""" Partly taken from https://github.com/jaberg/hyperopt/blob/smacout/hyperopt/smac.py """


import imp
import sys
import os

from functools import partial


class Cond(object):
    def __init__(self, name, val, op):
        self.op = op
        self.name = name
        self.val = val

    def __str__(self):
        return 'Cond{%s %s %s}' % (self.name, self.op, self.val)

    def __eq__(self, other):
        return self.op == other.op and self.name == other.name and self.val == other.val

    def __hash__(self):
        return hash((self.op, self.name, self.val))

    def __repr__(self):
        return str(self)

EQ = partial(Cond, op='=')


def expr_to_config(expr, conditions, hps):
    if conditions is None:
        conditions = ()
    assert isinstance(expr, base.Apply)
    if expr.name == 'switch':
        idx = expr.inputs()[0]
        options = expr.inputs()[1:]
        assert idx.name == 'hyperopt_param'
        assert idx.arg['obj'].name in ('randint',  # - in case of hp.choice
                                       'categorical'  # - in case of hp.pchoice
                                       )
        expr_to_config(idx, conditions, hps)
        for opt_idx, opt in enumerate(options):
            expr_to_config(opt,
                           conditions + (EQ(idx.arg['label'].obj, opt_idx),),
                           hps)
    elif expr.name == 'hyperopt_param':
        label = expr.arg['label'].obj
        if label in hps:
            if hps[label]['node'] != expr.arg['obj']:
                raise hyperopt.base.DuplicateLabel(label)
            hps[label]['conditions'].add(conditions)
        else:
            hps[label] = {'node': expr.arg['obj'],
                          'conditions': set((conditions,))}
    else:
        for ii in expr.inputs():
            expr_to_config(ii, conditions, hps)


def convert_tpe_to_smac_from_file(filename):

    space_name, ext = os.path.splitext(os.path.basename(filename))
    #noinspection PyBroadException
    try:
        space = imp.load_source(space_name, filename)
    except Exception, e:
        print("Could not find\n%s\n\tin\n%s\n\trelative to\n%s"
              % (space_name, filename, os.getcwd()))
        import traceback
        print traceback.format_exc()
        sys.exit(1)

    search_space = space.space

    expr = base.as_apply(search_space)

    hps = {}
    expr_to_config(expr, (True,), hps)
    new_space = ""
    # print expr
    for label, dct in hps.items():
        if dct['node'].name == "randint":
            assert len(dct['node'].inputs()) == 1
            #randint['x', 5] -> x [0, 4]i [0]
            upper = dct['node'].inputs()[0].eval()
            new_space += '%s {%s} [0]\n' % \
                  (label, ", ".join(["%s" % (i,) for i in range(upper)]))
        elif dct['node'].name == "categorical":
            # No difference to a randint node
            upper = dct['node'].inputs()[1].eval()
            new_space += '%s {%s} [0]\n' % \
                  (label, ", ".join(["%s" % (i,) for i in range(upper)]))
        elif dct['node'].name == "uniform":
            assert len(dct['node'].inputs()) == 2
            lower = dct['node'].inputs()[0].eval()
            upper = dct['node'].inputs()[1].eval()
            default = (lower+upper)/2.0
            new_space += '%s [%s, %s] [%s]\n' % (label, lower, upper, default)
        elif dct['node'].name == "quniform":
            # Assumption: q-value is always the last value
            assert len(dct['node'].inputs()) == 3
            lower = dct['node'].inputs()[0].eval()
            upper = dct['node'].inputs()[1].eval()
            q = dct['node'].inputs()[2].eval()
            default = (lower+upper)/2.0
            label = "Q%s_%s" % (q, label)
            new_space += '%s [%s, %s] [%s]\n' % (label, lower, upper, default)
        elif dct['node'].name == "loguniform":
            assert len(dct['node'].inputs()) == 2
            lower = dct['node'].inputs()[0].eval()
            upper = dct['node'].inputs()[1].eval()
            default = (lower+upper)/2.0
            label = "LOG_%s" % (label,)
            new_space += '%s [%s, %s] [%s]\n' % (label, lower, upper, default)
        elif dct['node'].name == "qloguniform":
            assert len(dct['node'].inputs()) == 3
            lower = dct['node'].inputs()[0].eval()
            upper = dct['node'].inputs()[1].eval()
            q = dct['node'].inputs()[2].eval()
            default = (lower+upper)/2.0
            label = "LOG_Q%s_%s" % (q, label)
            new_space += '%s [%s, %s] [%s]\n' % (label, lower, upper, default)
        elif dct['node'].name == "normal":
            assert len(dct['node'].inputs()) == 2
            mu = dct['node'].inputs()[0].eval()
            sigma = dct['node'].inputs()[1].eval()
            lower = mu-3*sigma
            upper = mu+3*sigma
            new_space += '%s [%s, %s] [%s]\n' % (label, lower, upper, mu)
        elif dct['node'].name == "qnormal":
            assert len(dct['node'].inputs()) == 3
            mu = dct['node'].inputs()[0].eval()
            sigma = dct['node'].inputs()[1].eval()
            lower = mu-3*sigma
            upper = mu+3*sigma
            q = dct['node'].inputs()[2].eval()
            label = "Q%s_%s" % (q, label)
            new_space += '%s [%s, %s] [%s]\n' % (label, lower, upper, mu)
        elif dct['node'].name == "lognormal":
            assert len(dct['node'].inputs()) == 2
            mu = dct['node'].inputs()[0].eval()
            sigma = dct['node'].inputs()[1].eval()
            lower = mu-3*sigma
            upper = mu+3*sigma
            label = "LOG_%s" % (label, )
            new_space += '%s [%s, %s] [%s]\n' % (label, lower, upper, mu)
        elif dct['node'].name == "qlognormal":
            assert len(dct['node'].inputs()) == 3
            mu = dct['node'].inputs()[0].eval()
            sigma = dct['node'].inputs()[1].eval()
            lower = mu-3*sigma
            upper = mu+3*sigma
            q = dct['node'].inputs()[2].eval()   
            label = "Q%s_LOG_%s" % (q, label)
            new_space += '%s [%s, %s] [%s]\n' % (label, lower, upper, mu)
        else:
            raise Exception("Node name %s not known" % dct['node'].name)

        # Now take care about conditions
        condict = dict()
        param_keys = list()
        # We allow only one varying param
        varying_param_name = ""
        varying_param = set()
        if dct['conditions']:
            # print "AAA", dct['conditions']
            for condseq in dct['conditions']:
                # print '##', label, condseq
                if len(condseq) == 1 and condseq[0] is True:
                    # Only true as a condition
                    continue
                else:
                    if len(condict.keys()) == 0:
                        # Once collect all keys
                        for condition in condseq[1:]:
                            param_keys.append(condition.name)
                            condict[condition.name] = condition.val
                    else:
                        # Now insert values
                        assert(param_keys.count(k) == 1 for k in condseq[1:])
                        for condition in condseq[1:]:
                            if condict[condition.name] != condition.val:
                                # We found a varying parameter, it is
                                if varying_param_name == condition.name:
                                    # either the one we already knew
                                    varying_param.add(condition.val)
                                elif varying_param_name == "":
                                    # or the first one
                                    varying_param_name = condition.name
                                    varying_param.add(condict[condition.name])
                                    varying_param.add(condition.val)
                                else:
                                    # or we cannot handle this
                                    raise Exception("This is not possible to handle:\n%s", (dct['conditions']))
        # Start printing conditions
        # print "CCC", varying_param, varying_param_name
        if varying_param_name != "":
            new_space += '%s | %s in {%s}\n' % (label, varying_param_name, ",".join(str(i) for i in varying_param))
        for key in condict.keys():
            if key != varying_param_name:
                new_space += '%s | %s in {%s}\n' % (label, key, condict[key])
    return new_space