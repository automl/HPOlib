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

import StringIO
import unittest

import sys
print sys.path

import HPOlib.format_converter.configuration_space as configuration_space
import HPOlib.format_converter.pcs_parser as pcs_parser

# More complex search space
classifier = configuration_space.CategoricalHyperparameter("classifier", ["svm", "nn"])
kernel = configuration_space.CategoricalHyperparameter("kernel", ["rbf", "linear"],
                                                conditions=[["classifier == svm", ]])
C = configuration_space.UniformFloatHyperparameter("C", 0.03125, 32768,  # base=2,
                                     conditions=[["classifier == svm", ]])
gamma = configuration_space.UniformFloatHyperparameter("gamma", 0.000030518, 8,  # base=2,
                                         conditions=[["kernel == rbf", "classifier == svm"]])
neurons = configuration_space.UniformIntegerHyperparameter("neurons", 16, 1024,  # q=16,
                                         conditions=[["classifier == nn", ]])
lr = configuration_space.UniformFloatHyperparameter("lr", 0.0001, 1.0,
                                      conditions=[["classifier == nn", ]])
preprocessing = configuration_space.CategoricalHyperparameter("preprocessing",
                                                       ["None", "pca"])
conditional_space = {"classifier": classifier,
                     "kernel": kernel,
                     "C": C,
                     "gamma": gamma,
                     "neurons": neurons,
                     "lr": lr,
                     "preprocessing": preprocessing}

float_a = configuration_space.UniformFloatHyperparameter("float_a", -1.23, 6.45)
e_float_a = configuration_space.UniformFloatHyperparameter("e_float_a", .5E-2, 4.3e+6)
int_a = configuration_space.UniformIntegerHyperparameter("int_a", -1, 6)
log_a = configuration_space.UniformFloatHyperparameter("log_a", -1.23, 6.45, base=10)
int_log_a = configuration_space.UniformIntegerHyperparameter("int_log_a", 1, 6, base=10)
cat_a = configuration_space.CategoricalHyperparameter("cat_a", ["a", "b", "c", "d"])
crazy = configuration_space.CategoricalHyperparameter("@.:;/\?!$%&_-<>*+1234567890", ["const"])
easy_space = {"float_a": float_a,
              "e_float_a": e_float_a,
              "int_a": int_a,
              "log_a": log_a,
              "int_log_a": int_log_a,
              "cat_a": cat_a,
              "@.:;/\?!$%&_-<>*+1234567890": crazy,
              }


class TestPCSConverter(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_read_configuration_space_basic(self):
        float_a_copy = configuration_space.UniformFloatHyperparameter("float_a", -1.23, 6.45)
        a_copy = {"a": float_a_copy, "b": int_a}
        a_real = {"b": int_a, "a": float_a}
        self.assertDictEqual(a_real, a_copy)

    def test_read_configuration_space_easy(self):
        expected = StringIO.StringIO()
        expected.write('# This is a \n')
        expected.write('   # This is a comment with a leading whitespace ### ffds \n')
        expected.write('\n')
        expected.write('float_a [-1.23, 6.45] [2.61] # bla\n')
        expected.write('e_float_a [.5E-2, 4.3e+6] [2.61]\n')
        expected.write('int_a [-1, 6] [4]i\n')
        expected.write('log_a [-1.23, 6.45] [2.61]l\n')
        expected.write('int_log_a [1, 6] [3]il\n')
        expected.write('cat_a {a,"b",c,d} [a]\n')
        expected.write('@.:;/\?!$%&_-<>*+1234567890 {"const"} ["const"]\n')
        expected.seek(0)
        cs = pcs_parser.read(expected)
        self.assertEqual(cs, easy_space)

    def test_read_configuration_space_conditional(self):
        # More complex search space as string array
        complex_cs = list()
        complex_cs.append("classifier {svm, nn} [svm]")
        complex_cs.append("kernel {rbf, linear} [rbf]")
        complex_cs.append("kernel | classifier in {svm}")
        complex_cs.append("C [0.03125, 32768] [100] # Should be base 2")
        complex_cs.append("C | classifier in {svm}")
        complex_cs.append("gamma [0.000030518, 8] [2] # Should be base=2")
        complex_cs.append("gamma | kernel in {rbf}")
        complex_cs.append("gamma | classifier in {svm}")
        complex_cs.append("neurons [16, 1024] [100]i # Should be Q16")
        complex_cs.append("neurons | classifier in {nn}")
        complex_cs.append("lr [0.0001, 1.0] [0.5]")
        complex_cs.append("lr | classifier in {nn}")
        complex_cs.append("preprocessing {None, pca} [None]")

        cs = pcs_parser.read(complex_cs)
        self.assertEqual(cs, conditional_space)

    def test_write_int(self):
        expected = "int_a [-1, 6] [2]i"
        sp = {"a": int_a}
        value = pcs_parser.write(sp)
        self.assertEqual(expected, value)

    def test_write_log_int(self):
        expected = "int_log_a [1, 6] [2]il"
        sp = {"a": int_log_a}
        value = pcs_parser.write(sp)
        self.assertEqual(expected, value)

    def test_write_q_int(self):
        expected = "Q16_int_a [16, 1024] [520]i"
        sp = {"a": configuration_space.UniformIntegerHyperparameter("int_a", 16, 1024, q=16)}
        value = pcs_parser.write(sp)
        self.assertEqual(expected, value)

    def test_write_q_float(self):
        expected = "Q16_float_a [16.1, 1024.1] [520.1]"
        sp = {"a": configuration_space.UniformFloatHyperparameter("float_a", 16.10, 1024.10, q=16)}
        value = pcs_parser.write(sp)
        self.assertEqual(expected, value)

    def test_write_log23(self):
        expected = "LOG23_a [1.0, 5.0] [3.0]"
        sp = {"a": configuration_space.UniformFloatHyperparameter("a", 23**1, 23**5, base=23)}
        value = pcs_parser.write(sp)
        self.assertEqual(expected, value)

    def test_write_log10(self):
        expected = "a [10.0, 1000.0] [100.0]l"
        sp = {"a": configuration_space.UniformFloatHyperparameter("a", 10, 1000, base=10)}
        value = pcs_parser.write(sp)
        self.assertEqual(expected, value)