import StringIO
import unittest

import HPOlib.format_converter.configuration_space as configuration_space
import HPOlib.format_converter.pyll_parser as pyll_parser


a = configuration_space.create_float("a", 0, 1)
cond_a = configuration_space.create_float('cond_a', 0, 1,
                                          conditions=[['a_or_b == a']])
a_int = configuration_space.create_int("a_int", 0, 1)
b = configuration_space.create_float("b", 0, 3, q=0.1)
cond_b = configuration_space.create_float('cond_b', 0, 3, q=0.1,
                                          conditions=[['a_or_b == b']])
b_int_1 = configuration_space.create_int("b_int", 0, 3, q=1.0)
b_int_2 = configuration_space.create_int("b_int", 0, 3, q=2.0)
c = configuration_space.create_float("c", 0, 1, base="LOG")
c_int = configuration_space.create_float("c_int", 0, 1, base="LOG")
d = configuration_space.create_float("d", 0, 3, q=0.1, base="LOG")
d_int_1 = configuration_space.create_int("d_int", 0, 3, q=1.0, base="LOG")
d_int_2 = configuration_space.create_int("d_int", 0, 3, q=2.0, base="LOG")
e = configuration_space.create_float("e", 0, 5,
                                     conditions=[['a_or_b in {a,b}']])
a_or_b = configuration_space.create_categorical("a_or_b", ["a", "b"])

conditional_space = {"a_or_b": a_or_b, "cond_a": cond_a, "cond_b": cond_b}
conditional_space_operator_in = {"a_or_b": a_or_b, "cond_a": cond_a,
                                 "cond_b": cond_b, "e": e}

# More complex search space
classifier = configuration_space.create_categorical("classifier", ["svm", "nn"])
kernel = configuration_space.create_categorical("kernel", ["rbf", "linear"],
    conditions=[["classifier == svm"]])
C = configuration_space.create_float("C", 0.03125, 32768, base=2,
    conditions=[["classifier == svm"]])
gamma = configuration_space.create_float("gamma", 0.000030518, 8, base=2,
    conditions=[["kernel == rbf"]])
neurons = configuration_space.create_int("neurons", 16, 1024, q=16,
    conditions=[["classifier == nn"]])
lr = configuration_space.create_float("lr", 0.0001, 1.0,
    conditions=[["classifier == nn"]])
preprocessing = configuration_space.create_categorical("preprocessing",
                                                       [None, "pca"])
config_space = {"classifier": classifier,
                "kernel": kernel,
                "C": C,
                "gamma": gamma,
                "neurons": neurons,
                "lr": lr,
                "preprocessing": preprocessing}

# A search space where a hyperparameter depends on two others:
gamma_2 = configuration_space.create_float("gamma_2", 0.000030518, 8, base=2,
    conditions=[["kernel == rbf", "classifier == svm"]])

config_space_2 = {"classifier": classifier,
                "kernel": kernel,
                "C": C,
                "gamma_2": gamma_2,
                "neurons": neurons,
                "lr": lr,
                "preprocessing": preprocessing}


class TestPyllUtil(unittest.TestCase):
    def test_convert_configuration_space(self):
        expected = StringIO.StringIO()
        expected.write('from hyperopt import hp\nimport hyperopt.pyll as pyll')
        expected.write('\n\n')
        expected.write('a = hp.uniform("a", 0.0, 1.0)\n')
        expected.write('b = hp.quniform("b", 0.0, 3.0, 0.1)\n\n')
        expected.write('space = {"a": a, "b": b}\n')
        simple_space = {"a": a, "b": b}
        cs = pyll_parser.write_configuration_space(simple_space)
        self.assertEqual(expected.getvalue(), cs)

    def test_convert_conditional_space(self):
        cs = pyll_parser.write_configuration_space(conditional_space)
        expected = StringIO.StringIO()
        expected.write('from hyperopt import hp\nimport hyperopt.pyll as pyll')
        expected.write('\n\n')
        expected.write('cond_a = hp.uniform("cond_a", 0.0, 1.0)\n')
        expected.write('cond_b = hp.quniform("cond_b", 0.0, 3.0, 0.1)\n')
        expected.write('a_or_b = hp.choice("a_or_b", [\n')
        expected.write('    {"a_or_b": "a", "cond_a": cond_a, },\n')
        expected.write('    {"a_or_b": "b", "cond_b": cond_b, },\n')
        expected.write('    ])\n\n')
        expected.write('space = {"a_or_b": a_or_b}\n')
        self.assertEqual(expected.getvalue(), cs)

    def test_convert_complex_space(self):
        cs = pyll_parser.write_configuration_space(config_space)
        expected = StringIO.StringIO()
        expected.write('from hyperopt import hp\nimport hyperopt.pyll as pyll')
        expected.write('\n\n')
        expected.write('LOG2_C = hp.uniform("LOG2_C", -5.0, 15.0)\n')
        expected.write('LOG2_gamma = hp.uniform("LOG2_gamma", -14.9999800563, 3.0)\n')
        expected.write('kernel = hp.choice("kernel", [\n')
        expected.write('    {"kernel": "linear", },\n')
        expected.write('    {"kernel": "rbf", "LOG2_gamma": LOG2_gamma, },\n')
        expected.write('    ])\n')
        expected.write('lr = hp.uniform("lr", 0.0001, 1.0)\n')
        expected.write('neurons = pyll.scope.int(hp.quniform("neurons", 16.0, 1024.0, 16.0))\n')
        expected.write('classifier = hp.choice("classifier", [\n')
        expected.write('    {"classifier": "nn", "lr": lr, "neurons": neurons, },\n')
        expected.write('    {"classifier": "svm", "LOG2_C": LOG2_C, "kernel": kernel, },\n')
        expected.write('    ])\n')
        expected.write('preprocessing = hp.choice("preprocessing", [\n')
        expected.write('    {"preprocessing": "None", },\n')
        expected.write('    {"preprocessing": "pca", },\n')
        expected.write('    ])\n\n')
        expected.write('space = {"classifier": classifier, "preprocessing": preprocessing}\n')
        self.assertEqual(expected.getvalue(), cs)

        expected.seek(0)
        cs = pyll_parser.write_configuration_space(config_space_2)
        self.assertEqual(expected.getvalue().replace("gamma", "gamma_2"), cs)

    def test_operator_in(self):
        cs = pyll_parser.write_configuration_space(conditional_space_operator_in)
        expected = StringIO.StringIO()
        expected.write('from hyperopt import hp\nimport hyperopt.pyll as pyll')
        expected.write('\n\n')
        expected.write('cond_a = hp.uniform("cond_a", 0.0, 1.0)\n')
        expected.write('cond_b = hp.quniform("cond_b", 0.0, 3.0, 0.1)\n')
        expected.write('e = hp.uniform("e", 0.0, 5.0)\n')
        expected.write('a_or_b = hp.choice("a_or_b", [\n')
        expected.write('    {"a_or_b": "a", "cond_a": cond_a, "e": e, },\n')
        expected.write('    {"a_or_b": "b", "cond_b": cond_b, "e": e, },\n')
        expected.write('    ])\n\n')
        expected.write('space = {"a_or_b": a_or_b}\n')
        self.assertEqual(expected.getvalue(), cs)

    def test_write_uniform(self):
        expected = ('a', 'a = hp.uniform("a", 0.0, 1.0)')
        value = pyll_parser.write_uniform(a)
        self.assertEqual(expected, value)

    def test_write_uniform_int(self):
        expected = ('a_int', 'a_int = pyll.scope.int(hp.uniform('
                             '"a_int", 0.0, 1.0, 1.0))')
        value = pyll_parser.write_uniform_int(a_int)
        self.assertEqual(expected, value)

    def test_write_quniform(self):
        expected = ("b", 'b = hp.quniform("b", 0.0, 3.0, 0.1)')
        value = pyll_parser.write_quniform(b)
        self.assertEqual(expected, value)

    def test_write_quniform_int(self):
        expected = ("b_int", 'b_int = pyll.scope.int(hp.quniform('
                    '"b_int", 0.0, 3.0, 1.0))')
        value = pyll_parser.write_quniform_int(b_int_1)
        self.assertEqual(expected, value)

        expected = ("b_int", 'b_int = pyll.scope.int(hp.quniform('
                    '"b_int", 0.0, 3.0, 2.0))')
        value = pyll_parser.write_quniform_int(b_int_2)
        self.assertEqual(expected, value)

    def test_write_loguniform(self):
        expected = ("c", 'c = hp.loguniform("c", 0.0, 1.0)')
        value = pyll_parser.write_loguniform(c)
        self.assertEqual(expected, value)

    def test_write_loguniform_int(self):
        expected = ("c_int", 'c_int = pyll.scope.int(hp.qloguniform('
                             '"c_int", 0.0, 1.0, 1.0))')
        value = pyll_parser.write_loguniform_int(c_int)
        self.assertEqual(expected, value)

    def test_write_qloguniform(self):
        expected = ("d", 'd = hp.qloguniform("d", 0.0, 3.0, 0.1)')
        value = pyll_parser.write_qloguniform(d)
        self.assertEqual(expected, value)

    def test_write_qloguniform_int(self):
        expected = ("d_int", 'd_int = pyll.scope.int(hp.qloguniform('
                    '"d_int", 0.0, 3.0, 1.0))')
        value = pyll_parser.write_qloguniform_int(d_int_1)
        self.assertEqual(expected, value)

        expected = ("d_int", 'd_int = pyll.scope.int(hp.qloguniform('
                    '"d_int", 0.0, 3.0, 2.0))')
        value = pyll_parser.write_qloguniform_int(d_int_2)
        self.assertEqual(expected, value)

    """
    def test_write_normal(self):
        parameter = configuration_space.create_int("d", 0, 1)
        expected = 'hp.normal("e", 0, 1)'
        value = pyll_parser.write_normal(parameter)
        self.assertEqual(expected, value)

    def test_write_qnormal(self):
        expected = 'hp.qnormal("f", 0, 1, 0.1)'
        value = pyll_parser.write_normal(parameter)
        self.assertEqual(expected, value)

    def test_write_qnormal_int(self):
        expected = 'pyll.scope.int(hp.qnormal("f_int", 0, 1, 1))'
        value = pyll_parser.write_normal_int(parameter)
        self.assertEqual(expected, value)

    def test_write_qlognormal(self):
        expected = 'hp.qlognormal("g", 0, 1, 0.1)'
        value = pyll_parser.write_qlognormal(parameter)
        self.assertEqual(expected, value)

    def test_write_qlognormal_int(self):
        expected = 'pyll.scope.int(hp.qlognormal("g_int", 0, 1, 1))'
        value = pyll_parser.write_qlognormal_int(parameter)
        self.assertEqual(expected, value)

    def test_write_swith(self):
        expected = 'hp.switch("h", [0, 1])'
        value = pyll_parser.write_switch(parameter)
        self.assertEqual(expected, value)
    """