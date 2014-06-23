import StringIO
import unittest

import sys
print sys.path

import hyperopt
from hyperopt import hp
import numpy as np

import HPOlib.format_converter.configuration_space as configuration_space
import HPOlib.format_converter.pyll_parser as pyll_parser



a = configuration_space.UniformFloatHyperparameter("a", 0, 1)
cond_a = configuration_space.UniformFloatHyperparameter(
    'cond_a', 0, 1, conditions=[['a_or_b == a']])
a_int = configuration_space.UniformIntegerHyperparameter("a_int", 0, 1)
b = configuration_space.UniformFloatHyperparameter("b", 0, 3, q=0.1)
cond_b = configuration_space.UniformFloatHyperparameter(
    'cond_b', 0, 3, q=0.1, conditions=[['a_or_b == b']])
b_int_1 = configuration_space.UniformIntegerHyperparameter("b_int", 0, 3, q=1.0)
b_int_2 = configuration_space.UniformIntegerHyperparameter("b_int", 0, 3, q=2.0)
c = configuration_space.UniformFloatHyperparameter("c", 0, 1, base=np.e)
c_int = configuration_space.UniformFloatHyperparameter("c_int", 0, 1, base=np.e)
d = configuration_space.UniformFloatHyperparameter("d", 0, 3, q=0.1, base=np.e)
d_int_1 = configuration_space.UniformIntegerHyperparameter("d_int", 0, 3, q=1.0, base=np.e)
d_int_2 = configuration_space.UniformIntegerHyperparameter("d_int", 0, 3, q=2.0, base=np.e)
e = configuration_space.UniformFloatHyperparameter("e", 0, 5,
                                     conditions=[['a_or_b in {a,b}']])
a_or_b = configuration_space.CategoricalHyperparameter("a_or_b", ["a", "b"])

conditional_space = {"a_or_b": a_or_b, "cond_a": cond_a, "cond_b": cond_b}
conditional_space_operator_in = {"a_or_b": a_or_b, "cond_a": cond_a,
                                 "cond_b": cond_b, "e": e}

# More complex search space
classifier = configuration_space.CategoricalHyperparameter("classifier", ["svm", "nn"])
kernel = configuration_space.CategoricalHyperparameter("kernel", ["rbf", "linear"],
    conditions=[["classifier == svm"]])
C = configuration_space.UniformFloatHyperparameter("C", 0.03125, 32768, base=2,
    conditions=[["classifier == svm"]])
gamma = configuration_space.UniformFloatHyperparameter("gamma", 0.000030518, 8, base=2,
    conditions=[["kernel == rbf"]])
neurons = configuration_space.UniformIntegerHyperparameter("neurons", 16, 1024, q=16,
    conditions=[["classifier == nn"]])
lr = configuration_space.UniformFloatHyperparameter("lr", 0.0001, 1.0,
    conditions=[["classifier == nn"]])
preprocessing = configuration_space.CategoricalHyperparameter("preprocessing",
                                                       [None, "pca"])
config_space = {"classifier": classifier,
                "kernel": kernel,
                "C": C,
                "gamma": gamma,
                "neurons": neurons,
                "lr": lr,
                "preprocessing": preprocessing}

# A search space where a hyperparameter depends on two others:
gamma_2 = configuration_space.UniformFloatHyperparameter("gamma_2", 0.000030518, 8, base=2,
    conditions=[["kernel == rbf", "classifier == svm"]])

config_space_2 = {"classifier": classifier,
                "kernel": kernel,
                "C": C,
                "gamma_2": gamma_2,
                "neurons": neurons,
                "lr": lr,
                "preprocessing": preprocessing}


class TestPyllReader(unittest.TestCase):
    def test_read_literal(self):
        pyll_parser.read_literal()

    def test_read_container(self):
        pyll_parser.read_container()

    def test_read_switch(self):
        pyll_parser.read_switch()

    # TODO: duplicate these tests for Integer
    def test_read_uniform(self):
        # 0 float
        # 1   hyperopt_param
        # 2     Literal{scale_mult1}
        # 3     uniform
        # 4       Literal{0.2}
        # 5       Literal{2}
        uniform = hp.uniform('scale_mult1', .2, 2).inputs()[0].inputs()[1]
        ret = pyll_parser.read_uniform(uniform, 'scale_mult1')
        expected = configuration_space.UniformFloatHyperparameter('scale_mult1', 0.2, 2)
        self.assertEqual(expected, ret)

    def test_read_loguniform(self):
        # 0 float
        # 1   hyperopt_param
        # 2     Literal{colnorm_thresh}
        # 3     loguniform
        # 4       Literal{-20.7232658369}
        # 5       Literal{-6.90775527898}
        loguniform = hp.loguniform('colnorm_thresh', np.log(1e-9),
            np.log(1e-3)).inputs()[0].inputs()[1]
        ret = pyll_parser.read_loguniform(loguniform, 'colnorm_thresh')
        expected = configuration_space.UniformFloatHyperparameter(
            'colnorm_thresh', -20.7232658369, -6.90775527898, base=np.e)
        self.assertEqual(expected, ret)

    def test_read_quniform(self):
        # 0 float
        # 1   hyperopt_param
        # 2     Literal{l0eg_fsize}
        # 3     quniform
        # 4       Literal{2.50001}
        # 5       Literal{8.5}
        # 6       Literal{1}
        quniform = hp.quniform('l0eg_fsize', 2.50001, 8.5, 1). \
            inputs()[0].inputs()[1]
        ret = pyll_parser.read_quniform(quniform, 'l0eg_fsize')
        expected = configuration_space.UniformFloatHyperparameter(
            'l0eg_fsize', 2.50001, 8.5, q=1)
        self.assertEqual(expected, ret)

    def test_read_qloguniform(self):
        # 0 float
        # 1   hyperopt_param
        # 2     Literal{nhid1}
        # 3     qloguniform
        # 4       Literal{2.77258872224}
        # 5       Literal{6.9314718056}
        # 6      q =
        # 7       Literal{16}
        qloguniform = hp.qloguniform('nhid1', np.log(16), np.log(1024), q=16). \
            inputs()[0].inputs()[1]
        ret = pyll_parser.read_qloguniform(qloguniform, 'nhid1')
        expected = configuration_space.UniformFloatHyperparameter(
            'nhid1', np.log(16), np.log(1024), q=16, base=np.e)
        self.assertEqual(expected, ret)

    def test_read_normal(self):
        # 0 float
        # 1   hyperopt_param
        # 2     Literal{l0eg_alpha}
        # 3     normal
        # 4       Literal{0.0}
        # 5       Literal{1.0}
        normal = hp.normal("l0eg_alpha", 0.0, 1.0).inputs()[0].inputs()[1]
        ret = pyll_parser.read_normal(normal, "l0eg_alpha")
        expected = configuration_space.NormalFloatHyperparameter(
            "l0eg_alpha", 0.0, 1.0)
        self.assertEqual(expected, ret)


    def test_read_lognormal(self):
        # 0 float
        # 1   hyperopt_param
        # 2     Literal{lr}
        # 3     lognormal
        # 4       Literal{-4.60517018599}
        # 5       Literal{3.0}
        lognormal = hp.lognormal('lr', np.log(.01), 3.).inputs()[0].inputs()[1]
        ret = pyll_parser.read_lognormal(lognormal, "lr")
        expected = configuration_space.NormalFloatHyperparameter(
            "lr", np.log(0.01), 3.0, base=np.e)
        self.assertEqual(expected, ret)

    def test_read_qnormal(self):
        # 0 float
        # 1   hyperopt_param
        # 2     Literal{qnormal}
        # 3     qnormal
        # 4       Literal{0.0}
        # 5       Literal{1.0}
        # 6       Literal{0.5}
        qnormal = hp.qnormal("qnormal", 0.0, 1.0, 0.5).inputs()[0].inputs()[1]
        ret = pyll_parser.read_qnormal(qnormal, "qnormal")
        expected = configuration_space.NormalFloatHyperparameter(
            "qnormal", 0.0, 1.0, q=0.5)
        self.assertEqual(expected, ret)

    def test_read_qlognormal(self):
        # 0 float
        # 1   hyperopt_param
        # 2     Literal{qlognormal}
        # 3     qlognormal
        # 4       Literal{0.0}
        # 5       Literal{1.0}
        # 6       Literal{0.5}
        qlognormal = hp.qlognormal("qlognormal", 0.0, 1.0, 0.5).inputs()[0].inputs()[1]
        ret = pyll_parser.read_qlognormal(qlognormal, "qlognormal")
        expected = configuration_space.NormalFloatHyperparameter(
            "l0eg_alpha", 0.0, 1.0, q=0.5, base=np.e)
        self.assertEqual(expected, ret)


class TestPyllWriter(unittest.TestCase):
    def setUp(self):
        self.pyll_writer = pyll_parser.PyllWriter()

    def test_convert_configuration_space(self):
        expected = StringIO.StringIO()
        expected.write('from hyperopt import hp\nimport hyperopt.pyll as pyll')
        expected.write('\n\n')
        expected.write('a = hp.uniform("a", 0.0, 1.0)\n')
        expected.write('b = hp.quniform("b", 0.0, 3.0, 0.1)\n\n')
        expected.write('space = {"a": a, "b": b}\n')
        simple_space = {"a": a, "b": b}
        cs = self.pyll_writer.write(simple_space)
        self.assertEqual(expected.getvalue(), cs)

    def test_convert_conditional_space(self):
        cs = self.pyll_writer.write(conditional_space)
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
        cs = self.pyll_writer.write(config_space)
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
        cs = self.pyll_writer.write(config_space_2)
        self.assertEqual(expected.getvalue().replace("gamma", "gamma_2"), cs)

    def test_operator_in(self):
        cs = self.pyll_writer.write(conditional_space_operator_in)
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
        value = self.pyll_writer.write_uniform(a)
        self.assertEqual(expected, value)

    def test_write_uniform_int(self):
        expected = ('a_int', 'a_int = pyll.scope.int(hp.uniform('
                             '"a_int", 0.0, 1.0, 1.0))')
        value = self.pyll_writer.write_uniform_int(a_int)
        self.assertEqual(expected, value)

    def test_write_quniform(self):
        expected = ("b", 'b = hp.quniform("b", 0.0, 3.0, 0.1)')
        value = self.pyll_writer.write_quniform(b)
        self.assertEqual(expected, value)

    def test_write_quniform_int(self):
        expected = ("b_int", 'b_int = pyll.scope.int(hp.quniform('
                    '"b_int", 0.0, 3.0, 1.0))')
        value = self.pyll_writer.write_quniform_int(b_int_1)
        self.assertEqual(expected, value)

        expected = ("b_int", 'b_int = pyll.scope.int(hp.quniform('
                    '"b_int", 0.0, 3.0, 2.0))')
        value = self.pyll_writer.write_quniform_int(b_int_2)
        self.assertEqual(expected, value)

    def test_write_loguniform(self):
        expected = ("c", 'c = hp.loguniform("c", 0.0, 1.0)')
        value = self.pyll_writer.write_loguniform(c)
        self.assertEqual(expected, value)

    def test_write_loguniform_int(self):
        expected = ("c_int", 'c_int = pyll.scope.int(hp.qloguniform('
                             '"c_int", 0.0, 1.0, 1.0))')
        value = self.pyll_writer.write_loguniform_int(c_int)
        self.assertEqual(expected, value)

    def test_write_qloguniform(self):
        expected = ("d", 'd = hp.qloguniform("d", 0.0, 3.0, 0.1)')
        value = self.pyll_writer.write_qloguniform(d)
        self.assertEqual(expected, value)

    def test_write_qloguniform_int(self):
        expected = ("d_int", 'd_int = pyll.scope.int(hp.qloguniform('
                    '"d_int", 0.0, 3.0, 1.0))')
        value = self.pyll_writer.write_qloguniform_int(d_int_1)
        self.assertEqual(expected, value)

        expected = ("d_int", 'd_int = pyll.scope.int(hp.qloguniform('
                    '"d_int", 0.0, 3.0, 2.0))')
        value = self.pyll_writer.write_qloguniform_int(d_int_2)
        self.assertEqual(expected, value)

    """
    def test_write_normal(self):
        parameter = configuration_space.create_int("d", 0, 1)
        expected = 'hp.normal("e", 0, 1)'
        value = self.pyll_writer.write_normal(parameter)
        self.assertEqual(expected, value)

    def test_write_qnormal(self):
        expected = 'hp.qnormal("f", 0, 1, 0.1)'
        value = self.pyll_writer.write_normal(parameter)
        self.assertEqual(expected, value)

    def test_write_qnormal_int(self):
        expected = 'pyll.scope.int(hp.qnormal("f_int", 0, 1, 1))'
        value = self.pyll_writer.write_normal_int(parameter)
        self.assertEqual(expected, value)

    def test_write_qlognormal(self):
        expected = 'hp.qlognormal("g", 0, 1, 0.1)'
        value = self.pyll_writer.write_qlognormal(parameter)
        self.assertEqual(expected, value)

    def test_write_qlognormal_int(self):
        expected = 'pyll.scope.int(hp.qlognormal("g_int", 0, 1, 1))'
        value = self.pyll_writer.write_qlognormal_int(parameter)
        self.assertEqual(expected, value)

    def test_write_swith(self):
        expected = 'hp.switch("h", [0, 1])'
        value = self.pyll_writer.write_switch(parameter)
        self.assertEqual(expected, value)
    """