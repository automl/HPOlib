import sys
import unittest

if sys.version_info[0] >= 3:
    import unittest.mock as mock
else:
    import mock

import ConfigSpace
import numpy as np

import hpolib.continuous_benchmark as continuous_benchmark


class Benchmark(continuous_benchmark.AbstractContinuousBenchmark):
    def objective_function(self, configuration):
        pass

    def objective_function_test(self, configuration):
        pass

    @staticmethod
    def get_lower_and_upper_bounds():
        return [0, 0], [1, 1]


class TestContinuousBenchmark(unittest.TestCase):
    @mock.patch.object(Benchmark, 'objective_function', autospec=True)
    def test_evaluate_array(self, benchmark_mock):
        sentinel = 'Sentinel'
        benchmark_mock.return_value = sentinel
        benchmark = Benchmark()
        rval = benchmark.evaluate_array([])
        self.assertEqual(rval, sentinel)
        self.assertEqual(benchmark_mock.call_count, 1)

    @mock.patch.object(Benchmark, 'objective_function_test', autospec=True)
    def test_evaluate_array_test(self, benchmark_mock):
        sentinel = 'Sentinel'
        benchmark_mock.return_value = sentinel
        benchmark = Benchmark()
        rval = benchmark.evaluate_array_test([])
        self.assertEqual(rval, sentinel)
        self.assertEqual(benchmark_mock.call_count, 1)

    @mock.patch.object(Benchmark, 'objective_function', autospec=True)
    @mock.patch.object(Benchmark, '_convert_dict_to_array', autospec=True)
    def test_evaluate_dict(self, convert_mock, benchmark_mock):
        sentinel = 'Sentinel'
        benchmark_mock.return_value = sentinel
        convert_mock.return_value = []
        benchmark = Benchmark()
        rval = benchmark.evaluate_dict({})
        self.assertEqual(rval, sentinel)
        self.assertEqual(benchmark_mock.call_count, 1)
        self.assertEqual(convert_mock.call_count, 1)
        self.assertEqual(convert_mock.call_args[0][1], {})

    @mock.patch.object(Benchmark, 'objective_function_test', autospec=True)
    @mock.patch.object(Benchmark, '_convert_dict_to_array', autospec=True)
    def test_evaluate_dict_test(self, convert_mock, benchmark_mock):
        sentinel = 'Sentinel'
        benchmark_mock.return_value = sentinel
        convert_mock.return_value = []
        benchmark = Benchmark()
        rval = benchmark.evaluate_dict_test({})
        self.assertEqual(rval, sentinel)
        self.assertEqual(benchmark_mock.call_count, 1)
        self.assertEqual(convert_mock.call_count, 1)
        self.assertEqual(convert_mock.call_args[0][1], {})

    def test__convert_dict_to_array(self):
        benchmark = Benchmark()
        configuration = {}
        self.assertRaisesRegex(
            ValueError, 'Configuration should have 2 elements, but has 0!',
            benchmark._convert_dict_to_array, configuration)

        configuration = {'X0': 0.1, 'X1': 0.2}
        rval = benchmark._convert_dict_to_array(configuration)
        self.assertIsInstance(rval, np.ndarray)
        np.testing.assert_allclose(rval, [0.1, 0.2])

    def test_get_configuration_space(self):
        fixture = ConfigSpace.ConfigurationSpace()
        fixture.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(
            'X0', 0, 1))
        fixture.add_hyperparameter(ConfigSpace.UniformFloatHyperparameter(
            'X1', 0, 1))

        benchmark = Benchmark()
        cs = benchmark.get_configuration_space()
        print(cs)
        print(fixture)

        self.assertEqual(cs, fixture)
