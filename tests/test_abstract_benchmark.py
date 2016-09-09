import sys
import unittest

if sys.version_info[0] >= 3:
    import unittest.mock as mock
else:
    import mock

import ConfigSpace

import hpolib.abstract_benchmark as abstract_benchmark


class Benchmark(abstract_benchmark.AbstractBenchmark):
    def objective_function(self, configuration):
        pass

    def objective_function_test(self, configuration):
        pass

    def get_configuration_space(cls):
        pass


class TestAbstractBenchmark(unittest.TestCase):

    @mock.patch.object(Benchmark, 'objective_function', autospec=True)
    def test_objective_function(self, benchmark_mock):
        sentinel = 'Sentinel'
        benchmark_mock.return_value = sentinel
        benchmark = Benchmark()
        rval = benchmark.objective_function({})
        self.assertEqual(rval, sentinel)
        self.assertEqual(benchmark_mock.call_count, 1)

    @mock.patch.object(Benchmark, 'objective_function_test', autospec=True)
    def test_objective_function_test(self, benchmark_mock):
        sentinel = 'Sentinel'
        benchmark_mock.return_value = sentinel
        benchmark = Benchmark()
        rval = benchmark.objective_function_test({})
        self.assertEqual(rval, sentinel)
        self.assertEqual(benchmark_mock.call_count, 1)

    @mock.patch.object(ConfigSpace.ConfigurationSpace, 'sample_configuration',
                       autospec=True)
    @mock.patch.object(Benchmark, 'objective_function', autospec=True)
    @mock.patch.object(Benchmark, 'objective_function_test', autospec=True)
    def test_test_function(self, test_mock, benchmark_mock, cs_mock):
        benchmark = Benchmark()
        cs_mock.return_value = dict()
        cs = ConfigSpace.ConfigurationSpace()
        benchmark.configuration_space = cs
        benchmark.test(10)
        self.assertEqual(cs_mock.call_count, 10)
        self.assertEqual(benchmark_mock.call_count, 10)
        self.assertEqual(test_mock.call_count, 10)



