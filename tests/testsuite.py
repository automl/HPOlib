import unittest

import unittests.test_benchmark_util as test_benchmark_util
import unittests.test_configuration_space as test_configuration_space
import unittests.test_optimization_interceptor as test_optimization_interceptor
import unittests.test_data_utils as test_data_utils
import unittests.test_dispatcher as test_dispatcher
import unittests.test_experiment as test_experiment
import unittests.test_pb_converter as test_pb_converter
import unittests.test_pcs_converter as test_pcs_converter
import unittests.test_plot_util as test_plot_util
import unittests.test_pyll_util as test_pyll_util
import unittests.test_runsolver_wrapper as test_runsolver_wrapper
import unittests.test_wrapping as test_wrapping
import unittests.test_wrapping_util as test_wrapping_util

import workflowtests.test_optimizer as test_optimizer


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(unittest.makeSuite(test_benchmark_util.BenchmarkUtilTest))
    _suite.addTest(unittest.makeSuite(test_configuration_space.TestConfigurationSpace))
    _suite.addTest(unittest.makeSuite(test_optimization_interceptor.OptimizationInterceptorTest))
    _suite.addTest(unittest.makeSuite(test_data_utils.DataUtilTest))
    _suite.addTest(unittest.makeSuite(test_dispatcher.DispatcherTest))
    _suite.addTest(unittest.makeSuite(test_experiment.ExperimentTest))
    _suite.addTest(unittest.makeSuite(test_pb_converter.TestPbConverter))
    _suite.addTest(unittest.makeSuite(test_pcs_converter.TestPCSConverter))
    _suite.addTest(unittest.makeSuite(test_plot_util.PlotUtilTest))
    _suite.addTest(unittest.makeSuite(test_pyll_util.TestPyllReader))
    _suite.addTest(unittest.makeSuite(test_pyll_util.TestPyllWriter))
    _suite.addTest(unittest.makeSuite(test_runsolver_wrapper.RunsolverWrapperTest))
    _suite.addTest(unittest.makeSuite(test_wrapping.WrappingTest))
    _suite.addTest(unittest.makeSuite(test_wrapping_util.WrappingTestUtil))

    _suite.addTest(unittest.makeSuite(test_optimizer.TestOptimizers))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    test_suite = suite()
    runner.run(suite())

