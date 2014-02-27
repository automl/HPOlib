import unittest

import test_benchmark_util
import test_cv
import test_experiment
import test_data_utils
import test_runsolver_wrapper
import test_wrapping


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(unittest.makeSuite(test_benchmark_util.BenchmarkUtilTest))
    _suite.addTest(unittest.makeSuite(test_cv.CVTest))
    _suite.addTest(unittest.makeSuite(test_experiment.ExperimentTest))
    _suite.addTest(unittest.makeSuite(test_data_utils.DataUtilTest))
    _suite.addTest(unittest.makeSuite(test_runsolver_wrapper.RunsolverWrapperTest))
    _suite.addTest(unittest.makeSuite(test_wrapping.WrappingTest))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = suite()
    runner.run(suite())

