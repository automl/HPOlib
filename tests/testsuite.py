import unittest

import unittests.test_benchmark_util as test_benchmark_util
# import unittests.test_cv as test_cv
import unittests.test_data_utils as test_data_utils
import unittests.test_experiment as test_experiment
#import unittests.test_gridsearch as test_gridsearch
import unittests.test_runsolver_wrapper as test_runsolver_wrapper
import unittests.test_wrapping as test_wrapping
import unittests.test_wrapping_util as test_wrapping_util


def suite():
    _suite = unittest.TestSuite()
    _suite.addTest(unittest.makeSuite(test_benchmark_util.BenchmarkUtilTest))
    # _suite.addTest(unittest.makeSuite(test_cv.CVTest))
    _suite.addTest(unittest.makeSuite(test_data_utils.DataUtilTest))
    _suite.addTest(unittest.makeSuite(test_experiment.ExperimentTest))
    #_suite.addTest(unittest.makeSuite(test_gridsearch.GridSearchTest))
    _suite.addTest(unittest.makeSuite(test_runsolver_wrapper.RunsolverWrapperTest))
    _suite.addTest(unittest.makeSuite(test_wrapping.WrappingTest))
    _suite.addTest(unittest.makeSuite(test_wrapping_util.WrappingTestUtil))
    return _suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = suite()
    runner.run(suite())

