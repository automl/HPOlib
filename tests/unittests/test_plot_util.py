import itertools
import unittest

import HPOlib.Plotting.plot_util as plot_util
from tests.unittests.experiments.branin_experiment import experiment as \
    branin_experiment

branin_expected = [24.129964413622268, 59.972610578348807, 16.787197168652479,
                   83.744395406186612, 113.65462688764023, 21.642525044091762,
                   39.455198792154548, 12.779763397329571, 18.437661047241164,
                   133.19805397287476, 2.6962005134978178, 104.23568552914563,
                   34.10506238840658, 7.4015507720119809, 6.6237514433108142,
                   89.527250891904131, 0.72639128754424731, 15.669849917551714,
                   25.994852509724662, 36.496205877869599]


class PlotUtilTest(unittest.TestCase):
    def assertListAlmostEqual(self, expected, actual):
        for element1, element2 in itertools.izip(expected, actual):
            print element1, element2
            self.assertAlmostEqual(element1, element2)

    def test_extract_trajectory(self):
        res = plot_util.extract_trajectory(experiment=branin_experiment,
                                           maxvalue=1000)
        expected_trajectory = ([1000.0] * 1) \
                              + ([24.129964413622268] * 2) \
                              + ([16.787197168652479] * 5) \
                              + ([12.779763397329571] * 3) \
                              + ([2.6962005134978178] * 6) \
                              + ([0.72639128754424731] * 4)
        self.assertListAlmostEqual(expected=expected_trajectory, actual=res)

        res = plot_util.extract_trajectory(branin_experiment, cut=22, maxvalue=1000)
        self.assertListAlmostEqual(expected_trajectory, res)

        res = plot_util.extract_trajectory(branin_experiment, cut=10, maxvalue=1000)
        self.assertListAlmostEqual(expected_trajectory[:10], res)
        self.assertAlmostEqual(12.779763397329571, res[-1])

        self.assertRaises(ValueError, plot_util.extract_trajectory,
                         branin_experiment, 0.5)

        self.assertRaises(ValueError, plot_util.extract_trajectory,
                          branin_experiment, 0)

    def test_extract_results(self):
        res = plot_util.extract_results(branin_experiment)
        self.assertEqual(res, branin_expected)

        res = plot_util.extract_results(branin_experiment, cut=22)
        self.assertEqual(res, branin_expected)

        res = plot_util.extract_results(branin_experiment, cut=10)
        self.assertEqual(len(res), 10)
        self.assertEqual(res, branin_expected[:10])

        self.assertRaises(ValueError, plot_util.extract_results,
                         branin_experiment, 0.5)

        self.assertRaises(ValueError, plot_util.extract_results,
                          branin_experiment, 0)

    def test_get_best(self):
        res = plot_util.get_best(branin_experiment)
        self.assertAlmostEqual(0.72639128754424731, res)

        res = plot_util.get_best(branin_experiment, cut=22)
        self.assertAlmostEqual(0.72639128754424731, res)

        res = plot_util.get_best(branin_experiment, cut=10)
        self.assertAlmostEqual(12.779763397329571, res)

        self.assertRaises(ValueError, plot_util.get_best,
                         branin_experiment, 0.5)

        self.assertRaises(ValueError, plot_util.get_best,
                          branin_experiment, 0)

    def test_get_best_value_and_index(self):
        res, index = plot_util.get_best_value_and_index(branin_experiment)
        self.assertAlmostEqual(0.72639128754424731, res)
        self.assertEqual(16, index)

        res, index = plot_util.get_best_value_and_index(branin_experiment, cut=22)
        self.assertAlmostEqual(0.72639128754424731, res)
        self.assertEqual(16, index)

        res, index = plot_util.get_best_value_and_index(branin_experiment, cut=11)
        self.assertAlmostEqual(2.6962005134978178, res)
        self.assertEqual(10, index)

        self.assertRaises(ValueError, plot_util.get_best_value_and_index,
                          branin_experiment, 0.5)

        self.assertRaises(ValueError, plot_util.get_best_value_and_index,
                          branin_experiment, 0)