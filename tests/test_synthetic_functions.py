import unittest
import numpy as np

from hpolib.benchmarks import synthetic_functions


class TestAbstractBenchmark(unittest.TestCase):

    def test_branin(self):
        f = synthetic_functions.Branin()

        for x in f.get_meta_information()["optima"]:
            np.testing.assert_approx_equal(f(x), f.get_meta_information()["f_opt"], significant=9)

    def test_hartmann3(self):
        f = synthetic_functions.Hartmann3()

        for x in f.get_meta_information()["optima"]:
            np.testing.assert_approx_equal(f(x), f.get_meta_information()["f_opt"], significant=9)

    def test_hartmann6(self):
        f = synthetic_functions.Hartmann6()

        for x in f.get_meta_information()["optima"]:
            np.testing.assert_approx_equal(f(x), f.get_meta_information()["f_opt"], significant=9)

    def test_camelback(self):
        f = synthetic_functions.Camelback()

        for x in f.get_meta_information()["optima"]:
            np.testing.assert_approx_equal(f(x), f.get_meta_information()["f_opt"], significant=9)

    def test_levy(self):
        f = synthetic_functions.Levy()

        for x in f.get_meta_information()["optima"]:
            assert np.isclose(f(x), f.get_meta_information()["f_opt"])

    def test_goldstein_price(self):
        f = synthetic_functions.GoldsteinPrice()

        for x in f.get_meta_information()["optima"]:
            assert np.isclose(f(x), f.get_meta_information()["f_opt"])

    def test_rosenbrock(self):
        f = synthetic_functions.Rosenbrock()

        for x in f.get_meta_information()["optima"]:
            np.testing.assert_approx_equal(f(x), f.get_meta_information()["f_opt"], significant=9)

    def test_sin_one(self):
        f = synthetic_functions.SinOne()

        for x in f.get_meta_information()["optima"]:
            np.testing.assert_approx_equal(f(x), f.get_meta_information()["f_opt"], significant=9)

    def test_sin_two(self):
        f = synthetic_functions.SinTwo()

        for x in f.get_meta_information()["optima"]:
            np.testing.assert_approx_equal(f(x), f.get_meta_information()["f_opt"], significant=9)

if __name__ == "__main__":
    unittest.main()
