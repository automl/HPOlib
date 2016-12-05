import unittest

import HPOlib.format_converter.configuration_space as configuration_space


class TestConfigurationSpace(unittest.TestCase):
    def test_uniform_float_to_int(self):
        # For some reason, the upper bound converts to an int
        param = configuration_space.UniformFloatHyperparameter(
            "@1:max-feature-time", 1.0, 600.0, q=1.0, base=10.0)
        param = param.to_integer()
        print param