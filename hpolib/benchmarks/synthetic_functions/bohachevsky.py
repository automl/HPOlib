
import numpy as np

import ConfigSpace as CS
from hpolib.abstract_benchmark import AbstractBenchmark


class Bohachevsky(AbstractBenchmark):
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x):
        y = 0.7 + x[0] ** 2 + 2.0 * x[1] ** 2
        y -= 0.3 * np.cos(3.0 * np.pi * x[0])
        y -= 0.4 * np.cos(4.0 * np.pi * x[1])

        return {'function_value': y}

    def objective_function_test(self, x):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Bohachevsky.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Bohachevsky',
                'num_function_evals': 200,
                'optima': ([[0, 0]]),
                'bounds': [[-100, 100], [-100, 100]],
                'f_opt': 0.0}
