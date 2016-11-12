
import numpy as np

import ConfigSpace as CS
from hpolib.abstract_benchmark import AbstractBenchmark


class Levy(AbstractBenchmark):
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x):
        z = 1 + ((x[0] - 1.) / 4.)
        s = np.power((np.sin(np.pi * z)), 2)
        y = (s + ((z - 1) ** 2) * (1 + np.power((np.sin(2 * np.pi * z)), 2)))

        return {'function_value': y}

    def objective_function_test(self, x):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Levy.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Levy',
                'num_function_evals': 200,
                'optima': ([[1.0]]),
                'bounds': [[-15, 10]],
                'f_opt': 0.0}
