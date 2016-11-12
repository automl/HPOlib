'''
Created on 2016/09/08

@author: Stefan Falkner
'''
import numpy as np

import ConfigSpace as CS
from hpolib.abstract_benchmark import AbstractBenchmark


class Branin(AbstractBenchmark):
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x):
        y = (x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10

        return {'function_value': y}

    def objective_function_test(self, x):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Branin.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Branin',
                'num_function_evals': 200,
                'optima': ([[-np.pi, 12.275],
                            [np.pi, 2.275],
                            [9.42478, 2.475]]),
                'bounds': [[-5, 10], [0, 15]],
                'f_opt': 0.39788735773}
