'''
Created on 08.09.2016

@author: Stefan Falkner
'''

import numpy as np

import ConfigSpace as CS
from hpolib.abstract_benchmark import AbstractBenchmark


class Hartmann6(AbstractBenchmark):

    alpha = [1.00, 1.20, 3.00, 3.20]
    A = np.array([[10.00, 3.00, 17.00, 3.50, 1.70, 8.00],
                      [0.05, 10.00, 17.00, 0.10, 8.00, 14.00],
                      [3.00, 3.50, 1.70, 10.00, 17.00, 8.00],
                      [17.00, 8.00, 0.05, 10.00, 0.10, 14.00]])
    P = 0.0001 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                               [2329, 4135, 8307, 3736, 1004, 9991],
                               [2348, 1451, 3522, 2883, 3047, 6650],
                               [4047, 8828, 8732, 5743, 1091, 381]])

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        """6d Hartmann test function
            input bounds:  0 <= xi <= 1, i = 1..6
            global optimum: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573),
            min function value = -3.32237
        """

        external_sum = 0
        for i in range(4):
            internal_sum = 0
            for j in range(6):
                internal_sum = internal_sum + self.A[i, j] * (x[j] - self.P[i, j]) ** 2
            external_sum = external_sum + self.alpha[i] * np.exp(-internal_sum)

        return {'function_value': -external_sum}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Hartmann6.get_meta_information()['bounds'])
        return(cs)
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Hartmann 3D',
				'num_function_evals': 200,
                'optima': ([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]),
                'bounds': [[0,1]]*6,
                'f_opt': -3.322368011391339}
