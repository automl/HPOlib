'''
Created on 08.09.2016

@author: Stefan Falkner
'''

import numpy as np

import ConfigSpace as CS
from hpolib.abstract_benchmark import AbstractBenchmark

class Hartmann3(AbstractBenchmark):

    alpha = [1.0, 1.2, 3.0, 3.2]
    A = np.array([[3.0, 10.0, 30.0],
                       [0.1, 10.0, 35.0],
                       [3.0, 10.0, 30.0],
                       [0.1, 10.0, 35.0]])
    P = 0.0001 * np.array([[3689, 1170, 2673],
                                [4699, 4387, 7470],
                                [1090, 8732, 5547],
                                [381, 5743, 8828]])



    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x):
        external_sum = 0
        for i in range(4):
            internal_sum = 0
            for j in range(3):
                internal_sum = internal_sum \
                            + self.A[i, j] * (x[j] \
                            - self.P[i, j]) ** 2
            external_sum = external_sum + self.alpha[i] * np.exp(-internal_sum)

        return {'function_value': -external_sum}

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x):
        return self.objective_function(x)
        
        
    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Hartmann3.get_meta_information()['bounds'])
        return(cs)
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Hartmann 3D',
				'num_function_evals': 200,
                'optima': ([[0.114614, 0.555649, 0.852547]]),
                'bounds': [[0,1]]*3,
                'f_opt': -3.86278}
