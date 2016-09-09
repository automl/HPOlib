'''
Created on 2016/09/08

@author: Stefan Falkner
'''

import numpy as np

import ConfigSpace as CS
from hpolib.abstract_benchmark import AbstractBenchmark

class Forrester(AbstractBenchmark):
    """ one dimensioal function with one global minimum, another local minimum and a zero gradient, inflection point.
    
    Simple 1d function for visualization. We added a different 'fidelity'
    than the listed Sources. We interpolate between the original function
    and a Least-Square cubic fit.
    
    One can add a an arbitray cost model externaly to simulate true
    multi-fidelity scenarios.
    
    
    Sources
    -------
    http://www.sfu.ca/~ssurjano/forretal08.html
    
    Forrester, A., Sobester, A., & Keane, A. (2008).
    Engineering design via surrogate modelling: a practical guide. Wiley.
    
    """
    
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, fidelity = 1, **kwargs):
        x = x[0]
        y1 = np.power(6*x-2, 2)* np.sin(12*x-4)
        
        # best least-squared fit with cubic polynomial
        y2 = 131.09227753 * (x**3) -164.50286816 * (x**2) + 50.7228373 * x  -2.84345244
        return {'function_value': fidelity*y1 + (1-fidelity)*y2,
                'cost': fidelity**2}

    def objective_function_test(self, x, **kwargs):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Forrester.get_meta_information()['bounds'])
        return(cs)

    @staticmethod
    def get_meta_information():
        return {'name': 'Branin',
                'num_function_evals': 20,
                'optima': ([[0.75724875]]),
                'bounds': [[0,1]],
                'f_opt': -6.02074}
