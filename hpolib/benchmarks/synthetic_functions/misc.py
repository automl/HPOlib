'''
Created on 2016/09/08

@author: Stefan Falkner, based on code from Aaron Klein
'''

import numpy as np

import ConfigSpace as CS
from hpolib.abstract_benchmark import AbstractBenchmark


class SyntheticNoiseAndCost(AbstractBenchmark):

    def __init__(self, original_benchmark, sigma_min, sigma_max,
                 c_min, c_max,
                 k_cost, k_noise):

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.c_min = c_min
        self.c_max = c_max
        self.k_cost = k_cost
        self.k_noise = k_noise

        self.original_benchmark = original_benchmark

        super().__init__()

    def sigma_function(self, dataset_fraction):
        sigma = self.sigma_min
        sigma += (self.sigma_max - self.sigma_min) * ((1 - dataset_fraction) ** self.k_noise)
        return sigma

    def cost_function(self, dataset_fraction):
        cost = self.c_min + (self.c_max - self.c_min) * (dataset_fraction ** self.k_cost)
        return cost

    def objective_function(self, configuration, dataset_fraction = 1, **kwargs):
        
        y = self.original_benchmark.objective_function(configuration, **kwargs)['function_value']
        y += self.sigma_function(dataset_fraction) * np.random.randn()

        cost = self.cost_function(dataset_fraction)

        return {'function_value': y, 'cost': cost}

    def objective_function_test(self, configuration, **kwargs):
        return self.original_benchmark.objective_function(configuration, **kwargs)

    #   can't be a @staticmethod anymore
    def get_configuration_space(self):
        return(self.original_benchmark.get_configuration_space())

    #   can't be a @staticmethod anymore
    def get_meta_information(self):
        d = self.original_task.get_meta_information()
        d['noise_model'] = [self.sigma_min, self.sigma_max, k_noise]
        d['cost_model'] = [self.c_min, self.c_max, k_cost]
        d['name'] = d['name'] + " + noise and cost model"
