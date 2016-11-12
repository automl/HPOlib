import numpy as np

import ConfigSpace as CS
from hpolib.abstract_benchmark import AbstractBenchmark


class SinTwo(AbstractBenchmark):
    """
    Two dimensional sin function introduced in the paper:

        K. Kawaguchi, L. P. Kaelbling, and T. Lozano-Perez.
        Bayesian Optimization with Exponential Convergence.
        In Advances in Neural Information Processing (NIPS), 2015

    """
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x):
        y = (0.5 * np.sin(13 * x[0]) * np.sin(27 * x[0]) + 0.5) * (0.5 * np.sin(13 * x[1]) * np.sin(27 * x[1]) + 0.5)

        return {'function_value': y}

    def objective_function_test(self, x):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(SinTwo.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'SinTwo',
                'num_function_evals': 200,
                'optima': ([[0.6330131633013163, 0.6330131633013163]]),
                'bounds': [[0, 1], [0, 1]],
                'f_opt': 0.042926342433644127 ** 2}
