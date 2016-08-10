import numpy as np

from hpolib.continuous_benchmark import AbstractContinuousBenchmark


class Branin(AbstractContinuousBenchmark):

    def objective_function(self, x):
        y = (x[1] - (5.1 / (4 * np.pi ** 2)) * x[0] ** 2 + 5 * x[0] / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10

        return {'function_value': y}

    def objective_function_test(self, x):
        return self.objective_function(x)

    @staticmethod
    def get_lower_and_upper_bounds():
        lower = np.array([-5, 0])
        upper = np.array([10, 15])
        return lower, upper

    def get_meta_information(self):
        return {'num_function_evals': 200,
                'optima': ([[-np.pi, 12.275],
                            [np.pi, 2.275],
                            [9.42478, 2.475]]),
                'f_opt': 0.397887}

