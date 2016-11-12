
import ConfigSpace as CS
from hpolib.abstract_benchmark import AbstractBenchmark


class Rosenbrock(AbstractBenchmark):
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x):
        y = 0
        d = 2
        for i in range(d - 1):
            y += 100 * (x[i + 1] - x[i] ** 2) ** 2
            y += (x[i] - 1) ** 2

        return {'function_value': y}

    def objective_function_test(self, x):
        return self.objective_function(x)

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(Rosenbrock.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Rosenbrock',
                'num_function_evals': 200,
                'optima': ([[1, 1]]),
                'bounds': [[-5, 10], [-5, 10]],
                'f_opt': 0.0}
