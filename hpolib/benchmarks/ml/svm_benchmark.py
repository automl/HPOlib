import time
import numpy as np

from sklearn import svm
from hpolib.abstract_benchmark import AbstractBenchmark

import ConfigSpace as CS


class SupportVectorMachineBenchmark(AbstractBenchmark):

    def __init__(self, path=None):
        self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets = self.get_data(path)

    def get_data(self, path):
        pass
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, **kwargs):
        start_time = time.time()

        C = np.exp(float(x[0]))
        gamma = np.exp(float(x[1]))
        clf = svm.SVC(gamma=gamma, C=C)
        clf.fit(self.train, self.train_targets)

        y = 1 - clf.score(self.valid, self.valid_targets)

        c = time.time() - start_time

        return {'function_value': y, "cost": c}
    
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        start_time = time.time()

        C = np.exp(float(x[0]))
        gamma = np.exp(float(x[1]))
        clf = svm.SVC(gamma=gamma, C=C)
        clf.fit(self.train, self.train_targets)

        y = 1 - clf.score(self.test, self.test_targets)

        c = time.time() - start_time

        return {'function_value': y, "cost": c}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds([
            [-10,10],
            [-10,10]
        ])
        return(cs)
