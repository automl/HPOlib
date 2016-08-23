import time
import numpy as np

from sklearn import svm
from hpolib.continuous_benchmark import AbstractContinuousBenchmark


class SupportVectorMachineBenchmark(AbstractContinuousBenchmark):

    def __init__(self, path=None):
        self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets = self.get_data(path)

    def get_data(self, path):
        pass

    def objective_function(self, x):
        start_time = time.time()

        C = np.exp(float(x[0]))
        gamma = np.exp(float(x[1]))
        clf = svm.SVC(gamma=gamma, C=C)
        clf.fit(self.train, self.train_targets)

        y = 1 - clf.score(self.valid, self.valid_targets)

        c = time.time() - start_time

        return {'function_value': y, "cost": c}

    def objective_function_test(self, x):
        start_time = time.time()

        C = np.exp(float(x[0, 0]))
        gamma = np.exp(float(x[0, 1]))
        clf = svm.SVC(gamma=gamma, C=C)
        clf.fit(self.train, self.train_targets)

        y = 1 - clf.score(self.test, self.test_targets)

        c = time.time() - start_time

        return {'function_value': y, "cost": c}

    @staticmethod
    def get_lower_and_upper_bounds():
        lower = np.array([-10, -10])
        upper = np.array([10, 10])
        return lower, upper
