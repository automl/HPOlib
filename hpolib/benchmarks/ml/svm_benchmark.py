import time
import numpy as np

from sklearn import svm
from hpolib.abstract_benchmark import AbstractBenchmark

import ConfigSpace as CS


class SupportVectorMachine(AbstractBenchmark):
    """
        Hyperparameter optimization task to optimize the regularization
        parameter C and the kernel parameter gamma of a support vector machine.
        Both hyperparameters are optimized on a log scale in [-10, 10].

        The test data set is only used for a final offline evaluation of
        a configuration. For that the validation and training data is
        concatenated to form the whole training data set.
    """
    def __init__(self, path=None):

        self.train, self.train_targets, self.valid, self.valid_targets, self.test, self.test_targets = self.get_data(path)

        # Use 10 time the number of classes as lower bound for the dataset fraction
        n_classes = np.unique(self.train_targets).shape[0]
        self.s_min = float(10 * n_classes) / self.train.shape[0]

    def get_data(self, path):
        pass

    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function(self, x, dataset_fraction=1, **kwargs):
        start_time = time.time()

        # Shuffle training data
        shuffle = np.random.permutation(self.train.shape[0])
        size = int(dataset_fraction * self.train.shape[0])

        # Split of dataset subset
        train = self.train[shuffle[:size]]
        train_targets = self.train_targets[shuffle[:size]]

        # Transform hyperparameters to linear scale
        C = np.exp(float(x[0]))
        gamma = np.exp(float(x[1]))

        # Train support vector machine
        clf = svm.SVC(gamma=gamma, C=C)
        clf.fit(train, train_targets)

        # Compute validation error
        y = 1 - clf.score(self.valid, self.valid_targets)
        c = time.time() - start_time

        return {'function_value': y, "cost": c}
    
    @AbstractBenchmark._check_configuration
    @AbstractBenchmark._configuration_as_array
    def objective_function_test(self, x, **kwargs):
        start_time = time.time()

        # Concatenate training and validation dataset
        train = np.concatenate((self.train, self.valid))
        train_targets = np.concatenate((self.train_targets, self.valid_targets))

        # Transform hyperparameters to linear scale
        C = np.exp(float(x[0]))
        gamma = np.exp(float(x[1]))

        # Train support vector machine
        clf = svm.SVC(gamma=gamma, C=C)
        clf.fit(train, train_targets)

        # Compute test error
        y = 1 - clf.score(self.test, self.test_targets)
        c = time.time() - start_time

        return {'function_value': y, "cost": c}

    @staticmethod
    def get_configuration_space():
        cs = CS.ConfigurationSpace()
        cs.generate_all_continuous_from_bounds(SupportVectorMachine.get_meta_information()['bounds'])
        return cs

    @staticmethod
    def get_meta_information():
        return {'name': 'Support Vector Machine',
                'bounds': [[-10, 10],  # C
                           [-10, 10]]  # gamma
                }
