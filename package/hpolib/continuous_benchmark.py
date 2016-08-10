import abc

import ConfigSpace
import numpy as np

import hpolib.abstract_benchmark as abstract_benchmark


class AbstractContinuousBenchmark(abstract_benchmark.AbstractBenchmark):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def objective_function(self, configuration):
        """Only here to have a different docstring"""
        pass

    @abc.abstractmethod
    def objective_function_test(self, configuration):
        """Only here to have a different docstring"""
        pass

    def evaluate_dict(self, configuration):
        """

        Parameters
        ----------
        configuration : dict-like

        Returns
        -------
        dict
        """
        configuration = self._convert_dict_to_array(configuration)

        # TODO do we want input checking here?
        rval = self.objective_function(configuration)
        # TODO do we want output checking here?

        return rval

    def evaluate_array(self, configuration):
        """

        Parameters
        ----------
        configuration : list-like

        Returns
        -------
        dict
        """
        # TODO do we want input checking here?
        rval = self.objective_function(configuration)
        # TODO do we want output checking here?
        return rval

    def evaluate_dict_test(self, configuration):
        """

        Parameters
        ----------
        configuration : dict-like

        Returns
        -------
        dict
        """
        configuration = self._convert_dict_to_array(configuration)

        # TODO do we want input checking here?
        rval = self.objective_function_test(configuration)
        # TODO do we want output checking here?

        return rval

    def evaluate_array_test(self, configuration):
        """

        Parameters
        ----------
        configuration : list-like

        Returns
        -------
        dict
        """
        # TODO do we want input checking here?
        rval = self.objective_function_test(configuration)
        # TODO do we want output checking here?
        return rval

    def _convert_dict_to_array(self, configuration):
        l, u = self.get_lower_and_upper_bounds()

        if len(l) != len(configuration):
            raise ValueError('Configuration should have %d elements, but has '
                             '%d!' % (len(l), len(configuration)))

        num_hyperparameters = len(l)
        array = np.ndarray((num_hyperparameters,))
        for i in range(num_hyperparameters):
            value = configuration['X%d' % i]
            array[i] = value

        return array

    @staticmethod
    @abc.abstractmethod
    def get_lower_and_upper_bounds():
        pass

    @classmethod
    def get_configuration_space(cls):
        lower, upper = cls.get_lower_and_upper_bounds()
        cs = ConfigSpace.ConfigurationSpace()
        for i, (l, u) in enumerate(zip(lower, upper)):
            hp = ConfigSpace.UniformFloatHyperparameter('X%d' % i, l, u)
            cs.add_hyperparameter(hp)
        return cs
