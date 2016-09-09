import abc

import numpy as np
import ConfigSpace


class AbstractBenchmark(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """Interface for benchmarks.

        A benchmark contains of two building blocks, the target function and
        the configuration space. Furthermore it can contain additional
        benchmark-specific information such as the location and the function
        value of the global optima. New benchmarks should be derived from
        this base class or one of its child classes.
        """
        self.configuration_space = self.get_configuration_space()

    @abc.abstractmethod
    def objective_function(self, configuration, **kwargs):
        """Objective function.

        Override this function to provide your benchmark function. This
        function will be called by one of the evaluate functions. For
        flexibility you have to return a dictionary with the only mandatory
        key being `function_value`, the objective function value for the
        configuration which was passed. By convention, all benchmarks are
        minization problems.

        Parameters
        ----------
        configuration : dict-like

        Returns
        -------
        dict
            Must contain at least the key `function_value`.
        """
        pass

    @abc.abstractmethod
    def objective_function_test(self, configuration, **kwargs):
        """
        If there is a different objective function for offline testing, e.g
        testing a machine learning on a hold extra test set instead
        on a validation set override this function here.

        Parameters
        ----------
        configuration : dict-like

        Returns
        -------
        dict
            Must contain at least the key `function_value`.
        """
        pass

    def _check_configuration(foo):
        """ Decorator to enable checking the input configuration

            Uses the check_configuration of the ConfigSpace class to ensure
            that all specified values are valid, and no conditionals are violated

            Can be combined with the _configuration_as_array decorator.
        """
        def wrapper(self, configuration, **kwargs):
            if not isinstance(configuration, ConfigSpace.Configuration):
                try:
                    squirtle = {k: configuration[i] for (i,k) in enumerate(self.configuration_space)}
                    wartortle = ConfigSpace.Configuration(self.configuration_space, squirtle)
                except Exception as e:
                    raise Exception('Error during the conversion of the provided'
                                    'into a ConfigSpace.Configuration object') from e
            else:
                wartortle = configuration
            self.configuration_space.check_configuration(wartortle)
            return(foo(self, configuration, **kwargs))
        return(wrapper)

    def _configuration_as_array(foo, data_type=np.float):
        """ Decorator to allow the first input argument to 'objective_function' to be an array.

            For all continuous benchmarks it is often required that the input to the benchmark
            can be a (NumPy) array. By adding this to the objective function, both inputs types,
            ConfigSpace.Configuration and array, are possible.

            Can be combined with the _check_configuration decorator.
        """
        def wrapper(self, configuration, **kwargs):
            if isinstance(configuration, ConfigSpace.Configuration):
                blastoise = np.array(
                    [ configuration[k] for k in configuration],
                    dtype=data_type
                )

            else:
                blastoise = configuration
            return( foo(self, blastoise, **kwargs) )
        return(wrapper)

    def __call__ (self, configuration, **kwargs):
        """ Provides interface to use, e.g., SciPy optimizers """
        return(self.objective_function(configuration, **kwargs)['function_value'])


    def test(self, n_runs=5):
        """ Draws some random configuration and call objective_fucntion(_test).

        Parameters
        ----------
        n_runs : int
            number of random configurations to draw and evaluate
        """
        for _ in range(n_runs):
            configuration = self.configuration_space.sample_configuration()
            self.objective_function(configuration)
            self.objective_function_test(configuration)

    @staticmethod
    @abc.abstractmethod
    def get_configuration_space():
        """ Defines the configuration space for each benchmark.

        Returns
        -------
        ConfigSpace.ConfigurationSpace
            A valid configuration space for the benchmark's parameters
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_meta_information(self):
        """ Provides some meta information about the benchmark.

        Returns
        -------
        dict
            some human-readable information

        """
        return {}
