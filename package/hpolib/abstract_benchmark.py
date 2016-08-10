import abc


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
    def objective_function(self, configuration):
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
            Mapping which at least contains the key `function_value`.
        """
        pass

    @abc.abstractmethod
    def objective_function_test(self, configuration):
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
        """
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
        # TODO do we want input checking here?
        rval = self.objective_function(configuration)
        # TODO do we want output checking here?

        return rval

    def evaluate_dict_test(self, configuration):
        """
        Wrapper function of the test objective function that can
        be used for an offline testing of incumbents.
        It rescales x from [0,1] to the original space and makes
        sure the x inside the bounds.

        Parameters
        ----------
        configuration : dict-like

        Returns
        -------
        dict
        """
        # TODO do we want input checking here?
        rval = self.objective_function_test(configuration)
        # TODO do we want output checking here?

        return rval

    def test(self, n_runs=5):
        for _ in range(n_runs):
            configuration = self.configuration_space.sample_configuration()
            self.evaluate_dict(configuration)
            self.evaluate_dict_test(configuration)

    @classmethod
    @abc.abstractmethod
    def get_configuration_space(cls):
        pass

    def get_meta_information(self):
        return {}



