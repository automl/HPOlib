from abc import ABCMeta, abstractmethod
import os
import logging
import sys

logger = logging.getLogger("HPOlib.optimizers_algorithm")


class OptimizerAlgorithm(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def check_dependencies(self):
        # Check if all dependencies of optimizer are available
        #
        # This method allows HPOlib to fail fast if something is wrong prior
        # to creating any new files. If necessary dependency cannot be found,
        # this method must raise a Exception with an appropriate error message.

        pass

    @abstractmethod
    def build_call(self):
        # Get parameters from config file and add them to command line

        pass

    @abstractmethod
    def manipulate_config(self, config, opt_name):
        # This callback allows to add further defaults to the config or
        # change the values of current config. Implementing this method is not mandatory.
        if not config.has_option(opt_name, 'params'):
            raise Exception(opt_name, ":params not specified in .cfg")

        path_to_optimizer = config.get(opt_name, 'path_to_optimizer')
        if not os.path.isabs(path_to_optimizer):
            path_to_optimizer = os.path.join(os.path.dirname(os.path.realpath(__file__)), path_to_optimizer)

        path_to_optimizer = os.path.normpath(path_to_optimizer)
        if not os.path.exists(path_to_optimizer):
            logger.critical("Path to optimizer not found: %s" % path_to_optimizer)
            sys.exit(1)

        config.set(opt_name, 'path_to_optimizer', path_to_optimizer)
        # pass

    @abstractmethod
    def main(self, config, options, experiment_dir, experiment_directory_prefix, **kwargs):
        # config:           Loaded .cfg file
        # options:          Options containing seed, restore_dir,
        # experiment_dir:   Experiment directory/Benchmark_directory
        # **kwargs:         Nothing so far

        # This method sets up an output directory for this experiment and builds
        # the command line call for the optimizer

        pass

    @abstractmethod
    def restore(self, config, optimizer_dir, **kwargs):
        # Returns the number of restored runs. This is the number of different configs
        # tested multiplied by the number of crossvalidation folds.
        # Optimizer might not have this function

        pass

if __name__ == "__main__":
    print("Optimizer algorithm class")
