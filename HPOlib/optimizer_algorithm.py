from abc import ABCMeta, abstractmethod
import os
import logging
import HPOlib.wrapping_util as wrapping_util


class OptimizerAlgorithm(object):
    """This is an abstract wrapper class defining methods needed to incorporate optimizers in HPOlib.

    Usage:  In wrapping.main(), check_before_start.check_optimizer() is called which initilizes subclass of this class depending
    on name of optimizer and return its object.
    """

    __metaclass__ = ABCMeta

    def __init__(self, opt_name):
        self.optimizer_name = None
        self.optimizer_dir = None
        self.logger = logging.getLogger("HPOlib.optimizers_algorithm")
        self.logger.info("opt path in obj initialization:%s" % self.optimizer_dir)
        self.logger.info("opt name:%s" % self.optimizer_name)

    @abstractmethod
    def check_dependencies(self):
        """Check if all dependencies of optimizer are available

        This method allows HPOlib to fail fast if something is wrong prior
        to creating any new files. If necessary dependency cannot be found,
        this method must raise a Exception with an appropriate error message.

        Returns
        -------

        """
        pass

    @abstractmethod
    def build_call(self, config, options, optimizer_dir):
        """Get parameters from config file and add them to command line

        Parameters
        ----------
        config          :   configuration file
        options         :   Options containing seed, restore_dir,
        optimizer_dir   :   directory path where experiment is run

        Returns
        -------
        cmd             :   optimizer command line call in string form.
        """
        pass

    @abstractmethod
    def custom_setup(self, config, options, experiment_dir, optimizer_dir):
        """setup directory where experiment will run.
        Creates the directory and copies all files to that directory needed by optimizer to run there.


        Parameters
        ----------
        config          :   configuration file
        options         :   Options containing seed, restore_dir,
        experiment_dir  :   benchmark directory
        optimizer_dir   :   directory path where experiment is run

        Returns
        -------
        optimizer_dir   :   generally same as input except for some special cases
                            where name of directory is changed based on some parameter setting.

        """
        pass

    @abstractmethod
    def manipulate_config(self, config):
        """This callback allows to add further defaults to the config or
        change the values of current config. Implementing this method is not mandatory.

        Parameters
        ----------
        config  :   configuration file

        Returns
        -------
        config  :   updated configuration file

        """
        pass

    def main(self, config, options, experiment_dir, **kwargs):
        """This method  sets up path name for directory where experiment is run, it then calls custum_setup() method
        to setup the directory with that name and all optimizer specific settings. Finally it calls build_call()
        method to create command line for executing optimizer with desired settings.

        Parameters
        ----------
        config          :   Loaded .cfg file
        options         :   Options containing seed, restore_dir,
        experiment_dir  :   Experiment directory/Benchmark_directory
        kwargs          : Nothing so far

        Returns
        -------
        cmd             :   command used to run optimizer with all its parameter settings
        optimizer_dir   :   directory path where experiment is run

        """

        time_string = wrapping_util.get_time_string()
        # optimizer_str = os.path.splitext(os.path.basename(__file__))[0]
        optimizer_str = self.optimizer_dir
        optimizer_dir = os.path.join(experiment_dir,
                                     optimizer_str + "_" +
                                     str(options.seed) + "_" + time_string)

        # setup directory where experiment will run
        optimizer_dir = self.custom_setup(config, options, experiment_dir, optimizer_dir)

        # Build call
        cmd = self.build_call(config, options, optimizer_dir)

        self.logger.info("### INFORMATION ############################################################################")
        self.logger.info("# You're running %35s" % config.get(self.optimizer_name, 'path_to_optimizer'))
        self.logger.info("optimization dir %s" % optimizer_dir)
        self.logger.info("optimization str %s" % optimizer_str)
        self.logger.info("##########################################################################################\n")
        # logger.info("exp dir:%s" % experiment_dir)
        return cmd, optimizer_dir


if __name__ == "__main__":
    print("Optimizer algorithm class")
