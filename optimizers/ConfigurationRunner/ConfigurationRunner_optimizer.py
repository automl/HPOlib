import argparse
from collections import OrderedDict
import csv
import logging
import subprocess
from multiprocessing import Pool

import HPOlib.optimization_interceptor


def command_line_function(params):
    call = construct_cli_call(params)
    ret = subprocess.call(call, shell=True)
    return ret


def construct_cli_call(params):
    optimization_interceptor_file = HPOlib.optimization_interceptor.__file__
    if optimization_interceptor_file.endswith("pyc"):
        optimization_interceptor_file = optimization_interceptor_file[:-1]
    cli_target = optimization_interceptor_file

    cli_call = []
    cli_call.append("python")
    cli_call.append(cli_target)
    cli_call.append("--params")
    params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))
    for param in params:
        value = params[param]
        if value:
            cli_call.append(" -" + param)
            cli_call.append("\"'" + str(params[param]) + "'\"")
    return " ".join(cli_call)


class ConfigurationRunner(object):
    """
    Run all hyperparameter configurations from a list sequentially.

    This optimizer can be used to do the following:
        * Perform a grid search
        * Evaluate a list of hyperparameter configurations

    It can read the following formats:
        * A csv in which every line specifies one hyperparameter configuration
    """
    def __init__(self, configurations_file, n_jobs):
        self.logger = logging.getLogger("ConfigurationRunner")
        self.logger.setLevel(logging.INFO)

        self.configurations_file = configurations_file
        self.n_jobs = n_jobs
        self.configurations = []

        with open(self.configurations_file) as fh:
            csv_reader = csv.DictReader(fh)
            for row in csv_reader:
                self.configurations.append(row)

        for configuration in self.configurations:
            self.logger.info(configuration)

    def run(self):
        pool = Pool(processes=4)
        pool.map(command_line_function, self.configurations)
        pool.close()
        pool.join()


if __name__ == "__main__":
    description = "Evaluate hyperparameter configurations via HPOlib."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("configurations_file", metavar="configurations-file",
                        help="List of configurations as a csv file.")
    parser.add_argument("--n-jobs", default=1, type=int,
                        help="Number of parallel function evaluations.")
    args = parser.parse_args()

    runner = ConfigurationRunner(args.configurations_file, args.n_jobs)
    runner.run()


