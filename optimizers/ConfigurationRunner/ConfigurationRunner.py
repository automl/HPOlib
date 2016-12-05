##
# HPOlib: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013-2015 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

#!/usr/bin/env python

import os

import HPOlib.wrapping_util as wrapping_util


version_info = ("0.1")
__authors__ = ["Matthias Feurer"]
__contact__ = "automl.org"

optimizer_str = "ConfigurationRunner"


def check_dependencies():
    # Check the dependencies. The ConfigurationRunner has no external
    # dependencies!
    pass


def build_call(config, options, optimizer_dir):
    # Build the call to the ConfigurationRunner
    this_file = __file__
    this_directory = os.path.dirname(this_file)
    ConfigurationRunner_file = os.path.join(this_directory,
                                            "ConfigurationRunner_optimizer.py")
    call = []
    call.append("python")
    call.append(ConfigurationRunner_file)
    call.append(os.path.join(optimizer_dir,
                os.path.basename(config.get("ConfigurationRunner",
                                            "configurations"))))
    call.append("--n-jobs")
    call.append(str(config.get("ConfigurationRunner", "n_jobs")))
    return " ".join(call)


#noinspection PyUnusedLocal
def restore(config, optimizer_dir, **kwargs):
    # This will not be used
    raise NotImplementedError("Restore is not implemented for the "
                              "ConfigurationRunner")


#noinspection PyUnusedLocal
def main(config, options, experiment_dir, experiment_directory_prefix, **kwargs):
    # config:           Loaded .cfg file
    # options:          Options containing seed, restore_dir,
    # experiment_dir:   Experiment directory/Benchmark_directory
    # **kwargs:         Nothing so far
    time_string = wrapping_util.get_time_string()

    path_to_optimizer = os.path.abspath(os.path.dirname(__file__))

    # Find experiment directory
    if options.restore:
        raise NotImplementedError("Restore is not implemented for the "
                                  "ConfigurationRunner")
    else:
        optimizer_dir = os.path.join(experiment_dir,
                                     experiment_directory_prefix
                                     + optimizer_str + "_" + time_string)

    # Set up experiment directory
    if not os.path.exists(optimizer_dir):
        os.mkdir(optimizer_dir)

    space = config.get('ConfigurationRunner', "configurations")
    abs_space = os.path.abspath(space)
    parent_space = os.path.join(experiment_dir, optimizer_str, space)
    if os.path.exists(abs_space):
        space = abs_space
    elif os.path.exists(parent_space):
        space = parent_space
    else:
        raise Exception("Configurations for the ConfigurationRunner not found. "
                        "Searched at %s and "
                        "%s" % (abs_space, parent_space))

    # if not os.path.exists(os.path.join(optimizer_dir, os.path.basename(space))):
    os.symlink(os.path.join(experiment_dir, optimizer_str, space),
               os.path.join(optimizer_dir, os.path.basename(space)))

    # Build call
    cmd = build_call(config, options, optimizer_dir)

    return cmd, optimizer_dir
