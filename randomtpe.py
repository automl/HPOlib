##
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
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

import cPickle
import os

from config_parser.parse import parse_config
import wrapping_util

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def buildRandomCall(config, options, optimizer_dir):
    #For TPE (and Random Search) we have to cd to the exp_dir
    call = 'cd ' + optimizer_dir + '\n' + \
            'python ' + os.path.dirname(os.path.realpath(__file__)) + \
            '/tpecall.py'
    call = ' '.join([call, '-p', config.get('TPE', 'space'), \
            '-a', config.get('DEFAULT', 'algorithm'), \
            '-m', config.get('DEFAULT', 'numberOfJobs'), \
            '-s', str(options.seed), '--random'])
    if options.restore:
        call = ' '.join([call, '-r'])
    return call


def restore(config, optimizer_dir, **kwargs):
    restore_file = os.path.join(optimizer_dir, 'state.pkl')
    if not os.path.exists(restore_file):
        print "Oups, this should have been checked before"
        raise Exception("%s does not exist" % (restore_file,))
        return -1

    fh = open(restore_file)
    state = cPickle.load(fh)
    fh.close()
    complete_runs = 0
    tpe_trials = state['trials']._trials
    for trial in tpe_trials:
        # Assumes that all states no valid state is marked crashed
        if trial['state'] == 2:
            complete_runs += 1
    restored_runs = complete_runs * config.getint('DEFAULT', 'numberCV')
    return restored_runs


def main(config, options, experiment_dir, **kwargs):
    # config:           Loaded .cfg file
    # options:          Options containing seed, restore, 
    # experiment_dir:   Experiment directory/Benchmarkdirectory
    # **kwargs:         Nothing so far
    time_string = wrapping_util.get_time_string()

    # Find experiment directory
    if options.restore:
        if not os.path.exists(options.restore):
            raise Exception("The restore directory %s does not exist" % (options.restore,))
        optimizer_dir = options.restore
    else:
        optimizer_dir = os.path.join(experiment_dir, "randomtpe_" + \
            str(options.seed) + "_" + time_string)

    # Build call
    cmd = buildRandomCall(config, options, optimizer_dir)

    # Set up experiment directory
    if not os.path.exists(optimizer_dir):
        os.mkdir(optimizer_dir)
        space = config.get('TPE', 'space')
        # Copy the hyperopt search space
        if not os.path.exists(os.path.join(optimizer_dir, space)):
            os.symlink(os.path.join(experiment_dir, "tpe", space),
                        os.path.join(optimizer_dir, space))

    return cmd, optimizer_dir