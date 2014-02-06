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

import cPickle
import os
import sys

import HPOlib.wrapping_util as wrapping_util



version_info = ("# %76s #\n" % "https://github.com/hyperopt/hyperopt/tree/486aebec8a4170e4781d99bbd6cca09123b12717")
__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

optimizer_str = "hyperopt_august2013_mod"


def build_tpe_call(config, options, optimizer_dir):
    #For TPE we have to cd to the exp_dir
    call = "cd " + optimizer_dir + "\n" + \
           "python " + os.path.dirname(os.path.realpath(__file__)) + \
           "/tpecall.py"
    call = ' '.join([call, '-p', config.get('TPE', 'space'),
                     "-a", config.get('DEFAULT', 'algorithm'),
                     "-m", config.get('TPE', 'numberEvals'),
                     "-s", str(options.seed)])
    if options.restore:
        call = ' '.join([call, '-r'])
    return call


#noinspection PyUnusedLocal
def restore(config, optimizer_dir, **kwargs):
    """
    Returns the number of restored runs. This is the number of different configs
    tested multiplied by the number of crossvalidation folds.
    """
    restore_file = os.path.join(optimizer_dir, 'state.pkl')
    if not os.path.exists(restore_file):
        print "Oups, this should have been checked before"
        raise Exception("%s does not exist" % (restore_file,))

    # Special settings for restoring
    fh = open(restore_file)
    state = cPickle.load(fh)
    fh.close()
    complete_runs = 0
    #noinspection PyProtectedMember
    tpe_trials = state['trials']._trials
    for trial in tpe_trials:
        # Assumes that all not valid states states are marked crashed
        if trial['state'] == 2:
            complete_runs += 1
    restored_runs = complete_runs * config.getint('DEFAULT', 'numberCV')
    return restored_runs


#noinspection PyUnusedLocal
def main(config, options, experiment_dir, **kwargs):
    # config:           Loaded .cfg file
    # options:          Options containing seed, restore_dir, 
    # experiment_dir:   Experiment directory/Benchmark_directory
    # **kwargs:         Nothing so far
    time_string = wrapping_util.get_time_string()
    cmd = ""
    SYSTEM_WIDE = 0
    AUGUST_2013_MOD = 1

    try:
        import hyperopt
        version = SYSTEM_WIDE
    except ImportError:
        try:
            cmd += "export PYTHONPATH=$PYTHONPATH:" + os.path.dirname(os.path.abspath(__file__)) + \
                "/optimizers/hyperopt_august2013_mod\n"
            import optimizers.hyperopt_august2013_mod.hyperopt as hyperopt
        except ImportError, e:
            import HPOlib.optimizers.hyperopt_august2013_mod.hyperopt as hyperopt
        version = AUGUST_2013_MOD

    path_to_optimizer = os.path.abspath(os.path.dirname(hyperopt.__file__))

    # Find experiment directory
    if options.restore:
        if not os.path.exists(options.restore):
            raise Exception("The restore directory does not exist")
        optimizer_dir = options.restore
    else:
        optimizer_dir = os.path.join(experiment_dir, optimizer_str + "_" +
                                     str(options.seed) + "_" +
                                     time_string)

    # Build call
    cmd += build_tpe_call(config, options, optimizer_dir)

    # Set up experiment directory
    if not os.path.exists(optimizer_dir):
        os.mkdir(optimizer_dir)
        space = config.get('TPE', 'space')
        # Copy the hyperopt search space
        if not os.path.exists(os.path.join(optimizer_dir, space)):
            os.symlink(os.path.join(experiment_dir, optimizer_str, space),
                       os.path.join(optimizer_dir, space))
    sys.stdout.write("### INFORMATION ################################################################\n")
    sys.stdout.write("# You're running %40s                      #\n" % path_to_optimizer)
    if version == SYSTEM_WIDE:
        pass
    else:
        sys.stdout.write("# To reproduce our results you need version 0.0.3.dev, which can be found here:#\n")
        sys.stdout.write("%s" % version_info)
        sys.stdout.write("# A newer version might be available, but not yet built in.                    #\n")
    sys.stdout.write("################################################################################\n")
    return cmd, optimizer_dir
