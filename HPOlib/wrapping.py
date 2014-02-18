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

from argparse import ArgumentParser
import imp
import os
import subprocess
import sys
import time

from config_parser.parse import parse_config

import HPOlib.check_before_start as check_before_start

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def calculate_wrapping_overhead(trials):
    wrapping_time = 0
    for times in zip(trials.cv_starttime, trials.cv_endtime):
        wrapping_time += times[1] - times[0]

    # We need to import numpy again
    import numpy as np
    benchmark_time = 0
    for t in trials.trials:
        benchmark_time += np.nansum(t['instance_durations'])
    wrapping_time = wrapping_time - benchmark_time
    return wrapping_time


def calculate_optimizer_time(trials):
    optimizer_time = []
    time_idx = 0
    
    optimizer_time.append(trials.cv_starttime[0] - trials.starttime[time_idx])
    
    for i in range(len(trials.cv_starttime[1:])):
        if trials.cv_starttime[i + 1] > trials.endtime[time_idx]:
            optimizer_time.append(trials.endtime[time_idx] -
                                  trials.cv_endtime[i])
            time_idx += 1
            optimizer_time.append(trials.cv_starttime[i + 1] -
                                  trials.starttime[time_idx])
        else:
            optimizer_time.append(trials.cv_starttime[i + 1] -
                                  trials.cv_endtime[i])
                
    optimizer_time.append(trials.endtime[time_idx] - trials.cv_endtime[-1])
    trials.optimizer_time = optimizer_time

    # We need to import numpy again
    import numpy as np
    return np.nansum(optimizer_time)


def use_option_parser():
    """Parse all options which can be handled by the wrapping script.
    Unknown arguments are ignored and returned as a list. It is useful to
    check this list in your program to handle typos etc.

    Returns:
        a tuple. The first element is an argparse.Namespace object,
        the second a list with all unknown arguments.
    """
    parser = ArgumentParser(description="Perform an experiment with the "
                                        "HPOlib")
    parser.add_argument("-o", "--optimizer", action="store", type=str,
                        help="Specify the optimizer name.", required=True)
    parser.add_argument("-p", "--print", action="store_true", dest="printcmd",
                        default=False,
                        help="If set print the command instead of executing it")
    parser.add_argument("-s", "--seed", dest="seed", default=1, type=int,
                        help="Set the seed of the optimizer")
    parser.add_argument("-t", "--title", dest="title", default=None,
                        help="A title for the experiment")
    restore_help_string = "Restore the state from a given directory"
    parser.add_argument("--restore", default=None, dest="restore",
                        help=restore_help_string)
    parser.add_argument("--silent", default=False, action="store_true",
                        dest="silent", help="Don't print the optimizer output")
    parser.add_argument("--verbose", default=False, action="store_true",
                        dest="verbose",
                        help="Print stderr/stdout for optimizer well, overrides --silent")

    args, unknown = parser.parse_known_args()
    return args, unknown


def parse_config_values_from_unknown_arguments(unknown_arguments, config):
    """Parse values not recognized by use_option_parser for config values.

    Args:
       unknown_arguments: A list of arguments which is returned by
           use_option_parser. It should only contain keys which are allowed
           in config files.
        config: A ConfigParser.SafeConfigParser object which contains all keys
           should be parsed from the unknown_arguments list.
    Returns:
        an argparse.Namespace object containing the parsed values.
    Raises:
        an error if an argument from unknown_arguments is not a key in config
    """
    further_possible_command_line_arguments = []
    for section in config.sections():
        for key in config.options(section):
            further_possible_command_line_arguments.append("--" + section +
                                                           ":" + key)
    for key in config.defaults():
        further_possible_command_line_arguments.append("--DEFAULT:" + key)

    parser = ArgumentParser()
    for argument in further_possible_command_line_arguments:
        parser.add_argument(argument)

    return parser.parse_args(unknown_arguments)


def override_config_with_cli_arguments(config, config_overrides):
    arg_dict = vars(config_overrides)
    for key in arg_dict:
        if arg_dict[key] is not None:
            split_key = key.split(":")
            if len(split_key) != 2:
                raise ValueError("Command line argument must be in the style "
                                 "of --SECTION:key. Please check that there "
                                 "is exactly one colon. You provided: %s" %
                                 key)
            if split_key[0] == 'DEFAULT' and split_key[1] == "optimizer_version":
                raise NotImplementedError("Overwriting the optimizer_version via a cli is a bad idea."
                                          "Try to do something else!\n"
                                          "You were trying to use %s instead of %s"
                                          % (arg_dict[key], config.get("DEFAULT", "optimizer_version")))
            config.set(split_key[0], split_key[1], arg_dict[key])

    return config


def save_config_to_file(file, config):
    for section in config.sections():
        file.write("[DEFAULT]\n")
        for key in config.defaults():
            if config.get(section, key) is None or \
               config.get(section, key) != "":
                file.write(key + " = " + config.get("DEFAULT", key) + "\n")

        file.write("[" + section + "]\n")
        for key in config.options(section):
            if config.get(section, key) is None or \
               config.get(section, key) != "":
                file.write(key + " = " + config.get(section, key) + "\n")


def main():
    """Start an optimization of the HPOlib. For documentation see the
    comments inside this function and the general HPOlib documentation."""
    experiment_dir = os.getcwd()
    check_before_start.check_zeroth(experiment_dir)

    # Now we can safely import non standard things
    import numpy as np
    import HPOlib.Experiment as Experiment

    args, unknown = use_option_parser()
    optimizer = args.optimizer

    config_file = os.path.join(experiment_dir, "config.cfg")
    config = parse_config(config_file, allow_no_value=True, optimizer_module=optimizer)

    # Check whether we're calling a optimizer version, that links to another
    # e.g. calling tpe, but running hyperopt_august2013_mod
    if optimizer != config.get('DEFAULT', 'optimizer_version'):
        print("[INFO] You called -o %s, but this is only a link to version %s" %
              (optimizer, config.get('DEFAULT', 'optimizer_version')))
        optimizer = config.get('DEFAULT', 'optimizer_version')

    # Override config values with command line values
    config_overrides = parse_config_values_from_unknown_arguments(unknown,
                                                                  config)
    config = override_config_with_cli_arguments(config, config_overrides)
    # Saving the config file is down further at the bottom, as soon as we get
    # hold of the new optimizer directory
    wrapping_dir = os.path.dirname(os.path.realpath(__file__))

    # Try adding runsolver to path
    os.putenv('PATH', os.environ['PATH'] + ":" + wrapping_dir + "/../runsolver/src/")

    # _check_runsolver, _check_modules()
    check_before_start.check_first(experiment_dir)

    # build call
    cmd = "export PYTHONPATH=$PYTHONPATH:" + wrapping_dir + "\n"

    # Load optimizer
    try:
        optimizer_module = imp.load_source(optimizer, wrapping_dir + "/" +
                                           optimizer + ".py")
    except (ImportError, IOError):
        print "Optimizer module", optimizer, "not found"
        import traceback
        print traceback.format_exc()
        sys.exit(1)
    optimizer_call, optimizer_dir = optimizer_module.main(config=config,
                                                          options=args,
                                                          experiment_dir=
                                                          experiment_dir)
    cmd += optimizer_call

    with open(os.path.join(optimizer_dir, "config.cfg"), "w") as f:
        config.set("DEFAULT", "is_not_original_config_file", "True")
        save_config_to_file(f, config)

    # _check_function
    check_before_start.check_second(experiment_dir, optimizer_dir)

    # initialize/reload pickle file
    if args.restore:
        try:
            os.remove(os.path.join(optimizer_dir, optimizer + ".pkl.lock"))
        except OSError:
            pass
    folds = config.getint('DEFAULT', 'numberCV')
    trials = Experiment.Experiment(optimizer_dir, optimizer, folds=folds,
                                   max_wallclock_time=config.get('DEFAULT',
                                                                 'cpu_limit'),
                                   title=args.title)
    trials.optimizer = optimizer

    # TODO: We do not have any old runs anymore. DELETE this!
    if args.restore:
        # Old versions did store NaNs instead of the worst possible result for
        # crashed runs in the instance_results. In order to be able to load
        # these files, these NaNs are replaced
        for i, trial in enumerate(trials.instance_order):
            _id, fold = trial
            instance_result = trials.get_trial_from_id(_id)['instance_results'][fold]
            if not np.isfinite(instance_result):
                # Make sure that we do delete the last job which was running but
                # did not finish
                if i == len(trials.instance_order) - 1 and \
                        len(trials.cv_starttime) != len(trials.cv_endtime):
                    # The last job obviously did not finish correctly, do not
                    # replace it
                    pass
                else:
                    trials.get_trial_from_id(_id)['instance_results'][fold] = \
                        config.getfloat('DEFAULT', 'result_on_terminate')
                    # Pretty sure we need this here:
                    trials.get_trial_from_id(_id)['instance_status'][fold] = \
                        Experiment.BROKEN_STATE

        #noinspection PyBroadException
        try:
            restored_runs = optimizer_module.restore(config=config,
                                                     optimizer_dir=optimizer_dir,
                                                     cmd=cmd)
        except:
            print "Could not restore runs for %s" % args.restore
            import traceback
            print traceback.format_exc()
            sys.exit(1)

        print "Restored %d runs" % restored_runs
        trials.remove_all_but_first_runs(restored_runs)
        fh = open(os.path.join(optimizer_dir, optimizer + ".out"), "a")
        fh.write("#" * 80 + "\n" + "Restart! Restored %d runs.\n" % restored_runs)
        fh.close()

        if len(trials.endtime) < len(trials.starttime):
            trials.endtime.append(trials.cv_endtime[-1])
        trials.starttime.append(time.time())
    else:
        trials.starttime.append(time.time())
    #noinspection PyProtectedMember
    trials._save_jobs()
    del trials
    sys.stdout.flush()

    # Run call
    if args.printcmd:
        print cmd
        return 0
    else:
        print cmd
        output_file = os.path.join(optimizer_dir, optimizer + ".out")
        fh = open(output_file, "a")
        # process = subprocess.Popen(cmd, stdout=fh, stderr=fh, shell=True, executable="/bin/bash")
        # ret = process.wait()

        print "-----------------------RUNNING----------------------------------"

        if args.verbose:
            # Print std and err output for optimizer
            proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            print 'Optimizer runs with PID (+1):', proc.pid

            while proc.poll() is None:
                next_line = proc.stdout.readline()
                fh.write(next_line)
                sys.stdout.write(next_line)
                sys.stdout.flush()

        elif args.silent:
            # Print nothing
            proc = subprocess.Popen(cmd, shell=True, stdin=fh, stdout=fh, stderr=fh)
            print 'Optimizer runs with PID (+1):', proc.pid

        else:
            # Print only stderr
            proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=fh, stderr=subprocess.PIPE)
            print 'Optimizer runs with PID (+1):', proc.pid

            while proc.poll() is None:
                next_line = proc.stderr.readline()
                fh.write(next_line)
                sys.stdout.write("[ERR]:" + next_line)
                sys.stdout.flush()

        ret = proc.returncode

        print "-----------------------END--------------------------------------"

        trials = Experiment.Experiment(optimizer_dir, optimizer)
        trials.endtime.append(time.time())
        #noinspection PyProtectedMember
        trials._save_jobs()
        # trials.finish_experiment()
        total_time = 0
        print "### Best result"
        print trials.get_best()
        print "###\n### Durations"
        try:
            for starttime, endtime in zip(trials.starttime, trials.endtime):
                total_time += endtime - starttime
            print "Needed a total of %f seconds" % total_time
            print "The optimizer %s took %10.5f seconds" %\
                  (optimizer, float(calculate_optimizer_time(trials)))
            print "The overhead of HPOlib is %f seconds" % \
                  (calculate_wrapping_overhead(trials))
            print "The benchmark itself took %f seconds" % \
                  trials.total_wallclock_time
        except Exception as e:
            import HPOlib.wrapping_util
            print HPOlib.wrapping_util.format_traceback(sys.exc_info())
            print "Experiment itself went fine, " \
                  "but calculating durations of optimization failed:", sys.exc_info()[0], e
        del trials
        print "###\nFinished with return code: " + str(ret)
        return ret

if __name__ == "__main__":
    main()