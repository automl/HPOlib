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
import logging
import os
from Queue import Queue, Empty
import signal
import shlex
import subprocess
import sys
from threading import Thread
import time

import HPOlib
import HPOlib.check_before_start as check_before_start
import HPOlib.wrapping_util as wrapping_util
# Experiment is imported after we check for numpy

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
hpolib_logger = logging.getLogger("HPOlib")
hpolib_logger.setLevel(logging.INFO)
logger = logging.getLogger("HPOlib.wrapping")


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


def use_arg_parser():
    """Parse all options which can be handled by the wrapping script.
    Unknown arguments are ignored and returned as a list. It is useful to
    check this list in your program to handle typos etc.

    Returns:
        a tuple. The first element is an argparse.Namespace object,
        the second a list with all unknown arguments.
    """
    description = "Perform an experiment with HPOlib. " \
                  "Call this script from experiment directory (containing 'config.cfg')"
    epilog = "Your are using HPOlib " + HPOlib.__version__
    prog = "path/from/Experiment/to/HPOlib/wrapping.py"

    parser = ArgumentParser(description=description, prog=prog, epilog=epilog)

    parser.add_argument("-o", "--optimizer", action="store", type=str,
                        dest="optimizer",
                        help="Specify the optimizer name.", required=True)
    parser.add_argument("-p", "--print", action="store_true",
                        dest="printcmd", default=False,
                        help="If set print the command instead of executing it")
    parser.add_argument("-s", "--seed", action="store", type=int,
                        dest="seed", default=1,
                        help="Set the seed of the optimizer")
    parser.add_argument("-t", "--title", action="store", type=str,
                        dest="title", default=None,
                        help="A title for the experiment")
    parser.add_argument("--cwd", action="store", type=str, dest="working_dir",
                        default=None, help="Change the working directory to "
                        "<working_directory> prior to running the experiment")
    parser.add_argument("-r", "--restore", action="store", type=str,
                        dest="restore", default=None,
                        help="Restore the state from a given directory")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-q", "--silent", action="store_true",
                       dest="silent", default=False,
                       help="Don't print anything during optimization")
    group.add_argument("-v", "--verbose", action="store_true",
                       dest="verbose", default=False,
                       help="Print stderr/stdout for optimizer")

    args, unknown = parser.parse_known_args()
    return args, unknown


def main():
    """Start an optimization of the HPOlib. For documentation see the
    comments inside this function and the general HPOlib documentation."""
    args, unknown_arguments = use_arg_parser()

    # Convert the path to the optimizer to be an absolute path, which is
    # necessary later when we change the working directory
    optimizer = args.optimizer
    if not os.path.isabs(optimizer):
        relative_path = optimizer
        optimizer = os.path.abspath(optimizer)
        logger.info("Converting relative optimizer path %s to absolute "
                    "optimizer path %s.", relative_path, optimizer)

    if args.working_dir:
        os.chdir(args.working_dir)

    experiment_dir = os.getcwd()
    check_before_start.check_first(experiment_dir)

    # Now we can safely import non standard things
    import numpy as np
    import HPOlib.Experiment as Experiment          # Wants numpy and scipy

    # Check how many optimizer versions are present and if all dependencies
    # are installed
    optimizer_version = check_before_start.check_optimizer(optimizer)

    logger.warning("You called -o %s, I am using optimizer defined in "
                   "%sDefault.cfg", optimizer, optimizer_version)
    optimizer = os.path.basename(optimizer_version)

    config = wrapping_util.get_configuration(experiment_dir,
                                             optimizer_version, unknown_arguments)

    # Saving the config file is down further at the bottom, as soon as we get
    # hold of the new optimizer directory
    wrapping_dir = os.path.dirname(os.path.realpath(__file__))

    # TODO: We don't need this anymore, if we install HPOlib
    # Try adding runsolver to path
    os.putenv('PATH', os.environ['PATH'] + ":" + wrapping_dir + "/../runsolver/src/")

    # Load optimizer
    try:
        optimizer_dir = os.path.dirname(os.path.realpath(optimizer_version))
        optimizer_module = imp.load_source(optimizer_dir, optimizer_version + ".py")
    except (ImportError, IOError):
        logger.critical("Optimizer module %s not found", optimizer)
        import traceback
        logger.critical(traceback.format_exc())
        sys.exit(1)
    optimizer_call, optimizer_dir_in_experiment = optimizer_module.main(config=config,
                                                          options=args,
                                                          experiment_dir=experiment_dir)
    cmd = optimizer_call

    with open(os.path.join(optimizer_dir_in_experiment, "config.cfg"), "w") as f:
        config.set("HPOLIB", "is_not_original_config_file", "True")
        wrapping_util.save_config_to_file(f, config, write_nones=True)

    # initialize/reload pickle file
    if args.restore:
        try:
            os.remove(os.path.join(optimizer_dir_in_experiment, optimizer + ".pkl.lock"))
        except OSError:
            pass
    folds = config.getint('HPOLIB', 'number_cv_folds')
    trials = Experiment.Experiment(optimizer_dir_in_experiment, optimizer, folds=folds,
                                   max_wallclock_time=config.get('HPOLIB',
                                                                 'cpu_limit'),
                                   title=args.title)
    trials.optimizer = optimizer_version

    if args.restore:
        #noinspection PyBroadException
        try:
            restored_runs = optimizer_module.restore(config=config,
                                                     optimizer_dir=optimizer_dir_in_experiment,
                                                     cmd=cmd)
        except:
            logger.critical("Could not restore runs for %s", args.restore)
            import traceback
            logger.critical(traceback.format_exc())
            sys.exit(1)

        logger.info("Restored %d runs", restored_runs)
        trials.remove_all_but_first_runs(restored_runs)
        fh = open(os.path.join(optimizer_dir_in_experiment, optimizer + ".out"), "a")
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
        logger.info(cmd)
        return 0
    else:
        # call target_function.setup()
        fn_setup = config.get("HPOLIB", "function_setup")
        if fn_setup:
            try:
                output = subprocess.check_output(fn_setup, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                logger.critical(e.output)
                sys.exit(1)

        logger.info(cmd)
        output_file = os.path.join(optimizer_dir_in_experiment, optimizer + ".out")
        fh = open(output_file, "a")
        cmd = shlex.split(cmd)
        print cmd
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, preexec_fn=os.setsid)
        logger.info("-----------------------RUNNING----------------------------------")
        # http://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python
        # How often is the experiment pickle supposed to be opened?
        if config.get("HPOLIB", "total_time_limit"):
            optimizer_end_time = time.time() + config.getint("HPOLIB", "total_time_limit")
        else:
            optimizer_end_time = sys.float_info.max

        last_output = time.time()
        printed_start_configuration = list()
        printed_end_configuration = list()
        current_best = -1
        sent_SIGTERM = False
        sent_SIGKILL = False
        # After the evaluation finished, we scan the experiment pickle twice
        # to print everything!
        minimal_runs_to_go = 2

        def enqueue_output(out, queue):
            for line in iter(out.readline, b''):
                queue.put(line)
            out.close()

        stderr_queue = Queue()
        stdout_queue = Queue()
        stderr_thread = Thread(target=enqueue_output, args=(proc.stderr, stderr_queue))
        stdout_thread = Thread(target=enqueue_output, args=(proc.stdout, stdout_queue))
        stderr_thread.daemon = True
        stdout_thread.daemon = True
        stderr_thread.start()
        stdout_thread.start()
        logger.info('Optimizer runs with PID: %d', proc.pid)

        while minimal_runs_to_go > 0:     # Think of this as a do-while loop...
            try:
                while True:
                    line = stdout_queue.get_nowait()
                    fh.write(line)

                    # Write to stdout only if verbose is on
                    if args.verbose:
                        sys.stdout.write(line)
                        sys.stdout.flush()
            except Empty:
                pass

            try:
                while True:
                    line = stderr_queue.get_nowait()
                    fh.write(line)

                    # Write always, except silent is on
                    if not args.silent:
                        sys.stderr.write("[ERR]:" + line)
                        sys.stderr.flush()
            except Empty:
                pass

            if time.time() > optimizer_end_time and not sent_SIGTERM:
                os.killpg(proc.pid, signal.SIGTERM)
                sent_SIGTERM = True

            if time.time() > optimizer_end_time + 200 and not sent_SIGKILL:
                os.killpg(proc.pid, signal.SIGKILL)
                sent_SIGKILL = True

            fh.flush()
            # necessary, otherwise HPOlib-run takes 100% of one processor
            time.sleep(0.1)

            if not (args.verbose or args.silent) and time.time() - last_output > 1:
                trials = Experiment.Experiment(optimizer_dir_in_experiment,
                                               optimizer)

                for i in range(len(printed_end_configuration), len(trials.instance_order)):
                    configuration = trials.instance_order[i][0]
                    fold = trials.instance_order[i][1]
                    if i + 1 > len(printed_start_configuration):
                        logger.info("Starting configuration %5d, fold %2d",
                                    configuration, fold)
                        printed_start_configuration.append(i)

                    if np.isfinite(trials.trials[configuration]
                                   ["instance_results"][fold]):
                        last_result = trials.trials[configuration] \
                            ["instance_results"][fold]
                        tmp_current_best = trials.get_arg_best()
                        if tmp_current_best <= i:
                            current_best = tmp_current_best
                        # Calculate current best
                        # Check if last result is finite, if not calc nanmean over all instances
                        dct_helper = trials.trials[current_best]
                        res = dct_helper["result"] if \
                            np.isfinite(dct_helper["result"]) \
                            else wrapping_util.nan_mean(dct_helper["instance_results"])
                            #np.nanmean(trials.trials[current_best]["instance_results"])
                            # nanmean does not work for all numpy version
                        logger.info("Result %10f, current best %10f",
                                    last_result, res)
                        printed_end_configuration.append(i)

                del trials
                last_output = time.time()

                if proc.poll() is not None:       # the end condition for the
                    minimal_runs_to_go -= 1       # do-while loop

            elif args.verbose or args.silent:
                if proc.poll() is not None:
                    minimal_runs_to_go -= 1

        ret = proc.returncode

        logger.info("-----------------------END--------------------------------------")
        fh.close()

        # call target_function.teardown()
        fn_teardown = config.get("HPOLIB", "function_teardown")
        if fn_teardown:
            try:
                output = subprocess.check_output(fn_teardown, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                logger.critical(e.output)
                sys.exit(1)

        trials = Experiment.Experiment(optimizer_dir_in_experiment, optimizer)
        trials.endtime.append(time.time())
        #noinspection PyProtectedMember
        trials._save_jobs()
        # trials.finish_experiment()
        total_time = 0
        logger.info("Best result")
        logger.info(trials.get_best())
        logger.info("Durations")
        try:
            for starttime, endtime in zip(trials.starttime, trials.endtime):
                total_time += endtime - starttime
            logger.info("Needed a total of %f seconds", total_time)
            logger.info("The optimizer %s took %10.5f seconds",
                  optimizer, float(calculate_optimizer_time(trials)))
            logger.info("The overhead of HPOlib is %f seconds",
                  calculate_wrapping_overhead(trials))
            logger.info("The benchmark itself took %f seconds" % \
                  trials.total_wallclock_time)
        except Exception as e:
            logger.error(HPOlib.wrapping_util.format_traceback(sys.exc_info()))
            logger.error("Experiment itself went fine, but calculating "
                         "durations of optimization failed: %s %s",
                         sys.exc_info()[0], e)
        del trials
        logger.info("Finished with return code: " + str(ret))
        return ret

if __name__ == "__main__":
    main()
