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
import psutil
import os
from Queue import Queue, Empty
import signal
import shlex
import shutil
import subprocess
import sys
from threading import Thread
import time
import warnings
import HPOlib
import HPOlib.check_before_start as check_before_start
import HPOlib.wrapping_util as wrapping_util
import HPOlib.dispatcher.runsolver_wrapper as runsolver_wrapper
# Import experiment only after the check for numpy succeeded

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"

INFODEVEL = """
##############################################################################
# Your are using the DEVELOPMENT version. This means we might change things  #
# on a daily basis, commit untested code and remove or add features without  #
# announcements. We do not intend to break any functionality, but cannot     #
# guarantee to not do it.                                                    #
##############################################################################
"""
IS_DEVELOPMENT = True

hpolib_logger = logging.getLogger("HPOlib")
logger = logging.getLogger("HPOlib.wrapping")


def calculate_wrapping_overhead(trials):
    wrapping_time = 0
    for times in zip(trials.cv_starttime, trials.cv_endtime):
        wrapping_time += times[1] - times[0]

    # We need to import numpy again
    import numpy as np
    benchmark_time = 0
    for t in trials.trials:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    if args.working_dir:
        experiment_dir = args.working_dir
    elif args.restore:
        args.restore = os.path.abspath(args.restore) + "/"
        experiment_dir = args.restore
    else:
        experiment_dir = os.getcwd()

    formatter = logging.Formatter('[%(levelname)s] [%(asctime)s:%(name)s] %('
                                  'message)s', datefmt='%H:%M:%S')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    hpolib_logger.addHandler(handler)
    hpolib_logger.setLevel(1)

    # First of all print the infodevel
    if IS_DEVELOPMENT:
        logger.critical(INFODEVEL)

    args, unknown_arguments = use_arg_parser()

    # Convert the path to the optimizer to be an absolute path, which is
    # necessary later when we change the working directory
    optimizer = args.optimizer
    print("opti:", optimizer)
    if not os.path.isabs(optimizer):
        relative_path = optimizer
        optimizer = os.path.abspath(optimizer)
        logger.info("Converting relative optimizer path %s to absolute "
                    "optimizer path %s.", relative_path, optimizer)

    os.chdir(experiment_dir)
    experiment_dir = os.getcwd()

    check_before_start.check_first(experiment_dir)

    # Now we can safely import non standard things
    import numpy as np
    import HPOlib.Experiment as Experiment          # Wants numpy and scipy

    # Check how many optimizer versions are present and if all dependencies
    # are installed also dynamically load optimizer obj
    optimizer_version, opt_obj = check_before_start.check_optimizer(optimizer)

    logger.warning("You called -o %s, I am using optimizer defined in "
                   "%sDefault.cfg", optimizer, optimizer_version)
    optimizer = os.path.basename(optimizer_version)

    config = wrapping_util.get_configuration(experiment_dir,
                                             optimizer_version, unknown_arguments, opt_obj)

    # DO NOT LOG UNTIL HERE UNLESS SOMETHING DRAMATIC HAS HAPPENED!!!
    loglevel = config.getint("HPOLIB", "HPOlib_loglevel")
    hpolib_logger.setLevel(loglevel)
    if args.silent:
        hpolib_logger.setLevel(60)
    if args.verbose:
        hpolib_logger.setLevel(10)

    # Saving the config file is down further at the bottom, as soon as we get
    # hold of the new optimizer directory
    # wrapping_dir = os.path.dirname(os.path.realpath(__file__))

    # Load optimizer
    try:
        optimizer_dir = os.path.dirname(os.path.realpath(optimizer_version))
        optimizer_module = imp.load_source(optimizer_dir, optimizer_version + ".py")
    except (ImportError, IOError):
        logger.critical("Optimizer module %s not found", optimizer)
        import traceback
        logger.critical(traceback.format_exc())
        sys.exit(1)

    # So the optimizer module can acces the seed from the config and
    config.set("HPOLIB", "seed", str(args.seed))
    experiment_directory_prefix = config.get("HPOLIB", "experiment_directory_prefix")

    optimizer_call, optimizer_dir_in_experiment = \
        opt_obj.main(config=config, options=args,
                     experiment_dir=experiment_dir)
                     # experiment_directory_prefix=experiment_directory_prefix)
    cmd = optimizer_call

    # Start the server for logging from subprocesses here, because its port must
    # be written to the config file.
    logging_host = config.get("HPOLIB", "logging_host")
    if logging_host:
        logging_receiver_thread = None
        default_logging_port = DEFAULT_TCP_LOGGING_PORT

        for logging_port in range(default_logging_port, 65535):
            try:
                logging_receiver = logging_server.LoggingReceiver(
                    host=logging_host, port=logging_port,
                    handler=logging_server.LogRecordStreamHandler)
                logging_receiver_thread = Thread(target=logging_receiver.serve_forever)
                logging_receiver_thread.daemon = True
                logger.info('%s started at %s' % (
                    logging_receiver.__class__.__name__,
                    logging_receiver.server_address))
                logging_receiver_thread.start()
                break
            # TODO I did not find any useful documentation about which Exceptions
            #  I should catch here...
            except Exception as e:
                logger.debug(e)
                logger.debug(e.message)

        if logging_receiver_thread is None:
            logger.critical("Could not create the logging server. Going to shut "
                            "down.")
            sys.exit(1)

        config.set("HPOLIB", "logging_port", str(logging_port))

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

    trials = Experiment.Experiment(expt_dir=optimizer_dir_in_experiment,
                                   expt_name=experiment_directory_prefix + optimizer,
                                   folds=folds,
                                   max_wallclock_time=config.get('HPOLIB',
                                                                 'cpu_limit'),
                                   title=args.title)
    trials.optimizer = optimizer_version

    optimizer_output_file = os.path.join(optimizer_dir_in_experiment, optimizer + wrapping_util.get_time_string() +
                                         "_" + str(args.seed) + ".out")
    if args.restore:
        # noinspection PyBroadException
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
        fh = open(optimizer_output_file, "a")
        fh.write("#" * 80 + "\n" + "Restart! Restored %d runs.\n" % restored_runs)
        fh.close()

        if len(trials.endtime) < len(trials.starttime):
            trials.endtime.append(trials.cv_endtime[-1])
        trials.starttime.append(time.time())
    else:
        trials.starttime.append(time.time())
    # noinspection PyProtectedMember
    trials._save_jobs()
    del trials
    sys.stdout.flush()

    # Run call
    if args.printcmd:
        logger.info(cmd)
        return 0
    else:
        # Create a second formatter and handler to customize the optimizer
        # output
        optimization_formatter = logging.Formatter(
            '[%(levelname)s] [%(asctime)s:%(optimizer)s] %(message)s',
            datefmt='%H:%M:%S')
        optimization_handler = logging.StreamHandler(sys.stdout)
        optimization_handler.setFormatter(optimization_formatter)
        optimization_logger = logging.getLogger(optimizer)
        optimization_logger.addHandler(optimization_handler)
        optimizer_loglevel = config.getint("HPOLIB", "optimizer_loglevel")
        optimization_logger.setLevel(optimizer_loglevel)

        # Use a flag which is set to true as soon as all children are
        # supposed to be killed
        exit_ = wrapping_util.Exit()
        signal.signal(signal.SIGTERM, exit_.signal_callback)
        signal.signal(signal.SIGABRT, exit_.signal_callback)
        signal.signal(signal.SIGINT, exit_.signal_callback)
        signal.signal(signal.SIGHUP, exit_.signal_callback)

        # Change into the current experiment directory
        # Some optimizer might expect this
        dir_before_exp = os.getcwd()

        temporary_output_dir = config.get("HPOLIB", "temporary_output_directory")
        if temporary_output_dir:
            last_part = os.path.split(optimizer_dir_in_experiment)[1]
            temporary_output_dir = os.path.join(temporary_output_dir, last_part)

            # Replace any occurence of the path in the command
            cmd = cmd.replace(optimizer_dir_in_experiment, temporary_output_dir)
            optimizer_output_file = optimizer_output_file.replace(optimizer_dir_in_experiment, temporary_output_dir)

            shutil.copytree(optimizer_dir_in_experiment, temporary_output_dir)

            # shutil.rmtree does not work properly with NFS
            # https://github.com/hashdist/hashdist/issues/113
            # Idea from https://github.com/ahmadia/hashdist/
            for rmtree_iter in range(5):
                try:
                    shutil.rmtree(optimizer_dir_in_experiment)
                    break
                except OSError, e:
                    time.sleep(rmtree_iter)

            optimizer_dir_in_experiment = temporary_output_dir

        # call target_function.setup()
        fn_setup = config.get("HPOLIB", "function_setup")
        if fn_setup:
            # if temporary_output_dir:
            #    logger.critical("The options 'temporary_output_directory' "
            #                    "and 'function_setup' cannot be used "
            #                    "together.")
            #    sys.exit(1)

            fn_setup_output = os.path.join(optimizer_dir_in_experiment,
                                           "function_setup.out")
            runsolver_cmd = runsolver_wrapper._make_runsolver_command(
                config, fn_setup_output)
            setup_cmd = runsolver_cmd + " " + fn_setup
            # runsolver_output = subprocess.STDOUT
            runsolver_output = open("/dev/null")
            runsolver_wrapper._run_command_with_shell(setup_cmd,
                                                      runsolver_output)

        os.chdir(optimizer_dir_in_experiment)

        logger.info(cmd)
        output_file = optimizer_output_file
        fh = open(output_file, "a")
        cmd = shlex.split(cmd)
        print cmd

        # See man 7 credentials for the meaning of a process group id
        # This makes wrapping.py useable with SGEs default behaviour,
        # where qdel sends a SIGKILL to a whole process group
        # logger.info(os.getpid())
        # os.setpgid(os.getpid(), os.getpid())    # same as os.setpgid(0, 0)
        # TODO: figure out why shell=True was removed in commit f47ac4bb3ffe7f70b795d50c0828ca7e109d2879
        # maybe it has something todo with the previous behaviour where a
        # session id was set...
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        global child_process_pid
        child_process_pid = proc.pid
        process = psutil.Process(os.getpid())

        logger.info("-----------------------RUNNING----------------------------------")
        # http://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python
        # How often is the experiment pickle supposed to be opened?
        if config.get("HPOLIB", "total_time_limit"):
            optimizer_end_time = time.time() + config.getint("HPOLIB", "total_time_limit")
        else:
            optimizer_end_time = sys.float_info.max

        sent_SIGINT = False
        sent_SIGINT_time = np.inf
        sent_SIGTERM = False
        sent_SIGTERM_time = np.inf
        sent_SIGKILL = False
        sent_SIGKILL_time = np.inf
        children_to_kill = list()

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
        if not (args.verbose or args.silent):
            logger.info('Optimizer runs with PID: %d', proc.pid)
            logger.info('We start in directory %s', os.getcwd())
        while True:
            # this implements the total runtime limit
            if time.time() > optimizer_end_time and not sent_SIGINT:
                logger.info("Reached total_time_limit, going to shutdown.")
                exit_.true()

            # necessary, otherwise HPOlib-run takes 100% of one processor
            time.sleep(0.25)

            try:
                while True:
                    line = stdout_queue.get_nowait()
                    fh.write(line)
                    fh.flush()

                    optimization_logger.info(line.replace("\n", ""),
                                             extra={'optimizer': optimizer})
            except Empty:
                pass

            try:
                while True:
                    line = stderr_queue.get_nowait()
                    fh.write(line)
                    fh.flush()

                    optimization_logger.error(line.replace("\n", ""),
                                              extra={'optimizer': optimizer})
            except Empty:
                pass

            ret = proc.poll()
            if ret is not None:
                # This does not include wrapping.py
                children = process.children()
                if len(children) == 0:
                    break
            # TODO: what happens if we have a ret but something is still
            # running?

            if exit_.get_exit() == True and not sent_SIGINT:
                logger.critical("Shutdown procedure: Sending SIGINT")
                wrapping_util.kill_processes(signal.SIGINT)
                sent_SIGINT_time = time.time()
                sent_SIGINT = True

            if exit_.get_exit() == True and not sent_SIGTERM and time.time() \
                    > sent_SIGINT_time + 5:
                logger.critical("Shutdown procedure: Sending SIGTERM")
                wrapping_util.kill_processes(signal.SIGTERM)
                sent_SIGTERM_time = time.time()
                sent_SIGTERM = True

            if exit_.get_exit() == True and not sent_SIGKILL and time.time() \
                    > sent_SIGTERM_time + 5:
                logger.critical("Shutdown procedure: Sending SIGKILL")
                wrapping_util.kill_processes(signal.SIGKILL)
                sent_SIGKILL_time = time.time()
                sent_SIGKILL = True

        logger.info("-----------------------END--------------------------------------")
        ret = proc.returncode
        logger.info("Finished with return code: %d", ret)
        del proc

        fh.close()

        # Change back into to directory
        os.chdir(dir_before_exp)

        # call target_function.setup()
        fn_teardown = config.get("HPOLIB", "function_teardown")
        if fn_teardown:
            # if temporary_output_dir:
            #    logger.critical("The options 'temporary_output_directory' "
            #                    "and 'function_teardown' cannot be used "
            #                    "together.")
            #    sys.exit(1)

            fn_teardown_output = os.path.join(optimizer_dir_in_experiment,
                                              "function_teardown.out")
            runsolver_cmd = runsolver_wrapper._make_runsolver_command(
                config, fn_teardown_output)
            teardown_cmd = runsolver_cmd + " " + fn_teardown
            # runsolver_output = subprocess.STDOUT
            runsolver_output = open("/dev/null")
            runsolver_wrapper._run_command_with_shell(teardown_cmd,
                                                      runsolver_output)

        if temporary_output_dir:
            # We cannot be sure that the directory
            # optimizer_dir_in_experiment in dir_before_exp got deleted
            # properly, therefore we append an underscore to the end of the
            # filename
            last_part = os.path.split(optimizer_dir_in_experiment)[1]
            new_dir = os.path.join(dir_before_exp, last_part)
            try:
                shutil.copytree(optimizer_dir_in_experiment, new_dir)
            except OSError as e:
                new_dir += "_"
                shutil.copytree(optimizer_dir_in_experiment, new_dir)

            # shutil.rmtree does not work properly with NFS
            # https://github.com/hashdist/hashdist/issues/113
            # Idea from https://github.com/ahmadia/hashdist/
            for rmtree_iter in range(5):
                try:
                    shutil.rmtree(optimizer_dir_in_experiment)
                    break
                except OSError, e:
                    time.sleep(rmtree_iter)

            optimizer_dir_in_experiment = new_dir

        trials = Experiment.Experiment(optimizer_dir_in_experiment,
                                       experiment_directory_prefix + optimizer)
        trials.endtime.append(time.time())
        # noinspection PyProtectedMember
        trials._save_jobs()
        # trials.finish_experiment()
        total_time = 0
        logger.info("Best result %f", trials.get_best())
        logger.info("Durations")
        try:
            for starttime, endtime in zip(trials.starttime, trials.endtime):
                total_time += endtime - starttime
            logger.info("  Needed a total of %f seconds", total_time)
            logger.info("  The optimizer %s took %10.5f seconds",
                        optimizer, float(calculate_optimizer_time(trials)))
            logger.info("  The overhead of HPOlib is %f seconds",
                        calculate_wrapping_overhead(trials))
            logger.info("  The benchmark itself took %f seconds" % \
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
