import argparse
import ConfigParser
import cPickle
import os
import StringIO
import sys

import HPOlib.Plotting.plot_util as plot_util
import HPOlib.wrapping_util as wrapping_util
from HPOlib.Locker import Locker


def get_num_terminates(result_on_terminate, trials):
    terminates = 0
    for trial in trials["trials"]:
        if trial["result"] >= result_on_terminate - 0.00001:
            terminates += 1

    return terminates


def collect_results(directory):
    locker = Locker()
    sio = StringIO.StringIO()
    sio.write("Statistics for %s\n" % directory)
    sio.write("%30s | %6s | %7s/%7s/%7s | %5s\n" %
             ("Optimizer", "Seed", "#conf",
             "#runs", "#crashs", "best"))


    subdirs = os.listdir(directory)
    subdirs.sort()
    results = []
    # Look for all sub-experiments
    for subdir in subdirs:
        if os.path.isdir(subdir):
            # Get the experiment pickle...
            for possible_experiment_pickle in os.listdir(subdir):
                # Some simple checks for an experiment pickle
                if possible_experiment_pickle[-4:] == ".pkl" and \
                        possible_experiment_pickle[:-4] in subdir:
                    exp_pkl = os.path.join(subdir, possible_experiment_pickle)
                    locker.lock(exp_pkl)
                    with open(exp_pkl) as fh:
                        try:
                            pkl = cPickle.load(fh)
                        except Exception as e:
                            print exp_pkl, type(e)
                            continue
                    locker.unlock(exp_pkl)

                    cfg = ConfigParser.ConfigParser()
                    cfg.read(os.path.join(subdir, "config.cfg"))

                    optimizer = pkl["optimizer"]
                    optimizer = optimizer.split("/")[-1]

                    configurations = len(pkl["trials"])
                    instance_runs = len(pkl["instance_order"])

                    # HPOlib < 0.1 don't have a seed in the config, try to
                    # infer it
                    try:
                        seed = cfg.getint("HPOLIB", "seed")
                    except:
                        seed = subdir.replace(possible_experiment_pickle[:-4],
                                              "")
                        seed = seed[1:].split("_")[0]
                        seed = int(seed)

                    worst_possible_result = cfg.getfloat\
                        ("HPOLIB", "result_on_terminate")
                    crashs = get_num_terminates(worst_possible_result, pkl)
                    try:
                        best_performance = plot_util.get_best(pkl)
                    except Exception as e:
                        print e, exp_pkl
                        continue
                    results.append([optimizer, int(seed), configurations,
                                    instance_runs, crashs, best_performance])

    def comparator(left, right):
        if left[0] < right[0]:
            return -1
        elif left[0] > right[0]:
            return 1
        else:
            if left[1] < right[1]:
                return -1
            elif left[1] > right[1]:
                return 1
            else:
                return 0

    results.sort(cmp=comparator)
    for result in results:
        sio.write("%30s | %6d | %7d/%7d/%7d | %5f\n"
                  % (result[0], result[1], result[2],
                     result[3], result[4], result[5]))
    return sio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print several statistics "
                                                 "for an experiment directory.")
    parser.add_argument('directory', type=str, default=os.getcwd(), nargs="?",
                        help="Directory for which statistics should be "
                             "printed. Default is the current working "
                             "directory.")
    args = parser.parse_args()
    sio = collect_results(args.directory)
    print sio.getvalue()