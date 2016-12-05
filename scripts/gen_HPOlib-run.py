#!/usr/bin/env python

from argparse import ArgumentParser
import os
import sys


def my_range(start, end, step):
    while start <= end:
        yield start
        start += step


def main():
    description = "Output's HPOlib-run commands for test functions"
    parser = ArgumentParser(description=description)

    # General Options
    parser.add_argument("-b", "--benchmark", dest="benchmark_dir",
                        default="../benchmarks/rkhs", help="specify benchmark name")
    parser.add_argument("-opt", "--optDir", dest="optimizerDir",
                        default="../optimizers", help="path to optimizers directory")
    parser.add_argument("-o", "--outputFile", dest="Output",
                        default="HPOlib-run_commands", help="outputFile")
    parser.add_argument("-start-seed", "--start_seed", default=1000,
                        type=int, help="seed start value")
    parser.add_argument("-step-seed", "--step_seed", default=1000,
                        type=int, help="seed step value")
    parser.add_argument("-stop-seed", "--stop_seed", default=10000,
                        type=int, help="seed stop value")
    parser.add_argument("--ignore-missing-optimizers", action="store_true",
                        help="Ignore missing optimizers. If not present, "
                             "this script will stop if an installed optimizer is"
                             "not found in the benchmark directory.")

    args, unknown = parser.parse_known_args()

    if os.path.isabs(args.optimizerDir):
        optimizers_dir = args.optimizerDir
    else:
        optimizers_dir = os.path.abspath(args.optimizerDir)

    if os.path.isabs(args.benchmark_dir):
        benchmark_dir = args.benchmark_dir
    else:
        benchmark_dir = os.path.abspath(args.benchmark_dir)

    output = args.Output
    start_seed = args.start_seed
    step_seed = args.step_seed
    stop_seed = args.stop_seed

    optimizers = {}
    try:
        opts = os.walk(optimizers_dir).next()[1]
    except StopIteration:
        print "Error! Optimizer Directory \"" + optimizers_dir + "\" doesn't exist. Please enter correct directory"
        sys.exit(1)

    for opt in opts:
        optVersions = []
        if opt != "ConfigurationRunner":
            optimizerDir = os.path.join(optimizers_dir, opt)
            insideOpt = os.walk(optimizerDir).next()[2]
            for s in filter(lambda x: ".cfg" in x, insideOpt):
                optVersions.append(s[0:-11])
            optimizers[opt] = optVersions

    if sum([len(v) for k, v in optimizers.iteritems()]) == 0:
        print "no *.cfg files found in given optimizers directory"
        sys.exit(1)

    outputfile = open(output, 'w')

    try:
        dirs = os.walk(benchmark_dir, followlinks=True).next()[1]
    except StopIteration:
        print "Error! Benchmark Directory \"" + benchmark_dir + "\" doesn't exist. Please enter correct directory"
        sys.exit(1)

    for seed in my_range(start_seed, stop_seed, step_seed):
        for optimizer in optimizers:
            for optVersion in optimizers[optimizer]:
                if optVersion in dirs:
                    path = os.path.join(optimizers_dir, optimizer, optVersion)
                    outputfile.write("HPOlib-run --cwd " + benchmark_dir + " -o " + path + " -s " + str(seed) + "\n")
                else:
                    print "Optimizer " + optVersion + " doesn't exist in directory \"" + benchmark_dir + "\""
                    if not args.ignore_missing_optimizers:
                        sys.exit(1)

    print "******************HPOlib-run Commands********************"
    print "Using Benchmark function directory: " + benchmark_dir
    print "Using Optimizers directory: " + optimizers_dir
    print "Optimizers found:"
    for i in [k for k, v in optimizers.iteritems()]:
        print " -" + i
    print "Results are output in file: " + os.path.abspath(args.Output)


if __name__ == '__main__':
    main()
