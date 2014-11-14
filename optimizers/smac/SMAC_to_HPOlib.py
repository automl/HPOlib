from collections import OrderedDict
import os
import re
import StringIO
import subprocess
import sys
import time

import numpy as np

from HPOlib.wrapping_util import remove_param_metadata
import HPOlib.optimization_interceptor


def construct_cli_call(cli_target, fold, params):
    cli_call = StringIO.StringIO()
    cli_call.write("python -m ")
    cli_call.write(cli_target)
    cli_call.write(" --fold %d" % fold)
    cli_call.write(" --params")
    params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))
    for param in params:
        cli_call.write(" -" + param + " \"'" + str(params[param]) + "'\"")
    return cli_call.getvalue()


def command_line_function(params, fold, cli_target):
    call = construct_cli_call(cli_target, fold, params)
    output = subprocess.check_output(call, shell=True)

    lines = output.split("\n")

    result = np.Inf
    for line in lines:
        pos = line.find("Result:")
        if pos != -1:
            result_string = line[pos:]
            result_array = result_string.split()
            result = float(result_array[1].strip(","))
            break

    # Parse the CLI
    return result


def parse_command_line():
    fold = int(sys.argv[1])
    seed = int(sys.argv[5])
    return fold, seed


def get_parameters():
    params = dict(zip(sys.argv[6::2], sys.argv[7::2]))
    # Now remove the leading minus
    for key in params.keys():
        new_key = re.sub('^-', '', key)
    params[new_key] = params[key]
    del params[key]
    remove_param_metadata(params)
    params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))
    return params


def format_return_string(status, runtime, runlength, quality, seed,
                         additional_data):
    return_string = "Result for ParamILS: %s, %f, %d, %f, %d, %s" % \
                    (status, runtime, runlength, quality, seed, additional_data)

    return return_string


def main():
    """Implement the Spearmint interface and then call HPOlib"""
    cli_target = HPOlib.optimization_interceptor.__file__
    cli_target = os.path.splitext(cli_target)[0]

    fold, seed = parse_command_line()
    params = get_parameters()

    starttime = time.time()
    result = command_line_function(params, fold, cli_target)
    endtime = time.time()
    print format_return_string("SAT", endtime - starttime, 1, result, seed, "")
    return result


if __name__ == "__main__":
    main()