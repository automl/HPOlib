from collections import OrderedDict
import os
import StringIO
import subprocess

import numpy as np

from HPOlib.wrapping_util import flatten_parameter_dict
import HPOlib.optimization_interceptor

def construct_cli_call(cli_target, params):
    cli_call = StringIO.StringIO()
    cli_call.write("python -m ")
    cli_call.write(cli_target)
    cli_call.write(" --params")
    params = flatten_parameter_dict(params)
    params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))
    for param in params:
        cli_call.write(" -" + param + " \"'" + str(params[param]) + "'\"")
    return cli_call.getvalue()


def command_line_function(params, cli_target):
    call = construct_cli_call(cli_target, params)
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


def main(job_id, params):
    """Implement the Spearmint interface and then call HPOlib"""
    cli_target = HPOlib.optimization_interceptor.__file__
    cli_target = os.path.splitext(cli_target)[0]
    result = command_line_function(params, cli_target)
    return result


