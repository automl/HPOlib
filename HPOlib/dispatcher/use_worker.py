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

import logging
import os
import re
import subprocess
import StringIO
import time
import json

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


logging.basicConfig(format='[%(levelname)s] [%(asctime)s:%(name)s] %('
                           'message)s', datefmt='%H:%M:%S')
hpolib_logger = logging.getLogger("HPOlib")
hpolib_logger.setLevel(logging.INFO)
logger = logging.getLogger("HPOlib.dispatcher.use_worker")

json_template = \
"""
[ {
  "@rc-id" : %d,
  "rc-algo-exec-config" : {
    "@algo-exec-config-id" : %d,
    "algo-exec" : "%s --__IGNORE__",
    "algo-exec-dir" : "%s",
    "algo-pcs" : {
      "@pcs-id" : %d,
      "pcs-filename" : "%s",
      "pcs-text" : "-param {--params} [--params]\\n%s",
      "pcs-subspace" : { }
    },
    "algo-cutoff" : 1.7976931348623157E308,
    "algo-deterministic" : true,
    "algo-tae-context" : { }
  },
  "rc-cutoff" : 1.7976931348623157E308,
  "rc-pisp" : {
    "@pisp-id" : %d,
    "pisp-pi" : {
      "@pi-id" : %d,
      "pi-name" : "no_instance",
      "pi-instance-specific-info" : "0",
      "pi-features" : { },
      "pi-instance-id(deprecated)" : 1
    },
    "pisp-seed" : 1
  },
  "rc-pc" : {
    "@pc-id" : %d,
    "pc-pcs" : {
      "@pcs-id" : %d
    },
    "pc-settings" : {
      "-param" : "--params",
      %s
    },
    "pc-active-parameters" : [],
    "pc-forbidden" : false,
    "pc-default" : true
  },
  "rc-sample-idx" : 0
} ]
"""

def make_command(cfg, fold, test=False):
    if test:
        fn = cfg.get("HPOLIB", "test_function")
        # TODO: test if test_function exists! probably in the startup script!
    else:
        fn = cfg.get("HPOLIB", "function")

    python_cmd = "runsolver -w /dev/null " + fn + " --fold %d --folds %d" % \
                      (fold, cfg.getint("HPOLIB", "number_cv_folds"))
    # Do not write the actual task in quotes because runsolver will not work
    # then; also we need use-pty and timestamp so that the "solver" output
    # is flushed to the output directory
    return python_cmd


def _make_worker_command(cfg, output_filename):
    cmd = cfg.get("HPOLIB", "leading_runsolver_info")
    cmd += " runsolver -o \"%s\" --timestamp --use-pty" % output_filename
    if cfg.get('HPOLIB', 'runsolver_time_limit'):
        cmd += " -W %d" % cfg.getint('HPOLIB', 'runsolver_time_limit')
    if cfg.get('HPOLIB', 'cpu_limit'):
        cmd += " -C %d" % cfg.getint('HPOLIB', 'cpu_limit')
    if cfg.get('HPOLIB', 'memory_limit'):
        cmd += " -M %d" % cfg.getint('HPOLIB', 'memory_limit')
    delay = 0
    if delay is not None:
        cmd += " -d %d" % int(delay)
    return cmd


def store_target_algorithm_calls(path, wallclock_time, result, additional_data,
                                 call):
    # Save the call to the target algorithm
    if not os.path.exists(path):
        fh = open(path, 'w')
        fh.write(",".join(["RESULT", "DURATION", "ADDITIONAL_INFO", "CALL"]) +
                 "\n")
    else:
            fh = open(path, "a")

    try:
        fh.write(",".join([str(result), str(wallclock_time),
                               str(additional_data), call]) + "\n")
    finally:
        fh.close()
    return


def dispatch(cfg, fold, params, test=False):
    #param_string = " ".join([key + " " + str(params[key]) for key in params])
    #time_string = wrapping_util.get_time_string()
    #run_instance_output = os.path.join(os.getcwd(),
    #                                   time_string + "_run_instance.out")
    #runsolver_output_file = os.path.join(os.getcwd(),
    #                                     time_string + "_runsolver.out")
    # cmd = make_command(cfg, fold, param_string, run_instance_output, test=test)
    import shlex

    # Values we need for json
    t = int(time.time() * 1000000) % 10000
    rc_id = os.getpid()
    algo_exec_config_id = t
    pcs_id = t
    pisp_id = t
    pi_id = t
    pc_id = t

    algo_exec = make_command(cfg=cfg, fold=fold, test=test)
    algo_exec_dir = os.getcwd()
    pcs_filename = str(t) + str(os.getpid())
    pcs_text = "\\n".join(["%s {%s} [%s]" % (p, params[p], params[p]) for p in params])
    pc_setting = ",\n".join(['"%s" : "%s"' % (p, params[p]) for p in params])

    json_str = json_template % (rc_id, algo_exec_config_id, algo_exec,
                                algo_exec_dir, pcs_id, pcs_filename, pcs_text,
                                pisp_id, pi_id, pc_id, pcs_id, pc_setting)
    cmd = ["json-executor", "--tae", "MYSQLDB", "--mysqldbtae-pool", "branin",
           "--mysqldbtae-username", "aad", "--mysqldbtae-password", "***",
           "--mysqldbtae-hostname", "metasql.rz.ki.privat", "--mysqldbtae-port",
           "3306", "--mysqldbtae-database", "braninSearch"]
    logger.critical(json_str)

    starttime = time.time()
    proc = subprocess.Popen(args=cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    stdoutdata, stderrdata = proc.communicate(json_str)
    endtime = time.time()

    logger.critical("JSON")
    logger.critical(json_str)
    logger.critical("STDERR")
    logger.critical(stderrdata)
    logger.critical("STDOUT")
    logger.critical(stdoutdata)

    startstring = "\n******JSON******\n"
    endstring = "\n********\n"

    json_start = stdoutdata.find(startstring)
    json_end = stdoutdata.find(endstring)

    logger.critical(json_start)
    logger.critical(json_end)

    if json_start != json_end and json_start < json_end:
        raw_json_return = stdoutdata[json_start + len(startstring):json_end]
        jason = json.loads(raw_json_return)
        logger.critical(jason)
        result = jason[0]["r-quality"]
        status = jason[0]["r-run-result"]
        additional_data = jason[0]["r-addl-run-data"]
        wallclock_time = jason[0]["r-wallclock-time"]
        runtime = jason[0]["r-runtime"]
        logger.critical(str(result) + " " + str(type(result)))
    else:
        logger.critical("Dispatcher 'use_worker' could not find any valid json "
                        "return string. This is what the json-executor "
                        "returned:\n %s" % stdoutdata)
        result = cfg.getfloat("HPOLIB", "result_on_terminate")
        status = "JSONERROR"
        wallclock_time = endtime - starttime
        additional_data = stdoutdata


    if cfg.getboolean("HPOLIB", "use_HPOlib_time_measurement") is True:
        # wallclock time is what we want, even in case of a JSON error
        time_return_value = wallclock_time
    else:
        if status != "JSONERROR":
            time_return_value = runtime
        else:
            time_return_value = cfg.getfloat(
                "HPOLIB", "runtime_on_terminate")

    if cfg.getboolean("HPOLIB", "store_target_algorithm_calls"):
        store_target_algorithm_calls(
            path=os.path.join(os.getcwd(), "target_algorithm_calls.csv"),
            wallclock_time=time_return_value, result=result,
            additional_data=additional_data, call=" ".join(cmd))

    if status == "JSONERROR":
        status = "CRASHED"

    return additional_data, result, status, time_return_value

