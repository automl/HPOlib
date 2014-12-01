import glob
import inspect
import os
import shlex
import subprocess
import shutil
import sys
import unittest

import HPOlib.wrapping_util as wrapping_util


class TestOptimizers(unittest.TestCase):
    """This test does not run in an IDE or similar environments."""
    def setUp(self):
        __file__ = inspect.getfile(TestOptimizers)
        self.hpolib_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        self.optimizer_dir = os.path.join(self.hpolib_dir, "optimizers")
        self.benchmarks_dir = os.path.join(self.hpolib_dir, "benchmarks")
        self.optimizers = ["smac/smac_2_06_01",
                           "smac/smac_2_08_00",
                           "spearmint/spearmint_april2013",
                           "tpe/hyperopt",
                           "tpe/random"]
        self.experiment_dir_prefix = wrapping_util.get_time_string()

    def tearDown(self):
        glob_string = os.path.join(self.benchmarks_dir, "*",
                                   self.experiment_dir_prefix + "*")
        runs = glob.glob(glob_string)
        for run in runs:
            shutil.rmtree(run)

    def test_run_smac(self):
        for optimizer in self.optimizers:
            cmd = "HPOlib-run -o %s/%s -s 10 --cwd %s/branin " \
                  "--HPOLIB:number_of_jobs 5 " \
                  "--HPOLIB:experiment_directory_prefix %s" % \
                  (self.optimizer_dir, optimizer, self.benchmarks_dir,
                   self.experiment_dir_prefix)

            proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                                    stdin=subprocess.PIPE,
                                    env=os.environ.copy())

            stdout, stderr = proc.communicate()
            self.assertEqual(0, proc.returncode)

