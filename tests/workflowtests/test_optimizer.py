import glob
import inspect
import os
import shlex
import subprocess
import shutil
import unittest

import HPOlib.wrapping_util as wrapping_util


class TestOptimizers(unittest.TestCase):
    """This test does not run in an IDE or similar environments."""
    def setUp(self):
        __file__ = inspect.getfile(TestOptimizers)
        self.hpolib_dir = os.path.join(os.path.dirname(__file__), "..", "..")
        self.optimizer_dir = os.path.join(self.hpolib_dir, "optimizers")
        self.benchmarks_dir = os.path.join(self.hpolib_dir, "benchmarks")
        # Add new optimizers only in this kind of format and keep in mind that
        # everything after the last '/' is treated as the optimizer name
        self.optimizers = ["smac/smac_2_06_01",
                           "smac/smac_2_08_00",
                           "spearmint/spearmint_april2013",
                           "tpe/hyperopt",
                           "tpe/random"]
        self.experiment_dir_prefix = wrapping_util.get_time_string()
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + \
            os.path.join(self.hpolib_dir, "runsolver/src")

    def tearDown(self):
        glob_string = os.path.join(self.benchmarks_dir, "*",
                                   self.experiment_dir_prefix + "*")
        runs = glob.glob(glob_string)
        for run in runs:
            shutil.rmtree(run)
            pass

    def test_run_and_test(self):
        for optimizer in self.optimizers:
            cmd = "python -m HPOlib.wrapping " \
                  "-o %s/%s -s 10 --cwd %s/logreg_on_grid " \
                  "--HPOLIB:number_of_jobs 5 " \
                  "--HPOLIB:experiment_directory_prefix %s" % \
                  (self.optimizer_dir, optimizer, self.benchmarks_dir,
                   self.experiment_dir_prefix)

            print
            print "#######################################"
            print cmd
            print

            proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                                    stdin=subprocess.PIPE)

            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                print stderr
                print stdout
            self.assertEqual(0, proc.returncode, stderr)

            # And now test that stuff

            testing_directory = os.path.join(self.benchmarks_dir,
                                             "logreg_on_grid",
                                             self.experiment_dir_prefix + "*" +
                                             optimizer.split("/")[-1] + "*")
            test_dir_glob = glob.glob(testing_directory)
            testing_directory = test_dir_glob[0]

            cmd = "HPOlib-testbest --all --cwd %s" % testing_directory
            print
            print "#######################################"
            print cmd
            print
            proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                                    stdin=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            self.assertEqual(0, proc.returncode)

    def test_crossvalidation(self):
        # This is only a dummy crossvalidation!
        cmd = "python -m HPOlib.wrapping" \
              " -o %s/%s -s 10 --cwd %s/logreg_on_grid " \
              "--HPOLIB:number_of_jobs 5 " \
              "--HPOLIB:experiment_directory_prefix %s " \
              "--HPOLIB:number_cv_folds 2" % \
              (self.optimizer_dir, self.optimizers[4], self.benchmarks_dir,
               self.experiment_dir_prefix)

        print
        print "#######################################"
        print cmd
        print

        proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                                stdin=subprocess.PIPE,
                                env=os.environ.copy())

        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print stderr
            print stdout
        self.assertEqual(0, proc.returncode)

