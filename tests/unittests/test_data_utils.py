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

import cPickle
import ConfigParser
from contextlib import contextmanager
import numpy as np
import os
import sys
import unittest

import HPOlib.data_util as data_util

@contextmanager
def redirected(out=sys.stdout, err=sys.stderr):
    saved = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out, err
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved


class DataUtilTest(unittest.TestCase):
    def setUp(self):
        # Change into the test directory
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        # Make sure there is no config file
        try:
            os.remove("./config.cfg")
        except os.error:
            pass

    def tearDown(self):
        try:
            os.remove("./test_get_trial_index.pkl")
        except OSError:
            pass

    def test_load_file(self):
        # Test numpy arrays
        train_data = np.zeros((100, 100))
        np.save("train_data.npy", train_data)
        data = data_util.load_file("train_data.npy", "numpy", 100)
        self.assertTrue((train_data == data).all())
        data = data_util.load_file("train_data.npy", "numpy", 10)
        self.assertTrue((train_data[:10] == data).all())
        os.remove("train_data.npy")

        # Test pickle files
        train_data = np.zeros((100, 100))
        fh = open("train_data.pkl", "w")
        cPickle.dump(train_data, fh)
        fh.close()
        data = data_util.load_file("train_data.pkl", "pickle", 100)
        self.assertTrue((train_data == data).all())
        data = data_util.load_file("train_data.pkl", "pickle", 10)
        self.assertTrue((train_data[:10] == data).all())
        os.remove("train_data.pkl")

        # Test zipped pickle files
        train_data = np.zeros((100, 100))
        from gzip import GzipFile as gfile
        fh = gfile("train_data.pkl.gz", "w")
        cPickle.dump(train_data, fh)
        fh.close()
        data = data_util.load_file("train_data.pkl.gz", "gfile", 100)
        self.assertTrue((train_data == data).all())
        data = data_util.load_file("train_data.pkl.gz", "gfile", 10)
        self.assertTrue((train_data[:10] == data).all())
        os.remove("train_data.pkl.gz")

        # Test wrong data file type
        self.assertRaises(IOError, data_util.load_file,
                          "test.sh", "uditare", 1)
        self.assertRaises(IOError, data_util.load_file,
                          "", "uditare", 1)

    @unittest.skip("Not implemented yet")
    def test_custom_split(self):
        self.fail()

    @unittest.skip("Not implemented yet")
    def test_prepare_cv_for_fold(self):
        self.fail()

    @unittest.skip("Not implemented yet")
    def test_remove_param_metadata(self):
        self.fail()