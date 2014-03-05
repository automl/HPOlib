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
from gzip import GzipFile as gfile
import logging
import os

import numpy as np

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


logger = logging.getLogger("HPOlib.data_util")


def load_file(filename, file_format, use_percentage):
    if not os.path.exists(filename):
        raise IOError("File %s not found", filename)

    if file_format == "gfile":
        logger.info("Loading file: %s", filename)
        fh = gfile(filename, "rb")
        data = cPickle.load(fh)
        if use_percentage >= 100.:
            pass
        else:
            max_data = int(len(data) / 100. * use_percentage)
            data = data[:max_data]
        fh.close()
        logger.info("Done loading file: %s has %d datapoints", filename,
                    len(data))

    elif file_format == "pickle":
        logger.info("Loading file: %s", filename)
        fh = open(filename, "r")
        data = cPickle.load(fh)
        if use_percentage >= 100.:
            pass
        else:
            data = data[:len(data) / 100. * use_percentage]
        fh.close()
        logger.info("Done loading file: %s has %d datapoints", filename,
                    len(data))

    elif file_format == "numpy":
        logger.info("Loading file: %s", filename)
        fh = open(filename, "r")
        data = np.load(fh)
        if use_percentage >= 100.:
            pass
        else:
            data = data[:len(data) / 100. * use_percentage]
        fh.close()
        logger.info("Done loading file: %s has %d datapoints", filename,
                                                                len(data))

    else:
        raise ValueError("%s is an unknown training_data_format", file_format)

    return data


def custom_split(data, n_train, n_valid):
    """
    Split the training data in such a way that the training data is divided into
    a training set with n_train samples and a validation set with n_valid
    samples.
    """
    if data is None:
        return None

    assert n_train + n_valid == len(data), ("Assertion failed, number of " +
        "training samples (%d) + validation samples (%d) != length of " +
        "data (%d)") % (n_train, n_valid, len(data))
    train = data[0:n_train]
    valid = data[n_train:]
    return train, valid


def prepare_cv_for_fold(data, fold, folds):
    """
    Split the data into training and validation data for a given fold.

    Arguments:
    data -- An array-like object
    fold -- Fold to split the data for
    folds -- Number of folds in total

    Returns:
    train -- Array-like object containing the training data
    valid -- Array-like object containing the validation data

    In case no data is handed over to the function, None is returned.

    """
    logger.info("Splitting data:")
    logger.info("Fold %s of %s folds", fold, folds)
    # Create an array with the split points
    if data is not None:
        data_len = len(data)
        splits = [data_len / folds * f for f in range(folds)]
        splits.append(data_len)
    else:
        return None

    if type(data) != np.ndarray:
        if isinstance(data[0], str):
            pass
            # Cannot be converted to a numpy array since this would blow up
            # memory
        else:
            data = np.array(data)

    if isinstance(data, np.ndarray):
        cv_split_mask = np.empty((data_len), dtype=np.bool)
        for i in range(folds):
            if i != fold:
                cv_split_mask[splits[i]:splits[i+1]] = 1
            else:
                cv_split_mask[splits[i]:splits[i+1]] = 0
        train = data[cv_split_mask]
        valid = data[~cv_split_mask]
        logger.info("%d, %d, %d, %d", data.shape, train.shape, valid.shape,
                    train.itemsize)
    else:
        train = []
        valid = []
        for i, datum in enumerate(data):
            if i >= splits[fold] and i <= splits[fold + 1]:
                valid.append(datum)
            else:
                train.append(datum)
    return train, valid
