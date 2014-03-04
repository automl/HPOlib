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

import os
import unittest
import numpy as np
import sys

import HPOlib.cv as cv
try:
    import hyperopt
except:
    # TODO: Remove this Hackiness when installation fully works!
    import HPOlib
    hyperopt_path = os.path.join(os.path.dirname(os.path.abspath(
        HPOlib.__file__)), "optimizers/hyperopt_august2013_mod")
    print hyperopt_path
    sys.path.append(hyperopt_path)
    import hyperopt


class CVTest(unittest.TestCase):
    def setUp(self):
        # Change into the parent of the test directory
        os.chdir(os.path.join("..", os.path.dirname(os.path.realpath(__file__))))

        # Make sure there is no config file
        try:
            os.remove("./config.cfg")
        except:
            pass

    def test_read_parameters_from_command_line(self):
        # Legal call
        sys.argv = ["test_cv.py", "-x", "'5'", "-name", "'Koenigsberghausen'",
                    "-y", "'5.0'", "-z", "'-3.0'"]
        params = cv.read_params_from_command_line()
        self.assertEqual(params, {'x': '5', 'name': 'Koenigsberghausen',
                                  'y': '5.0', 'z': '-3.0'})

        # illegal call, no - in front of parameter name
        sys.argv = ["test_cv.py", "x", "'5'"]
        self.assertRaises(ValueError, cv.read_params_from_command_line)

        # illegal call, no single quotation mark around second parameter
        sys.argv = ["test_cv.py", "-x", "5"]
        self.assertRaises(ValueError, cv.read_params_from_command_line)

         # illegal call, no parameter value
        sys.argv = ["test_cv.py", "-x", "-y"]
        self.assertRaises(ValueError, cv.read_params_from_command_line)

    def test_parameter_flattening(self):
        def naive_old_implementation(params):
            _params_to_check = list(params.keys())
            _new_dict = dict()
            while len(_params_to_check) != 0:
                p = _params_to_check.pop()
                if isinstance(params[p], dict):
                    _params_to_check.extend(params[p].keys())
                    params.update(params[p])
                elif isinstance(params[p], np.ndarray) or \
                    isinstance(params[p], list):
                    _new_dict[p] = params[p][0]
                else:
                    _new_dict[p] = params[p]
            return _new_dict

        # Branin
        import HPOlib.benchmarks.branin.tpe.space
        space = HPOlib.benchmarks.branin.tpe.space.space

        for i in range(100):
            sample = hyperopt.pyll.stochastic.sample(space)
            flatten = cv.flatten_parameter_dict(sample)
            flatten_old = naive_old_implementation(sample)
            self.assertEqual(len(flatten), 2)
            self.assertEqual(type(flatten), dict)
            self.assertEqual(type(flatten["x"]), float)
            self.assertEqual(type(flatten["y"]), float)
            self.assertEqual(flatten, flatten_old)

        # HPNnet
        import tests.search_spaces.nips2011
        space = tests.search_spaces.nips2011.space

        for i in range(100):
            sample = hyperopt.pyll.stochastic.sample(space)
            flatten = cv.flatten_parameter_dict(sample)
            flatten_old = naive_old_implementation(sample)
            # print flatten
            # print flatten_old
            self.assertLessEqual(len(flatten), 13)
            self.assertGreaterEqual(len(flatten), 10)
            self.assertEqual(type(flatten), dict)
            self.assertEqual(flatten, flatten_old)

        # AutoWEKA
        import tests.search_spaces.autoweka
        space = tests.search_spaces.autoweka.space
        for i in range(100):
            sample = hyperopt.pyll.stochastic.sample(space)
            flatten = cv.flatten_parameter_dict(sample)
            self.assertIn("attributesearch", flatten.keys())
            self.assertIn("targetclass", flatten.keys())
            # print flatten


if __name__ == "__main__":
    unittest.main()