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

import datetime
import math
import traceback

__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def get_time_string():
    local_time = datetime.datetime.today()
    time_string = "%d-%d-%d--%d-%d-%d-%d" % (local_time.year, local_time.month,
                  local_time.day, local_time.hour, local_time.minute,
                  local_time.second, local_time.microsecond)
    return time_string


def float_eq( a, b, eps=0.0001 ):
    return abs(math.log( a ) - math.log(b)) <=  eps


def format_traceback(exc_info):
    traceback_template = '''Traceback (most recent call last):
    File "%(filename)s", line %(lineno)s, in %(name)s
    %(type)s: %(message)s\n''' # Skipping the "actual line" item

    # Also note: we don't walk all the way through the frame stack in this example
    # see hg.python.org/cpython/file/8dffb76faacc/Lib/traceback.py#l280
    # (Imagine if the 1/0, below, were replaced by a call to test() which did 1/0.)

    exc_type, exc_value, exc_traceback = exc_info  # most recent (if any) by default

    '''
    Reason this _can_ be bad: If an (unhandled) exception happens AFTER this,
    or if we do not delete the labels on (not much) older versions of Py, the
    reference we created can linger.

    traceback.format_exc/print_exc do this very thing, BUT note this creates a
    temp scope within the function.
    '''

    traceback_details = {
                         'filename': exc_traceback.tb_frame.f_code.co_filename,
                         'lineno'  : exc_traceback.tb_lineno,
                         'name'    : exc_traceback.tb_frame.f_code.co_name,
                         'type'    : exc_type.__name__,
                         'message' : exc_value.message, # or see traceback._some_str()
                        }

    del(exc_type, exc_value, exc_traceback) # So we don't leave our local labels/objects dangling
    # This still isn't "completely safe", though!
    # "Best (recommended) practice: replace all exc_type, exc_value, exc_traceback
    # with sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]

    return "\n" + traceback.format_exc() + "\n\n" + traceback_template % traceback_details + "\n\n"

