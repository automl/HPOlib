================
Plotting results
================

.. role:: bash(code)
  :language: bash

.. role:: python(code)
  :language: python

.. role:: cfg(code)
  :language: cfg

.. _plotting:

Exporting results
=================
To process results with programming languages different than python we
provide a script called :bash:`HPOlib-export`, which can convert HPOlib
experiment pickles into different formats:

.. code:: bash

   HPOlib-export input output [-t|--type output_type]

Example
-------

.. code:: bash

    HPOlib-export benchmarks/branin/smac_2_06_01-dev_1_2014-11-24--16-6-19-290280/smac_2_06_01-dev.pkl output/smac_branin_seed1 -t json

The output looks something like this:

.. code:: json

    {"instance_order": [[0, 0], [1, 0]], "cv_endtime": [1416844799.094412, 1416844799.665029], "optimizer_time": [], "title": null, "folds": 1, "total_wallclock_time": 0.005697011947631836, "trials": [{"status": 3, "std": 0.0, "test_additional_data": {}, "test_duration": NaN, "instance_results": [24.129964413622268], "test_std": NaN, "instance_status": [3], "test_instance_durations": [NaN], "params": {"x": "2.5", "y": "7.5"}, "result": 24.129964413622268, "test_instance_status": [0], "additional_data": {"0": null}, "duration": 0.0037469863891601562, "test_status": 0, "test_instance_results": [NaN], "test_error": NaN, "instance_durations": [0.0037469863891601562]}, {"status": 3, "std": 0.0, "test_additional_data": {}, "test_duration": NaN, "instance_results": [59.972610578348807], "test_std": NaN, "instance_status": [3], "test_instance_durations": [NaN], "params": {"x": "1.2306023361226375", "y": "11.07583286377839"}, "result": 59.972610578348807, "test_instance_status": [0], "additional_data": {"0": null}, "duration": 0.0019500255584716797, "test_status": 0, "test_instance_results": [NaN], "test_error": NaN, "instance_durations": [0.0019500255584716797]}], "experiment_name": "smac_2_06_01-dev", "starttime": [1416844797.755915], "cv_starttime": [1416844799.035669, 1416844799.605281], "optimizer": "/home/feurerm/mhome/HPOlib/Software/HPOlib/optimizers/smac/smac_2_06_01-dev", "endtime": [1416844799.972342], "max_wallclock_time": ""}

Currently supported output types/formats are:

* `json <http://www.json.org/>`_
