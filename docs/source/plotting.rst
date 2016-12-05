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

    {"instance_order": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0]], "cv_endtime": [1416846588.037684, 1416846588.714215, 1416846589.185275, 1416846589.71545, 1416846590.240511, 1416846590.645061, 1416846591.157578, 1416846591.588725, 1416846592.075068, 1416846592.565032], "optimizer_time": [], "title": null, "folds": 1, "total_wallclock_time": 89.56732000000001, "trials": [{"status": 3, "std": 0.0, "test_additional_data": {"0": "../logreg.py"}, "test_duration": 3.9904310000000001, "instance_results": [0.0906], "test_std": 0.0, "additional_data": {"0": "../logreg.py"}, "test_instance_durations": [3.9904310000000001], "params": {"batchsize": "0", "l2_reg": "0", "lrate": "0", "n_epochs": "0"}, "result": 0.0906, "test_instance_status": [3], "duration": 3.9904310000000001, "test_status": 3, "test_result": 0.0906, "test_instance_results": [0.0906], "instance_status": [3], "instance_durations": [3.9904310000000001]}, {"status": 3, "std": 0.0, "test_additional_data": {"0": "../logreg.py"}, "test_duration": 2.6245590000000001, "instance_results": [0.20121], "test_std": 0.0, "additional_data": {"0": "../logreg.py"}, "test_instance_durations": [2.6245590000000001], "params": {"batchsize": "4", "l2_reg": "6", "lrate": "3", "n_epochs": "0"}, "result": 0.20121, "test_instance_status": [3], "duration": 2.6245590000000001, "test_status": 3, "test_result": 0.20121, "test_instance_results": [0.20121], "instance_status": [3], "instance_durations": [2.6245590000000001]}, {"status": 3, "std": 0.0, "test_additional_data": {"0": "../logreg.py"}, "test_duration": 3.3856489999999999, "instance_results": [0.15843699999999999], "test_std": 0.0, "additional_data": {"0": "../logreg.py"}, "test_instance_durations": [3.3856489999999999], "params": {"batchsize": "5", "l2_reg": "2", "lrate": "3", "n_epochs": "1"}, "result": 0.15843699999999999, "test_instance_status": [3], "duration": 3.3856489999999999, "test_status": 3, "test_result": 0.15843699999999999, "test_instance_results": [0.15843699999999999], "instance_status": [3], "instance_durations": [3.3856489999999999]}, {"status": 3, "std": 0.0, "test_additional_data": {"0": "../logreg.py"}, "test_duration": 16.756257000000002, "instance_results": [0.13843800000000001], "test_std": 0.0, "additional_data": {"0": "../logreg.py"}, "test_instance_durations": [16.756257000000002], "params": {"batchsize": "5", "l2_reg": "1", "lrate": "6", "n_epochs": "7"}, "result": 0.13843800000000001, "test_instance_status": [3], "duration": 16.756257000000002, "test_status": 3, "test_result": 0.13843800000000001, "test_instance_results": [0.13843800000000001], "instance_status": [3], "instance_durations": [16.756257000000002]}, {"status": 3, "std": 0.0, "test_additional_data": {"0": "../logreg.py"}, "test_duration": 2.7620979999999999, "instance_results": [0.13769999999999999], "test_std": 0.0, "additional_data": {"0": "../logreg.py"}, "test_instance_durations": [2.7620979999999999], "params": {"batchsize": "2", "l2_reg": "10", "lrate": "2", "n_epochs": "0"}, "result": 0.13769999999999999, "test_instance_status": [3], "duration": 2.7620979999999999, "test_status": 3, "test_result": 0.13769999999999999, "test_instance_results": [0.13769999999999999], "instance_status": [3], "instance_durations": [2.7620979999999999]}, {"status": 3, "std": 0.0, "test_additional_data": {"0": "../logreg.py"}, "test_duration": 3.5262229999999999, "instance_results": [0.272984], "test_std": 0.0, "additional_data": {"0": "../logreg.py"}, "test_instance_durations": [3.5262229999999999], "params": {"batchsize": "4", "l2_reg": "7", "lrate": "9", "n_epochs": "0"}, "result": 0.272984, "test_instance_status": [3], "duration": 3.5262229999999999, "test_status": 3, "test_result": 0.272984, "test_instance_results": [0.272984], "instance_status": [3], "instance_durations": [3.5262229999999999]}, {"status": 3, "std": 0.0, "test_additional_data": {"0": "../logreg.py"}, "test_duration": 2.293974, "instance_results": [0.28349999999999997], "test_std": 0.0, "additional_data": {"0": "../logreg.py"}, "test_instance_durations": [2.293974], "params": {"batchsize": "1", "l2_reg": "8", "lrate": "3", "n_epochs": "1"}, "result": 0.28349999999999997, "test_instance_status": [3], "duration": 2.293974, "test_status": 3, "test_result": 0.28349999999999997, "test_instance_results": [0.28349999999999997], "instance_status": [3], "instance_durations": [2.293974]}, {"status": 3, "std": 0.0, "test_additional_data": {"0": "../logreg.py"}, "test_duration": 2.1435740000000001, "instance_results": [0.23150000000000001], "test_std": 0.0, "additional_data": {"0": "../logreg.py"}, "test_instance_durations": [2.1435740000000001], "params": {"batchsize": "1", "l2_reg": "1", "lrate": "10", "n_epochs": "1"}, "result": 0.23150000000000001, "test_instance_status": [3], "duration": 2.1435740000000001, "test_status": 3, "test_result": 0.23150000000000001, "test_instance_results": [0.23150000000000001], "instance_status": [3], "instance_durations": [2.1435740000000001]}, {"status": 3, "std": 0.0, "test_additional_data": {"0": "../logreg.py"}, "test_duration": 5.0202150000000003, "instance_results": [0.275781], "test_std": 0.0, "additional_data": {"0": "../logreg.py"}, "test_instance_durations": [5.0202150000000003], "params": {"batchsize": "7", "l2_reg": "0", "lrate": "8", "n_epochs": "2"}, "result": 0.275781, "test_instance_status": [3], "duration": 5.0202150000000003, "test_status": 3, "test_result": 0.275781, "test_instance_results": [0.275781], "instance_status": [3], "instance_durations": [5.0202150000000003]}, {"status": 3, "std": 0.0, "test_additional_data": {"0": "../logreg.py"}, "test_duration": 2.2806799999999998, "instance_results": [0.13469999999999999], "test_std": 0.0, "additional_data": {"0": "../logreg.py"}, "test_instance_durations": [2.2806799999999998], "params": {"batchsize": "1", "l2_reg": "1", "lrate": "3", "n_epochs": "4"}, "result": 0.13469999999999999, "test_instance_status": [3], "duration": 2.2806799999999998, "test_status": 3, "test_result": 0.13469999999999999, "test_instance_results": [0.13469999999999999], "instance_status": [3], "instance_durations": [2.2806799999999998]}], "experiment_name": "smac_2_06_01-dev", "starttime": [1416846586.877219], "cv_starttime": [1416846587.882512, 1416846588.544788, 1416846589.009884, 1416846589.546839, 1416846590.061855, 1416846590.515719, 1416846590.983617, 1416846591.416308, 1416846591.905041, 1416846592.386514], "optimizer": "/home/feurerm/mhome/HPOlib/Software/HPOlib/optimizers/smac/smac_2_06_01-dev", "endtime": [1416846592.735704, 1416846608.662255], "max_wallclock_time": ""}

Currently supported output types/formats are:

* `json <http://www.json.org/>`_
