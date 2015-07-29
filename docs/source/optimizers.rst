=========================
Optimizeration algorithms
=========================

.. _bergstra and bengio, 2012: http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
.. _bergstra et al., 2011: http://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
.. _hutter et al., 2011: http://www.cs.ubc.ca/labs/beta/Projects/SMAC/papers/11-LION5-SMAC.pdf
.. _snoek et al. (2012): http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf

HPOlib ships several optimization packages by default. These are:

* :ref:`ConfigurationRunner <configurationrunner>`
    Executes configurations which are saved in a csv file.
* SMAC v2.06.01
    Includes the ROAR and SMAC algorithm (`Hutter et al., 2011`_).
* SMAC v2.08.00
    Includes the ROAR and SMAC algorithm (`Hutter et al., 2011`_).
* SMAC v2.10.00
    Includes the ROAR and SMAC algorithm (`Hutter et al., 2011`_).
* Spearmint (github clone from april 2013)
    Performs Bayesian optimization with Gaussian Processes as described in
    `Snoek et al. (2012)`_.
* Hyperopt (github clone from august 2013)
    Includes random search (`Bergstra and Bengio, 2012`_) and the Tree Parzen
    Estimator (`Bergstra et al., 2011`_)


.. _configurationrunner:

Configuration Runner
====================

The `ConfigurationRunner` is an optimizer which runs configurations saved in
a csv file. It is useful to evaluate configurations which do not come from an
optimization algorithm and still benefit from HPOlib's functionality.

By default, it expects a csv file called `configurations` as input. The first
line determines the names of the hyperparameters, every following line
determines a single configuration.

The following is an example file for the branin function::

    x,y
    0,0
    1,1
    2,2
    3,3
    4,4
    5,5
    6,6
    7,7
    8,8
    9,9
    10,10

**WARNING**: `ConfigurationRunner` does not check if the configurations
adhere to any configuration space. This must be done by the user.

Furthermore, `ConfigurationRunner` can execute the function evaluations in
parallel. This is governed by the argument `n_jobs` and only useful if the
target machine has enough processors/cores or the jobs are distributed across
several machines.



