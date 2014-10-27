======
Manual
======

.. role:: bash(code)
    :language: bash

.. role:: python(code)
    :language: python

.. role:: cfg(code)
    :language: cfg

..  <!-- #######################################################################
    HOWTO RUN BENCHMARKS
    ######################################################################## -->

.. _run_benchmarks:

How to run listed benchmarks
============================

After having succesfully installed the basic **HPOlib** you can download more
benchmarks or create your own. Each benchmarks comes with an algorithm and
(if necessary) a wrapper and data. If you want to use one of the benchmarks
listed here, follow these steps:

Let's say you want to run the logistic regression:

1.  Read the description for :ref:`logistic regression
    <algorithms_and_datasets.html#logreg>`
    and install the dependencies (which are `THEANO <http://deeplearning.net/software/theano/>`_
    and `scikit-data <http://jaberg.github.io/skdata/>`_) and maybe recommended
    software.
2.  Download the benchmark:

        :bash:`wget www.automl.org/logistic.tar.gz`

3.  Unpack:

        :bash:`tar -xf logistic.tar.gz`

4.  Inside the root directory you will find a script :bash:`wrappingLogistic.py`,
    three directories with the name of the optimizers (plus one directory for
    random search), two other directories, named :bash:`cv` and :bash:`nocv`
    and a script :bash:`theano_teardown.py`. Now choose if you want to run the experiment with
    crossvalidation or without. Change into the :bash:`cv` directory if you
    want to use crossvalidation, if not, change into the :bash:`nocv`
    directory. There you will find a :bash:`config.cfg` and which contains
    information about how long to run the experiment, how many cross validation
    folds to use etc.
5.  Be sure you are connected to the internet, because logistic regression uses
    `scikit-data <http://jaberg.github.io/skdata/>`_) to download data, when
    called for the first time. Then run:

        :bash:`HPOlib-run /path/to/optimizers/<tpe/hyperopt|smac|spearmint|tpe/random> [-s seed] [-t title]`

    from inside the :bash:`cv` or :bash:`nocv` folder to run one optimizer for
    as many evaluations as stated in :bash:`config.cfg`,
    (100 times in this example).
6.  More information about the :bash:`config.cfg` can be found :ref:`here
    <adjust_settings>`.

**NOTE:** Since calculations are done with the THEANO library, you can also run
this benchmark on a NVIDIA GPU. This is switched off by default, but you can
change this with the THEANO flags. You find them in :bash:`config.cfg` and
information on how to set the THEANO flags :ref:`here <configure_theano>`.

.. <!-- ########################################################################
   HOWTO RUN YOUR OWN BENCHMARKS
   ##########################################################################-->

.. _create_benchmarks:

How to run your own benchmarks
==============================
To run your own benchmark you basically need the software for the benchmark and
a search space description for the optimizers smac, spearmint and tpe. In order
to work with HPOlib you must put these files into a special directory structure.
It is the same directory structure as for the benchmarks which you can download
on this website and is explained in the list below. The following lines will
guide you through the creation of such a benchmark. Here is a rough guide on
what files you need:

* One **directory** having the name of the optimizer for each optimizer you want to use.
  Currently, these are :bash:`hyperopt_august2013_mod`,
  :bash:`random_hyperopt2013_mod`,
  :bash:`smac_2_06_01-dev` and :bash:`spearmint_april2013_mod`.
* One **search space** for each optimizer. This must be placed in the directory with the name of the optimizer.
  You can convert your searchspace to other formats with
  :ref:`HPOlib_convert <hpolib_convert>` from and to all three different
  optimizers.
* An **executable** which implements the HPOlib interface. Alternatively, this can
  be a wrapper which parser the command line arguments, calls your target algorithm
  and returns the result to the HPOlib.
* A **configuration file** `config.cfg`. See the section on
  :ref:`configuring the HPOlib <adjust_settings>` for details.

.. _create_benchmark_example:

Example
-------
First, create a directory :bash:`myBenchmark` inside the
:bash:`HPOlib/benchmarks` directory. The executable
:bash:`HPOlib/benchmarks/myBenchmark/myAlgo.py` with the target algorithm can
be as easy as

.. code:: python

    import math
    import time

    import HPOlib.benchmark_util as benchmark_util

    def myAlgo(params, **kwargs):
        # Params is a dict that contains the params
        # As the values are forwarded as strings you might want to convert and check them

        if not params.has_key('x'):
            raise ValueError("x is not a valid key in params")

        x = float(params["x"])

        if x < 0 or x > 3.5:
            raise ValueError("x not between 0 and 3.5: %s" % x)

        # **kwargs contains further information, like
        # for crossvalidation
        #    kwargs['folds'] is 1 when no cv
        #    kwargs['fold'] is the current fold. The index is zero-based

        # Run your algorithm and receive a result, you want to minimize
        result = -math.sin(x)

        return result

    if __name__ == "__main__":
        starttime = time.time()
        # Use a library function which parses the command line call
        args, params = benchmark_util.parse_cli()
        result = myAlgo(params, **args)
        duration = time.time() - starttime
        print "Result for ParamILS: %s, %f, 1, %f, %d, %s" % \
            ("SAT", abs(duration), result, -1, str(__file__))

As you can see, the script parses command line arguments, calls the target function
which is implemented in myAlgo, measures the runtime of the target algorithm and
prints a return string to the command line. This relevant information is extracted
by the HPOlib. If you write a new algorithm/wrapper script, you must parse the
following call:

.. code:: bash

    target_algorithm_executable --fold 0 --folds 1 --params [ [ -param1 value1 ] ]

The return string must take the following form:

.. code:: bash

    Result for ParamILS: SAT, <duration>, 1, <result>, -1, <additional information>

This return string is far from optimal and contains unnecessary and confusing
parts. It is therefore subject to change in one of the next versions of the HPOlib.

.. _config_example:

Next, create :bash:`HPOlib/benchmarks/myBenchmark/config.cfg`,
which is the configuration file. It tells the HPOlib what to do then looks like this:

.. code:: cfg

    [TPE]
    space = mySpace.py

    [HPOLIB]
    function = python ../myAlgo.py
    number_of_jobs = 200
    # worst possible result
    result_on_terminate = 0


Since the hyperparameter optimization algorithm must know about the variables
and their possible values for your target algorithms, the next step is to
specify these in a so-called search space. Create a new directory
:bash:`hyperopt_august2013_mod` inside the
:bash:`HPOlib/benchmarks/myBenchmark` directory and save
these two lines of python in a file called :bash:`mySpace.py`. If you look at
the :bash:`config.cfg`, we already the use of the newly created search space.
As problems get more complex, you may want to specify more complex search
spaces. It is recommended to do this in the TPE format, then translate it into
the SMAC format which can then be translated into the spearmint format.
More information on how to write search spaces in the TPE format
can be found in `this paper <http://www.coxlab.org/pdfs/2013_bergstra_hyperopt
.pdf>`_ and the `hyperopt wiki <https://github
.com/hyperopt/hyperopt/wiki/FMin>`_.

.. code:: python

    from hyperopt import hp
    space = {'x': hp.uniform('x', 0, 3.5)}

Now you can run your benchmark with tpe. The command (which has to be
executed from :bash:`HPOlib/benchmarks/myBenchmark`) is

.. code:: bash

    HPOlib-run -o ../../optimizers/tpe/hyperopt_august2013_mod

Further you can run your benchmark with the other optimizers:

.. code:: bash

    mkdir smac
    python path/to/hpolib/format_converter/TpeSMAC.py tpe/mySpace.py >> smac/params.pcs
    python path/to/wrapping.py smac
    mkdir spearmint
    python path/to/hpolib/format_converter/SMACSpearmint.py >> spearmint/config.pb
    python path/to/wrapping.py spearmint

.. _adjust_settings:

Configure the HPOlib
====================

The `config.cfg` is a file, which contains necessary settings about your
experiment. It is designed such that as little as possible information needs to be given.
This means all values for optimizers and the wrapping software are set to the default
values, except you want to change them. Default values are stored in a file called
:bash:`config_parser/generalDefault.cfg`. The following table describes the
values you must provide: The file is divided into sections. You only need to
fill in values for the [HPOLIB] section.

======================= ========================================================
Key                     Description
======================= ========================================================
function                The executeable for the target algorithm. The path can
                        either be either absolute or relative to an optimizer
                        directory in your benchmark folder (if the executeable
                        is not found you can try to prepend the parent directory
                        to the path)
number_of_jobs          number of evaluations that are performed by the
                        optimizers. **NOTE**:When using k-fold-crossvalidation,
                        SMAC will use :python:`k * number_of_jobs evaluations`
result_on_terminate     If your algorithms crashes, is killed, takes too long
                        etc. This result is given to the optimizer.
                        Should be the worst possible, but realistic result
                        for a problem
======================= ========================================================



An example can be found in the section [adding your own benchmark](manual.html#config_example).
The following parameters can be specified:

=========== =================================== =============== ====================================
Section      Parameter                          Default value   Description
=========== =================================== =============== ====================================
HPOLIB      number_cv_folds                     :cfg:`1`        number of folds for a crossvalidation
HPOLIB      max_crash_per_cv                    :cfg:`3`        If some runs of the crossvalidation fail, stop the crossvalidation for this configuration after max_crash_per_cv failed folds.
HPOLIB      remove_target_algorithm_output      :cfg:`True`     Per default, the target algorithm output is deleted. Set to False to keep the output. This is useful for debugging.
HPOLIB      console_output_delay                :cfg:`1.0`      HPOlib reads the experiment pickle periodically to print the current status to the command line interface.
                                                                Doing this often can inhibit performance of your hard-drive (espacially if perform a lot of HPOlib experiments in parallel)
                                                                so you might want to increase this number if you experience delay when accessing your hard drive.
HPOLIB      runsolver_time_limit,                               Enforce resource limits to a target algorithm run. If these limits are exceeded, the target algorithm will be killed by the runsolver. This can be used to ensure e.g. a runtime per algorithm or make sure an algorithm does not use too much space on a computing cluster.
            memory_limit, cpu_limit
HPOLIB      total_time_limit                                    Enforce a total time limit on the hyperparameter optimization.
HPOLIB      leading_runsolver_info                              Important when using THEANO and CUDA, see :ref:`configure_theano`
HPOLIB      use_own_time_measurement            :cfg:`True`     When set to True (the default), the runsolver time measurement is saved. Otherwise, the time measurement of the target algorithm is saved.
HPOLIB      number_of_concurrent_jobs           :cfg:`1`        WARNING: this only works for spearmint and SMAC and is not tested!
HPOLIB      function_setup                                      An executable which is called before the first target algorithm call. This can be for example check if everything is installed properly.
HPOLIB      function_teardown                                   An executable which is called after the last target algorithm call. This can be for example delete temporary directories.
HPOLIB      experiment_directory_prefix                         Adds a prefix to the automatically generated experiment directory. Can be useful if one experiments is run several times with different parameter settings.
HPOLIB      handles_cv                                          This flag determines whether cv.py or runsolver_wrapper.py is the proxy which a hyperparameter optimization package optimizes. This is only set to 1 for SMAC and must only be used by optimization algorithm developers.
=========== =================================== =============== ====================================



The following keys change the behaviour of the integrated hyperparameter
optimization packages:

=========== =================================== ==================================== ====================================
Section     Parameter                           Default value   Description
=========== =================================== ==================================== ====================================
TPE         space                               :cfg:`space.py`                      Name of the search space for tpe
TPE         path_to_optimizer                   :cfg:`./hyperopt_august2013_mod_src` Please consult the SMAC documentation.
SMAC        p                                   :cfg:`smac/params.pcs`               Please consult the SMAC documentation.
SMAC        run_obj                             :cfg:`QUALITY`                       Please consult the SMAC documentation.
SMAC        intra_instance_obj                  :cfg:`MEAN`                          Please consult the SMAC documentation.
SMAC        rf_full_tree_bootstrap              :cfg:`False`                         Please consult the SMAC documentation.
SMAC        rf_split_min                        :cfg:`10`                            Please consult the SMAC documentation.
SMAC        adaptive_capping                    :cfg:`false`                         Please consult the SMAC documentation.
SMAC        max_incumbent_runs                  :cfg:`2000`                          Please consult the SMAC documentation.
SMAC        num_iterations                      :cfg:`2147483647`                    Please consult the SMAC documentation.
SMAC        deterministic                       :cfg:`True`                          Please consult the SMAC documentation.
SMAC        retry_target_algorithm_run_count    :cfg:`0`                             Please consult the SMAC documentation.
SMAC        intensification_percentage          :cfg:`0`                             Please consult the SMAC documentation.
SMAC        validation                          :cfg:`false`                         Please consult the SMAC documentation.
SMAC        path_to_optimizer                   :cfg:`./smac_2_06_01-dev_src`        Please consult the SMAC documentation.
SPEARMINT   config                              :cfg:`config.pb`
SPEARMINT   method                              :cfg:`GPEIOptChooser`                The spearmint chooser to be used. Please consult the spearmint documentation for possible choices. WARNING: Only the GPEIOptChooser is tested!
SPEARMINT   method_args                                                              Pass arguments to the chooser method. Please consult the spearmint documentation for possible choices.
SPEARMINT   grid_size                           :cfg:`20000`                         Length of the Sobol sequence spearmint uses to optimize the Expected Improvement.
SPEARMINT   spearmint_polling_time              :cfg:`3.0`                           Spearmint reads its experiment pickle and checks for finished jobs periodically to find out whether a new job has to be started. For very short functions evaluations, this value can be decreased. Bear in mind that this puts load on your hard drive and can slow down your system if the experiment pickle becomes large (e.g. for the AutoWeka benchmark) or you run a lot of parallel jobs (>100).
SPEARMINT   path_to_optimizer                   :cfg:`./spearmint_april2013_mod_src`
=========== =================================== ==================================== ====================================

The config parameters can also be set via the command line. A use case for this
feature is to run the same experiment multiple times, but with different parameters.
The syntax is:

.. code:: bash

    HPOlib-run -o spearmint/spearmint_april2013_mod --SECTION:argument value

To set for example the spearmint grid size to 40000, use the following call

.. code:: bash

    HPOlib-run -o spearmint/spearmint_april2013_mod --SPEARMINT:grid_size 40000

If your target algorithm is a python script, you can also load the config file
from within your target algorithm. This allows you to specify extra parameters
for your target algorithm in the config file. Simply import
:bash:`HPOlib.wrapping_util` in your python script and call
:bash:`HPOlib.wrapping_util.load_experiment_config_file()`.
The return value is a `python config parser object <https://docs.python
.org/2/library/configparser.html>`_.

.. _configure_theano:

Configure theano for gpu and openBlas usage
-------------------------------------------

The `THEANO <http://deeplearning.net/software/theano/>`_-based benchmarks can
be speed-up by either running them on a `nvidia GPU <http://en.wikipedia
.org/wiki/CUDA>`_ or with an optimized `BLAS library <http://en.wikipedia
.org/wiki/Basic_Linear_Algebra_Subprograms>`_.
Theano is either configured with theano flags, by changing the value of a variable
in the target program (not recommended as you have to change source code)
or by using a :bash:`.theanorc` file. The :bash:`.theanorc` file is good for
global configurations and you can find more information on how to use it on the
[http://deeplearning.net/software/theano/library/config.html](theano config page).
For a more fine-grained control of theano you have to use theano flags.

Unfortunately, setting them in the shell before invoking :bash:`HPOlib-run`
does not work and therefore these parameters have to be added set via the
config variable :bash:`leading_runsolver_info`. This is already set to a
reasonable default for the respective benchmarks but has to be changed in order
to speed up calculations.

For openBlas, change the paths in the following paragraph and replace the value of the
config variable :bash:`leading_runsolver_info`. In case you want to change
more of the theano behaviour (e.g. the compile directory) you must append these
flags to the config variable.

.. code:: bash

    OPENBLAS_NUM_THREADS=2 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/the/openBLAS/lib LIBRARY_PATH=$LIBRARY_PATH:/path/to/the/openBLAS/lib THEANO_FLAGS=floatX=float32,device=cpu,blas.ldflags=-lopenblas

If you want to use CUDA on your nvidia GPU, you have to change
:bash:`device=cpu` to :bash:`device=gpu` and add
:bash:`cuda.root=/usr/local/cuda` to the THEANO flags. Change :bash:`cuda.root`
to your cuda installation directory if you did not install cuda to the
default location. For that, replace the path :bash:`cuda.root=/usr/local/cuda`
with the path to your CUDA installation.

.. _add_optimizer:

How to run your own optimizer
=============================

The interface to include your own optimizer is straight-forward. Let's assume
that you have written a hyperparameter optimization package called BayesOpt2.
You tell the HPOlib to use your software with the command line argument
:bash:`-o` or :bash:`--optimizer`. A call to
:bash:`HPOlib-run -o /path/to/BayesOpt2` should the run
an experiment with your newly written software.

But so far, the HPOlib does not know how to call your software. To let the HPOlib
know about the interface to your optimizer, you need to create the three following
files (replace BayesOpt2 if your optimization package has a different name):

* BayesOpt2.py: will create all files your optimization package needs in order
    to run
* BayesOpt2_parser.py: a parser which can change the configuration of your
    optimization algorithm based on HPOlib defaults
* BayesOpt2Default.cfg: default configuration for your optimization algorithm

The rest of this section will explain interface these scripts must provide and
the functionality which they must perform

BayesOpt2.py
------------

To run BayesOpt2, HPOlib will call the main function of the script
:bash:`bayesopt2.py`. The function signature is as follows:

.. code:: bash

    (call_string, directory) = optimizer_module.main(config=config, options=args, experiment_dir=experiment_dir, experiment_directory_prefix=experiment_directory_prefix)

Argument :bash:`config` is of type `ConfigParser <http://docs.python.org/2/library/configparser.html>`_,
:bash:`options` of type `ArgumentParser <https://docs.python.org/2/library/argparse.html>`_
and :bash:`experiment_dir` is a string to the experiment directory. The return
value is a tuple :bash:`(call_string, directory)`. :bash:`call_string` must
be a valid (bash) shell command which calls your hyperparameter optimization
package in the way you intend. You can construct the call string based on the
information in the config and the options you are provided with.
:bash:`directory` must be a new directory in which all experiment output will
be stored. :bash:`HPOlib-run` will the change in to the output directory
which your function returned and execute the call string. Your script must
therefore do the following in the :bash:`main` function:

1.  Set up an experiment directory and return the path to the experiment directory.
    It is highly recommended to create a directory with the following name:

    .. code:: bash

        <experiment_directory_prefix><bayesopt2><time_string>

2.  Return a valid bash shell command, which will be used to call your optimizer
    from the command line interface. The target algorithm you want to optimize
    is mostly called :bash:`cv.py`, except for SMAC which handles
    corssvalidation on its own. Calling :bash:`cv.py` allows optimizer
    independend bookkeeping. The actual function call is the invoked by the
    HPOlib. Its interface is

    .. code:: bash

        python cv.py -param_name1 'param_value' -x '5' -y '3.0'`

    etc... The function simply prints the loss to the command line.
    If your hyperparameter optimization package is written in python, you can
    also directly call the method :python:`doForTPE(params)`, where the params
    argument is a dictionary with all parameter values (both key and value being strings).

Have a look at the bundled scripts :bash:`smac_2_06_01-dev.py`,
:bash:`spearmint_april2013_mod.py` and :bash:`hyperopt_august2013_mod.py`
to get an idea what can/must be done.

..  <! If your hyperparameter optimization packages crashes for any reason
    (cluster timeout, computer freeze) it is handy to restart from the last available state.
    To do so, add a check in your `main`-function if the option `options.restore` exists.
    The value of `options.restore` is the experiment directory from which the experiment should be restored.
    The `call_string` you return should enable your hyperparemeter optimization
    software to restore from the given directory. In order to tell HPOlib how
    much you restored, you must add a function
    `restored_runs = optimizer_module.restore(config=config, optimizer_dir=optimizer_dir, cmd=cmd)`
    which returns an integer indicating the number of evaluations you restored.
    If you use crossvalidation, multiply this with the number of folds you use.!>

BayesOpt2_parser.py
-------------------

The parser file implements a simple interface which only allows the manipulation
of the config file:

.. code:: python

    config = manipulate_config(config)

See the `python documentation <http://docs.python.org/2/library/configparser.html>`_
for the documentation of the config object. Common usage of
:python:`manipulate_config` is to check if mandatory arguments are provided.
This is also the recommended place to convert values from the HPOLIB section to
the appropriate values of the optimization package.

BayesOpt2Default.cfg
--------------------

A configuration file for your optimization package as described in the
:ref:`configuration section <adjust_settings>`.


.. _hpolib_convert:

Convert Search Spaces
=====================


.. raw:: html

    <a href="https://github.com/automl/HPOlib"><img style="position: absolute; top: 0; right: 0; border: 0;" src="https://camo.githubusercontent.com/652c5b9acfaddf3a9c326fa6bde407b87f7be0f4/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f6f72616e67655f6666373630302e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_orange_ff7600.png"></a>
