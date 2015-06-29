= Version 0.2.0 =

=== Bugfixes ===
+ REMOVED THE LEADING MINUS IN THE EXPRIMENT.PKL and providing a method to fix 'broken' .pkls

=== Functionality ===

* HPOlib/Plotting/results.py: New result print script which can be used while the experiment is still running
* HPOlib/Plotting/doFanovaPlots.py;HPOlib/scripts/HPOlib-pyFanova: New plotting script that calls pyfanova and saves plots
* HPOlib/benchmarks: added a matlab examples running the branin textfunctions, contains an argumentparser

=== Other ===

* `HPOlib/dispatcher`: Replace `HPOlib/runsolver_wrapper`. Can then choose
    different dispatcher to run the target experiment. These are:
    * `python_file`: Import a python file and execute a function
    * `runsolver_wrapper`: Run a target algorithm through the runsolver wrapper.
* `HPOlib/dispatcher`: The additional run data field of the return string,
    which was previously unused is now stored in the experiment pickle file.
* `HPOlib/dispatcher/runsolver_wrapper.py`: The return string also accepts the
 new SMAC return string, which starts with `Result for this algorithm run`
* `HPOlib/wrapping.py`: One can now specify a temporary output directory where
    the experiment output is temporarily stored and transfered back to the
    experiment directory when the experiment is finished. This is useful for
    fast (<1s) jobs which are executed in high number in parallel (>100) when
    all computers write to a central file system.
* `HPOlib/wrapping.py` Enhanced cluster stability by removing setpgid() of
    wrapping.py. Thus, a qdel on a SGE cluster kills the whole process tree
    except the runsolver target process, which should end itself after some time.
* `HPOlib/cv.py` can now handle spearmint runs with parametersizes > 1.
    It converts a parameter `k` of size `n` to n-paramenters k_1, .., k_n.
* `HPOlib/format_converter`: Replace current converter scripts with one parser
    for each format. A parser reads a searchspace into a dict with the format
    specified in `configuration_space.py`. The `write` method outputs a
    searchspace maitaining as much information as possible.
* `scripts/HPOlib-testbest`: New script which takes a directory created by a
    previous optimization run. It then runs a test function on all
    configurations which were evaluated.
* `scripts/HPOlib-export`: New script which outputs the content of an
    experiment pickle in a format readable by other programming languages.
* It is now possible to store the call for each target algorithm evaluation.
    The option can be set in the config file.

=== Benchmarks ===

* Add new benchmark logistic regression on a grid. This is the first
    benchmark which is bundled with the HPOlib and has a test function included.
* Moved all benchmark related stuff into a new package `benchmark`.

=== Internals ===

* HPOlib/Experiment.py: the experiment pickle is only saved when the function _save_jobs() is invoked.
* Renamed several options:
* `HPOlib/cv.py` does no longer call `HPOlib/dispatcher.py` via CLI,
    but imports it.

=== Minor ===
* Print a warning, when a user uses a development version
* Print a warning, when user tries to run smac and does not have java version 1.7.0_65
* Remove leading_algo_info from config.cfg; Information can be set with the target algorithm call

=== Evaluation scripts ===
* getTopK outputs the k best (worst) configurations
* plotting scripts are more flexible and allow to set properties for matplotlib
+ You can nos choose between plottin mean or median

