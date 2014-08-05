= Version 0.2.0 =

=== Bugfixes ===

=== Functionality ===

* HPOlib/Plotting/results.py: New result print script which can be used while the experiment is still running

=== Other ===

* HPOlib/dispatcher: Replace HPOlib/runsolver_wrapper. Can then choose different dispatcher to run the target experiment. These are:
    * python_file: Import a python file and execute a function
    * runsolver_wrapper: Run a target algorithm through the runsolver wrapper.
* HPOlib/wrapping.py: One can now specify a temporary output directory where the experiment output is temporarily stored and 
    transfered back to the experiment directory when the experiment is finished. This is useful for fast (<1s) jobs which are executed
    in high number in parallel (>100) when all computers write to a central file system.
* HPOlib/wrapping.py Enhanced cluster stability by removing setpgid() of wrapping.py.
    Thus, a qdel on a SGE cluster kills the whole process tree except the
    runsolver target process, which should end itself after some time.
* `HPOlib/cv.py` can now handle spearmint runs with parametersizes > 1. It converts a parameter `k` of size `n` to n-paramenters k_1, .., k_n.
* `HPOlib/format_converter`: Replace current converter scripts with one parser for each format. A parser reads a searchspace into a dict with the format specified in `configuration_space.py`. The `write` method outputs a searchspace maitaining as much information as possible.
=== Internals ===

* HPOlib/Experiment.py: the experiment pickle is only saved when the function _save_jobs() is invoked.