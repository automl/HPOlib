= Version 0.2.0 =

=== Bugfixes ===

=== Functionality ===

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
    
=== Internals ===

* HPOlib/Experiment.py: the experiment pickle is only saved when the function _save_jobs() is invoked.
