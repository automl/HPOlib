.. _install:

====================================
INSTALLATION INSTRUCTIONS FOR HPOlib
====================================

.. role:: bash(code)
    :language: bash

.. role:: python(code)
    :language: python

First:

.. code:: bash

    git clone https://github.com/automl/HPOlib.git

Installing inside an virtualenv
===============================

1.  Get `virtualenv <http://www.virtualenv.org/en/latest/virtualenv.html#installation>`_,
    then load a freshly created virtualenv. (If you are not familiar with virtualenv,
    you might want to read `more <http://www.virtualenv.org/en/latest/virtualenv.html)>`_ about it)

    .. code:: bash

        pip install virtualenv
        virtualenv virtualHPOlib
        source virtualHPOlib/bin/activate

3.  Install :bash:`numpy`, :bash:`scipy`, :bash:`matplotlib`, as this doesn't
    work through setup.py.

    .. code:: bash

        easy_install -U distribute
        pip install numpy
        pip install scipy==0.13.2
        pip install matplotlib

    This may take some time. Afterwards you can verify having those libs installed with:

    .. code:: bash

        pip freeze
        argparse==1.2.1
        backports.ssl-match-hostname==3.4.0.2
        distribute==0.7.3
        matplotlib==1.3.1
        nose==1.3.0
        numpy==1.8.0
        pyparsing==2.0.1
        python-dateutil==2.2
        scipy==0.13.2
        six==1.5.2
        tornado==3.2
        wsgiref==0.1.2

4.  run setup.py

    .. code:: python

        python setup.py install

    This will install HPOlib and some requirements (:bash:`networkx`,
    :bash:`protobuf`, :bash:`pymongo`). Be sure your system is
    **connected to the internet**, so :bash:`setup.py` can download
    optimizer and runsolver code. Your environment now looks like that

    .. code:: bash

        pip freeze
            HPOlib==0.0.1
            argparse==1.2.1
            backports.ssl-match-hostname==3.4.0.2
            distribute==0.7.3
            matplotlib==1.3.1
            networkx==1.8.1
            nose==1.3.0
            numpy==1.8.0
            protobuf==2.5.0
            pymongo==2.6.3
            pyparsing==2.0.1
            python-dateutil==2.2
            scipy==0.13.3
            six==1.5.2
            tornado==3.2
            wsgiref==0.1.2

    and

    .. code:: bash

        ls optimizers/smac
            smac_2_06_01-dev_parser.py   smac_2_06_01-dev.py   smac_2_06_01-dev_src    smac_2_06_01-devDefault.cfg

5.  You can now run, e.g. smac with 200 evaluations on the branin function:

    .. code:: bash

        cd benchmarks/branin
        HPOlib-run -o ../../optimizers/smac/smac -s 23

    This takes depending on your machine ~2 minutes. You can now plot the results of your first experiment:

    .. code:: bash

        HPOlib-plot FIRSTRUN smac_2_06_01-dev_23_*/smac_*.pkl -s `pwd`/Plots/

    You can test the other optimizers (spearmint will take quite longer 30min):

    .. code:: bash

        HPOlib-run -o ../../optimizers/tpe/h -s 23
        HPOlib-run -o ../../optimizers/spearmint/spearmint_april2013 -s 23

    and again:

    .. code:: bash

        HPOlib-plot SMAC smac_2_06_01-dev_23_*/smac_*.pkl TPE hyperopt_august2013_mod_23_*/hyp*.pkl SPEARMINT spearmint_april2013_mod_23_*/spear*.pkl -s `pwd`/Plots/

    and to check the general performance on this super complex benchmark:

    .. code:: bash

        HPOlib-plot BRANIN smac_2_06_01-dev_23_*/smac_*.pkl hyperopt_august2013_mod_23_*/hyp*.pkl spearmint_april2013_mod_23_*/spear*.pkl -s `pwd`/Plots/

Usage without installation
==========================

If you decide to not install HPOlib, you need to download the optimizer code by yourself

.. code:: bash

    cd optimizers
    wget http://www.automl.org/hyperopt_august2013_mod_src.tar.gz
    wget http://www.automl.org/smac_2_06_01-dev_src.tar.gz
    wget http://www.automl.org/spearmint_april2013_mod_src.tar.gz

    tar -xf hyperopt_august2013_mod_src.tar.gz
    mv hyperopt_august2013_mod_src tpe/

    tar -xf smac_2_06_01-dev_src.tar.gz
    mv smac_2_06_01-dev_src.tar.gz smac/

    tar -xf spearmint_april2013_mod_src.tar.gz
    mv spearmint_april2013_mod_src spearmint/

    cd ../

And you need to install all requirements:

* numpy
* matplotlib
* networkx
* protobuf
* scipy
* pymongo

e.g. with

.. code:: bash

    sudo apt-get install python-numpy python-scipy mongodb python-networkx python-protobuf

Also you need the runsolver

.. code:: bash

    wget http://www.cril.univ-artois.fr/~roussel/runsolver/runsolver-3.3.2.tar.bz2
    tar -xf runsolver-3.3.2.tar.bz2
    cd runsolver/src
    make

as this might not work, you can change the makefile via

.. code:: bash

    sed -i 's/\/usr\/include\/asm\/unistd/\/usr\/include\/unistd/g' ./Makefile
    make

then you need to add runsolver (and HPOlib) to your PATH (PYTHONPATH):

.. code:: bash

    cd ../../
    export PATH=$PATH:/path/to/runsolver/src/
    export PYTHONPATH=$PYTHONPATH:`pwd`

then you can run a benchmark like in step 5.) from installing with setup.py with replacing
:bash:`HPOlib-run` with :bash:`../../scripts/HPOlib-run` and
:bash:`HPOlib-plot` with :bash:`../../scripts/HPOlib-plot`.

**FOR FURTHER DETAILS VISIT:** `<http://www.automl.org/hpolib>`_


**Problems during installation**

:bash:`python setup.py` crashes with :python:`ImportError: cannot import name Feature`
during installing pymongo. This happens due to pymongo using a deprecated feature
:python:'Feature', which is not available in the setuptools version (>2.2).
This error is fixed, but not yet available on PYPI.

Solution: Downgrade :bash:`setuptools` with :bash:`pip install setuptools==2.2`
and try again or install :bash:`pymongo` manually.


.. raw:: html

    <a href="https://github.com/automl/HPOlib"><img style="position: absolute; top: 0; right: 0; border: 0;" src="https://camo.githubusercontent.com/652c5b9acfaddf3a9c326fa6bde407b87f7be0f4/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f6f72616e67655f6666373630302e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_orange_ff7600.png"></a>
