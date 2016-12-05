.. _install:

====================================
Installation Instructions For HPOlib
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

        pip install numpy
        pip install scipy
        pip install matplotlib

    This may take some time. Afterwards you can verify having those libs installed with:

    .. code:: bash

        pip freeze

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
            smac_2_10_00-dev_parser.py   smac_2_10_00-dev.py   smac_2_10_00-dev_src    smac_2_10_00-devDefault.cfg

5.  You can now run, e.g. smac with 200 evaluations on the branin function:

    .. code:: bash

        cd benchmarks/branin
        HPOlib-run -o ../../optimizers/smac/smac_2_10_00-dev -s 23

    This takes depending on your machine ~2 minutes. You can now plot the results of your first experiment:

    .. code:: bash

        HPOlib-plot FIRSTRUN smac_2_10_00-dev_23_*/smac_*.pkl -s `pwd`/Plots/

    You can test the other optimizers (spearmint will take quite longer 30min):

    .. code:: bash

        HPOlib-run -o ../../optimizers/tpe/h -s 23
        HPOlib-run -o ../../optimizers/spearmint/spearmint_april2013 -s 23

    and again:

    .. code:: bash

        HPOlib-plot SMAC smac_2_10_00-dev_23_*/smac_*.pkl TPE hyperopt_august2013_mod_23_*/hyp*.pkl SPEARMINT spearmint_april2013_mod_23_*/spear*.pkl -s `pwd`/Plots/

    and to check the general performance on this super complex benchmark:

    .. code:: bash

        HPOlib-plot BRANIN smac_2_10_00-dev_23_*/smac_*.pkl hyperopt_august2013_mod_23_*/hyp*.pkl spearmint_april2013_mod_23_*/spear*.pkl -s `pwd`/Plots/


**Problems during installation**

:bash:`python setup.py` crashes with :python:`ImportError: cannot import name Feature`
during installing pymongo. This happens due to pymongo using a deprecated feature
:python:'Feature', which is not available in the setuptools version (>2.2).
This error is fixed, but not yet available on PYPI.

Solution: Downgrade :bash:`setuptools` with :bash:`pip install setuptools==2.2`
and try again or install :bash:`pymongo` manually.


.. raw:: html

    <a href="https://github.com/automl/HPOlib"><img style="position: absolute; top: 0; right: 0; border: 0;" src="https://camo.githubusercontent.com/652c5b9acfaddf3a9c326fa6bde407b87f7be0f4/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f6f72616e67655f6666373630302e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_orange_ff7600.png"></a>


**Updating optimizers**
We also provide an updated and adjusted version of spearmint. To also install this version do:

.. code:: bash

    cd optimizers
    rm spearmint_gitfork_mod_src
    git clone https://github.com/automl/spearmint.git
    mv spearmint spearmint_gitfork_mod_src
