=== INSTALLATION INSTRUCTIONS FOR HPOlib ===

git clone https://github.com/automl/HPOlib.git

=== Installing inside an virtualenv ===
1.) Get virtualenv (http://www.virtualenv.org/en/latest/virtualenv.html#installation)

    pip install virtualenv

2.) Create and load an virtualenv

    virtualenv virtualHPOlib
    source virtualHPOlib/bin/activate

You're now in an virtual python environment

3.) Install numpy, scipy, matplotlib, as this doesn't work through setup.py

    easy_install -U distribute
    pip install numpy
    pip install scipy
    pip install matplotlib

This may take some time. Aftwerwards you can verify having those libs installed with:

    pip freeze
        argparse==1.2.1
        backports.ssl-match-hostname==3.4.0.2
        distribute==0.7.3
        matplotlib==1.3.1
        nose==1.3.0
        numpy==1.8.0
        protobuf==2.5.0
        pyparsing==2.0.1
        python-dateutil==2.2
        scipy==0.13.3
        six==1.5.2
        tornado==3.2
        wsgiref==0.1.2

4.) run setup.py

   python setup.py install

This will install HPOlib and some requirements ('networkx', 'protobuf', 'pymongo').
Be sure your system is connected to the internet, so setup.py can download
optimizer and runsolver code. Your environment now looks like that

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

  ls optimizers/smac
    smac_2_06_01-dev_parser.py   smac_2_06_01-dev.py   smac_2_06_01-dev_src
    smac_2_06_01-devDefault.cfg

5.) You can now run, e.g. smac with 200 evaluations on the branin function:

    cd benchmarks/branin
    HPOlib-run -o ../../optimizers/smac/sma -s 23

This takes depending on your machine ~2 minutes

    HPOlib-plot FIRSTRUN smac_2_06_01-dev_23_*/smac_*.pkl -s `pwd`/Plots/

gives you some nice plots in './Plots'. You can test the other optimizers (spearmint will take quite longer 30min),
but don't be scared by messy output of the optimizer preceded by [ERR]:

    HPOlib-run -o ../../optimizers/tpe/h -s 23
    HPOlib-run -o ../../optimizers/spearmint/s -s 23

and again:

    HPOlib-plot SMAC smac_2_06_01-dev_23_*/smac_*.pkl TPE hyperopt_august2013_mod_23_*/hyp*.pkl SPEARMINT spearmint_april2013_mod_23_*/spear*.pkl -s `pwd`/Plots/
and to check the general performance on this super complex benchmark:

    HPOlib-plot BRANIN smac_2_06_01-dev_23_*/smac_*.pkl hyperopt_august2013_mod_23_*/hyp*.pkl spearmint_april2013_mod_23_*/spear*.pkl -s `pwd`/Plots/

=== Using without installation ===
If you decide to not install HPOlib, you need to download the optimizer code by yourself

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

    numpy
    matplotlib
    networkx
    protobuf
    scipy
    pymongo

e.g. with

    sudo apt-get install python-numpy python-scipy mongodb python-networkx python-protobuf

Also you need the runsolver

    wget http://www.cril.univ-artois.fr/~roussel/runsolver/runsolver-3.3.2.tar.bz2
    tar -xf runsolver-3.3.2.tar.bz2
    cd runsolver/src
    make

as this might not work, you can change the makefile via

    sed -i 's/\/usr\/include\/asm\/unistd/\/usr\/include\/unistd/g' ./Makefile
    make

then you need to add runsolver to your PATH

    cd ../../
    export PATH=$PATH:/path/to/runsolver/src/

and HPOlib to your PYTHONPATH

    export PYTHONPATH=$PYTHONPATH:`pwd`

then you can run a benchmark like in step 5.) from installing with setup.py with replacing
''HPOlib-run'' with ''../../scripts/HPOlib-run'' and ''HPOlib-plot'' with ''../../scripts/HPOlib-plot''


== FOR FURTHER DETAILS VISIT: automl.org ==