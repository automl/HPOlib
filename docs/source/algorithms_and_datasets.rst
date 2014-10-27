=======================
Algorithms and Datasets
=======================

.. role:: bash(code)
    :language: bash

.. |br| raw:: html

    <br />

Benchmarks Overview
===================

To run these algorithms and datasets with hyperparameter optimizers you need to install

1. the **HPOlib** software from :ref:`here <install>`
2. the benchmark data: An algorithm and depending on the benchmark a wrapper and/or data

Then the benchmarks can easily be used, as described :ref:`here <run_benchmarks>`;
Our software allows to integrate your own benchmarks as well. Here is the
:ref:`HowTo <create_benchmarks>`.

**NOTE:** For all bechmarks crossvalidation is possible, but not extra listed.
Although possible, it obviously makes no sense to do crossvalidation on
functions like Branin and pre-computed results like the LDA ongrid.
Whether it makes sense to do so is indicated in the column CV.

.. raw:: html

    <table border="1" frame="hsides" rules="groups" width="100%" align="center">
        <caption>Available Benchmarks</caption>
        <thead>
        <tr>
            <th>Algorithm</th>
            <th># hyperparams(condition.)</th>
            <th>contin./discr.</th>
            <th>Dataset</th>
            <th>Size(Train/Valid/Test)</th>
            <th>runtime</th>
            <th>programming language</th>
            <th>CV</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td align="center"><a href="algorithms_and_datasets.html#braninhar6camel">Branin</a></td>
            <td align="center">2(-)</td>
            <td align="center">2/-</td>
            <td align="center">-</td>
            <td align="center">-</td>
            <td align="center">&lt; 1s</td>
            <td align="center">Python</td>
            <td aligh="center">no</td>
        </tr>
        <tr>
            <td align="center"><a href="algorithms_and_datasets.html#braninhar6camel">Camelback function</a></td>
            <td align="center">2(-)</td>
            <td align="center">2/-</td>
            <td align="center">-</td>
            <td align="center">-</td>
            <td align="center">&lt; 1s</td>
            <td align="center">Ruby</td>
            <td aligh="center">no</td>
        </tr>
        <tr>
            <td align="center"><a href="algorithms_and_datasets.html#braninhar6camel">Hartmann 6d</a></td>
            <td align="center">6(-)</td>
            <td align="center">6/-</td>
            <td align="center">-</td>
            <td align="center">-</td>
            <td align="center">&lt; 1s</td>
            <td align="center">Python</td>
            <td aligh="center">no</td>
        </tr>
        </tbody><tbody>
        <tr>
            <td align="center"><a href="algorithms_and_datasets.html#svmlda">LDA ongrid<a></td>
            <td align="center">3(-)</td>
            <td align="center">-/3</td>
            <td align="center">wikipedia articles</td>
            <td align="center">-</td>
            <td align="center">&lt;1s</td>
            <td align="center">Python</td>
            <td aligh="center">no</td>
        </tr>
        <tr>
            <td align="center"><a href="algorithms_and_datasets.html#svmlda">SVM ongrid</a></td>
            <td align="center">3(-)</td>
            <td align="center">-/3</td>
            <td align="center">UniPROBE</td>
            <td align="center">-</td>
            <td align="center">&lt;1s</td>
            <td align="center">Python</td>
            <td aligh="center">no</td>
        </tr>
        <tr>
            <td align="center"><a href="algorithms_and_datasets.html#logreg">Logistic Regression</a></td>
            <td align="center">4(-)</td>
            <td align="center">4/-</td>
            <td align="center">MNIST</td>
            <td align="center">50k/10k/10k</td>
            <td align="center">&lt;1m (Intel Xeon E5-2650 v2; OpenBlas@2cores)</td>
            <td align="center">Python</td>
            <td aligh="center">yes</td>
        </tr>
        </tbody><tbody>
        <tr>
            <td align="center"><a href="algorithms_and_datasets.html#nnetdbnet">hp-nnet</a></td>
            <td align="center">14(4)</td>
            <td align="center">7/7</td>
            <td align="center">MRBI<br>convex</td>
            <td align="center">10k/2k/50k<br>6.5k/1.5k/50k</td>
            <td align="center">&#126;25m (GPU, NVIDIA Tesla M2070)<br>&#126;6m (GPU, NVIDIA Tesla M2070)</td>
            <td align="center">Python</td>
            <td aligh="center">yes</td>
        </tr>
        <tr>
            <td align="center"><a href="algorithms_and_datasets.html#nnetdbnet">hp-dbnet</a></td>
            <td align="center">38(29)</td>
            <td align="center">19/17</td>
            <td align="center">MRBI<br>convex</td>
            <td align="center">10k/2k/50k<br>6.5k/1.5k/50k</td>
            <td align="center">&#126;15m (GPU, Gefore GTX780)<br>&#126;10m (GPU, Gefore GTX780)</td>
            <td align="center">Python</td>
            <td aligh="center">yes</td>
        </tr>
        </tbody> <tbody>
        <tr>
            <td align="center"><a href="algorithms_and_datasets.html#autoweka">autoweka</a></td>
            <td align="center">786(784)</td>
            <td align="center">296/490</td>
            <td align="center">convex</td>
            <td align="center">6.5k/1.5k/50k</td>
            <td align="center">&#126;15m</td>
            <td align="center">Python/Java</td>
            <td aligh="center">yes</td>
        </tr></tbody>
    </table>

Description
===========

.. _braninhar6camel:

Branin, Hartmann 6d and Camelback Function
------------------------------------------

This benchmark already comes with the basic *HPOlib* bundle.

**Dependencies:** None |br|
**Recommended:** None

Branin, Camelback and the Hartmann 6d function are three simple test functions,
which are easy and cheap to evaluate. More test functions can be found
`here <http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page364.htm>`_
|br|
Branin has three global minima at (-pi, 12.275), (pi, 2.275), (9.42478, 2.475) where f(x)=0.397887.
|br|
Camelback has two global minima at (0.0898, -0.7126) and (-0.0898, 0.7126) where f(x) = -1.0316
|br|
Hartmann 6d is more difficult with 6 local minima and one global optimum at
(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573) where f(x)=3.32237.

.. _svmlda:

LDA ongrid/SVM ongrid
---------------------

This benchmark already comes with the basic *HPOlib* bundle.

**Dependencies:** None |br|
**Recommended:** None

Online Latent Dirichlet Allocation (LDA) is a very expensive algorithm to evaluate.
To make this less time consuming, a 6x6x8 grid of hyperparameter configurations
resulting in 288 data points was preevaluated. This grid forms the search space.

Same holds for the Support Vector Machine task, which has 1400 evaluated configurations.

The Online LDA code is written by Hoffman et. al. and the procedure is explained
in `Online Learning for Latent Dirichlet Allocation <http://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf>`_.
Latent Structured Support Vector Machine code is written by Kevin Mill et. al.
and explained in the paper `Max-Margin Min-Entropy Models <http://jmlr.org/proceedings/papers/v22/miller12/miller12.pdf>`_.
The grid search was performed by Jasper Snoek and previously used in
`Practical Bayesian Optimization of Machine Learning Algorithms <http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms>`_.

.. _logreg:

Logistic Regression
___________________

**Dependencies:** `theano <http://deeplearning.net/software/theano/>`_,
`scikit-data <http://jaberg.github.io/skdata/>`_ |br|
**Recommended:** `CUDA <https://developer.nvidia.com/cuda-downloads>`_

**NOTE:** *scikit-data* downloads the dataset from
the internet when using the benchmark for the first time. |br|
**NOTE:** This benchmarks can use a gpu, but this
feature is switched off to run it off-the-shelf. To use a gpu you need to
change the THEANO flags in :bash:`config.cfg`. See the :ref:`HowTo <configure_theano>`
for changing to gpu and for further information about the THEANO configuration
`here <http://deeplearning.net/software/theano/library/config.html#envvar-THEANO_FLAGS>`_ |br|
**NOTE:** In order to run the benchmark you must adjust the paths in the config files.

You can download this benchmark by clicking `here <http://www.automl.org/logreg.tar.gz>`_ or
running this command from a shell:

.. code:: bash

    wget http://www.automl.org/logreg.tar.gz
    tar -xf logistic.tar.gz

This benchmark performs a logistic regression to classifiy the popular MNIST
dataset. The implementation is Theano based, so that a GPU can be used.
The software is written by Jasper Snoek and was first used in the paper
`Practical Bayesian Optimization of Machine Learning Algorithms <http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms>`_.

**NOTE:** This benchmark comes with the version of
`hyperopt-nnet <https://github.com/hyperopt/hyperopt-nnet>`_ which we used for
our experiments. There might be a newer version with improvements.

.. _nnetdbnet:

HP-NNet and HP-DBNet
____________________

**Dependencies:** `theano <http://deeplearning.net/software/theano/>`_,
`scikit-data <http://jaberg.github.io/skdata/>`_ |br|
**Recommended:** `CUDA <https://developer.nvidia.com/cuda-downloads>`_

**NOTE:** This benchmark comes with the version of
`hyperopt-nnet <https://github.com/hyperopt/hyperopt-nnet>`_ which we used for
our experiments. There might be a newer version with improvements. |br|
**NOTE:** `scikit-data` downloads the dataset
from the internet when using the benchmark for the first time. |br|
**NOTE:** In order to run the benchmark you must adjust the paths in the
config files. |br|

You can download this benchmark by clicking `here <http://www.automl.org/hpnnet.tar.gz>`_ or running
this command from a shell:

.. code::bash

    wget http://www.automl.org/hpnnet.tar.gz
    tar -xf hpnnet.tar.gz


The HP-Nnet (HP-DBNet) is a Theano based implementation of a (deep) neural network.
It can be run on a CPU, but is drastically faster on a GPU (please follow the
theano flags instructions of the
:ref:`logistic regression <logreg>` example).
Both of them are written by James Bergstra and were used in the papers
`Random Search for Hyper-Parameter Optimization <http://jmlr.org/papers/v13/bergstra12a.html>`_
and `Algorithms for Hyper-Parameter Optimization <http://books.nips.cc/papers/files/nips24/NIPS2011_1385.pdf>`_.

.. _autoweka:

AutoWEKA
________

**NOTE:** AutoWEKA is not yet available for download!

..
    {#You can download this benchmark by clicking [here](autoweka.tar.gz) and
    [here](http://www.cs.ubc.ca/labs/beta/Projects/autoweka/datasets/convex.zip)
    or running this command from a shell:


    wget www.automl.org/autoweka.tar.gz
    tar -xf autoweka.tar.gz
    wget http://www.cs.ubc.ca/labs/beta/Projects/autoweka/datasets/convex.zip
    unzip convex.zip
    mv train.arff `echo autoweka_*/AWExperiment`
    mv test.arff `echo autoweka_*/AWExperiment`

    In case you downloaded the files from within your browser you have to move the
    file `train.arff` and `test.arff` to the directory `AWExperiment`.

    **NOTE:** This benchmark was done with a different
    sobol lib implementation for spearmint which is included in a file called
    `sobol_lib_1111_dims.py`. Please rename the file located in the directory
    `optimizers/spearmint_april_2013mode` to `sobol_lib.py` to be able to run autoweka.#}

    [AutoWEKA][AutoWEKA] is a software package which combines the machine learning toolbox [WEKA](http://www.cs.waikato.ac.nz/ml/weka/)
    with hyperparameter optimization software. But AutoWEKA goes one step further
    and also includes model selection inside the hyperparameter optimization.
    It can choose from 27 classifiers which are implemented in the WEKA toolbox.


.. raw:: html

    <a href="https://github.com/automl/HPOlib"><img style="position: absolute; top: 0; right: 0; border: 0;" src="https://camo.githubusercontent.com/652c5b9acfaddf3a9c326fa6bde407b87f7be0f4/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f6f72616e67655f6666373630302e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_orange_ff7600.png"></a>
