from __future__ import print_function
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand
from setuptools.command.install import install as InstallCommand
import io
import os

import HPOlib

here = os.path.abspath(os.path.dirname(__file__))

desc = 'A software that can be used to evaluate either black boxes or hyperparameter optimization algorithms'
long_desc = "HPOlib is a hyperparameter optimization library. It provides a common interface to three state of  \n", \
            "the art hyperparameter optimization packages: SMAC, spearmint and hyperopt. Moreover, the library  \n", \
            "provides optimization benchmarks which can be used to compare different hyperparameter optimization\n", \
            "packages and to establish standard test routines for hyperparameter optimization packages. "
keywords = 'hyperparameter optimization empirical evaluation black box'


def get_find_packages():
    packages = ['HPOlib',
                'HPOlib.config_parser',
                'HPOlib.benchmarks',
                'HPOlib.optimizers',
                'HPOlib.benchmarks.branin',
                'HPOlib.benchmarks.har6']
    #find_packages('.', exclude=('*.optimizers.*',))
    return packages

package_dir = {'HPOlib': 'HPOlib',
               'HPOlib.config_parser': 'HPOlib/config_parser',
               'HPOlib.benchmarks': 'HPOlib/benchmarks',
               'HPOlib.optimizers': 'HPOlib/optimizers',
               'HPOlib.benchmarks.branin': 'HPOlib/benchmarks/branin',
               'HPOlib.benchmarks.har6': 'HPOlib/benchmarks/har6'}
package_data = {'HPOlib': ['*/params.txt', '*/space.py', '*/config.pb'],
                'HPOlib.benchmarks.branin': ['*.cfg', '*/params.txt', '*/space.py', '*/config.pb'],
                'HPOlib.benchmarks.har6': ['*.cfg', '*/params.txt', '*/space.py', '*/config.pb'],
                'HPOlib.config_parser': ['*.cfg']}
scripts = ['HPOlib-run']


def read(*filenames, **kwargs):
    encoding = kwargs.get('encoding', 'utf-8')
    sep = kwargs.get('sep', '\n')
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)


class InstallRunsolver(InstallCommand):
    """Is supposed to install the runsolver later on"""
    def run(self):
        print("Hello User,\n"
              "you need to install runsolver on your own :-)")


class PyTest(TestCommand):
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print("Hello User,\n"
              "so far we have not included any tests. Sorry")

setup(
    name='HPOlib',
    version=HPOlib.__version__,
    url='https://github.com/automl/HPOlib',
    license='LGPLv3',
    platforms=['Linux'],
    author=HPOlib.__authors__,
    tests_require=['unittest'],
    install_requires=['argparse==1.2.1',
                      'matplotlib==1.3.1',
                      'networkx==1.8.1',
                      'numpy==1.8.0',
                      'protobuf==2.5.0',
                      'scipy==0.13.2',
                      ],
    cmdclass={'test': PyTest},
    author_email='eggenspk@informatik.uni-freiburg.de',
    description=desc,
    long_description=long_desc,
    keywords=keywords,
    packages=get_find_packages(),
    package_dir=package_dir,
    package_data=package_data,
    #include_package_data=True,
    scripts=scripts,
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ]
)
