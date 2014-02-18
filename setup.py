import hashlib
from setuptools import setup
from setuptools.command.test import test as test_command
from setuptools.command.install import install as install_command
import shutil
import urllib
import os
import subprocess
import tarfile

import HPOlib

here = os.path.abspath(os.path.dirname(__file__))
desc = 'A software that can be used to evaluate either black boxes or hyperparameter optimization algorithms'
keywords = 'hyperparameter optimization empirical evaluation black box'

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
data_files = []
scripts = ['HPOlib-run', 'runsolver/src/runsolver']


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_find_packages():
    packages = ['HPOlib',
                'HPOlib.config_parser',
                'HPOlib.benchmarks',
                'HPOlib.optimizers',
                'HPOlib.benchmarks.branin',
                'HPOlib.benchmarks.har6']
    return packages


class InstallRunsolver(install_command):
    """Is supposed to install the runsolver later on"""
    runsolver_needs_to_be_installed = False
    download_url = 'http://www.cril.univ-artois.fr/~roussel/runsolver/runsolver-3.3.2.tar.bz2'
    md5_should = '62d139a6d2b1b376eb9ed59ccb5d6726'
    save_as = os.path.join(os.getcwd(), "runsolver-3.3.2.tar.bz2")
    extract_as = os.getcwd()

    def _check_for_runsolver(self):
        name = 'runsolver'
        flags = os.X_OK
        path = os.environ.get('PATH', None)
        if path is None:
            return False
        for p in os.environ.get('PATH', '').split(os.pathsep):
            p = os.path.join(p, name)
            if os.access(p, flags):
                print "Copy runsolver from %s to runsolver/src/runsolver" % p
                shutil.copy(p, os.path.join(os.getcwd(), 'runsolver/src/runsolver'))
                return p
        return False

    def _download_runsolver(self):
        try:
            urllib.urlretrieve(self.download_url, filename=self.save_as)
        except Exception, e:
            print("Error downloading %s: %s" % (self.download_url, e))
            return False
        md5 = hashlib.md5(open(self.save_as).read()).hexdigest()
        if md5 != self.md5_should:
            print "md5 checksum has changed: %s to %s" % (self.md5_should, md5)
            return False
        return True

    def _extract_runsolver(self):
        if tarfile.is_tarfile(self.save_as):
            try:
                tfile = tarfile.open(self.save_as)
                tfile.extractall(self.extract_as)
            except Exception, e:
                print("Error occurred during extraction: %s" % e)
                return False
            return True
        return False

    def _build(self):
        print("Building runsolver")
        cur_pwd = os.getcwd()
        os.chdir(os.path.join(self.extract_as, "runsolver", "src"))
        try:
            subprocess.check_call("make")
        except subprocess.CalledProcessError, e:
            print "Error during building runsolver: %s" % e
            os.chdir(cur_pwd)
            return False
        os.chdir(cur_pwd)
        return True

    def run(self):
        downloaded, extracted, built = (False, False, False)
        runsolver_needs_to_be_installed = False

        print "Check whether you already have installed runsolver"
        runsolver_path = self._check_for_runsolver()
        if not runsolver_path:
            print "'runsolver' not found, so we try to install it"
            runsolver_needs_to_be_installed = True

        download_url = 'http://www.cril.univ-artois.fr/~roussel/runsolver/runsolver-3.3.2.tar.bz2'
        save_as = os.path.join(os.getcwd(), "runsolver-3.3.2.tar.bz2")
        extract_as = os.getcwd()
        if runsolver_needs_to_be_installed:
            print("Downloading runsolver from %s, saving to %s" % (download_url, save_as))
            downloaded = self._download_runsolver()

        if runsolver_needs_to_be_installed and downloaded:
            print("Extracting runsolver to %s" % extract_as)
            extracted = self._extract_runsolver()

        if runsolver_needs_to_be_installed and extracted:
            print("Building runsolver")
            built = self._build()
            if not built:
                print "Try a second time and replace '/usr/include/asm/unistd.h' with '/usr/include/unistd.h'"
                call = "sed -i 's/\/usr\/include\/asm\/unistd/\/usr\/include\/unistd/g' runsolver/src/Makefile"
                try:
                    subprocess.check_call(call, shell=True)
                except subprocess.CalledProcessError, e:
                    print "Replacing did not work: %s" % e
                built = self._build()

        install_command.run(self)

        # Give detailed output to user
        if runsolver_needs_to_be_installed and not built:
            print "[ERROR] Please install runsolver on your own! You can download it from:\n%s" % download_url
            print "[ERROR] Be sure 'runsolver' can be found during runtime in your $PATH variable!"
        if not runsolver_needs_to_be_installed and not built:
            print "[INFO] 'runsolver' has been found on this system and was copied from: %s" % runsolver_path
        if built:
            print "'runsolver' has been downloaded and installed"


class PyTest(test_command):
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
                      'pymongo==2.6.3',
                      ],
    author_email='eggenspk@informatik.uni-freiburg.de',
    description=desc,
    long_description=read("README"),
    keywords=keywords,
    packages=get_find_packages(),
    package_dir=package_dir,
    package_data=package_data,
    data_files=data_files,
    scripts=scripts,
    cmdclass={
        'install': InstallRunsolver,
        'test': PyTest
    },
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
