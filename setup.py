import hashlib
from setuptools import setup
from setuptools.command.install import install
import shutil
import urllib
import os
import subprocess
import sys
import socket
import tarfile

import HPOlib

# Otherwise installing the HPOlib when having no internet connection can take
# very long because urllib will block while trying to download a file
socket.setdefaulttimeout(10)

here = os.path.abspath(os.path.dirname(__file__))
desc = 'A software that can be used to evaluate either black boxes or hyperparameter optimization algorithms'
keywords = 'hyperparameter optimization empirical evaluation black box'

package_dir = {'HPOlib': 'HPOlib',
               'HPOlib.config_parser': 'HPOlib/config_parser',
               'HPOlib.Plotting': 'HPOlib/Plotting',
               'HPOlib.format_converter': 'HPOlib/format_converter',
               'HPOlib.dispatcher': 'HPOlib/dispatcher',
               'HPOlib.benchmarks': 'HPOlib/benchmarks'}
package_data = {'HPOlib.config_parser': ['*.cfg']}

data_files = [] + \
             [(d, [os.path.join(d, f) for f in files])
              for d, folders, files in os.walk("HPOlib/dispatcher/MySQLDBTAE")]

scripts = ['scripts/HPOlib-run', 'scripts/HPOlib-plot',
           'runsolver/src/runsolver', 'scripts/HPOlib-convert',
           'scripts/remove_minus.py', 'scripts/HPOlib-testbest',
           'scripts/HPOlib-pyFanova']


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_find_packages():
    packages = ['HPOlib',
                'HPOlib.config_parser',
                'HPOlib.Plotting',
                'HPOlib.format_converter',
                'HPOlib.dispatcher',
                'HPOlib.benchmarks']
    return packages


def download_source(download_url, md5, save_as):
    try:
        urllib.urlretrieve(download_url, filename=save_as)
    except Exception, e:
        sys.stdout.write("Error downloading %s: %s\n" % (download_url, e))
        return False
    md5_new = hashlib.md5(open(save_as).read()).hexdigest()
    if md5_new != md5:
        sys.stdout.write("md5 checksum has changed: %s to %s\n" % (md5, md5_new))
        return False
    return True


def extract_source(filename, extract_as):
    if tarfile.is_tarfile(filename):
        try:
            tfile = tarfile.open(filename)
            tfile.extractall(extract_as)
        except Exception, e:
            sys.stdout.write("Error occurred during extraction: %s\n" % e)
            return False
        return True
    return False


def copy_folder(new_dir, old_dir):
    try:
        shutil.copytree(old_dir, new_dir, symlinks=True,
                        ignore=shutil.ignore_patterns('*.pyc', '*_src*'))
    except shutil.Error as e:
        sys.stderr.write("[shutil.ERROR] Could not copy folder from %s to %s\n" % (old_dir, new_dir))
        sys.stderr.write("%s\n" % e.message)
        return False
    except Exception, e:
        sys.stderr.write("[ERROR] Could not copy folder from %s to %s\n" % (old_dir, new_dir))
        sys.stderr.write("%s\n" % e.message)
        return False
    return True


def make_dir(new_dir):
    try:
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        else:
            sys.stdout.write("[INFO] %s is already a directory, we use it for our installation\n" %
                             new_dir)
    except Exception, e:
        sys.stdout.write("[ERROR] Something (%s) went wrong, please create %s and try again\n" %
                         (e.message, new_dir))


class AdditionalInstall(install):
    """Is supposed to install the runsolver, download optimizers,
     and copy benchmarks"""

    def _check_for_runsolver(self):
        name = 'runsolver'
        flags = os.X_OK
        path = os.environ.get('PATH', None)
        if path is None:
            return False
        for p in os.environ.get('PATH', '').split(os.pathsep):
            p = os.path.join(p, name)
            if os.access(p, flags) and not os.path.isdir(p):
                if not os.path.isdir(os.path.join(os.getcwd(), 'runsolver/src/')):
                    os.mkdir(os.path.join(os.getcwd(), 'runsolver/src/'))
                sys.stdout.write("Copy runsolver from %s to runsolver/src/runsolver\n" % p)
                target = os.path.join(os.getcwd(), 'runsolver/src/runsolver')
                if not os.path.samefile(p, target):
                    shutil.copy(p, target)
                return p
        return False

    def _build(self, build_dir):
        sys.stdout.write("Building runsolver\n")
        cur_pwd = os.getcwd()
        os.chdir(build_dir)
        try:
            subprocess.check_call("make")
        except subprocess.CalledProcessError, e:
            sys.stdout.write("Error while building runsolver: %s\n" % e)
            os.chdir(cur_pwd)
            return False
        os.chdir(cur_pwd)
        return True

    def _copy_and_download_optimizer(self, optimizer_dir, optimizer_name, optimizer_tar_name,
                                     url, md5):
        load, extract = (False, False)

        load = download_source(download_url=url, md5=md5,
                               save_as=os.path.join(optimizer_dir, optimizer_tar_name))

        if load:
            extract = extract_source(os.path.join(optimizer_dir, optimizer_tar_name),
                                     extract_as=os.path.join(optimizer_dir, optimizer_name))
            os.remove(os.path.join(optimizer_dir, optimizer_tar_name))

        if load and extract:
            return True
        else:
            return False

    def run(self):
        # RUNSOLVER STUFF
        here_we_are = os.getcwd()
        runsolver_tar_name = "runsolver-3.3.4.tar.bz2"
        runsolver_name = "runsolver-3.3.4"
        if sys.version_info < (2, 7, 0) or sys.version_info >= (2, 8, 0):
            sys.stderr.write("HPOlib requires Python 2.7.0\n")
            sys.exit(-1)

        downloaded, extracted, built = (False, False, False)
        runsolver_needs_to_be_installed = False

        runsolver_path = self._check_for_runsolver()
        if not runsolver_path:
            sys.stdout.write("[INFO] 'runsolver' not found, so we try to install it\n")
            runsolver_needs_to_be_installed = True

        if runsolver_needs_to_be_installed:
            sys.stdout.write("Downloading runsolver from %s%s, saving to %s/%s\n" %
                             ('http://www.cril.univ-artois.fr/~roussel/runsolver/',
                              runsolver_tar_name, os.getcwd(), runsolver_tar_name))
            downloaded = download_source(download_url='http://www.cril.univ-artois.fr/~roussel/runsolver/%s' %
                                                      runsolver_tar_name,
                                         md5="5a9511266489c87f4a276b9e54ea4082",
                                         save_as=os.path.join(here_we_are, runsolver_tar_name))

        if runsolver_needs_to_be_installed and downloaded:
            sys.stdout.write("Extracting runsolver to %s\n" % os.path.join(here_we_are, runsolver_name))
            extracted = extract_source(filename=os.path.join(here_we_are, runsolver_tar_name), extract_as=here_we_are)

        if runsolver_needs_to_be_installed and extracted:
            sys.stdout.write("Building runsolver\n")
            sys.stdout.write("Replace a line, which seems to cause trouble on 32 bit systems'\n")
            call = "sed -i 's|gnu/asm/unistd_$(WSIZE).h|gnu/asm/unistd_$(WSIZE).h  /usr/include/i386-linux-gnu/asm/unistd_$(WSIZE).h|' runsolver/src/Makefile"
            try:
                subprocess.check_call(call, shell=True)
            except subprocess.CalledProcessError, e:
                sys.stdout.write("Replacing did not work: %s\n" % e)
            md5_new = hashlib.md5(open("runsolver/src/Makefile").read()).hexdigest()
            runsolver_md5 = "4870722e47a6f74d5167376d371f1730"
            if md5_new != runsolver_md5:
                sys.stdout.write("md5 checksum has changed: %s to %s, not compiling runsolver\n" % (runsolver_md5, md5_new))
            else:
                built = self._build(build_dir=os.path.join(here_we_are, "runsolver", "src"))

        # COPY/DOWNLOAD OPTIMIZER TO ROOT FOLDER
        optimizer_dir = os.path.join(here_we_are, "optimizers")

        tpe, smac, spearmint = (False, False, False)
        tpe = self._copy_and_download_optimizer(optimizer_dir=optimizer_dir,
                                                optimizer_name='tpe',
                                                optimizer_tar_name="hyperopt_august2013_mod_src.tar.gz",
                                                url="http://www.automl.org/hyperopt_august2013_mod_src.tar.gz",
                                                md5='15ce1adf9d32bb7f71dcfb9a847c55c7')
        # If one optimizer fails, others are likely to fail as well
        if tpe:
            smac = self._copy_and_download_optimizer(optimizer_dir=optimizer_dir,
                                                     optimizer_name='smac',
                                                     optimizer_tar_name="smac_2_06_01-dev_src.tar.gz",
                                                     url="http://www.automl.org/smac_2_06_01-dev_src.tar.gz",
                                                     md5='30ab1d09696de47efac77ed163772c0a')

            spearmint = self._copy_and_download_optimizer(optimizer_dir=optimizer_dir,
                                                          optimizer_name='spearmint',
                                                          optimizer_tar_name="spearmint_april2013_mod_src.tar.gz",
                                                          url="http://www.automl.org/spearmint_april2013" +
                                                              "_mod_src.tar.gz",
                                                          md5='6c89c387b2062cd8319a4b4936a1dded') # '340fc0da97a30454d633ce9781b05369')

            smac_2_08 = self._copy_and_download_optimizer(optimizer_dir=optimizer_dir,
                                                          optimizer_name='smac',
                                                          optimizer_tar_name="smac_2_08_00-master_src.tar.gz",
                                                          url="http://www.automl.org/smac_2_08_00-master_src.tar.gz",
                                                          md5='2be626a5437b56da2eba1b67b7a94367')


        downloaded_aeatk, extracted_aeatk = (False, False)
        # Download the AEATK/MySQLTAE-bundle
        aeatk_bundle_base_url = "http://automl.org/"
        aeatk_bundle_filename = "aeatk-v2.08.01-development-4.tar.gz"
        aeatk_url = aeatk_bundle_base_url + aeatk_bundle_filename
        sys.stdout.write("[INFO] downloading MySQLDBTAE from %s\n" % aeatk_url)
        downloaded_aeatk = download_source(aeatk_url,
                                           "59b6947d3e76b9ec0d60f0702570f710",
            save_as=os.path.join(here_we_are, "MySQLDBTAE.tar.gz"))
        if downloaded_aeatk:
            extracted_aeatk = extract_source(
                os.path.join(here_we_are, "MySQLDBTAE.tar.gz"),
                extract_as=os.path.join(here_we_are, 'HPOlib', 'dispatcher'))
        if not (downloaded_aeatk and extracted_aeatk):
            sys.stderr.write("[ERROR] MySQLDBTAE could not be installed "
                             "properly. You will not be able to use the "
                             "MySQLDBTAE dispatcher.")

        # TODO: Normally one wants to call run(self), but this runs distutils and ignores install_requirements for unknown reasons
        # if anyone knows a better way, feel free to change
        install.do_egg_install(self)

        # Give detailed output to user
        if not tpe or not smac or not spearmint or not smac_2_08:
            sys.stderr.write("[ERROR] Something went wrong while copying and downloading optimizers." +
                             "Please do the following to be ready to start optimizing:\n\n" +
                             "cd optimizers\n" +
                             "wget http://www.automl.org/hyperopt_august2013_mod_src.tar.gz \n" +
                             "wget http://www.automl.org/smac_2_06_01-dev_src.tar.gz \n" +
                             "wget http://www.automl.org/smac_2_08_00-master_src.tar.gz \n" +
                             "wget http://www.automl.org/spearmint_april2013_mod_src.tar.gz \n" +
                             "tar -xf hyperopt_august2013_mod_src.tar.gz \n" +
                             "mv hyperopt_august2013_mod_src tpe/ \n" +
                             "tar -xf smac_2_06_01-dev_src.tar.gz \n" +
                             "mv smac_2_06_01-dev_src.tar.gz smac/ \n" +
                             "tar -xf smac_2_08_00-master_src.tar.gz \n" +
                             "mv smac_2_08_00-master_src.tar.gz smac/ \n" +
                             "tar -xf spearmint_april2013_mod_src.tar.gz \n" +
                             "mv spearmint_april2013_mod_src spearmint/ \n\n" +
                             "Thank You!\n")
        if not extracted_aeatk:
            sys.stdout.write("[ERROR] MySQLDBTAE for the dispatcher "
                             "use_workers could not be installed.\n")
        if runsolver_needs_to_be_installed and not built:
            sys.stdout.write("[ERROR] Please install runsolver on your own! You can download it from:\n%s%s\n" % \
                  ('http://www.cril.univ-artois.fr/~roussel/runsolver/%s', runsolver_tar_name))
            sys.stdout.write("[ERROR] Be sure 'runsolver' can be found during runtime in your $PATH variable!\n")
        if not runsolver_needs_to_be_installed and not built:
            sys.stdout.write("[INFO] 'runsolver' has been found on this system and was copied from: %s\n" %
                             runsolver_path)
        if built:
            sys.stdout.write("'runsolver' has been downloaded and installed\n")

setup(
    name='HPOlib',
    version=HPOlib.__version__,
    url='https://github.com/automl/HPOlib',
    license='LGPLv3',
    platforms=['Linux'],
    author=HPOlib.__authors__,
    test_suite="tests.testsuite.suite",
    # setup_requires=['numpy'],
    # We require scipy, but this does not automatically install numpy,
    # so the user needs to make sure numpy is already installed
    install_requires=['argparse','numpy',
                      'matplotlib',
                      'networkx',
                      'protobuf',
                      'scipy>=0.13.2',
                      'pymongo',
                      'psutil',
                      'pyparsing'
                      ],
    author_email='eggenspk@informatik.uni-freiburg.de',
    description=desc,
    long_description=read("README.md"),
    keywords=keywords,
    packages=get_find_packages(),
    package_dir=package_dir,
    package_data=package_data,
    data_files=data_files,
    scripts=scripts,
    cmdclass={'install': AdditionalInstall},
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Development Status :: 3 - Alpha',
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
