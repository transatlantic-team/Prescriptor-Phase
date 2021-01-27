import os

import sys
from setuptools import setup, find_packages
LIBRARY_VERSION = '0.1'

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 6)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write('''
==========================
Unsupported Python version
==========================
This version of esp-sdk requires Python {}.{}, but you're trying to
install it on Python {}.{}.
'''.format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)

CUR_DIRECTORY_PATH = os.path.abspath(os.path.dirname(__file__))

# Python doesn't allow hyphens in package names so use underscore instead
PACKAGE_NAME = 'transat_rl'
LIB_NAME = 'transat-rl'


def read(fname):
    """
    Read file contents into a string
    :param fname: File to be read
    :return: String containing contents of file
    """
    with open(os.path.join(os.path.dirname(__file__), fname)) as file:
        return file.read()

setup(name=LIB_NAME,
      version=LIBRARY_VERSION,
      python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
      packages=find_packages(),
      package_dir={PACKAGE_NAME: PACKAGE_NAME},  # the one line where all the magic happens
      author="Transatlantic Team",
      url="https://github.com/transatlantic-team",
      # packages=["transat_rl"],
      install_requires=[])
