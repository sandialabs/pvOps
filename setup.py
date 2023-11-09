import os
import re

try:
    from setuptools import setup, find_packages
except ImportError:
    raise RuntimeError('setuptools is required')

DESCRIPTION = ('pvops is a python library for the analysis of ' +
               'field collected operational data for photovoltaic systems.')

LONG_DESCRIPTION = """
pvops is a python package for PV operators & researchers. It is
a collection of functions for working with text-based data
from photovoltaic power systems. The library includes functions for
processing text data as well as fusion of the text information with
time series data for visualization of contextual details for data
analysis.

Documentation: https://pvops.readthedocs.io/en/latest/index.html

Source code: https://github.com/sandialabs/pvOps

"""

DISTNAME = 'pvops'
MAINTAINER = "Thushara Gunda"
MAINTAINER_EMAIL = 'tgunda@sandia.gov'
AUTHOR = 'pvOps Developers'
LICENSE = 'BSD 3-Clause License'
URL = 'https://github.com/sandialabs/pvops'

TESTS_REQUIRE = [
    'pytest',
]

INSTALL_REQUIRES = [
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn',
    'nltk',
    'datefinder',
    'matplotlib',
    'seaborn',
    'plotly',
    'gensim',
    'networkx',
    'pvlib',
    'pvanalytics',
    'timezonefinder',
]

DOCS_REQUIRE = [
    'sphinx==7.2.6',
    'coverage==7.2.3',
    'ipykernel==6.22.0',
    'nbconvert==7.3.1',
    'nbformat==5.8.0',
    'nbsphinx==0.9.3',
    'nbsphinx-link==1.3.0',
    'sphinx-copybutton==0.5.2',
    'sphinxcontrib-bibtex==2.5.0',
    'sphinx_rtd_theme==1.3.0',
]

IV_REQUIRE = [
    'keras',
    'tensorflow',
    'pyDOE',
]

EXTRAS_REQUIRE = {
    'iv': IV_REQUIRE,
    'test': TESTS_REQUIRE,
    'doc': DOCS_REQUIRE
}

EXTRAS_REQUIRE['all'] = sorted(set(sum(EXTRAS_REQUIRE.values(), [])))

SETUP_REQUIRES = ['setuptools_scm']

CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering'
]

PACKAGES = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])

# get version from __init__.py
file_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(file_dir, 'pvops', '__init__.py')) as f:
    version_file = f.read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        VERSION = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name=DISTNAME,
    use_scm_version=True,
    packages=PACKAGES,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=TESTS_REQUIRE,
    setup_requires=SETUP_REQUIRES,
    ext_modules=[],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    url=URL,
    version=VERSION
)
