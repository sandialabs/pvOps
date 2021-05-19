# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import mock
MODULES = ['numpy', 'nltk', 'sklearn.pipeline', 'sklearn.model_selection',
           'scipy.sparse', 'pandas', 'scipy', 'sklearn.base',
           'gensim.models.doc2vec', 'nltk.tokenize', 'datefinder',
           'text_remove_nondate_nums', 'text_remove_numbers_stopwords',
           'get_dates', 'gensim.models', 'sklearn.svm', 'sklearn.tree',
           'sklearn.neural_network', 'sklearn.linear_model',
           'sklearn.ensemble', "sklearn.cluster", "networkx", "matplotlib",
           "matplotlib.pyplot", "gensim.models.doc2vec",
           "sklearn.feature_extraction.text", "nltk.tokenize",
           "plotly.graph_objects", "scipy.signal", 'matplotlib.colors',
           'seaborn', 'matplotlib.ticker', 'scipy.signal.find_peaks',
           'pvlib', 'pvanalytics', 'timezonefinder', 'sklearn', "pyDOE",
           "sklearn.metrics", "scipy.interpolate", "keras", "keras.layers",
           "sklearn.utils", "sklearn.preprocessing", "keras.models",
           "keras.utils"]

for module in MODULES:
    sys.modules[module] = mock.Mock()

sys.path.insert(0, os.path.abspath("../pvops"))
sys.path.insert(0, os.path.abspath("../pvops/text2time"))
sys.path.insert(0, os.path.abspath("../pvops/text"))
sys.path.insert(0, os.path.abspath("../pvops/timeseries"))
sys.path.insert(0, os.path.abspath("../pvops/iv"))


# -- Project information -----------------------------------------------------

project = "pvops"
copyright = "2021, Thushara Gunda"
author = "Thushara Gunda"

# The full version, including alpha/beta/rc tags
release = "0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc", "nbsphinx", "nbsphinx_link"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
