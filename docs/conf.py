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
from pvops import __version__

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = u"pvops"
copyright = u"2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS)"
author = u"pvOps Developers"
version = __version__
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

language = 'en'

extensions = [
    "sphinx.ext.autodoc",
    # pull in documentation from docstrings in a semi-automatic way.
    "nbsphinx",
    # nbsphinx is a Sphinx extension that provides a source parser
    # for *.ipynb files
    "nbsphinx_link",
    # A sphinx extension for including notebook files from outside
    # the sphinx source root.
    "sphinx_copybutton",
    # adds copy button to code blocks
    "sphinx.ext.coverage",
    # `make coverage` summarizes what has docstrings
    'sphinx.ext.doctest',
    # allows for testing of code snippets
    'sphinx.ext.viewcode',
    # add links to highlighted source code
    'sphinx.ext.napoleon',
    # add parsing for google/numpy style docs
    'sphinxcontrib.bibtex',
    # for bibtex referencing
]


coverage_show_missing_items = True
napoleon_numpy_docstring = True  # use numpy style
napoleon_google_docstring = False  # not google style
napoleon_use_rtype = False  # option for return section formatting
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
napoleon_use_ivar = True  # option for attribute section formatting
napoleon_use_param = False  # option for parameter section formatting
viewcode_import = True  # tries to find the source files
bibtex_bibfiles = ['refs/pvops.bib']

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
html_style = 'css/my_style.css'
