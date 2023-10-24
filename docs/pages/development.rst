.. _development:

Developing pvOps
=====================

Installation
------------

To maintain a local installation, developers should use the following commands::
    
    git clone https://github.com/sandialabs/pvOps.git
    cd pvops
    pip install -e .

Testing
-------
To test locally, run::

    pytest pvops

at the root of the repository. Note that this requires the installation
of pytest.

Linting
-------

Pvops uses flake8 to maintain code standards. To lint locally using 
the same filters required by pvops CI/CD pipeline, run the following
command at the root of the repository::

    flake8 . --count --statistics --show-source --ignore=E402,E203,E266,E501,W503,F403,F401,E402,W291,E302,W391,W292,F405,E722,W504,E121,E125,E712

Note that this requires the installation of flake8.

Documentation
------------------

Building docs
^^^^^^^^^^^^^^^

To build docs locally, navigate to ``pvops/docs`` and run::

    make html

After building, the static html files can be found in ``_build/html``.

Docstrings
^^^^^^^^^^^

The pvOps documentation adheres to NumPy style docstrings. Not only does this
help to keep a consistent style, but it is also necessary for the API documentation
to be parsed and displayed correctly. For an example of what this should look like::

    def func(arg1, arg2):
    """Summary line.

    Extended description of function.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        Description of return value

    """
    return True

Additional examples can be found in the 
`napoleon documentation <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html>`_.

Extending Documentation
^^^^^^^^^^^^^^^^^^^^^^^

When adding new functionality to the repository, it is important
to check that it is being properly documented in the API documentation.
Most of this is automatic. For example, if a function is added to 
``pvops.text.visualize`` with a proper docstring, there is no more work to do.
However, when new files are created they must be added to the appropriate page
in ``docs/pages/apidoc`` so that the automatic documentation recognizes it.

New pages should be placed into ``docs/pages``, and linked to in
``index.html``, or another page. It is recommended to use absolute paths
(starting from the root of the documentation) when linking anything.
