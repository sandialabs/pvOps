Installation
=============

pvops is tested on Python version 3.7 and 3.8 and depends on a variety of
packages. See :ref:`reqs` for more information.

The latest release of pvops is accessible via PYPI using the following
command line prompt::

    $ pip install pvops

Alternatively, the package can be installed using github::

    $ git clone https://github.com/sandialabs/pvOps.git
    $ cd pvops
    $ pip install .

NLTK data
----------

Functions in the text package relies on the "punkt" dataset for the nltk package.
After proper installation of pvops, run the commands::
    >>> import nltk
    >>> nltk.download('punkt')
    >>> nltk.download('stopwords')

Those operating under a proxy may have difficulty with this installation.
This `stack exchange post <https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed>`_
may help.