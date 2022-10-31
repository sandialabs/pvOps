- http://www.nltk.org/nltk_data/

https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed/39142816#39142816

BEST soln: https://stackoverflow.com/questions/41691327/ssl-sslerror-ssl-certificate-verify-failed-certificate-verify-failed-ssl-c/41692664#41692664

.. code-block::python:

>>> import ssl

>>> ssl._create_default_https_context = ssl._create_unverified_context

>>> import nltk

>>> nltk.download()