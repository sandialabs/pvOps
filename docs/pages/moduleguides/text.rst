Text Guide
============

Module Overview
----------------

This module aims to support the consistent extraction of key features
in O&M data:

* timestamp information
* characteristic categorical information
* a concise synopsis of the issue for context

Implemented functions include those for filling in data gaps (text.preprocess submodule),
machine learning analyses to fill in gaps in categorical information and to
generate concise summary strings (text.classify submodule), functions
to prepare data for natural language processing (text.nlp_utils submodule),
and a visualization suite (text.visualize submodule).

An example implementation of all capabilities can be found in 
`text_class_example.py <https://github.com/sandialabs/pvOps/blob/master/tutorials/text_class_example.py>`_ 
for specifics, and `tutorial_textmodule.ipynb <https://github.com/sandialabs/pvOps/blob/master/tutorials/tutorial_textmodule.ipynb>`_ for basics.

Text pre-processing
^^^^^^^^^^^^^^^^^^^^^

:py:mod:`~pvops.text.preprocess`

These functions process the O&M data into concise, machine learning-ready documents. 
Additionally, there are options to extract dates from the text.

* :py:func:`~pvops.text.preprocess.preprocessor` acts as a wrapper function, 
  utilizing the other preprocessing functions, which prepares the data for machine learning. 

  * See ``text_class_example.prep_data_for_ML`` for an example.

* :py:func:`~pvops.text.preprocess.preprocessor` should be used with the keyword argument
  `extract_dates_only = True` if the primary interest is date extraction
  instead continuing to use the data for machine learning.

  * See ``text_class_example.extract_dates`` module for an example.


Text classification
^^^^^^^^^^^^^^^^^^^^^

:py:mod:`~pvops.text.classify`

These functions process the O&M data to make an inference on the specified event descriptor.

* :py:func:`~pvops.text.classify.classification_deployer` is used to conduct supervised 
  or unsupervised classification of text documents. 
  This function conducts a grid search across the passed classifiers and hyperparameters. 

  * The :py:func:`~pvops.text.defaults.supervised_classifier_defs` and 
    :py:func:`~pvops.text.defaults.unsupervised_classifier_defs`
    functions return default values for conducting the grid search.
    
  * See ``text_class_example.classify_supervised`` or ``text_class_example.classify_unsupervised`` 
    modules for an example.

* Once the model is built and selected, classification (for supervised ML) 
  or clustering (for unsupervised ML) analysis can be conducted on the best model returned from the pipeline object.

  * See ``text_class_example.predict_best_model`` module for an example.


Utils
^^^^^^^^^^^^^^^^^^^^^

:py:mod:`~pvops.text.utils`

These helper functions focus on performing exploratory or secondary processing activities for the O&M data.

* :py:func:`pvops.text.nlp_utils.remap_attributes` is used to reorganize an attribute column into a new set of labels.

NLP Utils
^^^^^^^^^^^^

:py:mod:`~pvops.text.utils`

These helper functions focus on processing in preparation for NLP activities.

* :py:func:`~pvops.text.nlp_utils.summarize_text_data` prints summarized contents of the O&M data.
* :py:class:`~pvops.text.nlp_utils.Doc2VecModel` performs a gensim Doc2Vec 
  transformation of the input documents to create embedded representations of the documents.
* :py:class:`~pvops.text.nlp_utils.DataDensifier` is a data structure transformer which converts sparse data to dense data. 
* :py:func:`~pvops.text.nlp_utils.create_stopwords` concatenates a list of stopwords using both words grabbed from nltk and user-specified words


Visualizations
^^^^^^^^^^^^^^^^^^^^^
These functions create visualizations to get a better understanding about your documents.

* :py:func:`~pvops.text.visualize.visualize_attribute_connectivity` visualizes the connectivity of two attributes.
  
  .. image:: ../../assets/vis_attr_connect_example.svg
    :width: 600

* :py:func:`~pvops.text.visualize.visualize_attribute_timeseries` evaluates the density of an attribute over time. 
  
  .. image:: ../../assets/vis_attr_timeseries_example.svg
    :width: 600

* :py:func:`~pvops.text.visualize.visualize_cluster_entropy` observes the performance of different text embeddings.
  
  .. image:: ../../assets/vis_cluster_entropy_example.svg
    :width: 600

* :py:func:`~pvops.text.visualize.visualize_document_clusters` visualizes popular words in clusters after a cluster analysis is ran.
  
  .. image:: ../../assets/vis_doc_clusters_example.svg
    :width: 600

* :py:func:`~pvops.text.visualize.visualize_word_frequency_plot` visualizes word frequencies in the associated attribute column of O&M data.
  
  .. image:: ../../assets/vis_freq_plot_example.svg
    :width: 600


.. Example Code
.. --------------