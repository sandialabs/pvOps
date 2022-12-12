text Guide
============

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
`text_class_example.py` for specifics and `tutorial_textmodule.ipynb` for basics.

Text pre-processing
^^^^^^^^^^^^^^^^^^^^^
These functions process the O&M data into concise, machine learning-ready documents. Additionally, extract dates from the text.

* :py:func:`text.preprocess.preprocessor` acts as a wrapper function, utilizing the other preprocessing functions, which prepares the data for machine learning. 

    * See ``text_class_example.prep_data_for_ML`` module for an example.

* ``preprocessor.preprocessor(..., extract_dates_only = True)`` should be used if the primary interest is date extraction,
  rather than continuing to use the data for machine learning.

    * See ``text_class_example.extract_dates`` module for an example.


Text classification
^^^^^^^^^^^^^^^^^^^^^
These functions process the O&M data to make an inference on the specified event descriptor.

* ``classify.classification_deployer`` is used to conduct supervised or unsupervised classification of text documents. 
  This function conducts a grid search across the passed classifiers and hyperparameters. 

    * The ``defaults.supervised_classifier_defs`` and ``defaults.unsupervised_classifier_defs`` 
      functions contain default values for conducting the grid search.
    
    * See ``text_class_example.classify_supervised`` or ``text_class_example.classify_unsupervised`` 
      modules for an example.

* Once the model is built and selected, classification (for supervised ML) 
  or clustering (for unsupervised ML) analysis can be conducted on the best model returned from the pipeline object.

    * See ``text_class_example.predict_best_model`` module for an example.


Utils
^^^^^^^^^^^^^^^^^^^^^
These helper functions focus on performing exploratory or secondary processing activities for the O&M data

*  ``summarize_text_data`` is used to print summarized contents of the O&M data.
*  ``remap_attributes`` is used to reorganize an attribute column into a new set of labels.

Visualizations
^^^^^^^^^^^^^^^^^^^^^
These functions create visualizations to get a better understanding about your documents.

*  ``visualize_attribute_connectivity`` can be used to visualize the connectivity of two attributes.
*  ``visualize_attribute_timeseries`` can be used to evaluate the density of an attribute over time. 
*  ``visualize_cluster_entropy`` can be used to observe the performance of different text embeddings.
*  ``visualize_document_clusters`` can be used after clustering to visualize popular words in each cluster.
*  ``visualize_word_frequency_plot`` can be used to visualize word frequencies in the associated attribute column of O&M data.
