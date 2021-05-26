.. _index:

.. image:: assets/pvops_full_logo.svg
  :width: 400


Introduction
============
pvops is a python package for PV operators & researchers. It consists a set of documented functions for supporting operations research of photovoltaic energy systems.
The library leverages advances in machine learning and visualization tools to extract and visualize actionable information from common PV data including Operations & Maintenance (O&M) text data, timeseries production data, and current-voltage (IV) curves.

How to access pvops
========================
+------------------------+
|>  ``pip install pvops``|
+------------------------+

If you are interested in contributing to pvops, then you can also access this package through the GitHub repository: ``git clone https://github.com/sandialabs/pvOps.git``.

Text Subpackage Layout
======================

The text package can be broken down into three main components: **text processing, text classification, and visualizations**.

An example implementation of all capabilities can be found in `text_class_example.py` (for specifics) and `tutorial_textmodule.ipynb` (for basics).

Text pre-processing
-------------------
These functions process the O&M data into concise, machine learning-ready documents. Additionally, extract dates from the text.

* ``preprocess.preprocessor`` acts as a wrapper function, utilizing the other preprocessing functions, which prepares the data for machine learning. 

    * See ``text_class_example.prep_data_for_ML`` module for an example.

* ``preprocessor.preprocessor(..., extract_dates_only = True)`` should be used if the primary interest is date extraction (and not continue to preprocess all text for machine learning).

    * See ``text_class_example.extract_dates`` module for an example.


Text classification
-------------------
These functions process the O&M data to make an inference on the specified event descriptor.

* ``classify.classification_deployer`` is used to conduct supervised or unsupervised classification of text documents. This function conducts a grid search across the passed classifiers and hyperparameters. 

    * The ``defaults.supervised_classifier_defs`` and ``defaults.unsupervised_classifier_defs`` functions contain default values for conducting the grid search.
    
    * See ``text_class_example.classify_supervised`` or ``text_class_example.classify_unsupervised`` modules for an example.

* Once the model is built and selected, classification (for supervised ML) or clustering (for unsupervised ML) analysis can be conducted on the best model returned from the pipeline object.

    * See ``text_class_example.predict_best_model`` module for an example.


Utils
------
These helper functions focus on performing exploratory or secondary processing activities for the O&M data

*  ``summarize_text_data`` is used to print summarized contents of the O&M data 
*  ``remap_attributes`` is used to reorganize an attribute column into a new set of labels

Visualizations
--------------
These functions create visualizations to get a better understanding about your documents.

*  ``visualize_attribute_connectivity`` can be used to visualize the connectivity of two attributes
*  ``visualize_attribute_timeseries`` can be used to evaluate the density of an attribute over time  
*  ``visualize_cluster_entropy`` can be used to observe the performance of different text embeddings  
*  ``visualize_document_clusters`` can be used after clustering to visualize popular words in each cluster
*  ``visualize_word_frequency_plot`` can be used to visualize word frequencies in the associated attribute column of O&M data

Text2Time Subpackage Layout
===========================

The text2time package can be broken down into three main components: `data pre-processing`, `utils`, and `visualizations`.

Data pre-processing
-------------------
These functions focus on pre-processing user O&M and production data to create visualizations of the merged data.

*  ``om_date_convert`` and ``prod_date_convert`` convert dates in string format to date-time objects in the O&M and production data respectively.
*  ``data_site_na`` is used to handle missing site IDs in the user data.  This function can be used for both O&M and production data.
*  ``om_datelogic_check`` is used to detect/correct issues with the logic of the O&M date, specifically when the conclusion of an event occurs before it begins.
*  ``om_nadate_process`` and ``prod_nadate_process`` are used to detect/correct any missing time-stamps in the O&M and production data respectively.

Utils
-----
These helper functions focus on performing secondary calcuations from the O&M and production data to create visualizations of the merged data.

*  ``iec_calc`` is used to calculate a comparison dataset for the production data based on an irradiance as calculated by IEC calculation
*  ``summarize_overlaps`` is used to summarize the overlapping production and O&M data.
*  ``om_summary_stats`` is used to summarize statistics (e.g., event duration and month of occurrence) of O&M data
*  ``overlapping_data`` is used to trim the production and O&M data frames and only retain the data where both datasets overlap in time.
*  ``prod_anomalies`` is used to detect/correct issues when the production data is input in cumulative format and unexpected dips show up in the data.
*  ``prod_quant`` is used to calculate a comparison between the actual production data and a baseline (e.g. the IEC calculation)

Visualizations
--------------
These functions focus on visualizing the processed O&M and production data

*  ``visualize_categorical_scatter`` generates categorical scatter plots of chosen variable based on specified category (e.g. site ID) for the O&M data.
*  ``visualize_counts`` generates a count plot of categories based on a chosen categorical variable column for the O&M data.  If that variable is the user's site ID for every ticket, a plot for total count of events can be generated.
*  ``visualize_om_prod_overlap`` creates a visualization that overlays the O&M data on top of the coinciding production data.

Timeseries Subpackage Layout
============================
These funcions focus on timeseries preprocessing and modeling. 

Preprocess
----------
* ``prod_inverter_clipping_filter`` is used to filter out production periods with inverter clipping. The core method was adopted from `pvlib/pvanalytics`.

Model
-----
* ``modeller`` is a wrapper method used to model timeseries data. This method gives multiple options for the learned model structure

iv Subpackage Layout
====================
These functions focus on current-voltage (IV) curve simulation and classification.

*  ``extractor.py`` has an object called `BruteForceExtractor` which extracts diode parameters from IV curves (even outdoor-collected).
*  ``physics_utils.py`` contains methods which match aid the IV Simulator's physics-based calculations and the preprocessing pipeline's 
   correction calculations.
*  ``preprocess.py`` contains the preprocessing function which corrects a set of data according to irradiance and temperature and normalizes
   the curves so they are easily compared.
*  ``simulator.py`` holds the `IV Simulator` which can simulate current-voltage (IV) curves under different environmental and fault conditions.
*  ``utils.py`` holds a utility function which connects to the CEC database hosted by pvLib for cell-level and module-level parameters.


.. toctree::
    :maxdepth: 2
    :caption: Available resources:
 
    modules
    examples
    references
    whatsnew
    contributing 

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
