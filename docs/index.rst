.. _index:

******************************************************
pvops: a python package for PV operators & researchers
******************************************************

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   modules
   contributing

Introduction
============
pvops is a python package for PV operators & researchers. It is a collection of functions for working with text-based data from photovoltaic power systems. The library includes functions for processing text data as well as fusion of the text information with time series data for visualization of contextual details for data analysis. 

How to access pvops
========================
+------------------------+
|>  ``pip install pvops``|
+------------------------+

If you are interested in contributing to pvops, then you can also access this package through the GitHub repository: ``git clone https://github.com/tgunda/pvOps.git``.

Text Subpackage Layout
======================

The text package can be broken down into three main components: **text processing, text classification, and visualizations**.

An example implementation of all capabilities can be found in `text_class_example.py` (for specifics) and `tutorial_textmodule.ipynb` (for basics).

Text Processing
---------------
Process the documents into concise, machine learning-ready documents. Additionally, extract dates from the text.

* ``preprocessor.py`` acts as a wrapper function, utilizing the other preprocessing functions, which preps the data for machine learning. 
* If all you want to do is extract dates (and not continue to preprocess all text for machine learning), then call preprocessor as ``preprocessor.preprocessor(..., extract_dates_only = True)``
    * see ``text_class_example.extract_dates()`` module for an example

* To prep documents for machine learning, utilize the ``preprocessor.preprocessor()``
    * see ``text_class_example.prep_data_for_ML()`` module for an example

Text classification
-------------------
The written tickets are used to make an inference on the specified event descriptor.

* Conduct supervised or unsupervised classification of text documents utilizing ``classification_deployer.classification_deployer()``. This function conducts a grid search across the passed classifiers and hyperparameters.
    * See ``text_class_example.classify_supervised()`` or ``text_class_example.classify_unsupervised()`` modules for an example
* Once the model is built and selected, one can conduct classification (for supervised ML) or clustering (for unsupervised ML) by conducting a prediction on the returned pipeline object. 
    * See ``text_class_example.predict_best_model()`` module for an example

Visualizations
--------------
Create visualizations to get a better understanding about your documents.

*  Observe brief description about the passed documents by calling ``summarize_text_data.summarize_text_data()``
*  Observe attribute ticket densities across time using ``visualize_ticket_publication_timeseries``
*  Observe the performance of different text embeddings by calling ``visualize_cluster_entropy``
*  Observe word frequencies in the passed attribute's documents by calling ``visualize_freqPlot``
*  Observe graph which indicates the connectivity of two attributes by calling ``visualize_attribute_connectivity``
*  After clustering, utilize ``visualize_document_clusters`` to observe popular words in each cluster


Text2Time Subpackage Layout
===========================

The text2time package can be broken down into two main components: `data pre-processing` and `visualizations`.

Data pre-processing
-------------------
These functions focus on pre-processing user O&M and production data to create visualizations of the merged data.

*  ``om_date_convert`` and ``prod_date_convert`` convert dates in string format to date-time objects in the O&M and production data respectively.
*  ``data_site_na`` is used to handle missing site IDs in the user data.  This function can be used for both O&M and production data.
*  ``om_datelogic_check`` is used to detect/correct issues with the logic of the O&M date, specifically when the conclusion of an event occurs before it begins.
*  ``om_nadate_process`` and ``prod_nadate_process`` are used to detect/correct any missing time-stamps in the O&M and production data respectively.

Utils
------
These functions focus on pre-processing user O&M and production data to create visualizations of the merged data.
*  ``iec_calc`` is used to calculate a comparison dataset for the production data based on an irradiance as calculated by IEC calculation
*  ``summarize_overlaps`` is used to summarize the overlapping production and O&M data.
* ``om_summary_stats`` is used to summarize statistics (e.g., event duration and month of occurrence) of O&M data
*  ``overlapping_data`` is used to trim the production and O&M data frames and only retain the data where both datasets overlap in time.
*  ``prod_anomalies`` is used to detect/correct issues when the production data is input in cumulative format and unexpected dips show up in the data.
*  ``prod_quant`` is used to calculate a comparison between the actual production data and a baseline (e.g. the IEC calculation)


Visualizations
--------------
These functions focus on visualizing the processed O&M and production data

*  ``visualize_categorical_scatter`` generates categorical scatter plots of chosen variable based on specified category (e.g. site ID) for the O&M data.
*  ``visualize_counts`` generates a count plot of categories based on a chosen categorical variable column for the O&M data.  If that variable is the user's site ID for every ticket, a plot for total count of events can be generated.
*  ``visualize_om_prod_overlap`` creates a visualization that overlays the O&M data on top of the coinciding production data.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
