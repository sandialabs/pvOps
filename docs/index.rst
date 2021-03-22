.. _index:

******************************************************
pvOps: a python package for PV operators & researchers
******************************************************

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Introduction
============
pvOps contains a series of functions to facilitate fusion of text-based data with time series production data collected at photovoltaic sites. The package also contains example datasets and tutorials to help demonstrate how the functions can be used.

Installation
=============
pvOps can be installed using `pip`.


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

*  ``om_date_convert`` and ``prod_date_convert`` convert dates in string format to date-time objects
*  ``data_site_na`` is used to handle missing site IDs in the user data.  This function can be used for both O&M and production data.
*  ``om_datelogic_check`` is used to detect/correct issues with the logic of the O&M date, specifically when the conclusion of an event occurs before it begins.
*  ``prod_anomalies`` is used to detect/correct issues when the production data is input in cumulative format and unexpected dips show up in the data.
*  ``prod_nadate_process`` is used to detect/correct any missing time-stamps in the production data.
*  ``om_nadate_process`` is used to detect/correct any missing time-stamps in the O&M data.
*  ``summarize_overlaps`` is used to summarize the overlapping production and O&M data.
*  ``overlapping_data`` is used to trim the production and O&M data frames and only retain the data where both datasets overlap in time.
*  ``iec_calc`` is used to calculate a comparison dataset for the production data based on an irradiance as calculated by IEC calculation
*  ``prod_quant`` is used to calculate a comparison between the actual production data and a baseline (e.g. the IEC calculation)

Visualizations
--------------
These functions focus on visualizing the processed O&M and production data

*  ``visualize_om_prod_overlap`` creates a visualization that overlays the O&M data on top of the coinciding production data.
*  ``visualize_categorical_scatter`` generates categorical scatter plots of chosen variable based on specified category (e.g. site ID) for the O&M data.
*  ``visualize_counts`` generates a count plot of categories based on a chosen categorical variable column for the O&M data.  If that variable is the user's site ID for every ticket, a plot for total count of events can be generated.


Contributing
============

The long-term success of pvOps requires community support. One can make contributions by submitting issues and pull requests. Please follow the PR template

Copyright and License
=====================

pvOps is copyright through Sandia National Laboratories. The software is distributed under the Revised BSD License.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
