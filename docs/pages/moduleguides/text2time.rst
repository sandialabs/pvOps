Text2Time Guide
================

Module Overview
----------------

Aligning production data with O&M tickets is not a trivial task since 
intersection of dates and identification of anomalies depends on the nuances 
within the two datasets. This set of functions facilitate this 
data fusion. Key features include:

* conducting quality checks and controls on data.
* identification of overlapping periods between O&M and production data.
* generation of baseline values for production loss estimations.
* calculation of losses from production anomalies for specific time periods.

An example of usage can be found in 
`tutorial_text2time_module.ipynb <https://github.com/sandialabs/pvOps/blob/master/tutorials/tutorial_text2time_module.ipynb>`_.


The text2time package can be broken down into three main components: 
`data pre-processing`, `utils`, and `visualizations`.

Data pre-processing
^^^^^^^^^^^^^^^^^^^^^

:py:mod:`text2time.preprocess module <pvops.text2time.preprocess>`

These functions pre-process user O&M and production data to prepare them for 
further analyses and visualizations.

* :py:func:`~pvops.text2time.preprocess.om_date_convert` and 
  :py:func:`~pvops.text2time.preprocess.prod_date_convert`
  convert dates in string format to date-time objects in the O&M and 
  production data respectively.
* :py:func:`~pvops.text2time.preprocess.data_site_na` 
  handles missing site IDs in the user data.  This function can 
  be used for both O&M and production data.
* :py:func:`~pvops.text2time.preprocess.om_datelogic_check` 
  detects and handles issues with the logic of the O&M date, specifically 
  when the conclusion of an event occurs before it begins.
* :py:func:`~pvops.text2time.preprocess.om_nadate_process` and 
  :py:func:`~pvops.text2time.preprocess.prod_nadate_process` 
  detect and handle any missing time-stamps in the O&M and 
  production data respectively.

Utils
^^^^^^^^^^^^^^^^^^^^^

:py:mod:`text2time.utils module <pvops.text2time.utils>`

These functions perform secondary calcuations
on the O&M and production data to aid in data analyses and visualizations.

* :py:func:`~pvops.text2time.utils.iec_calc` calculates a 
  comparison dataset for the production data based on an irradiance as 
  calculated by IEC calculation.
* :py:func:`~pvops.text2time.utils.summarize_overlaps` summarizes 
  the overlapping production and O&M data.
* :py:func:`~pvops.text2time.utils.om_summary_stats` summarizes 
  statistics (e.g., event duration and month of occurrence) of O&M data.
* :py:func:`~pvops.text2time.utils.overlapping_data` trims the 
  production and O&M data frames and only retain the data where both datasets 
  overlap in time.
* :py:func:`~pvops.text2time.utils.prod_anomalies` detects and handles 
  issues when the production data is input in cumulative format and unexpected 
  dips show up in the data.
* :py:func:`~pvops.text2time.utils.prod_quant` calculates a 
  comparison between the actual production data and a baseline 
  (e.g. from a model from :ref:`timeseries models`).

Visualizations
^^^^^^^^^^^^^^^^^^^^^

:py:mod:`text2time.visualize module <pvops.text2time.visualize>`

These functions visualize the processed O&M and production data:

* :py:func:`~pvops.text2time.visualize.visualize_categorical_scatter` 
  generates categorical scatter plots of chosen variable based on specified 
  category (e.g. site ID) for the O&M data.

  .. image:: ../../assets/vis_cat_scatter_example.svg
    :width: 600

* :py:func:`~pvops.text2time.visualize.visualize_counts` 
  generates a count plot of categories based on a chosen categorical variable
  column for the O&M data.  
  If that variable is the user's site ID for every ticket, a plot for total 
  count of events can be generated.

  .. image:: ../../assets/vis_counts_example.svg
    :width: 600

* :py:func:`~pvops.text2time.visualize.visualize_om_prod_overlap` 
  creates a visualization that overlays the O&M data on top of the 
  coinciding production data.

  .. image:: ../../assets/vis_overlap_example.png
    :width: 600

Example Code
--------------

Load in OM data and convert dates to python date-time objects

.. doctest::

  >>> import pandas as pd
  >>> import os
  >>> from pvops.text2time import preprocess
  
  >>> example_OMpath = os.path.join('example_data', 'example_om_data2.csv')
  >>> om_data = pd.read_csv(example_OMpath, on_bad_lines='skip', engine='python')
  >>> om_col_dict = {
  ... 'siteid': 'randid',
  ... 'datestart': 'date_start',
  ... 'dateend': 'date_end',
  ... 'workID': 'WONumber',
  ... 'worktype': 'WOType',
  ... 'asset': 'Asset',
  ... 'eventdur': 'EventDur', #user's name choice for new column (Repair Duration)
  ... 'modatestart': 'MonthStart', #user's name choice for new column (Month when an event begins)
  ... 'agedatestart': 'AgeStart'} #user's name choice for new column (Age of system when event begins)
  >>> om_data_converted = preprocess.om_date_convert(om_data, om_col_dict)