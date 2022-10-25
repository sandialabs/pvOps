=================
Text2Time Example
=================
This example walks through the cleaning, processing, and
analysis of production and O&M data in tandem.

Setup the environment
---------------------

.. doctest::
    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> import shutil
    >>> import os
    >>> from pvops.text2time import visualize, utils, preprocess
    >>> from pvops.timeseries.models import linear, iec

Define paths to csv data.
.. doctest::

    >>> example_OMpath = os.path.join('example_data', 'example_om_data2.csv')
    >>> example_prodpath = os.path.join('example_data', 'example_prod_data_cumE2.csv')
    >>> example_metapath = os.path.join('example_data', 'example_metadata2.csv')

Read in csv data

.. doctest::

    >>> prod_data = pd.read_csv(example_prodpath, error_bad_lines=False, engine='python')
    >>> om_data = pd.read_csv(example_OMpath, error_bad_lines=False, engine='python')
    >>> metadata = pd.read_csv(example_metapath, error_bad_lines=False, engine='python')

Prepare data for analysis
-------------------------

Setup dictionaries to assign user's column names with pvOps names

.. doctest::

    >>> prod_col_dict = {
    ...     'siteid': 'randid', 
    ...     'timestamp': 'Date', 
    ...     'energyprod': 'Energy',
    ...     'irradiance':'Irradiance',
    ...     'baseline': 'IEC_pstep', #user's name choice for new column (baseline expected energy defined by user or calculated based on IEC)
    ...     'dcsize': 'dcsize', #user's name choice for new column (System DC-size, extracted from meta-data)
    ...     'compared': 'Compared',#user's name choice for new column
    ...     'energy_pstep': 'Energy_pstep' #user's name choice for new column
    ...     } 

    >>> om_col_dict = {
    ...     'siteid': 'randid', 
    ...     'datestart': 'date_start',
    ...     'dateend': 'date_end',
    ...     'workID': 'WONumber',
    ...     'worktype': 'WOType',
    ...     'asset': 'Asset',
    ...     'eventdur': 'EventDur', #user's name choice for new column (Repair Duration)
    ...     'modatestart': 'MonthStart', #user's name choice for new column (Month when an event begins)
    ...     'agedatestart': 'AgeStart' #user's name choice for new column (Age of system when event begins)
    ...     }
    >>> metad_col_dict = {
    ...     'siteid': 'randid',
    ...     'dcsize': 'DC_Size_kW',
    ...     'COD': 'COD'
    ...     }