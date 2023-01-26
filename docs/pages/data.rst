Loading Data into pvOps
===========================

pvOps handles three types of photovoltaic data:

Types of Data
^^^^^^^^^^^^^^

Production
----------
.. Description of production style data
.. Explanation of column dictionaries
.. doctest for loading and viewing data

Operation and maintenance
-------------------------
.. Description of OM style data
.. Explanation of column dictionaries
.. doctest for loading and viewing data

IV curves
---------
.. Description of IV style data
.. Explanation of column dictionaries
.. doctest for loading and viewing data

Column Dictionaries
^^^^^^^^^^^^^^^^^^^

Many functions in the pvOps library make assumptions about the column
names of the DataFrames which are passed into them. To allow users to
keep a potentially different naming scheme, pvOps uses column dictionaries
to translate between user column names and pvOps column names.
The format for dictionaries is {pvops variable: user-specific column names}. 
For example,

.. code-block::

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

