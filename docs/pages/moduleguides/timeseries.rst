Timeseries Guide
==================

Module Overview
-----------------

These funcions provide processing and modelling capabilities for timeseries 
production data. Processing functions prepare data to train two 
types of expected energy models:

* AIT: additive interaction trained model, see :cite:t:`app12041872`
  for more information.
* Linear: a high flexibility linear regression model.

Additionally, the ability to generate expected energy via IEC 
standards (iec 61724-1) is implemented in the :py:mod:`~pvops.timeseries.models.iec`
module.

An example of usage can be found in 
`tutorial_timeseries_module.ipynb <https://github.com/sandialabs/pvOps/blob/master/tutorials/tutorial_timeseries_module.ipynb>`.

Preprocess
^^^^^^^^^^^^^^^^^^^^^
* :py:func:`pvops.timeseries.preprocess.prod_inverter_clipping_filter` 
  filters out production periods with inverter clipping. 
  The core method was adopted from `pvlib/pvanalytics`.
* :py:func:`pvops.timeseries.preprocess.normalize_production_by_capacity` 
  normalizes power by site capacity.
* :py:func:`pvops.timeseries.preprocess.prod_irradiance_filter` 
  filters rows of production data frame according to performance and data 
  quality. NOTE: this method is currently in development.
* :py:func:`pvops.timeseries.preprocess.establish_solar_loc`
  adds solar position data to production data using
  pvLib.

Models
^^^^^^^^^^^^^^^^^^^^^
* :py:func:`pvops.timeseries.models.linear.modeller` is a wrapper method 
  used to model timeseries data using a linear model. 
  This method gives multiple options for the 
  learned model structure.
* :py:func:`pvops.timeseries.models.AIT.AIT_calc` Calculates expected energy 
  using measured irradiance based on trained regression model from field data.
* :py:func:`pvops.timeseries.models.iec.iec_calc` calculates expected energy using measured irradiance
  based on IEC calculations.

Example Code
--------------

load in data and run some processing functions