IV package
===============

These functions focus on current-voltage (IV) curve simulation and 
classification.

extractor
----------

* `~pvops.iv.extractor` primarily features the 
  :py:class:`pvops.iv.extractor.BruteForceExtractor` class, which 
  extracts diode parameters from IV curves (even outdoor-collected).

physics_utils
-------------

:py:mod:`~pvops.iv.physics_utils` contains methods which aid the IV 
Simulator's physics-based calculations and the preprocessing pipeline's 
correction calculations.

* :py:func:`~pvops.iv.physics_utils.calculate_IVparams` calculates
  key parameters of an IV curve.
* :py:func:`~pvops.iv.physics_utils.smooth_curve` smooths
  IV curve using a polyfit.
* :py:func:`~pvops.iv.physics_utils.iv_cutoff` cuts off IV curve
  greater than a given voltage value.
* :py:func:`~pvops.iv.physics_utils.intersection` computes
  the intersection between two curves.
* :py:func:`~pvops.iv.physics_utils.T_to_tcell` calculates
  a cell temperature given ambient temperature via NREL weather-correction
  tools.
* :py:func:`~pvops.iv.physics_utils.bypass` limits voltage
  to above a minimum value.
* :py:func:`~pvops.iv.physics_utils.add_series` adds two
  IV curves in series.
* :py:func:`~pvops.iv.physics_utils.voltage_pts`
  provides voltage points for an IV curve.
* :py:func:`~pvops.iv.physics_utils.gt_correction` corrects IV
  trace using irradiance and temperature using one of three
  available options.



preprocess
----------

:py:mod:`~pvops.iv.preprocess` contains the preprocessing function which 
corrects a set of data according to irradiance and temperature and 
normalizes the curves so they are comparable.

simulator
---------

:py:mod:`~pvops.iv.simulator` holds the `IV Simulator` which can simulate 
current-voltage (IV) curves under different environmental and fault 
conditions.

utils
-------

:py:mod:`~pvops.iv.utils` holds a utility function which connects to the CEC 
database hosted by pvLib for cell-level and module-level parameters.

timeseries_simulator
----------------------

:py:mod:`~pvops.iv.timeseries_simulator.py` holds a timeseries wrapper of the 
IV Simulator which allows users to specify time-based failure degradation 
patterns.
