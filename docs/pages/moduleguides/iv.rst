IV Guide
===============

Module Overview
----------------

These functions focus on current-voltage (IV) curve simulation and 
classification.

.. note::
  To use the capabilites in this module, pvOps must be installed with the ``iv`` option:
  ``pip install pvops[iv]``.


Tutorials that exemplify usage can be found at:
  - `tutorial_iv_classifier.ipynb <https://github.com/sandialabs/pvOps/blob/master/tutorials/tutorial_iv_classifier.ipynb>`_.
  - `tutorial_iv_diode_extractor.ipynb <https://github.com/sandialabs/pvOps/blob/master/tutorials/tutorial_iv_diode_extractor.ipynb>`_.
  - `tutorial_iv_simulator.ipynb <https://github.com/sandialabs/pvOps/blob/master/tutorials/tutorial_iv_simulator.ipynb>`_.

extractor
^^^^^^^^^^^^^^^^^^^^^

* :py:mod:`~pvops.iv.extractor` primarily features the 
  :py:class:`~pvops.iv.extractor.BruteForceExtractor` class, which 
  extracts diode parameters from IV curves (even outdoor-collected).

physics_utils
^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^

:py:mod:`~pvops.iv.preprocess` contains the preprocessing function 
* :py:func:`~pvops.iv.preprocess.preprocess` which 
corrects a set of data according to irradiance and temperature and 
normalizes the curves so they are comparable.

simulator
^^^^^^^^^^^^^^^^^^^^^

:py:mod:`~pvops.iv.simulator` holds the 
:py:class:`~pvops.iv.simulator.IV Simulator` class which can simulate 
current-voltage (IV) curves under different environmental and fault 
conditions. There is also a utility function 
:py:func:`~pvops.iv.simulator.create_df` for building an IV curve dataframe
from a set of parameters.

utils
^^^^^^^^^^^^^^^^^^^^^

:py:mod:`~pvops.iv.utils` holds the utility function 
:py:func:`~pvops.iv.utils.get_CEC_params` which connects to the 
California Energy Commission (CEC) 
database hosted by pvLib for cell-level and module-level parameters.

timeseries_simulator
^^^^^^^^^^^^^^^^^^^^^

:py:mod:`~pvops.iv.timeseries_simulator` contains 
:py:class:`~pvops.iv.timeseries_simulator.IVTimeseriesGenerator`, 
a subclass of the IV Simulator,
which allows users to specify time-based failure degradation 
patterns. The class 
:py:class:`~pvops.iv.timeseries_simulator.TimeseriesFailure`
is used to define the time-based failures.

.. Example Code
.. --------------
