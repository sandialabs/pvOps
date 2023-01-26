Beta 
-----------------------

New features and bug fixes are predominant in the beta versions.

New features
~~~~~~~~~~~~

* IV trace classification framework built according to literature (PR #25)
* Timeseries IV simulation for highly customizable degradation of system parameters (PR #28)
* Leverage pvlib solarposition package to populate content per site (PR #32)
* Add coefficient-level evaluations linear models (PR #32)
* Give user ability to input own test-train splits to linear modeller (PR #32)
* Remap attributes function must retain the unaltered attributes (PR #32)
* Interpolate O&M data onto production data where overlaps exist (PR #32)

Bug fixes
~~~~~~~~~

* Basic package fixes to README (PR #27) and documentation configuration (PR #24)
* Fix IV simulator bug for edge case where two IV curves added have equal I_{sc} (PR #30)
* Neural network configuration referencing in 1D CNN (PR #32)

Docs
~~~~

* Update how to reference pvOps (PR #33)

Tests
~~~~~
* Removed python 3.6 test support due to https://github.com/actions/setup-python/issues/162.
