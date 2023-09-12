.. _index:

.. image:: assets/pvops_full_logo.svg
  :width: 400

Overview
============
pvops is a python package for PV operators & researchers. 
It consists of a set of documented functions for supporting operations 
research of photovoltaic (PV) energy systems.
The library leverages advances in machine learning, natural
language processing and visualization 
tools to extract and visualize actionable information from common 
PV data including Operations & Maintenance (O&M) text data, timeseries 
production data, and current-voltage (IV) curves.

.. list-table:: Module Overview
   :widths: 25 25 50
   :header-rows: 1

   * - Module
     - Type of data
     - Highlights of functions
   * - text
     - O&M records
     - - fill data gaps in dates and categorical records
       - visualize word clusters and patterns over time
   * - timeseries
     - Production data
     - - estimate expected energy with multiple models
       - evaluate inverter clipping
   * - text2time 
     - O&M records and production data 
     - - analyze overlaps between O&M and production (timeseries) records
       - visualize overlaps between O&M records and production data
   * - iv
     - IV records 
     - - simulate IV curves with physical faults
       - extract diode parameters from IV curves 
       - classify faults using IV curves
    
.. toctree::
    :maxdepth: 1
    :caption: Available resources:
    
    Overview <self>
    pages/installation
    pages/userguide
    pages/examples
    pages/modules
    pages/references
    pages/abbreviations
    pages/development
    pages/contributing
    pages/releasenotes

