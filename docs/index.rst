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

Statement of Need
=================

Continued interest in PV deployment across the world has resulted in increased awareness of needs associated 
with managing reliability and performance of these systems during operation. Current open-source packages for 
PV analysis focus on theoretical evaluations of solar power simulations (e.g., `pvlib`; :cite:p:`holmgren2018pvlib`), 
specific use cases of empirical evaluations (e.g., `RdTools`; :cite:p:`deceglie2018rdtools` and `Pecos`; :cite:p:`klise2016performance`
for degradation analysis), or analysis of electroluminescene images (e.g., `PVimage`; :cite:p:`pierce2020identifying`). However, 
a general package that can support data-driven, exploratory evaluations of diverse field collected information is currently lacking. 
To address this gap, we present `pvOps`, an open-source, Python package that can be used by  researchers and industry 
analysts alike to evaluate different types of data routinely collected during PV field operations. 

PV data collected in the field varies greatly in structure (i.e., timeseries and text records) and quality 
(i.e., completeness and consistency). The data available for analysis is frequently semi-structured. 
Furthermore, the level of detail collected between different owners/operators might vary. 
For example, some may capture a general start and end time for an associated event whereas others might include 
additional time details for different resolution activities. This diversity in data types and structures often 
leads to data being under-utilized due to the amount of manual processing required. To address these issues, 
`pvOps` provides a suite of data processing, cleaning, and visualization methods to leverage insights across a 
broad range of data types, including operations and maintenance records,  production timeseries, and IV curves. 
The functions within `pvOps` enable users to better parse available data to understand patterns in outages and production losses. 

    
.. toctree::
    :maxdepth: 1
    :caption: Available resources:
    
    Overview <self>
    pages/userguide
    pages/tutorials
    pages/modules
    pages/development
    pages/contributing
    pages/releasenotes
    pages/references


