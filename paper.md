---
title: 'pvOps: a Python package for empirical analysis of photovoltaic field data'
tags:
  - Python
  - photovoltaic
  - time series 
  - machine learning
  - natural language processing
authors:
  - name: Kirk L. Bonney
    corresponding: true # (This is how to denote the corresponding author)
    orcid: 0009-0006-2383-1634
    affiliation: 1 
  - name: Thushara Gunda
    orcid: 0000-0003-1945-4064
    affiliation: 1 
  - name: Michael W. Hopwood
    orcid: 0000-0001-6190-1767 
    affiliation: 2 
  - name: Hector Mendoza
    orcid: 0009-0007-5812-606X
    affiliation: 1 
  - name: Nicole D. Jackson
    orcid: 0000-0002-3814-9906
    affiliation: 1
affiliations:
 - name: Sandia National Laboratories, USA
   index: 1
 - name: University of Central Florida, USA
   index: 2
date: 4 April 2023
bibliography: paper.bib
---

<!--[pvOps](figures/pvops_full_logo.svg) Perhaps we could ask the journal if there's a way to include the pvOps icon in the title. I'm ok with it being excluded.-->

[GitHub repository]: https://github.com/sandialabs/pvOps
[package documentation]: https://pvops.readthedocs.io/en/latest/

# Summary

The purpose of `pvOps` is to support empirical evaluations of data collected in the field related to the operations and maintenance (O&M) of photovoltaic (PV) power plants. `pvOps` presently contains modules that address the diversity of field data, including text-based maintenance logs, current-voltage (IV) curves, and timeseries of production information. The package functions leverage machine learning, visualization, and other techniques to enable cleaning, processing, and fusion of these datasets. These capabilities are intended to facilitate easier evaluation of field patterns and extraction of relevant insights to support reliability-related decision-making for PV sites. The open-source code, examples, and instructions for installing the package through PyPI can be accessed through the [GitHub repository]. 

# Statement of Need

Continued interest in PV deployment across the world has resulted in increased awareness of needs associated with managing reliability and performance of these systems during operation. Current open-source packages for PV analysis focus on theoretical evaluations of solar power simulations (e.g., `pvlib` [@holmgren2018pvlib]), data cleaning and feature development for production data (e.g. `pvanalytics` [@perry2022pvanalytics]), specific use cases of empirical evaluations (e.g., `RdTools` [@deceglie2018rdtools] and `Pecos` [@klise2016performance] for degradation analysis), or analysis of electroluminescene images (e.g., `PVimage` [@pierce2020identifying]); see [openpvtools](https://openpvtools.readthedocs.io/en/latest/) for a list of additional open source PV packages. However, a general package that can support data-driven, exploratory evaluations of diverse field collected information is currently lacking. For example, a maintenance log that describes an inverter failure may be temporally correlated to a dip in production levels. Identifying such relationships across different types of field data can improve understanding of the impacts of certain types of failures on a PV plant. To address this gap, we present `pvOps`, an open-source Python package that can be used by  researchers and industry analysts alike to evaluate and extract insights from different types of data routinely collected during PV field operations. 

PV data collected in the field varies greatly in structure (e.g., timeseries and text records) and quality (e.g., completeness and consistency). The data available for analysis is frequently semi-structured. Furthermore, the level of detail collected between different owners/operators might vary. For example, some may capture a general start and end time for an associated event whereas others might include additional time details for different resolution activities. This diversity in data types and structures often leads to data being under-utilized due to the amount of manual processing required. To address these issues, `pvOps` provides a suite of data processing, cleaning, and visualization methods to leverage insights across a broad range of data types, including operations and maintenance records,  production timeseries, and IV curves. The functions within `pvOps` enable users to better parse available data to understand patterns in outages and production losses. 

# Package Overview 
The following table summarizes the four modules within `pvOps` by presenting: the type of data they analyze, example data features, and highlights of relevant functions. 

\textbf{Table 1. Summary of modules and functions within `pvOps`}

Module | Type of data | Example data features | Highlights of functions
------- | ------ | --------- | -----------
text | O&M records | *timestamps*, *issue description*, *issue classification* | fill data gaps in dates and categorical records, visualize word clusters and patterns over time
 | | | 
timeseries | Production data | *site*, *timestamp*, *power production*, *irradiance* | estimate expected energy with multiple models, evaluate inverter clipping
 | | | 
text2time | O&M records and production data | see entries for `text` and  `timeseries` modules above | analyze overlaps between O&M and production (timeseries) records, visualize overlaps between O&M records and production data
 | | | 
iv | IV records | *current*, *voltage*, *irradiance*, *temperature*  | simulate IV curves with physical faults, extract diode parameters from IV curves, classify faults using IV curves

The functions within each module can be used to build pipelines that integrate relevant data processing, fusion, and visualization capabilities to support user endgoals. For example, a user with IV curve data could build a pipeline that leverages functions within the `iv` module to process and extract diode parameters within IV curves as well as train models to support classifications based on fault type. A pipeline could be also be built that leverages functions across modules if a user has access to multiple types of data (e.g., both O&M and production records). A sample end-to-end workflow using `pvOps` modules could be:

1. Use functions within the `text` module to systematically review data quality issues within O&M records, train a machine learning model on available records, and use the model to estimate possible labels for missing entries
2. Leverage the functions within the `timeseries` module, use machine learning to develop their own expected energy models for a given time series of irradiance and system size details, or use a pre-trained expected energy model [@hopwood2022generation] or leverage industry standard equations as a basis for evaluating possible production losses
3. Couple outputs from the above two analyses (using functions in the `text2time` module) based on timestamps to develop summaries and visualizations of production impacts observed during these periods

The [package documentation] for `pvOps` provides thorough examples exploring the various capabilities of each module. Additional details about the `iv` module capabilities are captured in [@hopwood2020neural; @hopwood2022physics] while more information about the design and development of the `text`, `timeseries`, and `text2time` modules are captured in [@mendoza2021pvops]. Key package dependencies of `pvOps` include `pandas` [@reback2020pandas], `sklearn` [@pedregosa2011sklearn], `nltk` [@bird2009nltk], and `keras` [@chollet2015keras] for analysis and `matplotlib` [@hunter2007matplotlib], `seaborn` [@waskom2021seaborn], and `plotly` [@plotly2015] for visualization.

# Ongoing Development
The `pvOps` functionality and documentation continues to be improved and updated as new empirical techniques are identified. For example, research efforts have demonstrated utility of natural language processing techniques (e.g., topic modeling) and survival analyses to support evaluation of patterns in O&M records  [@gunda2020machine]. Additional statistical methods, such as Hidden Markov Modeling, have also been successfully used to support classification of failures within production data [@hopwood2022classification]. These and other capabilities will continue to be added to the package to improve its utility for supporting empirical analyses of field data. 

# CRediT Authorship Statement

<!-- see: https://www.elsevier.com/authors/policies-and-guidelines/credit-author-statement -->

KLB: Writing - Original Draft, Software - Software Development, Software - Testing; TG: Conceptualization, Writing - Original Draft, Software - Design; MWH: Writing - Review & Editing, Software - Software Development; HM: Writing - Review & Editing, Software - Software Development; NDJ: Conceptualization, Funding Acquisition, Project Administration, Supervision, Writing - Review & Editing. 

# Acknowledgements
This material is supported by the U.S. Department of Energy, Office of Energy Efficiency and Renewable Energy - Solar Energy Technologies Office. Sandia National Laboratories, a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia LLC, a wholly owned subsidiary of Honeywell International Inc. for the U.S. Department of Energy’s National Nuclear Security Administration under contract DE-NA0003525.

# References

<!-- These will be formally checked and built during the review process -->
