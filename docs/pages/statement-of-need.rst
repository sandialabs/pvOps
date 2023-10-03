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
