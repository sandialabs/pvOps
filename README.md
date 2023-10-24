<img src="https://github.com/sandialabs/pvOps/blob/master/docs/assets/pvops_full_logo.svg" width="400"/>

[![GitHub version](https://badge.fury.io/gh/sandialabs%2FpvOps.svg)](https://badge.fury.io/gh/sandialabs%2FpvOps)
[![License](https://img.shields.io/pypi/l/pvOps?color=green)](https://github.com/sandialabs/pvOps/blob/master/LICENSE)
[![ActionStatus](https://github.com/sandialabs/pvOps/workflows/lint%20and%20test/badge.svg)](https://github.com/sandialabs/pvOps/actions)
[![DOI](https://zenodo.org/badge/289032705.svg)](https://zenodo.org/badge/latestdoi/289032705)
[![status](https://joss.theoj.org/papers/6c3554c98b1771125613cff94241847c/status.svg)](https://joss.theoj.org/papers/6c3554c98b1771125613cff94241847c)

pvops contains a series of functions to facilitate fusion of text-based data with time series production data collected at photovoltaic sites. The package also contains example datasets and tutorials to help demonstrate how the functions can be used.

Installation
=============
pvops can be installed using `pip`. See more information at [readthedocs](https://pvops.readthedocs.io/en/latest/).

Tutorials
=========
To get started with pvops we recommended working with the [tutorials](https://pvops.readthedocs.io/en/latest/pages/tutorials.html)


Package Layout and Documentation
==============

The package is delineated into the following directories.
```
├───docs                : Documentation directory
|
├───tutorials           : Contains tutorials of functionality
│   └─── example_data   : └─── Example data
|
└───pvops               : Source function library
    ├───tests           : ├─── Library stability tests
    ├───text            : ├─── Text processing functions
    ├───text2time       : ├─── Text2Timeseries functions
    ├───timeseries      : ├─── Timeseries functions
    └───iv              : └─── Current-voltage functions
```

More information about these modules is available at [readthedocs](https://pvops.readthedocs.io/en/latest/).

Citing
======

If using this package, please cite [our paper](https://ieeexplore.ieee.org/document/9518439) using the following:

**Citation:** 

```
H. Mendoza, M. Hopwood and T. Gunda, "pvOps: Improving Operational Assessments through Data Fusion," 2021 IEEE 48th Photovoltaic Specialists Conference (PVSC), 2021, pp. 0112-0119, doi: 10.1109/PVSC43889.2021.9518439.
```

**BibTex:**

```
@INPROCEEDINGS{9518439,
  author={Mendoza, Hector and Hopwood, Michael and Gunda, Thushara},
  booktitle={2021 IEEE 48th Photovoltaic Specialists Conference (PVSC)}, 
  title={pvOps: Improving Operational Assessments through Data Fusion}, 
  year={2021},
  volume={},
  number={},
  pages={0112-0119},
  doi={10.1109/PVSC43889.2021.9518439}}
```

Contributing
============

The long-term success of pvops requires community support. Please see the [Contributing page](https://pvops.readthedocs.io/en/latest/) for more on how you can contribute.

[![Contributors Display](https://badges.pufler.dev/contributors/sandialabs/pvOps?size=50&padding=5&bots=true)](https://badges.pufler.dev)

Logo Credit: [Daniel Rubinstein](http://www.danielrubinstein.com/)

Copyright and License
=======

pvops is copyright through Sandia National Laboratories. The software is distributed under the Revised BSD License. See [copyright and license](https://github.com/sandialabs/pvops/blob/master/LICENSE) for more information.
