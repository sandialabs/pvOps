============
IV Simulator
============
TODO: overview of simulator here

Preset definitions of faults
----------------------------

- Complete

- landscape

- Portrait

- Pole

- Bird droppings


Manual definition of faults
---------------------------

To define a fault manually, you must provide two specifications:
1. Mapping of cells onto a module, which we deem a _modcell_
2. Definition of cell conditions, stored in _condition_dict_

Example using 2D list where multiple conditions are input.

.. doctest::

    >>> modcells = {
        'another_example': [
            [0,0,0,0,0,0,0,0,0,0,
             1,1,1,1,1,1,1,1,1,1,
             1,1,1,0,0,0,0,1,1,1, 
             1,1,1,0,0,0,0,1,1,1,
             1,1,1,0,0,0,0,1,1,1,  
             0,0,0,0,0,0,0,0,0,0],

            [1,1,1,1,1,1,1,1,1,1,  
             0,0,0,0,0,0,0,0,0,0,
             0,0,0,1,1,1,1,0,0,0, 
             0,0,0,1,1,1,1,0,0,0,
             0,0,0,1,1,1,1,0,0,0,  
             1,1,1,1,1,1,1,1,1,1]
             ]
            }
    >>> condition_dict = {0: {},
                  1: {'identifier': 'heavy_shade',
                      'E': 405,
                     }                              
                 }

    >>> sim.add_manual_conditions(modcells, condition_dict)

    >>> sim.print_info()


Generating samples with latin hypercube sampling
------------------------------------------------




    Process
    -------
    A `pristine` condition is created automatically

    - Specify failure conditions either by 

        1) add a preset configuration

            - ``add_preset_conditions('complete', fault_condition)``
            - ``add_preset_conditions('landscape', fault_condition, rows_aff = 2)``
            - ``add_preset_conditions('portrait', fault_condition, cols_aff = 2)``
            - ``add_preset_conditions('pole', fault_condition, width = 2, pos = None)``
            - ``add_preset_conditions('bird_droppings', fault_condition, n_droppings = None)``

        2) add a manual configuration
            
            - add_manual_conditions(modcell, condition_dict)

        3) both

    - (Optional) Generate many definitions of a cell condition
      ``generate_many_samples(identifier, N, distributions = None, default_sample = None)``

    - (Optional) Define a string as a list of modules
      ``build_strings(config_dict)``

    - Simulate all levels of the designed PV system
      ``simulate(sample_limit)``

    - (Optional) Display information about the system
      ``print_info()``
      ``visualize(lim = False)``

    - Access simulations for your intended use

        1) Export simulations as dataframe, which has columns:
           ``df = sims_to_df(cutoff=False)``

        2) Access simulations manually
           Inspect ``Simulator().multilevel_ivdata``
           See `Attributes` above for information on multilevel_ivdata.