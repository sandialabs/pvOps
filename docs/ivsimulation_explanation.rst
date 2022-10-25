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