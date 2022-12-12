============
IV Simulator
============
TODO: overview of simulator here


Initializing a simulator object

.. doctest::
    
    >>> from pvops.iv.simulator import Simulator
    >>> sim = Simulator()

Preset definitions of faults
----------------------------

The IV Simulator comes with five built-in conditions:

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



Example
-------
.. code-block:: python

        sim = Simulator(
                    mod_specs = {
                                    'Jinko_Solar_Co___Ltd_JKM270PP_60': {'ncols': 6,
                                                                        'nsubstrings': 3
                                                                        }
                                },
                    pristine_condition = {
                                            'identifier': 'pristine',
                                            'E': 1000,
                                            'Tc': 50,
                                            'Rsh_mult': 1,
                                            'Rs_mult': 1,
                                            'Io_mult': 1,
                                            'Il_mult': 1,
                                            'nnsvth_mult': 1,
                                            },
                    # Optional, Determined by IVProcessor()
                    replacement_5params = {'I_L_ref': 9.06157444e+00,
                                            'I_o_ref': 1.67727320e-10, # 0.3e-10,
                                            'R_s': 5.35574950e-03,
                                            'R_sh_ref': 3.03330425e+00,
                                            'a_ref': 2.54553421e-02}
        )
        
        condition = {'identifier':'light_shade','E':925}
        sim.add_preset_conditions('complete', condition, save_name = f'Complete_lightshading')
        
        sim.build_strings({'Partial_lightshading': ['pristine']*6 + ['Complete_lightshading']*6})
        
        sim.simulate()
        
        sim.print_info()
        
        # Look at a result!
        Vsim = sim.multilevel_ivdata['string']['Partial_lightshading']['V'][0]
        Isim = sim.multilevel_ivdata['string']['Partial_lightshading']['I'][0]