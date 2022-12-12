Getting Started with IV Curve Simulator
=======================================
An object which simulates Photovoltaic (PV) current-voltage (IV) 
curves with failures.

Initializing a Simulator object
--------------------------------
First define initialization parameters

.. doctest::

    >>> mod_specs = {
    ...     'Jinko_Solar_Co___Ltd_JKM270PP_60': 
    ...     {'ncols': 6, 'nsubstrings': 3}
    ...     }
    
    >>> pristine_condition = {
    ...     'identifier': 'pristine',
    ...     'E': 1000,
    ...     'Tc': 50,
    ...     'Rsh_mult': 1,
    ...     'Rs_mult': 1,
    ...     'Io_mult': 1,
    ...     'Il_mult': 1,
    ...     'nnsvth_mult': 1,
    ...     }
    
    >>> replacement_5params = {
    ...     'I_L_ref': 9.06157444e+00,
    ...     'I_o_ref': 1.67727320e-10,
    ...     'R_s': 5.35574950e-03,
    ...     'R_sh_ref': 3.03330425e+00,
    ...     'a_ref': 2.54553421e-02
    ...     }

.. note:: 
    replacement_5params is optional, and can be determined 
    by IVProcessor().

Using the defined parameters, define a Simulator object

.. doctest::

    >>> import pvops

    >>> sim = pvops.iv.Simulator(
    ... mod_specs,
    ... pristine_condition,
    ... replacement_5params
        )


Define a condition and add it to the Simulator

.. doctest::

    >>> condition = {'identifier':'light_shade','E':925}
    
    >>> sim.add_preset_conditions('complete', condition, save_name = f'Complete_lightshading')
    
TODO: Add explanation here, not sure what these methods do

.. doctest::

    >>> sim.build_strings({'Partial_lightshading': ['pristine']*6 + ['Complete_lightshading']*6})
    
    >>> sim.simulate()
    
    >>> sim.print_info()
    
Visualize results
.. doctest::

    >>> Vsim = sim.multilevel_ivdata['string']['Partial_lightshading']['V'][0]
    
    >>> Isim = sim.multilevel_ivdata['string']['Partial_lightshading']['I'][0]



simulation_method : int
    Module simulation method (1 or 2)

    1) Avalanche breakdown model, as hypothesized in Ref. [1]_

    2) : Add-on to method 1, includes a rebalancing of the $I_sc$ prior to adding in series

    .. [1] "Computer simulation of the effects of electrical mismatches in photovoltaic cell 
        interconnection circuits" JW Bishop, Solar Cell (1988) DOI: 10.1016/0379-6787(88)90059-2

Attributes
----------
multilevel_ivdata : dict
    Dictionary containing the simulated IV curves

    - For nth-definition of string curves, 
        multilevel_ivdata['string']['STRING IDENTIFIER'][n]
    - For nth-definition of module curves,
        multilevel_ivdata['module']['MODULE IDENTIFIER'][n]
    - For nth-definition of substring (substr_id = 1,2,3,...) curves,
        multilevel_ivdata['module']['MODULE IDENTIFIER']['substr{sbstr_id}'][n]

pristine_condition : dict
    Dictionary of conditions defining the pristine case
module_parameters : dict
    Dictionary of module-level parameters
cell_parameters : dict
    Dictionary of cell-level parameters

Methods
-------

add_preset_conditions(fault_name, fault_condition, save_name = None, kwargs)
    Define a failure condition using a preset condition. See :py:meth:`add_preset_conditions`

add_manual_conditions(modcell, condition_dict)
    Define a failure by passing in modcell and cell condition definitons manually. See :py:meth:`add_manual_conditions`

generate_many_samples(identifier, N, distributions = None, default_sample = None)
    Generate `N` more definitions of the same failure cell condition by defining parameter `distributions`

build_strings(config_dict)
    Define a string as a list of modules which were defined in :py:meth:`add_preset_conditions` or :py:meth:`add_manual_conditions`

simulate(sample_limit)
    Simulate cell, substring, module, and string-level definitions

print_info()
    Display the number of definitions on the cell, module, and string levels

visualize(lim = False)
    Visualize the definitions and render parameter distributions

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


            Example:
        --------

        .. code-block:: python

            modcells  =  {'unique_shading':   [0,0,0,0,0,0,0,0,0,0,  # Using 1D list 
                                            1,1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,1, 
                                            1,1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,1,  
                                            0,0,0,0,0,0,0,0,0,0],
                        'another_example':  [[0,0,0,0,0,0,0,0,0,0,  # Using 2D list (aka, multiple conditions as input)
                                            1,1,1,1,1,1,1,1,1,1,
                                            1,1,1,0,0,0,0,1,1,1, 
                                            1,1,1,0,0,0,0,1,1,1,
                                            1,1,1,0,0,0,0,1,1,1,  
                                            0,0,0,0,0,0,0,0,0,0],
                                            [0,1,0,0,0,0,0,0,0,0,  
                                            1,1,1,1,1,1,1,1,1,1,
                                            1,1,1,1,1,1,1,1,1,1, 
                                            0,0,0,1,1,1,0,0,0,0,
                                            0,0,0,1,1,1,0,0,0,0,  
                                            0,0,0,0,0,0,0,0,0,0]]
                        }
            # All numbers used in modcells must be defined here
            # If defining a pristine condition, pass a blank dictionary
            # If making edits to a pristine condition (e.g. dropping irradiance to 400) \\
            # you only need to a) specify the change made, and b) name an identifier string (for future reference)
            # The pristine condition can be changed when first creating the class object
            # To define a pristine, you can either pass an empty dictionary or pass {'identifier':'pristine'}
            condition_dict = {0: {},
                            1: {'identifier': 'shading_cond1',
                                'E': 400,
                                }                              
                            }
            add_manual_conditions(modcell, condition_dict)