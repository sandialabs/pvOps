==============================================
IV Failure Classification with Neural Networks
==============================================
TODO: Overview of the HOWTO

Initialize environment
----------------------

.. doctest::

    >>> from pvops.iv import simulator, extractor, preprocess

    >>> from pvops.iv.models import nn

    >>> iv_col_dict = {
    ...     "mode": "mode",
    ...     "current": "current",            # Populated in simulator
    ...     "voltage": "voltage",            # Populated in simulator
    ...     "irradiance": "E",               # Populated in simulator
    ...     "temperature": "T",              # Populated in simulator
    ...     "power": "power",                # Populated in preprocess
    ...     "derivative": "derivative",      # Populated in feature_generation
    ...     "current_diff": "current_diff",  # Populated in feature_generation
            }
    
    >>> sim = simulator.Simulator()

Collect the IV curves
---------------------
Define the pristine condition
.. doctest::

    >>> sim.pristine_condition = {
    ...     'identifier': 'pristine',
    ...     'E': E,
    ...     'Tc': Tc,
    ...     'Rsh_mult': 1,
    ...     'Rs_mult': 1,
    ...     'Io_mult': 1,
    ...     'Il_mult': 1,
    ...     'nnsvth_mult': 1,
    ...     }

    >>> condition = {'identifier':namer('weathered_pristine')}
    
    >>> sim.add_preset_conditions('complete', condition, save_name = namer('Complete_weathered_pristine'))
    
    >>> condition = {'identifier':namer('shade'),'Il_mult':0.6}
    
    >>> sim.add_preset_conditions('complete', condition, save_name = namer('Complete_shading'))
    
    >>> condition = {'identifier':namer('cracking'),'Rs_mult':1.5}
    
    >>> sim.add_preset_conditions('complete', condition, save_name = namer('Complete_cracking'))

    >>> dicts = {'Il_mult':{
    ...     'mean': 0.6,
    ...     'std': 0.7,
    ...     'low': 0.33,
    ...     'upp': 0.95,
    ...     }
    ... }

    
    >>> sim.generate_many_samples(namer('shade'), N_samples, dicts)
    
    >>> dicts = {
    ...     'Rs_mult':{'mean':1.3,
    ...                 'std':0.6,
    ...                 'low':1.1,
    ...                 'upp':1.8
    ...                 },
    ...     'Rsh_mult':{'mean':0.5,
    ...                 'std':0.6,
    ...                 'low':0.3,
    ...                 'upp':0.7
    ...                 }
    ...     }

    >>> sim.generate_many_samples(namer('cracking'), N_samples, dicts)

    >>> sim.build_strings({
    ...     namer('Partial Soiling (1M)'): [namer('Complete_weathered_pristine')]*11 + [namer('Complete_shading')]*1,
    ...     namer('Partial Soiling (6M)'): [namer('Complete_weathered_pristine')]*6 + [namer('Complete_shading')]*6,
    ...     namer('Cell cracking (4M)'): [namer('Complete_weathered_pristine')]*8 + [namer('Complete_cracking')]*4,
    ...     })