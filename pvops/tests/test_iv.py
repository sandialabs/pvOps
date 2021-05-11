import os
import sys

iv_directory = os.path.join("pvops","iv")
sys.path.append(iv_directory)

import random
import simulator, extractor

def test_simulation():
    random.seed(0)
    
    sim = simulator.Simulator()

    # test adding presets
    heavy_shading = {'identifier':'heavy_shade',
                    'E': 400,
                    'Tc': 20}
    light_shading = {'identifier':'light_shade',
                        'E': 800}
    sim.add_preset_conditions('landscape', heavy_shading, rows_aff = 2)
    sim.add_preset_conditions('portrait', heavy_shading, cols_aff = 2)
    sim.add_preset_conditions('pole', heavy_shading, light_shading = light_shading, width = 2, pos = None)

    # test adding manuals
    modcells = { 'another_example':  [[0,0,0,0,0,0,0,0,0,0,  # Using 2D list (aka, multiple conditions as input)
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
                                        1,1,1,1,1,1,1,1,1,1]]
                }
    condition_dict = {0: {},
                    1: {'identifier': 'heavy_shade',
                        'E': 405,
                        }                              
                    }
    sim.add_manual_conditions(modcells, condition_dict)

    # test generate many samples
    N = 2
    dicts = {'E':       {'mean': 400,
                            'std': 500,
                            'low': 200,
                            'upp': 600
                        },
            'Tc':      {'mean': 30,
                        'std': 10,
                        }
            }
    sim.generate_many_samples('heavy_shade', N, dicts)
    dicts = {'E':       {'mean': 800,
                            'std': 500,
                            'low': 600,
                            'upp': 1000
                        }
            }
    sim.generate_many_samples('light_shade', N, distributions = dicts)

    # test building strings
    sim.build_strings({'pole_bottom_mods': ['pristine', 'pristine', 'pristine', 'pristine', 'pristine', 'pristine',
                                            'pole_2width', 'pole_2width', 'pole_2width', 'pole_2width', 'pole_2width', 'pole_2width'],
                   'portrait_2cols_3bottom_mods': ['pristine', 'pristine', 'pristine', 'pristine', 'pristine', 'pristine',
                                            'pristine', 'pristine', 'pristine', 'portrait_2cols', 'portrait_2cols', 'portrait_2cols']})

    # test simulating
    sim.simulate()
    
    df = sim.sims_to_df(focus=['string', 'module'], cutoff=True)

    n_str_samples = 16
    n_mod_samples = 29
    
    assert len(df[df['level']=='string']) == n_str_samples
    assert len(df[df['level']=='module']) == n_mod_samples

    # test visualizing
    # sim.visualize()
    # truth_identity_max_len = 'heavy_shade'
    # truth_max_len = 28
    # assert sim.maxIdent == truth_identity_max_len
    # assert sim.maxL == truth_max_len