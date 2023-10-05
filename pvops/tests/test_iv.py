import random
import os
import pandas as pd
import numpy as np
import pvops.iv.timeseries_simulator
import pvops.iv.preprocess
import pvops.iv.simulator
from pvops.iv.models import nn

datadir = os.path.join('tutorials', 'example_data')
example_prodpath = os.path.join(
    datadir, 'example_prod_with_covariates.csv')


def test_simulation():
    random.seed(0)

    sim = pvops.iv.simulator.Simulator()

    # test adding presets
    heavy_shading = {'identifier': 'heavy_shade',
                     'E': 400,
                     'Tc': 20}
    light_shading = {'identifier': 'light_shade',
                     'E': 800}
    sim.add_preset_conditions('landscape', heavy_shading, rows_aff=2)
    sim.add_preset_conditions('portrait', heavy_shading, cols_aff=2)
    sim.add_preset_conditions('pole', heavy_shading,
                              light_shading=light_shading,
                              width=2, pos=None)

    # test adding manuals
    # Using 2D list (aka, multiple conditions as input)
    modcells = {'another_example': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
                                     1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
                                     1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
                                     0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
                                     0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
                                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
                }
    condition_dict = {0: {},
                      1: {'identifier': 'heavy_shade',
                          'E': 405,
                          }
                      }
    sim.add_manual_conditions(modcells, condition_dict)

    # test generate many samples
    N = 2
    dicts = {'E': {'mean': 400,
                   'std': 500,
                   'low': 200,
                   'upp': 600
                   },
             'Tc': {'mean': 30,
                    'std': 10,
                    }
             }
    sim.generate_many_samples('heavy_shade', N, dicts)
    dicts = {'E': {'mean': 800,
                   'std': 500,
                   'low': 600,
                   'upp': 1000
                   }
             }
    sim.generate_many_samples('light_shade', N, distributions=dicts)

    # test building strings
    sim.build_strings({'pole_bottom_mods': ['pristine', 'pristine', 'pristine',
                                            'pristine', 'pristine', 'pristine',
                                            'pole_2width', 'pole_2width',
                                            'pole_2width', 'pole_2width',
                                            'pole_2width', 'pole_2width'],
                       'portrait_2cols_3bottom_mods': ['pristine', 'pristine',
                                                       'pristine', 'pristine',
                                                       'pristine', 'pristine',
                                                       'pristine', 'pristine',
                                                       'pristine',
                                                       'portrait_2cols',
                                                       'portrait_2cols',
                                                       'portrait_2cols']})

    # test simulating
    sim.simulate()

    df = sim.sims_to_df(focus=['string', 'module'], cutoff=True)

    n_str_samples = 16
    n_mod_samples = 29

    assert len(df[df['level'] == 'string']) == n_str_samples
    assert len(df[df['level'] == 'module']) == n_mod_samples


def test_classification():

    sim = pvops.iv.simulator.Simulator()

    condition = {'identifier': 'shade', 'Il_mult': 0.6}
    sim.add_preset_conditions('complete', condition,
                              save_name='Complete_shading')
    dicts = {'Il_mult': {'mean': 0.6,
                         'std': 0.7,
                         'low': 0.33,
                         'upp': 0.95,
                         }
             }
    sim.generate_many_samples('shade', 100, dicts)

    sim.build_strings({'Pristine array': ['pristine'] * 12,
                       'Partial Soiling (1M)': ['pristine'] * 11 +
                                               ['Complete_shading'] * 1,
                       'Partial Soiling (6M)': ['pristine'] * 6 +
                                               ['Complete_shading'] * 6
                       }
                      )

    sim.simulate()
    df = sim.sims_to_df(focus=['string'], cutoff=True)

    iv_col_dict = {
        "mode": "mode",
        "current": "current",            # Populated in simulator
        "voltage": "voltage",            # Populated in simulator
        "irradiance": "E",               # Populated in simulator
        "temperature": "T",              # Populated in simulator
        "power": "power",                # Populated in preprocess
        "derivative": "derivative",      # Populated in feature_generation
        "current_diff": "current_diff",  # Populated in feature_generation
    }

    # Irradiance & Temperature correction, and normalize axes
    prep_df = pvops.iv.preprocess.preprocess(df, 0.05, iv_col_dict,
                                             resmpl_cutoff=0.03, correct_gt=True,
                                             normalize_y=False,
                                             CECmodule_parameters=sim.module_parameters,
                                             n_mods=12, gt_correct_option=3)
    # Shuffle
    bigdf = prep_df.sample(frac=1).reset_index(drop=True)
    bigdf.dropna(inplace=True)

    feat_df = nn.feature_generation(bigdf, iv_col_dict)

    nn_config = {
        # NN parameters
        "model_choice": "1DCNN",
        "params": ['current', 'power', 'derivative', 'current_diff'],
        "dropout_pct": 0.5,
        "verbose": 1,
        # Training parameters
        "train_size": 0.8,
        "shuffle_split": True,
        "balance_tactic": 'truncate',
        "n_CV_splits": 2,
        "batch_size": 10,
        "max_epochs": 100,
        # LSTM parameters
        "use_attention_lstm": False,
        "units": 50,
        # 1DCNN parameters
        "nfilters": 64,
        "kernel_size": 12,
    }

    iv_col_dict = {'mode': 'mode'}
    model, _, _ = nn.classify_curves(feat_df, iv_col_dict, nn_config)

    if model.test_accuracy > 0.9:
        assert True
    else:
        assert False


def test_timeseries_simulator():

    env_df = pd.read_csv(example_prodpath)
    env_df.index = pd.to_datetime(env_df["date"])
    env_df = env_df.sort_index()

    # Only simulate where irradiance > 200
    env_df = env_df[env_df['irrad_poa_Wm2'] > 600]
    # Two sites have data here so we choose one
    env_df = env_df[env_df['randid'] == 'R10']
    # Remove any NaN environmental specifications
    env_df = env_df.dropna(subset=['irrad_poa_Wm2', 'temp_amb_C'])

    # Reduce number of simulations for test
    env_df = env_df.iloc[0:100]

    failureA = pvops.iv.timeseries_simulator.TimeseriesFailure()
    longterm_fcn_dict = {
        'Rs_mult': "degrade"
    }
    annual_fcn_dict = {
        'Rs_mult': lambda x: (0.3 * np.sin(np.pi * x))
    }

    failureA.trend(longterm_fcn_dict=longterm_fcn_dict,
                   annual_fcn_dict=annual_fcn_dict,
                   degradation_rate=1.005)

    iv_col_dict = {'irradiance': 'irrad_poa_Wm2',
                   'temperature': 'temp_amb_C'
                   }

    env_df['identifier'] = env_df.index.strftime("%Y-%m-%d %H:%M:%S")

    time_simulator = pvops.iv.timeseries_simulator.IVTimeseriesGenerator()
    time_simulator.generate(
        env_df, [failureA], iv_col_dict, 'identifier', plot_trends=False)

    time_simulator.add_time_conditions('complete', nmods=12)
    time_simulator.simulate()

    sims_df = time_simulator.sims_to_df(focus=['string'], cutoff=True)

    assert len(sims_df) == 100
