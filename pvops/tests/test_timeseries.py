import os
import sys
import pandas as pd
import numpy as np
from pvops.timeseries.models import linear
from pvops.timeseries import preprocess as tprep
from pvops.text2time import preprocess as t2tprep

# Define csv paths
datadir = os.path.join('tutorials', 'example_data')
example_OMpath = os.path.join(datadir, 'example_om_data2.csv')
example_prodpath = os.path.join(datadir, 'example_perf_data.csv')
example_metapath = os.path.join(datadir, 'example_metadata2.csv')
example_prod2path = os.path.join(datadir, 'example_prod_with_covariates.csv')

# Assigning dictionaries to connect pvops variables with User's column names
# Format for dictionaries is {pvops variable: user-specific column names}
prod_col_dict = {'siteid': 'randid',
                 'timestamp': 'Date',
                 'powerprod': 'AC_POWER',
                 'energyprod': 'Energy',
                 'irradiance': 'POAirradiance',
                 'baseline': 'IEC_pstep',
                 'dcsize': 'dcsize',
                 'compared': 'Compared',
                 'energy_pstep': 'Energy_pstep',
                 'clearsky_irr': 'clearsky_irr'
                 }

om_col_dict = {'siteid': 'randid',
               'datestart': 'date_start',
               'dateend': 'date_end',
               'workID': 'WONumber',
               'worktype': 'WOType',
               'asset': 'Asset',
               'eventdur': 'EventDur',
               'modatestart': 'MonthStart',
               'agedatestart': 'AgeStart'}

metad_col_dict = {'siteid': 'randid',
                  'dcsize': 'DC_Size_kW',
                  'COD': 'COD',
                  'latitude': 'latitude',
                  'longitude': 'longitude'}


def test_prod_irradiance_filter():

    prod_df = pd.read_csv(example_prodpath)
    meta_df = pd.read_csv(example_metapath)

    prod_df = t2tprep.prod_date_convert(prod_df, prod_col_dict)
    prod_df.index = prod_df[prod_col_dict['timestamp']]
    prod_df['randid'] = 'R27'

    # Data is missing in the middle of this example, so only going to pass
    # The first set of rows
    prod_df = prod_df.iloc[0:200]

    prod_df_out, mask_series = tprep.prod_irradiance_filter(prod_df,
                                                            prod_col_dict,
                                                            meta_df,
                                                            metad_col_dict)

    true_detection_irradiance = [0, 44]
    assert sum(mask_series) in true_detection_irradiance


def test_prod_inverter_clipping_filter():

    prod_df = pd.read_csv(example_prodpath)
    meta_df = pd.read_csv(example_metapath)

    prod_df = t2tprep.prod_date_convert(prod_df, prod_col_dict)
    prod_df.index = prod_df[prod_col_dict['timestamp']]
    prod_df['randid'] = 'R27'

    # Data is missing in the middle of this example, so only going to pass
    # The first set of rows
    prod_df = prod_df.iloc[0:200]

    geometric = tprep.prod_inverter_clipping_filter(prod_df,
                                                    prod_col_dict,
                                                    meta_df, metad_col_dict,
                                                    model='geometric')

    threshold = tprep.prod_inverter_clipping_filter(prod_df,
                                                    prod_col_dict,
                                                    meta_df, metad_col_dict,
                                                    model='threshold')

    levels = tprep.prod_inverter_clipping_filter(prod_df,
                                                 prod_col_dict,
                                                 meta_df, metad_col_dict,
                                                 model='levels')

    true_detection_geometric = 0
    true_detection_threshold = 0
    true_detection_levels = 183

    assert sum(geometric['mask']) == true_detection_geometric
    assert sum(threshold['mask']) == true_detection_threshold
    assert sum(levels['mask']) == true_detection_levels


def test_linear_model():
    prod_df = pd.read_csv(example_prod2path)

    # Format for dictionaries is {pvops variable: user-specific column names}
    prod_col_dict = {'siteid': 'randid',
                     'timestamp': 'date',
                     'powerprod': 'generated_kW',
                     'irradiance': 'irrad_poa_Wm2',
                     'temperature': 'temp_amb_C',
                     'baseline': 'IEC_pstep',
                     'dcsize': 'dcsize',
                     'compared': 'Compared',
                     'energy_pstep': 'Energy_pstep'}

    prod_data_converted = t2tprep.prod_date_convert(prod_df, prod_col_dict)
    prod_data_datena_d, _ = t2tprep.prod_nadate_process(
        prod_data_converted, prod_col_dict, pnadrop=True)

    prod_data_datena_d.index = prod_data_datena_d[prod_col_dict['timestamp']]

    model_prod_data = prod_data_datena_d.dropna(subset=[
        'irrad_poa_Wm2', 'temp_amb_C', 'wind_speed_ms'] +
        [prod_col_dict['powerprod']
         ])
    model_prod_data = model_prod_data[model_prod_data['randid'] == 'R15']

    model, train_df, test_df = linear.modeller(prod_col_dict,
                                               kernel_type='default',
                                               time_weighted='month',
                                               X_parameters=[
                                                   'irrad_poa_Wm2',
                                                   'temp_amb_C'],
                                               prod_df=model_prod_data,
                                               test_split=0.05,
                                               degree=3,
                                               verbose=0)

    name = list(model.estimators.keys())[0]

    benchmark_r2 = 0.99
    benchmark_mse = 420000

    eval = model.estimators[name]['test_eval']

    assert eval['r2'] > benchmark_r2
    assert eval['mse'] < benchmark_mse


def test_establish_solar_loc():
    prod_df = pd.read_csv(example_prod2path)
    meta_df = pd.read_csv(example_metapath)
    # Test-specific changes
    meta_df['randid'] = ["R10", "R15"]
    meta_df.index = meta_df['randid']
    # Format for dictionaries is {pvops variable: user-specific column names}
    prod_col_dict = {'siteid': 'randid',
                     'timestamp': 'date',
                     'powerprod': 'generated_kW',
                     'irradiance': 'irrad_poa_Wm2',
                     'temperature': 'temp_amb_C',
                     'baseline': 'IEC_pstep',
                     'dcsize': 'dcsize',
                     'compared': 'Compared',
                     'energy_pstep': 'Energy_pstep'}
    prod_data_converted = t2tprep.prod_date_convert(prod_df, prod_col_dict)
    prod_data_datena_d, _ = t2tprep.prod_nadate_process(
        prod_data_converted, prod_col_dict, pnadrop=True)

    prod_data_datena_d.index = pd.to_datetime(prod_data_datena_d[prod_col_dict['timestamp']])

    prod_with_solar_pos = tprep.establish_solar_loc(prod_data_datena_d,
                                                    prod_col_dict,
                                                    meta_df,
                                                    metad_col_dict)

    positional_columns = ['apparent_zenith',
                          'zenith',
                          'apparent_elevation',
                          'elevation',
                          'azimuth',
                          'equation_of_time']
    answer = [142.081554, 142.081554, -52.081554,
              -52.081554, 140.635657, -3.925820]
    rounded_answer = [round(a, 2) for a in answer]

    expected = prod_with_solar_pos.iloc[0][positional_columns].values
    rounded_expected = [round(a, 2) for a in expected]

    assert len(rounded_answer) == len(rounded_expected)
    assert all([a == b for a, b in zip(rounded_answer,
                                       rounded_expected)])
