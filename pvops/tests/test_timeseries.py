import os
import sys
import pandas as pd

timeseries_directory = os.path.join("pvops")
sys.path.append(timeseries_directory)

from timeseries import preprocess as tprep
from text2time import preprocess as t2tprep

# Define csv paths
datadir = os.path.join('examples', 'example_data')
example_OMpath = os.path.join(datadir, 'example_om_data2.csv')
example_prodpath = os.path.join(datadir, 'example_perf_data.csv')
example_metapath = os.path.join(datadir, 'example_metadata2.csv')

# Assigning dictionaries to connect pvops variables with User's column names
# Format for dictionaries is {pvops variable: user-specific column names}
prod_col_dict = {'siteid': 'randid',
                 'timestamp': 'Date',
                 'power': 'AC_POWER',
                 'energyprod': 'Energy',
                 'irradiance': 'POAirradiance',
                 # user's name choice for new column (baseline expected energy defined by user or calculated based on IEC)
                 'baseline': 'IEC_pstep',
                 # user's name choice for new column (System DC-size, extracted from meta-data)
                 'dcsize': 'dcsize',
                 'compared': 'Compared',  # user's name choice for new column
                 'energy_pstep': 'Energy_pstep',  # user's name choice for new column
                 'clearsky_irr': 'clearsky_irr'  # user's name choice for new column
                 }

om_col_dict = {'siteid': 'randid',
               'datestart': 'date_start',
               'dateend': 'date_end',
               'workID': 'WONumber',
               'worktype': 'WOType',
               'asset': 'Asset',
               # user's name choice for new column (Repair Duration)
               'eventdur': 'EventDur',
               # user's name choice for new column (Month when an event begins)
               'modatestart': 'MonthStart',
               'agedatestart': 'AgeStart'}  # user's name choice for new column (Age of system when event begins)

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

    prod_df_out, mask_series = tprep.prod_irradiance_filter(prod_df, prod_col_dict,
                                                            meta_df, metad_col_dict)

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
