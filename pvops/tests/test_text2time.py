import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import sys
import os
import pandas.api.types as ptypes
from pvops.text2time import preprocess, utils

# from om_data_convert import om_data_convert
# Import modules
# from pvops.text2time

# import catscat_fig, count_fig, data_site_na, iec_calc,\
# om_date_convert, om_datelogic_check, om_nadate_process, om_summary_stats,\
# overlapping_data, prod_anomalies, prod_date_convert, prod_quant,\
# summarize_overlaps, viz_om_prod, prod_nadate_process


# Define csv paths
datadir = os.path.join('tutorials', 'example_data')
test_datadir = os.path.join('pvops', 'tests')
example_OMpath = os.path.join(datadir, 'example_om_data2.csv')
example_prodpath = os.path.join(datadir, 'example_prod_data_cumE2.csv')
example_metapath = os.path.join(datadir, 'example_metadata2.csv')

# Assigning dictionaries to connect pvops variables with User's column names
# Format for dictionaries is {pvops variable: user-specific column names}
prod_col_dict = {'siteid': 'randid',
                 'timestamp': 'Date',
                 'energyprod': 'Energy',
                 'irradiance': 'Irradiance',
                 # user's name choice for new column (baseline expected energy defined by user or calculated based on IEC)
                 'baseline': 'IEC_pstep',
                 # user's name choice for new column (System DC-size, extracted from meta-data)
                 'dcsize': 'dcsize',
                 'compared': 'Compared',  # user's name choice for new column
                 'energy_pstep': 'Energy_pstep'}  # user's name choice for new column

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
                  'COD': 'COD'}


# Read data
prod_data = pd.read_csv(
    example_prodpath, on_bad_lines='skip', engine='python')
om_data = pd.read_csv(example_OMpath, on_bad_lines='skip', engine='python')
metadata = pd.read_csv(
    example_metapath, on_bad_lines='skip', engine='python')


def check_same(df1, df2, col):
    if (df1[col].isnull().all() and df2[col].isnull().all()):
        # If both are all nan in col, then assert True
        assert True
    if ptypes.is_numeric_dtype(df1[col]):
        # If numeric, do rounding
        df1 = df1.round({col: 2})
        df2 = df2.round({col: 2})
    assert df1[col].equals(df2[col])


def test_om_data_convert_s():
    om_data_converted = preprocess.om_date_convert(om_data, om_col_dict)
    assert ptypes.is_datetime64_dtype(
        om_data_converted[om_col_dict['datestart']])


def test_om_data_convert_e():
    om_data_converted = preprocess.om_date_convert(om_data, om_col_dict)
    assert ptypes.is_datetime64_dtype(
        om_data_converted[om_col_dict['dateend']])


def test_prod_data_convert():
    prod_data_converted = preprocess.prod_date_convert(
        prod_data, prod_col_dict)
    assert ptypes.is_datetime64_dtype(
        prod_data_converted[prod_col_dict['timestamp']])

# def test_data_site_na():
#     om_data_converted = preprocess.om_date_convert(om_data, om_col_dict)
#     om_data_sitena, addressed = preprocess.data_site_na(om_data_converted, om_col_dict)
#     assert sum(om_data_sitena.loc[prod_col_dict['siteid']].isna())==0


def test_om_datelogic_s():
    om_data_converted = preprocess.om_date_convert(om_data, om_col_dict)
    om_data_sitena, addressed = preprocess.data_site_na(
        om_data_converted, om_col_dict)
    om_data_checked, addressed = preprocess.om_datelogic_check(
        om_data_sitena, om_col_dict, 'swap')
    mask = om_data_checked.loc[:, om_col_dict['dateend']
                               ] < om_data_checked.loc[:, om_col_dict['datestart']]
    assert sum(mask) == 0


def test_om_datelogic_d():
    om_data_converted = preprocess.om_date_convert(om_data, om_col_dict)
    om_data_sitena, addressed = preprocess.data_site_na(
        om_data_converted, om_col_dict)
    om_data_checked, addressed = preprocess.om_datelogic_check(
        om_data_sitena, om_col_dict, 'drop')
    mask = om_data_checked.loc[:, om_col_dict['dateend']
                               ] < om_data_checked.loc[:, om_col_dict['datestart']]
    assert sum(mask) == 0


def test_prod_anomalies_ffT():
    threshold = 1.0
    prod_data_converted = preprocess.prod_date_convert(
        prod_data, prod_col_dict)
    prod_data_anom, addressed = utils.prod_anomalies(prod_data_converted,
                                                     prod_col_dict, threshold,
                                                     np.nan, ffill=True)
    mask = prod_data_anom[prod_col_dict['energyprod']] < 1.0
    assert sum(mask) == 0


def test_prod_anomalies_ffF():
    threshold = 1.0
    prod_data_converted = preprocess.prod_date_convert(
        prod_data, prod_col_dict)
    prod_data_anom, addressed = utils.prod_anomalies(prod_data_converted,
                                                     prod_col_dict, threshold,
                                                     np.nan, ffill=False)
    mask = prod_data_anom[prod_col_dict['energyprod']] < 1.0
    assert sum(mask) == 0


def test_prod_nadate_process_d():
    threshold = 1.0
    prod_data_converted = preprocess.prod_date_convert(
        prod_data, prod_col_dict)
    prod_data_anom, addressed = utils.prod_anomalies(prod_data_converted,
                                                     prod_col_dict, threshold,
                                                     np.nan, ffill=True)
    prod_data_datena, addressed = preprocess.prod_nadate_process(prod_data_anom,
                                                                 prod_col_dict,
                                                                 pnadrop=True)
    mask = prod_data_datena[prod_col_dict['timestamp']].isna()
    assert sum(mask) == 0


def test_prod_nadate_process_id():
    threshold = 1.0
    prod_data_converted = preprocess.prod_date_convert(
        prod_data, prod_col_dict)
    prod_data_anom, addressed = utils.prod_anomalies(prod_data_converted,
                                                     prod_col_dict, threshold,
                                                     np.nan, ffill=True)
    prod_data_datena, addressed = preprocess.prod_nadate_process(prod_data_anom,
                                                                 prod_col_dict,
                                                                 pnadrop=False)
    mask = prod_data_datena[prod_col_dict['timestamp']].isna()
    assert sum(mask) > 0


def test_om_nadate_process_d():
    om_data_converted = preprocess.om_date_convert(om_data, om_col_dict)
    om_data_sitena, addressed = preprocess.data_site_na(
        om_data_converted, om_col_dict)
    om_data_checked_s, addressed = preprocess.om_datelogic_check(om_data_sitena,
                                                                 om_col_dict, 'swap')
    om_data_datena_d, addressed = preprocess.om_nadate_process(om_data_checked_s,
                                                               om_col_dict,
                                                               om_dendflag='drop')
    mask = om_data_datena_d[om_col_dict['dateend']].isna()
    assert sum(mask) == 0


def test_om_nadate_process_t():
    om_data_converted = preprocess.om_date_convert(om_data, om_col_dict)
    om_data_sitena, addressed = preprocess.data_site_na(
        om_data_converted, om_col_dict)
    om_data_checked_s, addressed = preprocess.om_datelogic_check(om_data_sitena,
                                                                 om_col_dict, 'swap')
    om_data_datena_t, addressed = preprocess.om_nadate_process(om_data_checked_s,
                                                               om_col_dict,
                                                               om_dendflag='today')
    mask = om_data_datena_t[om_col_dict['dateend']].isna()
    assert sum(mask) == 0


def test_summarize_overlaps():
    # Prod data => Note the flags used
    threshold = 1.0
    prod_data_converted = preprocess.prod_date_convert(
        prod_data, prod_col_dict)
    prod_data_anom, addressed = utils.prod_anomalies(prod_data_converted,
                                                     prod_col_dict, threshold,
                                                     np.nan, ffill=True)
    prod_data_datena_d, addressed = preprocess.prod_nadate_process(prod_data_anom,
                                                                   prod_col_dict,
                                                                   pnadrop=True)

    # O&M data => Note the flags used
    om_data_converted = preprocess.om_date_convert(om_data, om_col_dict)
    om_data_sitena, addressed = preprocess.data_site_na(
        om_data_converted, om_col_dict)
    om_data_checked_s, addressed = preprocess.om_datelogic_check(om_data_sitena,
                                                                 om_col_dict, 'swap')
    om_data_datena_d, addressed = preprocess.om_nadate_process(om_data_checked_s, om_col_dict,
                                                               om_dendflag='drop')

    # summarize overlaps
    prod_summary, om_summary = utils.summarize_overlaps(
        prod_data_datena_d, om_data_datena_d, prod_col_dict, om_col_dict)

    # import expected pickled DFs
    prod_summ_pick = pd.read_pickle(
        os.path.join(test_datadir, 'prod_summ_pick.pkl'))
    om_summ_pick = pd.read_pickle(
        os.path.join(test_datadir, 'om_summ_pick.pkl'))

    assert prod_summary.equals(
        prod_summ_pick) and om_summary.equals(om_summ_pick)


def test_overlapping_data():
    # Prod data
    threshold = 1.0
    prod_data_converted = preprocess.prod_date_convert(
        prod_data, prod_col_dict)
    prod_data_anom, addressed = utils.prod_anomalies(
        prod_data_converted, prod_col_dict, threshold, np.nan, ffill=True)
    prod_data_datena_d, addressed = preprocess.prod_nadate_process(
        prod_data_anom, prod_col_dict, pnadrop=True)

    # O&M data
    om_data_converted = preprocess.om_date_convert(om_data, om_col_dict)
    om_data_sitena, addressed = preprocess.data_site_na(
        om_data_converted, om_col_dict)
    om_data_checked_s, addressed = preprocess.om_datelogic_check(
        om_data_sitena, om_col_dict, 'swap')
    om_data_datena_d, addressed = preprocess.om_nadate_process(
        om_data_checked_s, om_col_dict, om_dendflag='drop')

    # trim DFs
    prod_data_clean, om_data_clean = utils.overlapping_data(
        prod_data_datena_d, om_data_datena_d, prod_col_dict, om_col_dict)

    assert len(prod_data_clean) == 1020 and len(om_data_clean) == 7


# def test_iec_calc():
#     # Prod data
#     threshold = 1.0
#     prod_data_converted = preprocess.prod_date_convert(prod_data, prod_col_dict)
#     prod_data_anom, addressed = utils.prod_anomalies(prod_data_converted, prod_col_dict, threshold, np.nan, ffill=True)
#     prod_data_datena_d, addressed = preprocess.prod_nadate_process(prod_data_anom, prod_col_dict, pnadrop=True)

#     # O&M data
#     om_data_converted = preprocess.om_date_convert(om_data, om_col_dict)
#     om_data_sitena, addressed = preprocess.data_site_na(om_data_converted, om_col_dict)
#     om_data_checked_s, addressed = preprocess.om_datelogic_check(om_data_sitena, om_col_dict, 'swap')
#     om_data_datena_d, addressed = preprocess.om_nadate_process(om_data_checked_s, om_col_dict, om_dendflag='drop')

#     # trim DFs
#     prod_data_clean, om_data_clean = utils.overlapping_data(prod_data_datena_d, om_data_datena_d, prod_col_dict, om_col_dict)

#     # IEC calc
#     prod_data_clean_iec = utils.iec_calc(prod_data_clean, prod_col_dict, metadata, metad_col_dict, gi_ref=1000.)

#     # import expected pickled DFs
#     prod_data_clean_iec_pick = pd.read_pickle(os.path.join(test_datadir, 'prod_data_clean_iec_pick.pkl'))

#     print(prod_data_clean_iec.dtypes)
#     print(prod_data_clean_iec_pick.dtypes)

#     for col in prod_data_clean_iec_pick.columns:
#         check_same(prod_data_clean_iec, prod_data_clean_iec_pick, col)


# def test_prod_quant():

#     #Prod data
#     threshold = 1.0
#     prod_data_converted = preprocess.prod_date_convert(prod_data,
#                                                        prod_col_dict)
#     prod_data_anom, addressed = utils.prod_anomalies(prod_data_converted,
#                                                      prod_col_dict, threshold,
#                                                      np.nan, ffill=True)
#     prod_data_datena_d, addressed = preprocess.prod_nadate_process(prod_data_anom,
#                                                                    prod_col_dict,
#                                                                    pnadrop=True)

#     #O&M data
#     om_data_converted = preprocess.om_date_convert(om_data, om_col_dict)
#     om_data_sitena, addressed = preprocess.data_site_na(om_data_converted,
#                                                         om_col_dict)
#     om_data_checked_s, addressed = preprocess.om_datelogic_check(om_data_sitena,
#                                                                  om_col_dict, 'swap')
#     om_data_datena_d, addressed = preprocess.om_nadate_process(om_data_checked_s,
#                                                                om_col_dict,
#                                                                om_dendflag='drop')

#     #trim DFs
#     prod_data_clean, om_data_clean = utils.overlapping_data(prod_data_datena_d,
#                                                             om_data_datena_d,
#                                                             prod_col_dict,
#                                                             om_col_dict)

#     #IEC calc
#     prod_data_clean_iec = utils.iec_calc(prod_data_clean, prod_col_dict,
#                                          metadata, metad_col_dict,
#                                          gi_ref=1000.)

#     prod_data_quant = utils.prod_quant(prod_data_clean_iec, prod_col_dict,
#                                        comp_type='norm', ecumu=True)

#     #import expected pickled DFs
#     prod_data_quant_pick = pd.read_pickle(os.path.join(test_datadir,
#                                                        'prod_data_quant_pick.pkl'))

#     print(prod_data_quant)#.dtypes)
#     print(prod_data_quant_pick)#.dtypes)

#     for col in prod_data_quant_pick.columns:
#         check_same(prod_data_quant,prod_data_quant_pick,col)


def test_om_summary_stats():
    # Prod data
    threshold = 1.0
    prod_data_converted = preprocess.prod_date_convert(
        prod_data, prod_col_dict)
    prod_data_anom, addressed = utils.prod_anomalies(prod_data_converted,
                                                     prod_col_dict, threshold,
                                                     np.nan, ffill=True)
    prod_data_datena_d, addressed = preprocess.prod_nadate_process(prod_data_anom,
                                                                   prod_col_dict,
                                                                   pnadrop=True)

    # O&M data
    om_data_converted = preprocess.om_date_convert(om_data, om_col_dict)
    om_data_sitena, addressed = preprocess.data_site_na(om_data_converted,
                                                        om_col_dict)
    om_data_checked_s, addressed = preprocess.om_datelogic_check(om_data_sitena,
                                                                 om_col_dict,
                                                                 'swap')
    om_data_datena_d, addressed = preprocess.om_nadate_process(om_data_checked_s,
                                                               om_col_dict,
                                                               om_dendflag='drop')

    # trim DFs
    prod_data_clean, om_data_clean = utils.overlapping_data(prod_data_datena_d,
                                                            om_data_datena_d,
                                                            prod_col_dict,
                                                            om_col_dict)

    # OM Stats Calc
    om_data_update = utils.om_summary_stats(om_data_clean, metadata,
                                            om_col_dict, metad_col_dict)

    # import expected pickled DF
    om_data_update_pick = pd.read_pickle(os.path.join(test_datadir,
                                                      'om_data_update_pick.pkl'))
    om_data_update_pick = om_data_update_pick.round({'EventDur': 2})
    om_data_update = om_data_update.round({'EventDur': 2})

    om_data_update = om_data_update.astype({'MonthStart': 'int64'})

    for col in om_data_update_pick.columns:
        check_same(om_data_update, om_data_update_pick, col)

# test_iec_calc()
# test_prod_quant()
