import sys
import os
import pandas as pd

T2time_path = os.path.join("..", "pvOps", "text2time")
sys.path.append("..")
sys.path.append(T2time_path)

from overlapping_data import overlapping_data

def summarize_overlaps(prod_df, om_df, prod_col_dict, om_col_dict):
    """
    Provides general overview of the overlapping production and O&M data.


    Parameters

    ----------
    prod_df: DataFrame
        A data frame corresponding to the production
        data after having been processed by the perf_om_NA_qc function. This
        data frame needs the columns specified in prod_col_dict.

    om_df: DataFrame
        A data frame corresponding to the O&M data after
        having been processed by the perf_om_NA_qc function. This data frame
        needs the columns specified in om_col_dict.

    prod_col_dict: dict of {str : str}
        A dictionary that contains the column names relevant for the production data

        - **siteid** (*string*), should be assigned to associated site-ID column name in prod_df
        - **timestamp** (*string*), should be assigned to associated time-stamp column name in prod_df
        - **energyprod** (*string*), should be assigned to associated production column name in prod_df
        - **irradiance** (*string*), should be assigned to associated irradiance column name in prod_df

    om_col_dict: dict of {str : str}
        A dictionary that contains the column names relevant for the O&M data

        - **siteid** (*string*), should be assigned to associated site-ID column name in om_df
        - **datestart** (*string*), should be assigned to associated O&M event start-date column name in om_df
        - **dateend** (*string*), should be assigned to associated O&M event end-date column name in om_df

    Returns

    -------
    prod_output: DataFrame
        A data frame that includes statistics for the production data per site in the data frame.
        Two statistical parameters are calculated and assigned to separate columns:

        - **Actual # Time Stamps** (*datetime.datetime*), total number of overlapping production time-stamps
        - **Max # Time Stamps** (*datetime.datetime), maximum number of production time-stamps, including NANs

    om_out: DataFrame
        A data frame that includes statistics for the O&M data per site in the data frame.
        Three statistical parameters are calculated and assigned to separate columns:

        - **Earliest Event Start** (*datetime.datetime*), column that specifies timestamp of earliest start of all events per site.
        - **Latest Event End** (*datetime.datetime), column that specifies timestamp for latest conclusion of all events per site.
        - **Total Events** (*int*), column that specifies total number of events per site

    """

    # Obtaining new DFs to extract statistics by using overlapping_data function
    prod_df, om_df = overlapping_data(prod_df, om_df, prod_col_dict, om_col_dict)

    om_site = om_col_dict["siteid"]
    om_date_s = om_col_dict["datestart"]
    om_date_e = om_col_dict["dateend"]

    prod_site = prod_col_dict["siteid"]
    prod_ts = prod_col_dict["timestamp"]

    # total number of OM events per site
    num_om_events = om_df[[om_site, om_date_s]].groupby([om_site]).count()

    # earliest dates of O&M events per site
    min_date = om_df[[om_site, om_date_s]].groupby([om_site]).min()

    # earliest dates of O&M events per site
    max_date = om_df[[om_site, om_date_e]].groupby([om_site]).max()

    # concatenating
    om_output = pd.concat([min_date, max_date, num_om_events], axis=1)
    om_output.columns = ["Earliest Event Start", "Latest Event End", "Total Events"]

    # production data timestep frequency in number of hours
    prod_max_ts = prod_df[[prod_site, prod_ts]].groupby([prod_site]).size()
    prod_act_ts = prod_df[[prod_site, prod_ts]].groupby([prod_site]).count()

    prod_output = pd.concat([prod_act_ts, prod_max_ts], axis=1)
    prod_output.columns = ["Actual # Time Stamps", "Max # Time Stamps"]

    return prod_output, om_output
