import pandas as pd
import numpy as np


def om_summary_stats(om_df, meta_df, om_col_dict, meta_col_dict):
    """
    Adds columns to OM dataframe capturing statistics (e.g., event duration, month of occurrence, and age).
    Latter is calculated by using corresponding site commissioning date within the metadata dataframe.


    Parameters

    ----------
    om_df: DataFrame
        A data frame corresponding to the O&M data after having been pre-processed
        by the QC and overlappingDFs functions. This data frame needs
        to have the columns specified in om_col_dict.

    meta_df: DataFrame
        A data frame corresponding to the metadata that contains columns specified in meta_col_dict.

    om_col_dict: dict of {str : str}
        A dictionary that contains the column names relevant for the O&M data which consist of at least:

        - **siteid** (*string*), should be assigned to column name for associated site-ID
        - **datestart** (*string*), should be assigned to column name for associated O&M event start-date
        - **dateend** (*string*), should be assigned to column name for associated O&M event end-date
        - **eventdur** (*string*), should be assigned to column name desired for calculated event duration (calculated here, in hours)
        - **modatestart** (*string*), should be assigned to column name desired for month of event start (calculated here)
        - **agedatestart** (*string*), should be assigned to column name desired for calculated age of site when event started (calculated here, in days)

    meta_col_dict: dict
        A dictionary that contains the column names relevant for the meta-data

        - **siteid** (*string*), should be assigned to associated site-ID column name in meta_df
        - **COD** (*string*), should be asigned to column name corresponding to associated commisioning dates for all sites captured in om_df


    Returns

    -------
    om_df: DataFrame
        An updated version of the input dataframe, but with three new columns
        added for visualizations:  event duration, month of event occurrence, and
        age of system at time of event occurrence.  See om_col_dict for mapping
        of expected variables to user-defined variables.


    """

    # assigning dictionary items to local variables for cleaner code
    om_site = om_col_dict["siteid"]
    om_date_s = om_col_dict["datestart"]
    om_date_e = om_col_dict["dateend"]
    om_rep_dur = om_col_dict["eventdur"]
    om_mo_st = om_col_dict["modatestart"]
    om_age_st = om_col_dict["agedatestart"]

    meta_site = meta_col_dict["siteid"]
    meta_cod = meta_col_dict["COD"]

    # creating local dataframes to not modify originals
    meta_df = meta_df.copy()
    om_df = om_df.copy()

    # Setting randid as index
    om_df.set_index(om_site, inplace=True)

    # Calculating duration of repairs on OM data
    om_df[om_rep_dur] = om_df.loc[:][om_date_e] - om_df[:][om_date_s]

    # Converting Month on which OM-event Starts to an int-type for plotting
    # purposes (To make x-label show in int format)
    om_df[om_mo_st] = om_df[om_date_s].dt.month.astype(int)

    # Calculating age of system at time OM-event occurred using meta-data =>
    meta_df = meta_df.set_index(meta_site)

    # =========================================================================
    # Extracting commissioning dates of only the sites in the O&M data-frame (in case meta_df has more sites)
    cod_dates = pd.to_datetime(meta_df.loc[om_df.index.unique()][meta_cod].copy())

    # Adding age column to om_df, but first initiating a COD column in the
    # OM-data (using NANs) to be able to take the difference between two columns
    om_df[meta_cod] = np.nan
    for i in cod_dates.index:
        om_df.loc[i, meta_cod] = cod_dates[i]
    om_df[meta_cod] = pd.to_datetime(om_df[meta_cod])
    om_df[meta_cod] = om_df[meta_cod].dt.floor(
        "D"
    )  # hour on commisioning data is unimportant for this analysis
    om_df[om_age_st] = om_df.loc[:, om_date_s] - om_df.loc[:, meta_cod]
    # =========================================================================

    # Converting durations to Days
    # Rounding to # of whole days and converting to int (using .dt.days) to do
    # catplot (pandas won't plot timedeltas on y-axis)
    om_df[om_age_st] = om_df[om_age_st].dt.round("D").dt.days
    om_df[om_rep_dur] = om_df[om_rep_dur].dt.seconds / 3600.0

    # Resetting index before completion of function since DFs are mutable
    om_df.reset_index(inplace=True)

    return om_df
