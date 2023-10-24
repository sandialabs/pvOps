"""
These helper functions focus on performing secondary 
calcuations from the O&M and production data to create
visualizations of the merged data
"""
import pandas as pd
import numpy as np
import tqdm

def interpolate_data(prod_df, om_df, prod_col_dict, om_col_dict, om_cols_to_translate=["asset", "prod_impact"]):
    """
    Provides general overview of the overlapping production and O&M data.

    Parameters
    ----------
    prod_df : DataFrame
        A data frame corresponding to the production
        data after having been processed by the perf_om_NA_qc function. This
        data frame needs the columns specified in prod_col_dict.
    om_df : DataFrame
        A data frame corresponding to the O&M data after
        having been processed by the perf_om_NA_qc function. This data frame
        needs the columns specified in om_col_dict.
    prod_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the production data

        - **siteid** (*string*), should be assigned to associated site-ID column name in prod_df
        - **timestamp** (*string*), should be assigned to associated time-stamp column name in
          prod_df
        - **energyprod** (*string*), should be assigned to associated production column name in
          prod_df
        - **irradiance** (*string*), should be assigned to associated irradiance column name in
          prod_df

    om_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the O&M data

        - **siteid** (*string*), should be assigned to associated site-ID column name in om_df
        - **datestart** (*string*), should be assigned to associated O&M event start-date
          column name in om_df
        - **dateend** (*string*), should be assigned to associated O&M event end-date
          column name in om_df
        - Others specified in om_cols_to_translate

    om_cols_to_translate : list
        List of om_col_dict keys to translate into prod_df

    Returns
    -------
    prod_output : DataFrame
        A data frame that includes statistics for the production data per site in the data frame.
        Two statistical parameters are calculated and assigned to separate columns:

        - **Actual # Time Stamps** (*datetime.datetime*), total number of overlapping
          production time-stamps
        - **Max # Time Stamps** (*datetime.datetime*), maximum number of production time-stamps,
          including NANs

    om_out : DataFrame
        A data frame that includes statistics for the O&M data per site in the data frame.
        Three statistical parameters are calculated and assigned to separate columns:

        - **Earliest Event Start** (*datetime.datetime*), column that specifies timestamp of
          earliest start of all events per site.
        - **Latest Event End** (*datetime.datetime*), column that specifies timestamp for
          latest conclusion of all events per site.
        - **Total Events** (*int*), column that specifies total number of events per site

    """

    prod_df = prod_df.copy()
    om_df = om_df.copy()

    om_site = om_col_dict["siteid"]
    om_date_s = om_col_dict["datestart"]
    om_date_e = om_col_dict["dateend"]
    om_asset = om_col_dict["asset"]

    unique_assets = om_df[om_asset].unique()
    translation_keys = [om_col_dict[key] for key in om_cols_to_translate]

    # Prime the columns in prod_df
    for asset in unique_assets:
        for key in translation_keys:
            prod_df[str(asset) + "_" + key] = [[] for _ in range(len(prod_df))]

    prod_ts = prod_col_dict["timestamp"]
    prod_df["has_ticket"] = False

    # Obtaining new DFs to extract statistics by using overlapping_data function
    prod_df_overlap, om_df_overlap = overlapping_data(
        prod_df, om_df, prod_col_dict, om_col_dict)

    print(f'processing {len(om_df_overlap)} rows')
    for ind, row in tqdm.tqdm(om_df_overlap.iterrows()):
        mask = ((prod_df[prod_ts] >= row[om_date_s]) &
                (prod_df[prod_ts] < row[om_date_e]) &
                (prod_df[om_site] == row[om_site]))
        idxs = np.where(mask)[0]
        for key in translation_keys:
            for i in idxs:
                prod_df.iloc[i, :][str(row[om_asset]) + "_" + key
                                   ].append(row[key])
        prod_df.loc[mask, "has_ticket"] = True
    return prod_df, om_df_overlap


def summarize_overlaps(prod_df, om_df, prod_col_dict, om_col_dict):
    """
    Provides general overview of the overlapping production and O&M data.

    Parameters
    ----------
    prod_df : DataFrame
        A data frame corresponding to the production
        data after having been processed by the perf_om_NA_qc function. This
        data frame needs the columns specified in prod_col_dict.
    om_df : DataFrame
        A data frame corresponding to the O&M data after
        having been processed by the perf_om_NA_qc function. This data frame
        needs the columns specified in om_col_dict.

    prod_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the production data

        - **siteid** (*string*), should be assigned to associated site-ID column name in prod_df
        - **timestamp** (*string*), should be assigned to associated time-stamp column name in
          prod_df
        - **energyprod** (*string*), should be assigned to associated production column name in
          prod_df
        - **irradiance** (*string*), should be assigned to associated irradiance column name in
          prod_df

    om_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the O&M data

        - **siteid** (*string*), should be assigned to associated site-ID column name in om_df
        - **datestart** (*string*), should be assigned to associated O&M event start-date
          column name in om_df
        - **dateend** (*string*), should be assigned to associated O&M event end-date
          column name in om_df

    Returns
    -------
    prod_output : DataFrame
        A data frame that includes statistics for the production data per site in the data frame.
        Two statistical parameters are calculated and assigned to separate columns:

        - **Actual # Time Stamps** (*datetime.datetime*), total number of overlapping
          production time-stamps
        - **Max # Time Stamps** (*datetime.datetime*), maximum number of production time-stamps,
          including NANs

    om_out : DataFrame
        A data frame that includes statistics for the O&M data per site in the data frame.
        Three statistical parameters are calculated and assigned to separate columns:

        - **Earliest Event Start** (*datetime.datetime*), column that specifies timestamp of
          earliest start of all events per site.
        - **Latest Event End** (datetime.datetime*), column that specifies timestamp for
          latest conclusion of all events per site.
        - **Total Events** (*int*), column that specifies total number of events per site

    """

    # Obtaining new DFs to extract statistics by using overlapping_data function
    prod_df, om_df = overlapping_data(
        prod_df, om_df, prod_col_dict, om_col_dict)

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
    om_output.columns = ["Earliest Event Start",
                         "Latest Event End", "Total Events"]

    # production data timestep frequency in number of hours
    prod_max_ts = prod_df[[prod_site, prod_ts]].groupby([prod_site]).size()
    prod_act_ts = prod_df[[prod_site, prod_ts]].groupby([prod_site]).count()

    prod_output = pd.concat([prod_act_ts, prod_max_ts], axis=1)
    prod_output.columns = ["Actual # Time Stamps", "Max # Time Stamps"]

    return prod_output, om_output


def om_summary_stats(om_df, meta_df, om_col_dict, meta_col_dict):
    """
    Adds columns to OM dataframe capturing statistics (e.g., event duration, month of
    occurrence, and age).
    Latter is calculated by using corresponding site commissioning date within the
    metadata dataframe.

    Parameters
    ----------
    om_df : DataFrame
        A data frame corresponding to the O&M data after having been pre-processed
        by the QC and overlappingDFs functions. This data frame needs
        to have the columns specified in om_col_dict.

    meta_df : DataFrame
        A data frame corresponding to the metadata that contains columns specified in meta_col_dict.

    om_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the O&M data which consist of
        at least:

        - **siteid** (*string*), should be assigned to column name for associated site-ID
        - **datestart** (*string*), should be assigned to column name for associated O&M event
          start-date
        - **dateend** (*string*), should be assigned to column name for associated O&M event
          end-date
        - **eventdur** (*string*), should be assigned to column name desired for calculated
          event duration (calculated here, in hours)
        - **modatestart** (*string*), should be assigned to column name desired for month of
          event start (calculated here)
        - **agedatestart** (*string*), should be assigned to column name desired for calculated
          age of site when event started (calculated here, in days)

    meta_col_dict : dict
        A dictionary that contains the column names relevant for the meta-data

        - **siteid** (*string*), should be assigned to associated site-ID column name in meta_df
        - **COD** (*string*), should be asigned to column name corresponding to associated
          commisioning dates for all sites captured in om_df

    Returns
    -------
    om_df : DataFrame
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
    # Extracting commissioning dates of only the sites in the O&M data-frame
    # (in case meta_df has more sites)
    cod_dates = pd.to_datetime(
        meta_df.loc[om_df.index.unique()][meta_cod].copy())

    # Adding age column to om_df, but first initiating a COD column in the
    # OM-data (using NANs) to be able to take the difference between two columns
    om_df[meta_cod] = np.nan
    om_df[meta_cod] = om_df[meta_cod].astype("O")

    for i in cod_dates.index:
        om_df.loc[i, meta_cod] = cod_dates[i]
    om_df[meta_cod] = pd.to_datetime(om_df[meta_cod])
    om_df[meta_cod] = om_df[meta_cod].dt.floor(
        "D")  # hour on commisioning data is
    # unimportant for this analysis
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


def overlapping_data(prod_df, om_df, prod_col_dict, om_col_dict):
    """
    Finds the overlapping time-range between the production data and O&M data
    for any given site.  The outputs are a truncated version of the input data
    frames, that contains only data with overlapping dates between the two DFs.

    Parameters
    ----------
    prod_df : DataFrame
        A data frame corresponding to the production
        data after having been processed by the perf_om_NA_qc function. This
        data frame needs the columns specified in prod_col_dict. The
        time-stamp column should not have any NANs for proper operation
        of this function.
    om_df : DataFrame
        A data frame corresponding to the O&M data after
        having been processed by the perf_om_NA_qc function. This data frame needs
        the columns specified in om_col_dict. The time-stamp columns should not
        have any NANs for proper operation of this function.
    prod_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the production data

        - **siteid** (*string*), should be assigned to associated site-ID column name in prod_df
        - **timestamp** (*string*), should be assigned to associated time-stamp
          column name in prod_df
        - **energyprod** (*string*), should be assigned to associated production
          column name in prod_df
        - **irradiance** (*string*), should be assigned to associated irradiance
          column name in prod_df

    om_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the O&M data

        - **siteid** (*string*), should be assigned to associated site-ID column name in om_df
        - **datestart** (*string*), should be assigned to associated O&M event start-date
          column name in om_df
        - **dateend** (*string*), should be assigned to associated O&M event end-date
          column name in om_df

    Returns
    -------
    prod_df : DataFrame
        Production data frame similar to the input data frame, but truncated
        to only contain data that overlaps in time with the O&M data.
    om_df : DataFrame
        O&M data frame similar to the input data frame, but truncated to only
        contain data that overlaps in time with the production data.

    """
    # assigning dictionary items to local variables for cleaner code
    om_site = om_col_dict["siteid"]
    om_date_s = om_col_dict["datestart"]
    om_date_e = om_col_dict["dateend"]

    prod_site = prod_col_dict["siteid"]
    prod_ts = prod_col_dict["timestamp"]

    # creating local dataframes to not modify originals
    prod_df = prod_df.copy()
    om_df = om_df.copy()

    # setting randid as the index
    om_df = om_df.set_index(om_site)
    prod_df = prod_df.set_index(prod_site)

    # initializing new dataframes
    om_df_commondates = pd.DataFrame()
    prod_df_commondates = pd.DataFrame()

    # finding overlapping DFs
    for rid in prod_df.index.unique():
        if rid in om_df.index.unique():
            # OM Keepers:
            # Only OM tickets that have: (1) an end-date greater than the
            # earliest perf-date AND (2) a start-date less than the last perf-date
            omtail_gt_phead_mask = om_df.loc[rid, om_date_e] >= min(
                prod_df.loc[rid][prod_ts]
            )
            omhead_lt_ptail_mask = om_df.loc[rid, om_date_s] <= max(
                prod_df.loc[rid][prod_ts]
            )

            # Perf Keepers:
            # Only Perf data that has:  (1) a date greater than the START of
            # the earliest OM ticket AND (2) a date less than the END of the oldest OM ticket
            if isinstance(pd.to_datetime(om_df.loc[rid][om_date_s]), pd.Series):
                om_datestart_check = om_df.loc[rid][om_date_s]
                om_dateend_check = om_df.loc[rid][om_date_e]
            else:
                om_datestart_check = [om_df.loc[rid][om_date_s]]
                om_dateend_check = [om_df.loc[rid][om_date_e]]

            # To show production data for the full day if an event occurs
            perf_gt_omhead_mask = prod_df.loc[rid][prod_ts].dt.ceil("D") >= min(
                om_datestart_check
            )
            perf_lt_omtail_mask = prod_df.loc[rid][prod_ts].dt.floor("D") <= max(
                om_dateend_check
            )

            # Creating NEW DataFrames using masks generated above and concatenate to
            # "_commondates" DFs
            if isinstance(pd.to_datetime(om_df.loc[rid][om_date_s]), pd.Series):
                om_overlap_section = om_df.loc[rid][
                    (omtail_gt_phead_mask) & (omhead_lt_ptail_mask)
                ]
                om_df_commondates = pd.concat(
                    [om_df_commondates, om_overlap_section])
            else:
                om_overlap_section = om_df.loc[[rid]][
                    [(omtail_gt_phead_mask) & (omhead_lt_ptail_mask)]
                ]
                om_df_commondates = pd.concat(
                    [om_df_commondates, om_overlap_section])

            perf_overlap_section = prod_df.loc[rid][
                (perf_gt_omhead_mask) & (perf_lt_omtail_mask)
            ]
            prod_df_commondates = pd.concat(
                [prod_df_commondates, perf_overlap_section])

    # resetting index of DFs before return
    prod_df_commondates.reset_index(inplace=True)
    om_df_commondates.reset_index(inplace=True)

    prod_df = prod_df_commondates
    om_df = om_df_commondates

    return prod_df, om_df


def prod_anomalies(prod_df, prod_col_dict, minval=1.0, repval=np.nan, ffill=True):
    """
    For production data with cumulative energy entries, 1) addresses time-stamps where production
    unexpectedly drops to near zero and 2) replaces unexpected production drops with NANs or with
    user-specified value.  If unexpected production drops are replaced with NANs and if 'ffill'
    is set to 'True' in the input argument, a forward-fill method is used to replace the
    unexpected drops.

    Parameters
    ----------
    prod_df : DataFrame
        A data frame corresponding to production data were production is logged on
        a cumulative basis.
    prod_col_dict : dict of {str : str}
        A dictionary that contains the column names associated with the production data,
        which consist of at least:

        - **energyprod** (*string*), should be assigned to the associated cumulative
          production column name in prod_df

    minval : float
        Cutoff value for production data that determines where anomalies are defined. Any production
        values below minval will be addressed by this function. Default minval is 1.0
    repval : float
        Value that should replace the anomalies in a cumulative production data format.
        Default value is numpy's NAN.
    ffill : boolean
        Boolean flag that determines whether NANs in production column in prod_df
        should be filled using a forward-fill method.

    Returns
    -------
    prod_df : DataFrame
        An updated version of the input dataframe, but with zero production values
        converted to user's preference.
    addressed : DataFrame
        A data frame showing rows from the input that were addressed by this function.
    """

    prod_ener = prod_col_dict["energyprod"]

    prod_df = prod_df.copy()
    mask = prod_df.loc[:, prod_ener] < minval
    maskna = prod_df.loc[:, prod_ener].isna()
    addressed = prod_df[mask]
    addressedwna = prod_df[mask | maskna]
    prod_df.loc[mask, prod_ener] = repval

    if ffill:
        prod_df.loc[:, prod_ener].ffill(inplace=True)
        addressed = addressedwna

    return prod_df, addressed


def prod_quant(prod_df, prod_col_dict, comp_type, ecumu=True):
    """
    Compares performance of observed production data in relation to an expected baseline

    Parameters
    ----------
    prod_df : DataFrame
        A data frame corresponding to the production data after having been
        processed by the QC and overlappingDFs functions. This data
        frame needs at least the columns specified in prod_col_dict.
    prod_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the production data

        - **siteid** (*string*), should be assigned to associated site-ID column name in prod_df
        - **timestamp** (*string*), should be assigned to associated time-stamp
          column name in prod_df
        - **energyprod** (*string*), should be assigned to associated production
          column name in prod_df
        - **baseline** (*string*), should be assigned to associated expected baseline
          production column name in prod_df
        - **compared** (*string*), should be assigned to column name desired for
          quantified production data (calculated here)
        - **energy_pstep** (*string*), should be assigned to column name desired for
          energy per time-step (calculated here)

    comp_type : str
        Flag that specifies how the energy production should be compared to the
        expected baseline. A flag of 'diff' shows the subtracted difference between
        the two (baseline - observed). A flag of 'norm' shows the ratio of the two
        (observed/baseline)
    ecumu : bool
        Boolean flag that specifies whether the production (energy output)
        data is input as cumulative information ("True") or on a per time-step basis ("False").

    Returns
    -------
    DataFrame
        A data frame similar to the input, with an added column for the performance comparisons
    """

    prod_site = prod_col_dict["siteid"]
    prod_ener = prod_col_dict["energyprod"]
    baseline_ener = prod_col_dict["baseline"]
    quant_ener = prod_col_dict["compared"]
    pstep_ener = prod_col_dict["energy_pstep"]

    # creating local dataframes to not modify originals
    prod_df = prod_df.copy()
    prod_df.set_index(prod_site, inplace=True)

    for rid in prod_df.index.unique():
        # adding per timestep column for energy production if energy format is cumulative
        if ecumu:
            prod_df.loc[rid, pstep_ener] = prod_df.loc[rid, prod_ener].diff()
        else:
            prod_df.loc[rid, pstep_ener] = prod_df.loc[rid, prod_ener]

        if comp_type == "diff":
            prod_df.loc[rid, quant_ener] = (
                prod_df.loc[rid, baseline_ener] - prod_df.loc[rid, pstep_ener]
            )

        elif comp_type == "norm":
            prod_df.loc[rid, quant_ener] = (
                prod_df.loc[rid, pstep_ener] / prod_df.loc[rid, baseline_ener]
            )

    prod_df.reset_index(inplace=True)

    return prod_df
