import pandas as pd


def overlapping_data(prod_df, om_df, prod_col_dict, om_col_dict):
    """
    Finds the overlapping time-range between the production data and O&M data
    for any given site.  The outputs are a truncated version of the input data
    frames, that contains only data with overlapping dates between the two DFs.


    Parameters

    ----------
    prod_df: DataFrame
        A data frame corresponding to the production
        data after having been processed by the perf_om_NA_qc function. This
        data frame needs the columns specified in prod_col_dict. The
        time-stamp column should not have any NANs for proper operation
        of this function.

    om_df: DataFrame
        A data frame corresponding to the O&M data after
        having been processed by the perf_om_NA_qc function. This data frame needs
        the columns specified in om_col_dict. The time-stamp columns should not
        have any NANs for proper operation of this function.

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
    prod_df: DataFrame
        Production data frame similar to the input data frame, but truncated
        to only contain data that overlaps in time with the O&M data.

    om_df: DataFrame
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

            # Creating NEW DataFrames using masks generated above and concatenate to "_commondates" DFs
            if isinstance(pd.to_datetime(om_df.loc[rid][om_date_s]), pd.Series):
                om_overlap_section = om_df.loc[rid][
                    (omtail_gt_phead_mask) & (omhead_lt_ptail_mask)
                ]
                om_df_commondates = pd.concat([om_df_commondates, om_overlap_section])
            else:
                om_overlap_section = om_df.loc[[rid]][
                    [(omtail_gt_phead_mask) & (omhead_lt_ptail_mask)]
                ]
                om_df_commondates = pd.concat([om_df_commondates, om_overlap_section])

            perf_overlap_section = prod_df.loc[rid][
                (perf_gt_omhead_mask) & (perf_lt_omtail_mask)
            ]
            prod_df_commondates = pd.concat([prod_df_commondates, perf_overlap_section])

    # resetting index of DFs before return
    prod_df_commondates.reset_index(inplace=True)
    om_df_commondates.reset_index(inplace=True)

    prod_df = prod_df_commondates
    om_df = om_df_commondates

    return prod_df, om_df
