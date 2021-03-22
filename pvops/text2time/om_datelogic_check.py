import pandas as pd


def om_datelogic_check(om_df, om_col_dict, om_dflag="swap"):
    """
    Addresses issues with O&M dates where the start
    of an event is listed as occurring after its end.  These row are either dropped or the dates
    are swapped, depending on the user's preference.


    Parameters

    ----------
    om_df: DataFrame
        A data frame corresponding to O&M data.

    om_col_dict: dict of {str : str}
        A dictionary that contains the column names associated with the O&M data, which consist of at least:

          - **datestart** (*string*), should be assigned to column name for associated O&M event start date in om_df
          - **dateend** (*string*), should be assigned to column name for associated O&M event end date in om_df

    om_dflag: str
       A flag that specifies how to address rows where the start of an event occurs after its conclusion.
       A flag of 'drop' will drop those rows, and a flag of 'swap' swap the two dates for that row.

    Returns

    -------
    om_df: DataFrame
        An updated version of the input dataframe, but with O&M data quality issues addressed to ensure the start of an event
        precedes the event end date.

    addressed: DataFrame
        A data frame showing rows from the input that were addressed by this function.
    """

    # assigning dictionary items to local variables for cleaner code
    om_date_s = om_col_dict["datestart"]
    om_date_e = om_col_dict["dateend"]

    om_df = om_df.copy()

    # addressing cases where Date_EventEnd ocurrs before Date_EventStart
    mask = om_df.loc[:, om_date_e] < om_df.loc[:, om_date_s]
    addressed = om_df.loc[mask]
    if any(mask) and om_dflag == "swap":  # swap dates for rows where End < Start
        om_df.loc[mask, [om_date_s, om_date_e]] = om_df.loc[
            mask, [om_date_e, om_date_s]
        ].values[0]
    elif any(mask) and om_dflag == "drop":  # drop rows where End < Start
        om_df = om_df[~mask]

    return om_df, addressed
