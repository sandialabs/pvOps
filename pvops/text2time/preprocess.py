"""
These functions focus on pre-processing user O&M and production data to
create visualizations of the merged data
"""
from datetime import datetime
import pandas as pd


def data_site_na(pom_df, df_col_dict):
    """
    Drops rows where site-ID is missing (NAN) within either production 
    or O&M data.

    Parameters

    ----------
    pom_df: DataFrame
        A data frame corresponding to either the production or O&M 
        data.

    df_col_dict: dict of {str : str}
        A dictionary that contains the column names associated with 
        the 
        input `pom_df` and contains at least:

        - **siteid** (*string*), should be assigned to column name 
        for 
        user's site-ID

    Returns

    -------
    pom_df: DataFrame
        An updated version of the input data frame, where rows with 
        site-IDs of NAN are dropped.

    addressed: DataFrame
        A data frame showing rows from the input that were removed 
        by this function.
    """

    df_site = df_col_dict["siteid"]

    pom_df = pom_df.copy()

    namask = pom_df.loc[:, df_site].isna()
    addressed = pom_df.loc[namask]

    pom_df.dropna(subset=[df_site], inplace=True)

    return pom_df, addressed


def om_date_convert(om_df, om_col_dict, toffset=0.0):
    """
    Converts dates from string format to date time object in O&M
    dataframe.


    Parameters

    ----------
    om_df: DataFrame
        A data frame corresponding to O&M data.

    om_col_dict: dict of {str : str}
        A dictionary that contains the column names associated with
        the O&M data, which consist of at least:

          - **datestart** (*string*), should be assigned to column
            name for O&M event start date in om_df
          - **dateend** (*string*), should be assigned to column name
            for O&M event end date  in om_df

    toffset: float
       Value that specifies how many hours the O&M data should be
       shifted by in case time-stamps in production data and O&M data
       don't align as they should

    Returns

    -------
    DataFrame
        An updated version of the input dataframe, but with 
        time-stamps converted to localized (time-zone agnostic) 
        date-time objects.
    """

    om_df = om_df.copy()

    om_date_s = om_col_dict["datestart"]
    om_date_e = om_col_dict["dateend"]

    # Converting date-data from string data to DateTime objects
    om_df[om_date_s] = pd.to_datetime(
        om_df[om_date_s]) + pd.Timedelta(hours=toffset)
    om_df[om_date_e] = pd.to_datetime(
        om_df[om_date_e]) + pd.Timedelta(hours=toffset)

    # localizing timestamp
    om_df[om_date_s] = om_df[om_date_s].dt.tz_localize(None)
    om_df[om_date_e] = om_df[om_date_e].dt.tz_localize(None)

    return om_df


def om_datelogic_check(om_df, om_col_dict, om_dflag="swap"):
    """
    Addresses issues with O&M dates where the start
    of an event is listed as occurring after its end.  These row are
    either dropped or the dates are swapped, depending on the user's
    preference.


    Parameters

    ----------
    om_df: DataFrame
        A data frame corresponding to O&M data.

    om_col_dict: dict of {str : str}
        A dictionary that contains the column names associated with
        the O&M data, which consist of at least:

          - **datestart** (*string*), should be assigned to column
            name for associated O&M event start date in om_df
          - **dateend** (*string*), should be assigned to column name
            for associated O&M event end date in om_df

    om_dflag: str
       A flag that specifies how to address rows where the start of
       an event occurs after its conclusion. A flag of 'drop' will
       drop those rows, and a flag of 'swap' swap the two dates for
       that row.

    Returns

    -------
    om_df: DataFrame
        An updated version of the input dataframe, but with O&M data
        quality issues addressed to ensure the start of an event
        precedes the event end date.

    addressed: DataFrame
        A data frame showing rows from the input that were addressed
        by this function.
    """

    # assigning dictionary items to local variables for cleaner code
    om_date_s = om_col_dict["datestart"]
    om_date_e = om_col_dict["dateend"]

    om_df = om_df.copy()

    # addressing cases where Date_EventEnd ocurrs before Date_EventStart
    mask = om_df.loc[:, om_date_e] < om_df.loc[:, om_date_s]
    addressed = om_df.loc[mask]
    # swap dates for rows where End < Start
    if any(mask) and om_dflag == "swap":
        om_df.loc[mask, [om_date_s, om_date_e]] = om_df.loc[
            mask, [om_date_e, om_date_s]
        ].values[0]
    # drop rows where End < Start
    elif any(mask) and om_dflag == "drop":
        om_df = om_df[~mask]

    return om_df, addressed


def om_nadate_process(om_df, om_col_dict, om_dendflag="drop"):
    """
    Addresses issues with O&M dataframe where dates are missing
    (NAN). Two operations are performed: 1) rows are dropped 
    where start of an event is missing and (2) rows where the 
    conclusion of an event is NAN can either be dropped or marked 
    with the time at which program is run, depending on the user's
    preference.

    Parameters

    ----------
    om_df: DataFrame
        A data frame corresponding to O&M data.

    om_col_dict: dict of {str : str}
        A dictionary that contains the column names associated with
        the O&M data, which consist of at least:

          - **datestart** (*string*), should be assigned to column
            name for user's O&M event start-date
          - **dateend** (*string*), should be assigned to column name
            for user's O&M event end-date

    om_dendflag: str
       A flag that specifies how to address rows where the conclusion
       of an event is missing (NAN). A flag of 'drop' will drop those
       rows, and a flag of 'today' will replace the NAN with the time
       at which the program is run.  Any other value will leave the
       rows untouched.

    Returns

    -------
    om_df: DataFrame
        An updated version of the input dataframe, but with no
        missing time-stamps in the O&M data.

    addressed: DataFrame
        A data frame showing rows from the input that were addressed
        by this function.
    """

    om_df = om_df.copy()

    # assigning dictionary items to local variables for cleaner code
    om_date_s = om_col_dict["datestart"]
    om_date_e = om_col_dict["dateend"]

    # Dropping rows where om_date_s has values of NA in om_df
    mask1 = om_df.loc[:, om_date_s].isna()
    om_df.dropna(
        subset=[om_date_s], inplace=True
    )  # drops rows with om_date_e of NA in om_df

    # Addressing rows with 'om_date_e' values of NA in om_df
    mask2 = om_df.loc[:, om_date_e].isna()
    mask = mask1 | mask2
    addressed = om_df.loc[mask]

    if om_dendflag == "drop":
        om_df.dropna(
            subset=[om_date_e], inplace=True
        )  # drops rows with om_date_e of NA in om_df
    elif om_dendflag == "today":
        om_df[om_date_e].fillna(
            pd.to_datetime(str(datetime.now())[:20]), inplace=True
        )  # replacing NANs with today's date
    else:
        raise SyntaxError('Undefined om_dendflag')

    return om_df, addressed


def prod_date_convert(prod_df, prod_col_dict, toffset=0.0):
    """Converts dates from string format to datetime format in
    production dataframe.


    Parameters

    ----------
    prod_df: DataFrame
        A data frame corresponding to production data.

    prod_col_dict: dict of {str : str}
        A dictionary that contains the column names associated with 
        the production data, which consist of at least:

          - **timestamp** (*string*), should be assigned to user's
            time-stamp column name

    toffset: float
       Value that specifies how many hours the production data 
       should be shifted by in case time-stamps in production data
       and O&M data don't align as they should.

    Returns

    -------
    DataFrame
        An updated version of the input dataframe, but with
        time-stamps converted to localized (time-zone agnostic)
        date-time objects.
    """

    # creating local dataframes to not modify originals
    prod_df = prod_df.copy()

    prod_ts = prod_col_dict["timestamp"]

    # Converting date-data from string data to DateTime objects
    prod_df[prod_ts] = pd.to_datetime(
        prod_df[prod_ts]) + pd.Timedelta(hours=toffset)

    # localizing timestamp
    prod_df[prod_ts] = prod_df[prod_ts].dt.tz_localize(None)

    return prod_df


def prod_nadate_process(prod_df, prod_col_dict, pnadrop=False):
    """
    Processes rows of production data frame for missing time-stamp
    info (NAN).


    Parameters

    ----------
    prod_df: DataFrame
        A data frame corresponding to production data.

    prod_df_col_dict: dict of {str : str}
        A dictionary that contains the column names associated with
        the production data, which consist of at least:

          - **timestamp** (*string*), should be assigned to
            associated time-stamp column name in prod_df

    pnadrop: bool
        Boolean flag that determines what to do with rows where
        time-stamp is missing. A value of `True` will drop these
        rows.  Leaving the default value of `False` will identify
        rows with missing time-stamps for the user, but the function
        will output the same input data frame with no modifications.

    Returns

    -------
    prod_df: DataFrame
        The output data frame.  If pflag = 'drop', an updated version
        of the input data frame is output, but rows with missing
        time-stamps are removed. If default value is maintained, the
        input data frame is output with no modifications.

    addressed: DataFrame
        A data frame showing rows from the input that were addressed
        or identified by this function.
    """

    prod_df = prod_df.copy()

    # creating local dataframes to not modify originals
    prod_df = prod_df.copy()

    prod_ts = prod_col_dict["timestamp"]

    # Dropping rows
    mask = prod_df.loc[:, prod_ts].isna()
    addressed = prod_df[mask]
    if pnadrop:
        prod_df.dropna(subset=[prod_ts], inplace=True)

    return prod_df, addressed
