import pandas as pd


def om_date_convert(om_df, om_col_dict, toffset=0.0):
    """
    Converts dates from string format to date time object in O&M dataframe.


    Parameters

    ----------
    om_df: DataFrame
        A data frame corresponding to O&M data.

    om_col_dict: dict of {str : str}
        A dictionary that contains the column names associated with the O&M data, which consist of at least:

          - **datestart** (*string*), should be assigned to column name for O&M event start date in om_df
          - **dateend** (*string*), should be assigned to column name for O&M event end date  in om_df

    toffset: float
       Value that specifies how many hours the O&M data should be shifted by in
       case time-stamps in production data and O&M data don't align as they should

    Returns

    -------
    DataFrame
        An updated version of the input dataframe, but with time-stamps
        converted to localized (time-zone agnostic) date-time objects.
    """

    df = om_df.copy()

    om_date_s = om_col_dict["datestart"]
    om_date_e = om_col_dict["dateend"]

    # Converting date-data from string data to DateTime objects
    df[om_date_s] = pd.to_datetime(df[om_date_s]) + pd.Timedelta(hours=toffset)
    df[om_date_e] = pd.to_datetime(df[om_date_e]) + pd.Timedelta(hours=toffset)

    # localizing timestamp
    df[om_date_s] = df[om_date_s].dt.tz_localize(None)
    df[om_date_e] = df[om_date_e].dt.tz_localize(None)

    return df
