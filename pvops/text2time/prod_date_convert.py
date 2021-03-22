import pandas as pd


def prod_date_convert(prod_df, prod_col_dict, toffset=0.0):
    """
    Converts dates from string format to datetime format in production dataframe.


    Parameters

    ----------
    prod_df: DataFrame
        A data frame corresponding to production data.

    prod_col_dict: dict of {str : str}
        A dictionary that contains the column names associated with the production data, which consist of at least:

          - **timestamp** (*string*), should be assigned to user's time-stamp column name

    toffset: float
       Value that specifies how many hours the production data should be shifted by
       in case time-stamps in production data and O&M data don't align as they should.

    Returns

    -------
    DataFrame
        An updated version of the input dataframe, but with time-stamps
        converted to localized (time-zone agnostic) date-time objects.
    """

    # creating local dataframes to not modify originals
    df = prod_df.copy()

    prod_ts = prod_col_dict["timestamp"]

    # Converting date-data from string data to DateTime objects
    df[prod_ts] = pd.to_datetime(df[prod_ts]) + pd.Timedelta(hours=toffset)

    # localizing timestamp
    df[prod_ts] = df[prod_ts].dt.tz_localize(None)

    return df
