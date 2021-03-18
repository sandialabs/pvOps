def data_site_na(df, df_col_dict):
    """
    Drops rows where site-ID is missing (NAN) within either production or O&M data.


    Parameters

    ----------
    df: DataFrame
        A data frame corresponding to either the production or O&M data.

    df_col_dict: dict of {str : str}
        A dictionary that contains the column names associated with the input df
        and contains at least:

        - **siteid** (*string*), should be assigned to column name for user's site-ID

    Returns

    -------
    df: DataFrame
        An updated version of the input data frame, where rows with site-IDs of NAN are dropped.

    addressed: DataFrame
        A data frame showing rows from the input that were removed by this function.
    """

    df_site = df_col_dict["siteid"]

    df = df.copy()

    namask = df.loc[:, df_site].isna()
    addressed = df.loc[namask]

    df.dropna(subset=[df_site], inplace=True)

    return df, addressed
