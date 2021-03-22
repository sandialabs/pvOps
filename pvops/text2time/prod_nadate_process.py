import pandas as pd
from datetime import datetime


def prod_nadate_process(prod_df, prod_col_dict, pnadrop=False):
    """
    Processes rows of production data frame for missing time-stamp info (NAN).


    Parameters

    ----------
    prod_df: DataFrame
        A data frame corresponding to production data.

    prod_df_col_dict: dict of {str : str}
        A dictionary that contains the column names associated with the production data, which consist of at least:

          - **timestamp** (*string*), should be assigned to associated time-stamp column name in prod_df

    pnadrop: bool
        Boolean flag that determines what to do with rows where time-stamp is missing.
        A value of 'drop' (default) will drop these rows.  Leaving the default
        value of 'ID' will identify rows with missing time-stamps for the user,
        but the function will output the same input data frame with no modifications.

    Returns

    -------
    prod_df: DataFrame
        The output data frame.  If pflag = 'drop', an updated version of the input
        data frame is output, but rows with missing time-stamps are removed. If
        default value is maintained, the input data frame is output with no modifications.

    addressed: DataFrame
        A data frame showing rows from the input that were addressed or identified
        by this function.
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
    else:
        None

    return prod_df, addressed
