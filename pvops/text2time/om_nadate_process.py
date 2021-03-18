import pandas as pd
from datetime import datetime


def om_nadate_process(om_df, om_col_dict, om_dendflag="drop"):
    """
    Addresses issues with O&M dataframe where dates are missing (NAN). Two operations are performed:
    1) rows are dropped where start of an event is missing and (2) rows where the conclusion of an event is NAN
    can either be dropped or marked with the time at which program is run, depending on the user's preference.

    Parameters

    ----------
    om_df: DataFrame
        A data frame corresponding to O&M data.

    om_col_dict: dict of {str : str}
        A dictionary that contains the column names associated with the O&M data, which consist of at least:

          - **datestart** (*string*), should be assigned to column name for user's O&M event start-date
          - **dateend** (*string*), should be assigned to column name for user's O&M event end-date

    om_dendflag: str
       A flag that specifies how to address rows where the conclusion of an event is missing (NAN).
       A flag of 'drop' will drop those rows, and a flag of 'today' will replace the NAN
       with the time at which the program is run.  Any other value will leave the rows untouched.

    Returns

    -------
    om_df: DataFrame
        An updated version of the input dataframe, but with no
        missing time-stamps in the O&M data.

    addressed: DataFrame
        A data frame showing rows from the input that were addressed by this function.
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
        None

    return om_df, addressed
