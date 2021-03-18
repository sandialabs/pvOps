import pandas as pd
import numpy as np


def prod_anomalies(prod_df, prod_col_dict, minval=1.0, repval=np.nan, ffill=True):
    """
    For production data with cumulative energy entries, 1) addresses time-stamps where production
    unexpectedly drops to near zero and 2) replaces unexpected production drops with NANs or with
    user-specified value.  If unexpected production drops are replaced with NANs and if 'ffill'
    is set to 'True' in the input argument, a forward-fill method is used to replace the unexpected drops.


    Parameters

    ----------
    prod_df: DataFrame
        A data frame corresponding to production data were production is logged on
        a cumulative basis.

    prod_col_dict: dict of {str : str}
        A dictionary that contains the column names associated with the production data, which consist of at least:

        - **energyprod** (*string*), should be assigned to the associated cumulative production column name in prod_df

    minval: float
        Cutoff value for production data that determines where anomalies are defined. Any production
        values below minval will be addressed by this function. Default minval is 1.0

    repval: float
       Value that should replace the anomalies in a cumulative production data format.
       Default value is numpy's NAN.

    ffill: boolean
        Boolean flag that determines whether NANs in production column in prod_df
        should be filled using a forward-fill method.

    Returns

    -------
    prod_df: DataFrame
        An updated version of the input dataframe, but with zero production values
        converted to user's preference.

    addressed: DataFrame
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
        prod_df.loc[:, prod_ener].fillna(method="ffill", inplace=True)
        addressed = addressedwna

    return prod_df, addressed
