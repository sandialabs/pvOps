import numpy as np

def iec_calc(prod_df, prod_col_dict, meta_df, meta_col_dict,
             gi_ref=1000.0):
    """Calculates expected energy using measured irradiance
    based on IEC calculations. 

    Parameters
    ----------
    prod_df : DataFrame
        A data frame corresponding to the production data
        after having been processed by the perf_om_NA_qc
        and overlappingDFs functions. This data frame needs
        at least the columns specified in prod_col_dict.

    prod_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant
        for the production data

        - **siteid** (*string*), should be assigned to 
          site-ID column name in prod_df
        - **timestamp** (*string*), should be assigned to
          time-stamp column name in prod_df
        - **irradiance** (*string*), **plane-of-array**. Should be assigned to
          irradiance column name in prod_df, where data
          should be in [W/m^2].
        - **baseline** (*string*), should be assigned to
          preferred column name to capture IEC calculations
          in prod_df
        - **dcsize**, (*string*), should be assigned to
          preferred column name for site capacity in prod_df

    meta_df : DataFrame
        A data frame corresponding to site metadata.
        At the least, the columns in meta_col_dict be
        present.

    meta_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant
        for the meta-data

        - **siteid** (*string*), should be assigned to site-ID
          column name
        - **dcsize** (*string*), should be assigned to
          column name corresponding to site capacity, where
          data is in [kW]

    gi_ref : float
        reference plane of array irradiance in W/m^2 at
        which a site capacity is determined (default value
        is 1000 [W/m^2])

    Returns
    -------
    DataFrame
        A data frame for production data with a new column,
        iecE, which is the predicted energy calculated
        based on the IEC standard using measured irradiance
        data

    """
    # assigning dictionary items to local variables for cleaner code
    prod_site = prod_col_dict["siteid"]
    prod_ts = prod_col_dict["timestamp"]
    prod_irr = prod_col_dict["irradiance"]
    prod_iec = prod_col_dict["baseline"]
    prod_dcsize = prod_col_dict["dcsize"]

    meta_site = meta_col_dict["siteid"]
    meta_size = meta_col_dict["dcsize"]

    # creating local dataframes to not modify originals
    prod_df = prod_df.copy()
    meta_df = meta_df.copy()

    # setting index for metadata for alignment to production data
    meta_df = meta_df.set_index(meta_site)

    # Creating new column in production data corresponding to site size (in terms of KW)
    prod_df[prod_dcsize] = prod_df.loc[:, prod_site].apply(
        lambda x: meta_df.loc[x, meta_size]
    )

    # iec calculation

    for sid in prod_df.loc[:, prod_site].unique():
        mask = prod_df.loc[:, prod_site] == sid
        tstep = prod_df.loc[mask, prod_ts].iloc[1] - \
            prod_df.loc[mask, prod_ts].iloc[0]
        tstep = tstep / np.timedelta64(
            1, "h"
        )  # Converting the time-step to float (representing hours) to
        # arrive at kWh for the iecE calculation

        prod_df.loc[mask, prod_iec] = (
            prod_df.loc[mask, prod_dcsize]
            * prod_df.loc[mask, prod_irr]
            * tstep
            / gi_ref
        )
    prod_df.drop(columns=[prod_dcsize], inplace=True)

    return prod_df
