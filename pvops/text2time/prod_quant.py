import pandas as pd


def prod_quant(prod_df, prod_col_dict, comp_type, ecumu=True):
    """
    Compares performance of observed production data in relation to an expected baseline


    Parameters

    ----------
    prod_df: DataFrame
        A data frame corresponding to the production data after having been
        processed by the QC and overlappingDFs functions. This data
        frame needs at least the columns specified in prod_col_dict.

    prod_col_dict: dict of {str : str}
        A dictionary that contains the column names relevant for the production data

        - **siteid** (*string*), should be assigned to associated site-ID column name in prod_df
        - **timestamp** (*string*), should be assigned to associated time-stamp column name in prod_df
        - **energyprod** (*string*), should be assigned to associated production column name in prod_df
        - **baseline** (*string*), should be assigned to associated expected baseline production column name in prod_df
        - **compared** (*string*), should be assigned to column name desired for quantified production data (calculated here)
        - **energy_pstep** (*string*), should be assigned to column name desired for energy per time-step (calculated here)

    comp_type: str
        Flag that specifies how the energy production should be compared to the
        expected baseline. A flag of 'diff' shows the subtracted difference between
        the two (baseline - observed). A flag of 'norm' shows the ratio of the two
        (observed/baseline)

    ecumu: bool
         Boolean flag that specifies whether the production (energy output)
        data is input as cumulative information ("True") or on a per time-step basis ("False").

     Returns

     -------
     DataFrame
        A data frame similar to the input, with an added column for the performance comparisons
    """

    prod_site = prod_col_dict["siteid"]
    prod_ts = prod_col_dict["timestamp"]
    prod_ener = prod_col_dict["energyprod"]
    baseline_ener = prod_col_dict["baseline"]
    quant_ener = prod_col_dict["compared"]
    pstep_ener = prod_col_dict["energy_pstep"]

    # creating local dataframes to not modify originals
    prod_df = prod_df.copy()
    prod_df.set_index(prod_site, inplace=True)

    for rid in prod_df.index.unique():
        # adding per timestep column for energy production if energy format is cumulative
        if ecumu:
            prod_df.loc[rid, pstep_ener] = prod_df.loc[rid, prod_ener].diff()
        else:
            prod_df.loc[rid, pstep_ener] = prod_df.loc[rid, prod_ener]

        if comp_type == "diff":
            prod_df.loc[rid, quant_ener] = (
                prod_df.loc[rid, baseline_ener] - prod_df.loc[rid, pstep_ener]
            )

        elif comp_type == "norm":
            prod_df.loc[rid, quant_ener] = (
                prod_df.loc[rid, pstep_ener] / prod_df.loc[rid, baseline_ener]
            )

    prod_df.reset_index(inplace=True)

    return prod_df
