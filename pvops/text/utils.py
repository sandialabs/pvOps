import pandas as pd
import numpy as np


def remap_attributes(om_df, remapping_df, remapping_col_dict,
                     allow_missing_mappings=False):
    """A utility function which remaps the attributes of om_df using columns
       within remapping_df.

    Parameters

    ---------
    om_df : DataFrame
        A pandas dataframe containing O&M data, which needs to be remapped.
    remapping_df : dataframe
        Holds columns that define the remappings
    remapping_col_dict: dict of {str : str}
        A dictionary that contains the column names that describes how
        remapping is going to be done
        - **attribute_col** (*string*), should be assigned to associated
          column name in om_df which will be remapped
        - **remapping_col_from** (*string*), should be assigned
          to associated column name in remapping_df that matches
          original attribute of interest in om_df
        - **remapping_col_to** (*string*), should be assigned to
          associated column name in remapping_df that contains the
          final mapped entries
    allow_missing_mappings : bool
        If True, allow attributes without specified mappings to exist in
        the final dataframe.
        If False, only attributes specified in `remapping_df` will be in
        final dataframe.

    Returns

    -------
    DataFrame
        dataframe with remapped columns populated
    """
    df = om_df.copy()
    ATTRIBUTE_COL = remapping_col_dict["attribute_col"]
    REMAPPING_COL_FROM = remapping_col_dict["remapping_col_from"]
    REMAPPING_COL_TO = remapping_col_dict["remapping_col_to"]

    # Lower all columns
    df[ATTRIBUTE_COL] = df[ATTRIBUTE_COL].str.lower()
    print(remapping_df[REMAPPING_COL_FROM].tolist())
    remapping_df[REMAPPING_COL_FROM] = remapping_df[REMAPPING_COL_FROM].str.lower()
    remapping_df[REMAPPING_COL_TO] = remapping_df[REMAPPING_COL_TO].str.lower()

    print("Original attribute distribution:")
    print(df[ATTRIBUTE_COL].value_counts())

    # append remapping for if nan, make "Missing"
    remapping_df = remapping_df.append(pd.DataFrame({np.NaN, "Missing"}))
    remapping_df = remapping_df.append(pd.DataFrame({None, "Missing"}))

    if allow_missing_mappings:
        print(remapping_df)
        # Find attributes not considered in mapping
        missing_mappings = list(set(df[ATTRIBUTE_COL].tolist())
                                ^ set(remapping_df[REMAPPING_COL_FROM]))
        temp_remapping_df = pd.DataFrame()
        temp_remapping_df[REMAPPING_COL_FROM] = missing_mappings
        temp_remapping_df[REMAPPING_COL_TO] = missing_mappings
        print("Add missing:")
        print(temp_remapping_df)
        remapping_df = remapping_df.append(temp_remapping_df)
        print("Full remap:")
        print(remapping_df)

    renamer = dict(
        zip(remapping_df[REMAPPING_COL_FROM], remapping_df[REMAPPING_COL_TO])
    )
    df[ATTRIBUTE_COL] = df[ATTRIBUTE_COL].map(renamer)

    print("Final attribute distribution:")
    print(df[ATTRIBUTE_COL].value_counts())

    return df
