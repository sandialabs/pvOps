import pandas as pd
import numpy as np


def remap_attributes(om_df, remapping_df, remapping_col_dict,
                     allow_missing_mappings=False, print_info=False):
    """A utility function which remaps the attributes of om_df using columns
       within remapping_df.

    Parameters
    ----------
    om_df : DataFrame
        A pandas dataframe containing O&M data, which needs to be remapped.
    remapping_df : DataFrame
        Holds columns that define the remappings
    remapping_col_dict : dict of {str : str}
        A dictionary that contains the column names that describes how
        remapping is going to be done

        - attribute_col : string, should be assigned to associated
          column name in om_df which will be remapped
        - remapping_col_from : string, should be assigned
          to associated column name in remapping_df that matches
          original attribute of interest in om_df
        - remapping_col_to : string, should be assigned to
          associated column name in remapping_df that contains the
          final mapped entries
    allow_missing_mappings : bool
        If True, allow attributes without specified mappings to exist in
        the final dataframe.
        If False, only attributes specified in `remapping_df` will be in
        final dataframe.
    print_info : bool
        If True, print information about remapping.

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

    if print_info:
        print("Initial value counts:")
        print(df[ATTRIBUTE_COL].value_counts())

    remapping_df[REMAPPING_COL_FROM] = remapping_df[REMAPPING_COL_FROM].str.lower()
    remapping_df[REMAPPING_COL_TO] = remapping_df[REMAPPING_COL_TO].str.lower()

    if allow_missing_mappings:
        # Find attributes not considered in mapping
        unique_words_in_data = set(df[ATTRIBUTE_COL].tolist())
        missing_mappings = list(unique_words_in_data
                                ^ set(remapping_df[REMAPPING_COL_FROM]))
        missing_mappings = [word for word in missing_mappings
                            if word in unique_words_in_data]
        temp_remapping_df = pd.DataFrame()
        temp_remapping_df[REMAPPING_COL_FROM] = missing_mappings
        temp_remapping_df[REMAPPING_COL_TO] = missing_mappings
        remapping_df = pd.concat([remapping_df, temp_remapping_df]) 

    if print_info:
        print("All mappings:\n", remapping_df)
    renamer = dict(
        zip(remapping_df[REMAPPING_COL_FROM], remapping_df[REMAPPING_COL_TO])
    )
    df[ATTRIBUTE_COL] = df[ATTRIBUTE_COL].map(renamer)

    if print_info:
        print("Final attribute distribution:")
        print(df[ATTRIBUTE_COL].value_counts())

        print(f"Number of nan definitions of {ATTRIBUTE_COL}:"
              "{sum(df[ATTRIBUTE_COL].isna())}")

    return df

def remap_words_in_text(om_df, remapping_df, remapping_col_dict):
    """A utility function which remaps a text column of om_df using columns
       within remapping_df.

    Parameters
    ----------
    om_df : DataFrame
        A pandas dataframe containing O&M note data
    remapping_df : DataFrame
        Holds columns that define the remappings
    remapping_col_dict : dict of {str : str}
        A dictionary that contains the column names that describes how
        remapping is going to be done

        - data : string, should be assigned to associated
          column name in om_df which will have its text tokenized and remapped
        - remapping_col_from : string, should be assigned
          to associated column name in remapping_df that matches
          original attribute of interest in om_df
        - remapping_col_to : string, should be assigned to
          associated column name in remapping_df that contains the
          final mapped entries

    Returns
    -------
    DataFrame
        dataframe with remapped columns populated
    """
    df = om_df.copy()
    TEXT_COL = remapping_col_dict["data"]
    REMAPPING_COL_FROM = remapping_col_dict["remapping_col_from"]
    REMAPPING_COL_TO = remapping_col_dict["remapping_col_to"]

    # drop any values where input value is equal to output value
    remapping_df = remapping_df[remapping_df[REMAPPING_COL_FROM] != remapping_df[REMAPPING_COL_TO]]

    # case-sensitive
    remapping_df[REMAPPING_COL_FROM] = remapping_df[REMAPPING_COL_FROM].str.lower()
    remapping_df[REMAPPING_COL_TO] = remapping_df[REMAPPING_COL_TO].str.lower()
    df[TEXT_COL] = df[TEXT_COL].str.lower()

    renamer = dict(
        zip(remapping_df[REMAPPING_COL_FROM], remapping_df[REMAPPING_COL_TO])
    )

    df[TEXT_COL] = df[TEXT_COL].replace(renamer, regex=True)

    return df