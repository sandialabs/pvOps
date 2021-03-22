import pandas as pd
import numpy as np

from gensim.models import Word2Vec
import numpy as np


def summarize_text_data(om_df, colname):
    """Display information about a set of documents located in a dataframe, including
    the number of samples, average number of words, vocabulary size, and number of words
    in total.

    Parameters

    ---------
    om_df : DataFrame
        A pandas dataframe containing O&M data, which contains at least the colname of interest
    colname : str
        Column name of column with text

    Returns

    ------
    None
    """
    df = om_df.copy()
    text = df[colname].tolist()

    nonan_text = [x for x in text if (str(x) != "nan" and x != None)]

    tokenized = [sentence.split() for sentence in nonan_text]
    avg_n_words = np.array([len(tokens) for tokens in tokenized]).mean()
    sum_n_words = np.array([len(tokens) for tokens in tokenized]).sum()
    model = Word2Vec(tokenized, min_count=1, size=64)

    # Total vocabulary
    vocab = model.wv.vocab

    # Bold title.
    print("\033[1m" + "DETAILS" + "\033[0m")

    info = {
        "n_samples": len(df),
        "n_nan_docs": len(df) - len(nonan_text),
        "n_words_doc_average": avg_n_words,
        "n_unique_words": len(vocab),
        "n_total_words": sum_n_words,
    }

    # Display information.
    print(f'  {info["n_samples"]} samples')
    print(f'  {info["n_nan_docs"]} invalid documents')
    print("  {:.2f} words per sample on average".format(info["n_words_doc_average"]))
    print(f'  Number of unique words {info["n_unique_words"]}')
    print("  {:.2f} total words".format(info["n_total_words"]))

    return info


def remap_attributes(om_df, remapping_df, remapping_col_dict):
    """A utility function which remaps the attributes of om_df using columns within remapping_df.

    Parameters

    ---------
    om_df : DataFrame
        A pandas dataframe containing O&M data, which needs to be remapped.
    remapping_df : dataframe
        Holds columns that define the remappings
    remapping_col_dict: dict of {str : str}
        A dictionary that contains the column names that describes how remapping is going to be done
        - **attribute_col** (*string*), should be assigned to associated column name in om_df which will be remapped
        - **remapping_col_from** (*string*), should be assigned to associated column name in remapping_df that matches original attribute of interest in om_df
        - **remapping_col_to** (*string*), should be assigned to associated column name in remapping_df that contains the final mapped entries

    Returns

    -------
    DataFrame
        dataframe with remapped columns populated
    """
    df = om_df.copy()
    ATTRIBUTE_COL = remapping_col_dict["attribute_col"]
    REMAPPING_COL_FROM = remapping_col_dict["remapping_col_from"]
    REMAPPING_COL_TO = remapping_col_dict["remapping_col_to"]

    print("Original attribute distribution:")
    print(df[ATTRIBUTE_COL].value_counts())

    # append remapping for if nan, make "Missing"
    remapping_df = remapping_df.append(pd.DataFrame({np.NaN, "Missing"}))

    renamer = dict(
        zip(remapping_df[REMAPPING_COL_FROM], remapping_df[REMAPPING_COL_TO])
    )
    df[ATTRIBUTE_COL] = df[ATTRIBUTE_COL].map(renamer)

    print("Final attribute distribution:")
    print(df[ATTRIBUTE_COL].value_counts())

    return df