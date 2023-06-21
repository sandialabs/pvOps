import pandas as pd
from nltk.tokenize import word_tokenize

from pvops.text.classify_regex import add_keyword_labels, EQUIPMENT_DICT

LABEL_COLUMN = 'equipment_label'
NEW_LABEL_COLUMN = 'new_equipment_label'
NOTES_COLUMN = 'notes'

def get_sample_data(filename):
    """Function to read .csv file of data.

    Parameters
    ----------
    filename : str

    Returns
    -------
    df: pd.DataFrame
    """

    df = pd.read_csv(filename)
    df = df[~df[NOTES_COLUMN].isna()] # drop any logs without notes
    return df[[NOTES_COLUMN, LABEL_COLUMN]]

class Example:
    def __init__(self, df):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Must have columns LABEL_COLUMN, and NOTES_COLUMN
        """
        self.df = df

        # tokenize notes to words
        self.df[NOTES_COLUMN] = self.df[NOTES_COLUMN].apply(word_tokenize)

    def add_equipment_labels(self):
        """Add new equipment labels.
        """
        self.df = add_keyword_labels(self.df,
                                     text_col=NOTES_COLUMN,
                                     new_col=NEW_LABEL_COLUMN,
                                     reference_dict=EQUIPMENT_DICT)

if __name__ == "__main__":
    # python -m examples.text_classify_regex_example
    df = get_sample_data(filename='~/data/charity/doe_data/sm_logs_notes_cleaned.csv')
    e = Example(df)
    e.add_equipment_labels()