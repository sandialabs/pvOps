from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics import accuracy_score

from pvops.text import utils
from pvops.text.classify import get_labels_from_keywords
from examples.example_data.reference_dict import EQUIPMENT_DICT

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
    df = df[~df[NOTES_COLUMN].isna()]  # drop any logs without notes

    # remap assets
    REMAPPING_COL_FROM = 'in'
    REMAPPING_COL_TO = 'out_'
    remapping_df = pd.read_csv('~/pvOps/examples/example_data/remappings_asset.csv')
    remapping_col_dict = {
        'attribute_col': LABEL_COLUMN,
        'remapping_col_from': REMAPPING_COL_FROM,
        'remapping_col_to': REMAPPING_COL_TO
    }

    df = utils.remap_attributes(df, remapping_df, remapping_col_dict, allow_missing_mappings=True)
    return df[[NOTES_COLUMN, LABEL_COLUMN]]

class Example:
    def __init__(self, om_df, col_dict):
        """
        Parameters
        ----------
        om_df : pd.DataFrame
            Must have columns LABEL_COLUMN, and NOTES_COLUMN
        """
        self.om_df = om_df
        self.col_dict = col_dict

        # tokenize notes to words
        self.om_df[NOTES_COLUMN] = self.om_df[NOTES_COLUMN].apply(word_tokenize)

    def add_equipment_labels(self):
        """Add new equipment labels.
        """
        self.om_df = get_labels_from_keywords(self.om_df,
                                              col_dict=self.col_dict,
                                              reference_dict=EQUIPMENT_DICT)

    def get_metrics(self):
        """Get accuracy measures and count metrics.
        """
        # entries with some keyword over interest, over all entries
        label_count = self.om_df[NEW_LABEL_COLUMN].count() / len(self.om_df)

        # replace 'Other' values with 'Unknown'
        self.om_df[LABEL_COLUMN] = self.om_df[LABEL_COLUMN].replace('other', 'unknown')
        # replace NaN values to use accuracy score
        self.om_df[[LABEL_COLUMN, NEW_LABEL_COLUMN]] = self.om_df[[LABEL_COLUMN, NEW_LABEL_COLUMN]].fillna('unknown')

        acc_score = accuracy_score(y_true=self.om_df[LABEL_COLUMN], y_pred=self.om_df[NEW_LABEL_COLUMN])

        msg = f'{label_count:.2%} of entries had a keyword of interest, with {acc_score:.2%} accuracy.'
        print(msg)


if __name__ == "__main__":
    # python -m examples.text_classify_regex_example
    # TODO: update this with existing pvops data
    om_df = get_sample_data(filename='~/data/charity/doe_data/sm_logs_notes_cleaned.csv')
    e = Example(om_df, col_dict={'data': NOTES_COLUMN,
                                 'regex_label': NEW_LABEL_COLUMN})
    e.add_equipment_labels()
    e.get_metrics()