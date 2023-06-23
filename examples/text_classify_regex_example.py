from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from pvops.text import utils
from pvops.text.classify import add_keyword_labels
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
    df = df[~df[NOTES_COLUMN].isna()] # drop any logs without notes

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
    
    def get_metrics(self):
        """Get accuracy measures and count metrics.
        """
        # entries with some keyword over interest, over all entries
        label_count = self.df[NEW_LABEL_COLUMN].count() / len(self.df)

        # replace 'Other' values with 'Unknown'
        self.df[LABEL_COLUMN] = self.df[LABEL_COLUMN].replace('other', 'unknown')
        # replace NaN values to use accuracy score
        self.df[[LABEL_COLUMN, NEW_LABEL_COLUMN]] = self.df[[LABEL_COLUMN, NEW_LABEL_COLUMN]].fillna('unknown')

        acc_score = accuracy_score(y_true=self.df[LABEL_COLUMN], y_pred=self.df[NEW_LABEL_COLUMN])
    
        msg = f'{label_count:.2%} of entries had a keyword of interest, with {acc_score:.2%} accuracy.'
        print(msg)

        # TODO: clean up visualization function here
        labels = list(set(self.df[LABEL_COLUMN]).union(set(self.df[NEW_LABEL_COLUMN])))
        cm_display = ConfusionMatrixDisplay.from_predictions(y_true=self.df[LABEL_COLUMN], y_pred=self.df[NEW_LABEL_COLUMN], display_labels=labels, xticks_rotation='vertical', normalize='true', values_format='.0%')
        fig, ax = plt.subplots(figsize=(10,10))
        cm_display.plot(ax=ax)

        cm_display.figure_.savefig('examples/conf_mat.png', dpi=600)

if __name__ == "__main__":
    # python -m examples.text_classify_regex_example
    # TODO: update this with existing pvops data
    df = get_sample_data(filename='~/data/charity/doe_data/sm_logs_notes_cleaned.csv')
    e = Example(df)
    e.add_equipment_labels()
    e.get_metrics()