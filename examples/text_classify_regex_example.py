import pandas as pd
from nltk.tokenize import word_tokenize

from pvops.text.classify_regex import add_equipment_labels, EQUIPMENT_DICT

LABEL_COLUMN = 'equipment_label'
REGEX_LABEL_COLUMN = 'regex_equipment_label'
NOTES_COLUMN = 'notes'

def get_sample_data():
    filename = '~/data/charity/doe_data/sm_logs_notes_cleaned.csv'
    cols = [NOTES_COLUMN, LABEL_COLUMN] # TODO: add more columns of interest later
    df = pd.read_csv(filename)
    # drop any logs without notes or resolution notes
    df = df[~df[NOTES_COLUMN].isna()]
    return df[cols]

class Example:
    def __init__(self, df, REGEX_LABEL_COLUMN):
        """
        where df is expected input
        """
        self.REGEX_LABEL_COLUMN = REGEX_LABEL_COLUMN
        self.df = df

        # tokenize notes to words
        self.df[NOTES_COLUMN] = self.df[NOTES_COLUMN].apply(word_tokenize)

    def add_equipment_labels(self):
        text_col = NOTES_COLUMN
        new_col = REGEX_LABEL_COLUMN
        reference_dict = EQUIPMENT_DICT
        self.df = add_equipment_labels(self.df, text_col, new_col, reference_dict)

if __name__ == "__main__":
    # python -m examples.classify_regex_example
    df = get_sample_data()
    e = Example(df=df, REGEX_LABEL_COLUMN=REGEX_LABEL_COLUMN)
    e.add_equipment_labels()
    import pdb; pdb.set_trace()
