import pandas as pd

# TODO: compare input, output of classify module

"""
1. get expected input (CCRE data)
2. run through regex function
3. get expected output
"""

def get_sample_data():
    filename = '~/data/charity/doe_data/sm_logs_notes_only.csv'
    cols = ['Problem Description', 'Problem Level',] # TODO: add more columns of interest later
    df = pd.read_csv(filename)
    return df[cols]

class Example:
    def __init__(self, df, LABEL_COLUMN):
        """
        where df is expected input
        """
        self.LABEL_COLUMN = LABEL_COLUMN
        self.df = df

if __name__ == "__main__":
    # python -m pvops.text.classify_regex
    df = get_sample_data()
    e = Example(df=df, LABEL_COLUMN='Problem Level')