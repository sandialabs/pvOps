import pandas as pd

# TODO: compare input, output of classify module

"""
1. get expected input (CCRE data)
2. run through regex function
3. get expected output
"""
LABEL_COLUMN = 'Problem Level'
NOTES_COLUMN = 'Problem Description'

# first pass of equipment dictionary is based on
# https://www.osti.gov/servlets/purl/1872704
# TODO: only currently takes single words, not phrases: 'string inverter', 'met station', etc.
EQUIPMENT_DICT = {'combiner': ['combiner', 'comb', 'cb'],
                  'battery': ['battery', 'bess',], # this should be energy storage, if we could handle n-grams
                  'inverter': ['inverter', 'invert', 'inv',],
                  'met': ['met'], # this should be met station, if we could handle n-grams
                  'meter': ['meter'],
                  'module': ['module', 'mod'],
                  'recloser': ['recloser', 'reclose'],
                  'relay': ['relay'],
                  'substation': ['substation'],
                  'switchgear': ['switchgear', 'switch'],
                  'tracker': ['tracker'],
                  'transformer': ['transformer', 'xfmr'],
                  'wiring': ['wiring', 'wire', 'wires']
                  }

def get_sample_data():
    filename = '~/data/charity/doe_data/sm_logs_notes_only.csv'
    cols = [NOTES_COLUMN, LABEL_COLUMN] # TODO: add more columns of interest later
    df = pd.read_csv(filename)
    return df[cols]

def get_all_keywords_of_interest(list_of_txt, custom_dict=None):
    """
    if keywords of interest are in the list of text, return the keyword category
    for example, if 'inverter' are in the list of text, return ['inverter']

    Parameters
    ----------
    list_of_txt: list
        list of strings
    
    Returns
    -------
    included_equipment: list
        list of strings (included equipment named)

    """
    text_to_search = set(list_of_txt)
    
    if custom_dict is not None:
        EQUIPMENT_DICT.update(custom_dict)

    equipment_keywords = set(EQUIPMENT_DICT.keys())
    included_equipment = list(text_to_search.intersection(equipment_keywords))

    return included_equipment

class Example:
    def __init__(self, df, LABEL_COLUMN):
        """
        where df is expected input
        """
        self.LABEL_COLUMN = LABEL_COLUMN
        self.df = df

    def add_equipment_labels(self):
        """
        manually add labels to mor text logs
        dataframe with additional 'equipment_label' column
        dependent on entries in EQUIPMENT_DICT
        """
        self.df[LABEL_COLUMN] = self.df[NOTES_COLUMN].apply(get_all_keywords_of_interest)

        # each multi-category now in its own row
        # some logs have multiple equipment issues
        self.df = self.df.explode(LABEL_COLUMN)

if __name__ == "__main__":
    # python -m pvops.text.classify_regex
    df = get_sample_data()
    e = Example(df=df, LABEL_COLUMN=LABEL_COLUMN)