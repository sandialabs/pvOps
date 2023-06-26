from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics import accuracy_score

from pvops.text import utils, preprocess
from pvops.text.classify import get_attributes_from_keywords
from pvops.text.visualize import visualize_classification_confusion_matrix
from examples.example_data.reference_dict import EQUIPMENT_DICT


class Example:
    def __init__(self, om_df, col_dict):
        """
        Parameters
        ----------
        om_df : pd.DataFrame
            Must have columns in col_dict
        col_dict : dict of {str : str}
            A dictionary that contains the column names associated with 
            the 
            input `om_df` and contains at least:

            - **data** (*string*)
            - **attribute_col** (*string*)
            - **predicted_col** (*string*)
        """
        self.om_df = om_df
        self.col_dict = col_dict
        self.DATA_COL = self.col_dict['data']
        self.LABEL_COL = self.col_dict['attribute_col']
        self.PREDICT_COL = self.col_dict['predicted_col']

        # tokenize notes to words
        self.om_df[self.DATA_COL] = self.om_df[self.DATA_COL].apply(word_tokenize)

    def add_equipment_labels(self):
        """Add new equipment labels.
        """
        self.om_df = get_attributes_from_keywords(self.om_df,
                                                  col_dict=self.col_dict,
                                                  reference_dict=EQUIPMENT_DICT)

    def plot_confusion_matrix(self, output_filepath, title):
        """Plot confusion matrix for actual and predicted labels

        Parameters
        ----------
        output_filepath : str
            path to write plot image to
        title : str
            Optional, title of plot
        """
        fig = visualize_classification_confusion_matrix(self.om_df, self.col_dict, title)
        fig.figure_.savefig(output_filepath, dpi=600, pad_inches=2)

    def get_metrics(self):
        """Get accuracy measures and count metrics.
        """
        # entries with some keyword over interest, over all entries
        label_count = self.om_df[self.PREDICT_COL].count() / len(self.om_df)

        # replace 'Other' values with 'Unknown'
        self.om_df[self.LABEL_COL] = self.om_df[self.LABEL_COL].replace('other', 'unknown')
        # replace NaN values to use accuracy score
        self.om_df[[self.LABEL_COL, self.PREDICT_COL]] = self.om_df[[self.LABEL_COL, self.PREDICT_COL]].fillna('unknown')
        acc_score = accuracy_score(y_true=self.om_df[self.LABEL_COL], y_pred=self.om_df[self.PREDICT_COL])

        msg = f'{label_count:.2%} of entries had a keyword of interest, with {acc_score:.2%} accuracy.'
        print(msg)


if __name__ == "__main__":
    # python -m examples.text_classify_regex_example

    om_df = pd.read_csv('examples/example_data/example_ML_ticket_data.csv')
    col_dict = {
        "data" : "CompletionDesc",
        "eventstart" : "Date_EventStart",
        "save_data_column" : "processed_data",
        "save_date_column" : "processed_date",
        "attribute_col" : "Asset",
        "predicted_col" : "Keyword_Asset",
        "remapping_col_from": "in",
        "remapping_col_to": "out_"
    }

    # remap assets
    remapping_df = pd.read_csv('~/pvOps/examples/example_data/remappings_asset.csv')
    om_df = utils.remap_attributes(om_df, remapping_df, col_dict, allow_missing_mappings=True)

    # preprocessing steps
    om_df = preprocess.preprocessor(om_df, lst_stopwords=[], col_dict=col_dict, print_info=False, extract_dates_only=False)
    om_df[col_dict['attribute_col']] = om_df.apply(lambda row: row[col_dict['attribute_col']].lower(), axis=1)

    e = Example(om_df=om_df, col_dict=col_dict)
    e.add_equipment_labels()
    e.get_metrics()
    e.plot_confusion_matrix(output_filepath='examples/example_conf_mat.png', title='Confusion Matrix of Actual and Predicted Asset Labels')
