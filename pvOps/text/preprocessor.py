from datetime import datetime
import numpy as np
import traceback

from text_remove_nondate_nums import text_remove_nondate_nums
from text_remove_numbers_stopwords import text_remove_numbers_stopwords
from get_dates import get_dates


def preprocessor(
    om_df, lst_stopwords, col_dict, print_info=False, extract_dates_only=False
):
    """Preprocessing function which processes the raw text data into processed text data and extracts dates

    Parameters

    ----------
    om_df : DataFrame
        A pandas dataframe containing O&M data, which contains at least the columns within col_dict.
    lst_stopwords : list
        List of stop words which will be filtered in final preprocessing step
    col_dict: dict of {str : str}
        A dictionary that contains the column names relevant for the get_dates fn
        - **data** (*string*), should be assigned to associated column which stores the text logs
        - **eventstart** (*string*), should be assigned to associated column which stores the log submission datetime
        - **save_data_column** (*string*), should be assigned to associated column where the processed text should be stored
        - **save_date_column** (*string*), should be assigned to associated column where the extracted dates from the text should be stored
    print_info : bool
        Flag indicating whether to print information about the preprocessing progress
    extract_dates_only : bool
        If True, return after extracting dates in each ticket
        If False, return with preprocessed text and extracted dates

    Returns

    -------
    DataFrame
        Contains the original columns as well as the processed data, located in columns defined by the inputs
    """

    DATA_COLUMN = col_dict["data"]
    EVENTSTART_COLUMN = col_dict["eventstart"]
    SAVE_DATA_COLUMN = col_dict["save_data_column"]
    SAVE_DATE_COLUMN = col_dict["save_date_column"]
    df = om_df.copy()

    dates_extracted = []
    clean_corpus = []
    # basedate_extracted = []

    n_nans = 0
    # n_success_date = 0
    n_fails_date = 0
    n_total = len(df.index)
    n_fails_prep = 0

    tally = 0
    lens = []

    df = om_df.copy()

    df.reset_index(drop=True, inplace=True)

    for ind, row in df.iterrows():

        document = row[DATA_COLUMN]
        if document == np.nan:
            n_nans += 1
        try:
            document = str(document).lower()
            document = text_remove_nondate_nums(document, PRINT_INFO=print_info)
            dts = get_dates(document, df, ind, col_dict, print_info)
            if print_info:
                print("Dates: ", dts)

            lens.append(len(dts))

        except Exception as e:
            print(e)
            dts = np.nan
            n_fails_date += 1

            lens.append(np.nan)

        dates_extracted.append(dts)

        if not extract_dates_only:
            try:
                out = text_remove_numbers_stopwords(document, lst_stopwords)
                clean_corpus.append(out)
            except Exception as e:
                print(traceback.format_exc())
                clean_corpus.append("")
                n_fails_prep += 1

    if print_info:
        print(
            f"len clean corpus: {len(clean_corpus)}, len deduced dates: {len(dates_extracted)}"
        )
        print(
            f"num_total {n_total}, num_nans {n_nans}, num_fails_date {n_fails_date}, num_fails_prep {n_fails_prep}, tally {tally}"
        )

    df[SAVE_DATE_COLUMN] = dates_extracted
    if not extract_dates_only:
        df[SAVE_DATA_COLUMN] = clean_corpus

    filtered_dates = []
    for ind, row in df.iterrows():
        nlp_dates = row[SAVE_DATE_COLUMN]

        if len(nlp_dates) == 0:
            filtered_dates.append(nlp_dates)
            continue

        try:
            date = datetime.strptime(row[EVENTSTART_COLUMN], "%Y-%m-%d %H:%M:%S")

            fltrd = []
            for dt in nlp_dates:
                # d = datetime.strptime(dt, '%m-%d-%Y %H:%M:%S')

                # if less than a year, include
                if abs((date - dt).total_seconds()) < 3.154e7:
                    fltrd.append(dt)
            filtered_dates.append(fltrd)

        except:
            # NaN values
            filtered_dates.append(nlp_dates)

    df[SAVE_DATE_COLUMN] = filtered_dates

    return df