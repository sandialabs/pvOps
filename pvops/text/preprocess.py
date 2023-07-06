import re
import nltk

import numpy as np
import datefinder
import traceback
from datetime import datetime, timedelta

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
    col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the get_dates fn

        - data : string, should be assigned to associated column which stores the text logs
        - eventstart : string, should be assigned to associated column which stores the log submission datetime
        - save_data_column : string, should be assigned to associated column where the processed text should be stored
        - save_date_column : string, should be assigned to associated column where the extracted dates from the text should be stored

    print_info : bool
        Flag indicating whether to print information about the preprocessing progress
    extract_dates_only : bool
        If True, return after extracting dates in each ticket
        If False, return with preprocessed text and extracted dates

    Returns
    -------
    df : DataFrame
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
            document = text_remove_nondate_nums(
                document, PRINT_INFO=print_info)
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
            except:
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
            date = datetime.strptime(
                row[EVENTSTART_COLUMN], "%Y-%m-%d %H:%M:%S")

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


def get_dates(
    document, om_df, ind, col_dict, print_info, infer_date_surrounding_rows=True
):
    """Extract dates from the input document.

    This method is utilized within ``preprocessor.py``. For an easy way to extract dates, utilize the preprocessor and set
    extract_dates_only = True.

    Parameters
    ----------
    document : str
        String representation of a document
    om_df : DataFrame
        A pandas dataframe containing O&M data, which contains at least the columns within col_dict.
    ind : integer
        Designates the row of the dataframe which is currently being observed. This is required because if the
        current row does not have a valid date in the `eventstart`, then an iterative search is conducted
        by first starting at the nearest rows.
    col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the get_dates fn

        - data : string, should be assigned to associated column which stores the text logs
        - eventstart : string, should be assigned to associated column which stores the log submission datetime

    print_info : bool
        Flag indicating whether to print information about the preprocessing progress
    infer_date_surrounding_rows : bool
        If True, utilizes iterative search in dataframe to infer the datetime from surrounding rows if the current row's date value is nan
        If False, does not utilize the base datetime. Consequentially, today's date is used to replace the missing parts of the datetime.
        Recommendation: set True if you frequently publish documents and your dataframe is ordered chronologically

    Returns
    -------
    list
        List of dates found in text
    """

    DATA_COLUMN = col_dict["data"]
    EVENTSTART_COLUMN = col_dict["eventstart"]

    try:
        row = om_df.iloc[ind]
        if print_info:
            print("Start time: ", row[EVENTSTART_COLUMN])

        no_base_date_found = False
        if isinstance(row[EVENTSTART_COLUMN], float) and np.isnan(
            row[EVENTSTART_COLUMN]
        ):
            # Was given a NaN value as event start date, so look before an after this row for a date

            if infer_date_surrounding_rows:
                no_base_date_found = True

            else:
                if print_info:
                    print("found nan")
                find_valid = False

                w = 1
                om_df_len = len(om_df.index)

                while find_valid is False and no_base_date_found is False:
                    ind_behind = ind - w
                    ind_ahead = ind + w

                    if ind_behind >= 0:
                        if print_info:
                            print("checking index: ", ind_behind)
                        row_behind = om_df.iloc[ind_behind]
                        if isinstance(
                            row_behind[EVENTSTART_COLUMN], float
                        ) and np.isnan(row_behind[EVENTSTART_COLUMN]):
                            pass
                        else:
                            basedate = list(
                                datefinder.find_dates(
                                    row_behind[EVENTSTART_COLUMN])
                            )[0]
                            find_valid = True
                            continue

                    if ind_ahead < om_df_len:
                        if print_info:
                            print("checking index: ", ind_ahead)
                        row_ahead = om_df.iloc[ind_ahead]
                        if isinstance(row_ahead[EVENTSTART_COLUMN], float) and np.isnan(
                            row_ahead[EVENTSTART_COLUMN]
                        ):
                            pass
                        else:
                            basedate = list(
                                datefinder.find_dates(
                                    row_ahead[EVENTSTART_COLUMN])
                            )[0]
                            find_valid = True
                            continue  # not needed but consistent syntax

                    if ind_ahead > om_df_len and ind_behind < 0:
                        no_base_date_found = True
                    w += 1

        else:
            basedate = list(datefinder.find_dates(row[EVENTSTART_COLUMN]))[0]

        if no_base_date_found:
            matches = list(datefinder.find_dates(document))
        else:
            matches = list(datefinder.find_dates(document, base_date=basedate))

    except Exception as e:
        matches = []
        if print_info:
            print(traceback.format_exc())
            print("\n")
            print("date")
            print(row[EVENTSTART_COLUMN])
            print("proc")
            print(document)
            print("raw")
            print(om_df.iloc[[ind]][DATA_COLUMN].tolist()[0])
            print(ind)
            print(e)
            print(traceback.format_exc())

    valid_matches = []
    # valid_inds = []
    for mtch in matches:
        try:
            if (mtch > datetime.strptime("01/01/1970", "%m/%d/%Y")) and (
                mtch < datetime.now() + timedelta(days=365 * 100)
            ):

                valid_matches.append(mtch)

        except Exception as e:
            if print_info:
                print(e)

    return valid_matches


def text_remove_nondate_nums(document, PRINT_INFO=False):
    """Conduct initial text processing steps to prepare the text for date
    extractions. Function mostly uses regex-based text substitution to
    remove numerical structures within the text, which may be mistaken
    as a date by the date extractor.

    Parameters
    ----------
    document : str
        String representation of a document
    PRINT_INFO : bool
        Flag indicating whether to print information about the preprocessing
        progress

    Returns
    -------
    string
        string of processed document
    """

    if PRINT_INFO:
        print()
        print()
        print("IN: ", document)

    # Remove URLs
    find_URL = r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))"""
    document = re.sub(find_URL, " ", document)

    regexs = [
        r"\d+(\%|\s\%|\bpercent\b|\s\bpercent\b)",  # Take out 'd%', 'd %'
        r"(#\s|#)\d+",  # '#d', '# d'
        # r'-?\d+(,\d+)+(].]d*)?', # take out lists of numbers with no space
        r"-?\d+(,\d+)+(\.\d*)?",  # take out lists of numbers with no space
        # r'-?\d+(,\s\d+)+(].]d*)?', # take out list of numbers with space
        r"\s\d{3}\s",  # numeric with 3 digits
        r"\b(0|00|1[3-9]|[2-9]\d)\b-\d{4}",  # [1-12]-4DIGIT  allowed only
        r"\s+\d+\.\d+\s+",  # e.g.: 10.1 and 10.2 with space before and after
        # Take out numbers longer than 8 digits (8 because datetimes 20190320 should stay)
        r"\d{9,}",
        r"\d+[.]+\d+[.]\d+[.][\d?]",  # Take out IP numbers
        # Take out single digit-hyphen trios e.g. 3-1-4  but leave 10-20-18 (possible date)
        r"\d-\d-\d",
        # Take out single digit-hyphen trios e.g. 3-1-4  but leave 10-20-18 (possible date)
        r"\d[.]\d[.]\d",
        r"\d+(\.\d*)?\s*[kK]?[wW]\s",
        r"\b(?!([jJ]an(uary)?|[fF]eb(r)?(uary)?|[mM]ar(ch)?|[aA]pr(il)?|[mM]ay|[jJ]un(e)?|[jJ]ul(y)?|[aA]ug(ust)?|[sS]ep(t)?(ember)?|[oO]ct(ober)?|[nN]ov(ember)?|[dD]ec(ember)?\b))[a-zA-Z]+-\d+",
        # ^ take out e.g. webbox-10
        r"[\w\.-]+@[\w\.-]+\.\w+",  # take out email addresses
        # take out phone numbers
        r"(\s\d{3}[-\.\s]?\d{3}[-\.\s]?\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]?\d{4}|\s\d{3}[-\.\s]\d{4})",
        # e.g. neff - cb 2.1b.16 - forced outage ; unknown. at 1645 26-jun cb 2.1b.16 offline.. 0000 - unknown
        r"\s\d+[.]\d+\D+[.]\d+\s",
        r"\s\D[.]\d[,\s]",
    ]

    replacements = [
        "",
        "",
        "",
        " ",
        " ",
        " ",
        "",
        "",
        "",
        "",
        " kW",
        " ",
        " ",
        " ",
        " ",
        " ",
    ]
    document = document.center(len(document) + 2)  # add spaces on either side
    for regex, repl in zip(regexs, replacements):
        # print('\t',regex)
        document = re.sub(regex, repl, document)
        # print('\t',words)

    if PRINT_INFO:
        print("SUB1:", document)
    # Decision to change all hyphens (-) to 'to'
    # to get rid of invalid timezone extrapolations

    document = str(document).lower()
    # print('prechk:',words)
    document = nltk.word_tokenize(document)

    if PRINT_INFO:
        print("TOKENED: ", document)

    # Remove single-character tokens (mostly punctuation)
    document = [word for word in document if len(word) > 1]
    if PRINT_INFO:
        print("FLTRD: ", document)
    document = " ".join(document)
    if PRINT_INFO:
        print("JOINED: ", document)

    # print('chkpt: ',words)
    regexs = [
        r"\d+(\%|\s\%|\bpercent\b|\s\bpercent\b)",  # Take out 'd%', 'd %'
        r"(#\s|#)\d+",  # '#d', '# d'
        r"\d+(,\d+)+(].]d*)?",  # take out lists of numbers with no space
        r"\d+(,\s\d+)+(].]d*)?",  # take out list of numbers with space
        r"\s\d{3}\s",  # numeric with 3 digits
        # [1-12]-4DIGIT  allowed only: 91-1010
        r"\s\b(0|00|1[3-9]|[2-9]\d)\b[-/]\d{4}\s",
        # 4DIGIT-[1-12]  allowed only: 4301/43
        r"\s\d{4}[-/]\b(0|00|1[3-9]|[2-9]\d)\b\s",
        r"\s\d+\[.]\d+\s",  # e.g.: ' 10.1 ' and 10.2
        # Take out numbers longer than 8 digits (8 because datetimes
        # 20190320 should stay)
        r"\d{9,}",
        r"\s\D[.]\s",  # Take out " m. " for maybe, ' c. ' for cerca, etc.
        # Take out 123/29 because not a date format, usually indicating
        # temperature/etc.
        r"\s\d{3}\/\d{2}\s",
        r"\s[a-zA-Z]+-[a-zA-Z]+\d\s",  # this and next one: e-a4 they are e7-1
        r"\s[a-zA-Z]\d+-\d+\s",
        r"\s[a-zA-Z]\d+\s",  # take out examples like `j23`
    ]
    replacements = ["", "", "", "", " ", " ",
                    " ", " ", "", " ", " ", " ", " ", " "]

    document = document.center(len(document) + 2)  # add spaces on either side
    for regex, repl in zip(regexs, replacements):
        document = re.sub(regex, repl, document)

    if PRINT_INFO:
        print("TO DFINDER: ", document)

    return document


def text_remove_numbers_stopwords(document, lst_stopwords):
    """Conduct final processing steps after date extraction

    Parameters
    ----------
    document : str
        String representation of a document
    lst_stopwords : list
        List of stop words which will be filtered in final preprocessing step

    Returns
    -------
    string
        string of processed document
    """

    for char in "<>,.*?!/\\:\"'@#$%^&(){}[]|~`_-":
        document = document.replace(char, " ")

    # many documents use ; or - as sentence partitioners
    # for char in ';-':
    # document = document.replace(char,'')

    rem_num = re.sub("[0-9]+", "", document)

    # remove all spaces
    document_tok = nltk.word_tokenize(rem_num)
    document = [i for i in document_tok if i not in lst_stopwords]
    document = " ".join(document)

    return document


def get_keywords_of_interest(document_tok, reference_df, reference_col_dict):
    """Find keywords of interest in list of strings from reference dict.

    If keywords of interest given in a reference dict are in the list of
    strings, return the keyword category, or categories. For example,
    if the string 'inverter' is in the list of text, return ['inverter'].

    Parameters
    ----------
    document_tok : list of str
        Tokenized text, functionally a list of string values.
    reference_df : DataFrame
        Holds columns that define the reference dictionary to search for keywords of interest,
        Note: This function can currently only handle single words, no n-gram functionality.
    reference_col_dict : dict of {str : str}
        A dictionary that contains the column names that describes how
        referencing is going to be done

        - reference_col_from : string, should be assigned to
          associated column name in reference_df that are possible input reference values
          Example: pd.Series(['inverter', 'invert', 'inv'])
        - reference_col_to : string, should be assigned to
          associated column name in reference_df that are the output reference values
          of interest
          Example: pd.Series(['inverter', 'inverter', 'inverter'])

    Returns
    -------
    included_equipment: list of str
        List of keywords from reference_dict found in list_of_txt, can be more than one value.
    """
    REFERENCE_COL_FROM = reference_col_dict["reference_col_from"]
    REFERENCE_COL_TO = reference_col_dict["reference_col_to"]

    reference_dict = dict(
        zip(reference_df[REFERENCE_COL_FROM], reference_df[REFERENCE_COL_TO])
    )

    # keywords of interest
    overlap_keywords = reference_dict.keys() & document_tok
    included_keywords = list({reference_dict[x] for x in overlap_keywords})
    return included_keywords
