import numpy as np
import datefinder
import traceback
from datetime import datetime, timedelta


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
    col_dict: dict of {str : str}
        A dictionary that contains the column names relevant for the get_dates fn
        - **data** (*string*), should be assigned to associated column which stores the text logs
        - **eventstart** (*string*), should be assigned to associated column which stores the log submission datetime
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
                                datefinder.find_dates(row_behind[EVENTSTART_COLUMN])
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
                                datefinder.find_dates(row_ahead[EVENTSTART_COLUMN])
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
