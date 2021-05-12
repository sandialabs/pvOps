# Standard
import datetime
import warnings

# Third party

import pandas as pd
import scipy

warnings.filterwarnings("ignore")


def find_and_break_days_or_hours(df, filter_bool, min_count_per_day=8, frequency='days', print_info=False):
    index_list = []
    day_hour_list = []
    prev = 0
    for index, j in enumerate(df.index):
        if str(type(j)) != "<class 'str'>":
            print(type(j))
            print(j)
            print(df.loc[j])
        if frequency == "days":
            curr = int(datetime.datetime.strptime(
                j, "%m/%d/%Y %H:%M:%S %p").strftime("%d"))
            frq = datetime.datetime.strptime(
                j, "%m/%d/%Y %H:%M:%S %p").strftime("%m/%d/%Y")
        elif frequency == "hours":
            curr = int(datetime.datetime.strptime(
                j, "%m/%d/%Y %H:%M:%S %p").strftime("%H"))
            frq = datetime.datetime.strptime(
                j, "%m/%d/%Y %H:%M:%S %p").strftime("%m/%d/%Y %H")

        if curr != prev:
            index_list.append(index)
            day_hour_list.append(frq)
        prev = curr

    cut_results = []
    # Break df into days
    for k in range(len(index_list)):
        if k == (len(index_list)-1):
            # append last day
            cut_results.append(df[index_list[k]:-1])
        else:
            cut_results.append(df[index_list[k]:index_list[k+1]])

    cut_results[-1] = pd.concat([cut_results[-1], df.iloc[[-1]]])

    if filter_bool:
        # NUMBER OF POINTS PER DAY MUST BE AT CERTAIN THRESHOLD
        checked_cut_results = []
        checked_index_list = []
        checked_day_hour_list = []
        dropped_days = []
        for i in range(len(cut_results)):
            # TODO: also check standard deviation for flatlining
            if len(cut_results[i]) > min_count_per_day:
                checked_cut_results.append(cut_results[i])
                checked_index_list.append(index_list[i])
                checked_day_hour_list.append(day_hour_list[i])
            else:
                dropped_days.append(cut_results[i].index[0])

        cut_results = checked_cut_results
        index_list = checked_index_list
        day_hour_list = checked_day_hour_list

        if len(dropped_days) > 0:
            df = pd.concat(cut_results)

        else:
            if print_info:
                print("No need to alter df because no dropped days detected")

    return index_list, day_hour_list, cut_results, df
