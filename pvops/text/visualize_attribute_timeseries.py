import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd


def visualize_attribute_timeseries(
    om_df, om_col_dict, date_structure="%Y-%m", figsize=(12, 6), cmap_name="brg"
):
    """Visualize stacked bar chart of attribute frequency over time, where x-axis is time and y-axis is count, displaying separate bars
    for each label within the label column

    Parameters

    ----------
    om_df : DataFrame
        A pandas dataframe of O&M data, which contains columns in om_col_dict
    om_col_dict: dict of {str : str}
        A dictionary that contains the column names relevant for the get_dates fn
        - **label** (*string*), should be assigned to associated column name for the label/attribute of interest in om_df
        - **date** (*string*), should be assigned to associated column name for the dates relating to the documents in om_df
    date_structure : str
        Controls the resolution of the bar chart's timeseries
        Default: "%Y-%m". Can change to include finer resolutions (e.g., by including day, "%Y-%m-%d")
        or coarser resolutions (e.g., by year, "%Y")
    figsize : tuple
        Optional, figure size
    cmap_name : str
        Optional, color map name in matplotlib

    Returns

    -------
    Matplotlib figure instance
    """
    df = om_df.copy()
    LABEL_COLUMN = om_col_dict["label"]
    DATE_COLUMN = om_col_dict["date"]

    def restructure(vals, inds, ind_set):
        out = np.zeros(len(ind_set))
        for ind, val in zip(inds, vals):
            loc = ind_set.index(ind)
            out[loc] = val
        return out

    fig = plt.figure(figsize=figsize)
    asset_set = list(set(df[LABEL_COLUMN].tolist()))

    dates = df[DATE_COLUMN].tolist()
    assets_list = df[LABEL_COLUMN].tolist()

    full_date_list = [i.strftime(date_structure) for i in dates]
    datetime_list = [
        datetime.datetime.strptime(i, date_structure) for i in full_date_list
    ]
    date_set = list(set(datetime_list))
    date_set = sorted(date_set)
    date_set = [i.strftime(date_structure) for i in date_set]
    assets_list = np.array(assets_list)

    asset_sums = []
    index_sums = []
    for dt in date_set:
        inds = [i for i, x in enumerate(full_date_list) if x == dt]
        alist = assets_list[inds]

        index_sums += [dt] * len(alist)
        asset_sums += list(alist)

    asset_set = list(set(asset_sums))

    newdf = pd.DataFrame()
    newdf[LABEL_COLUMN] = asset_sums
    newdf[DATE_COLUMN] = index_sums

    cmap = plt.cm.get_cmap(cmap_name, len(asset_set))

    graphs = []
    for i, a in enumerate(asset_set):
        iter_ = newdf[newdf[LABEL_COLUMN] == a]
        valcounts = iter_[DATE_COLUMN].value_counts()
        valcounts.sort_index(inplace=True)
        vals = restructure(valcounts.values, valcounts.index, date_set)
        p = plt.bar(date_set, vals, color=cmap(i))
        graphs.append(p[0])

    plt.legend(graphs, list(asset_set))
    plt.xlabel("Month")
    plt.ylabel(f"Affected {LABEL_COLUMN} counts")
    plt.xticks(rotation=45)
    return fig