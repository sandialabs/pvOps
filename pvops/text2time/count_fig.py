import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator


def count_fig(om_df, om_col_dict, count_var, sv_nm, fig_sets):
    """
    Produces a seaborn countplot of an O&M categorical column using sns.countplot()


    Parameters

    ----------
    om_df: DataFrame
        A data frame corresponding to the O&M data after having been processed
        by the perf_om_NA_qc and overlappingDFs functions. This data frame needs
        to have the columns specified in om_col_dict.

    om_col_dict: dict of {str : str}
        A dictionary that contains the column names relevant for the O&M data

        - **siteid** (*string*), should be assigned to column name for associated site-ID in om_df
        - **modatestart** (*string*), should be assigned to column name desired for month of event start (calculated)


    count_var:str
        Column name that contains categorical variable to be plotted

    sv_nm: str
        The name under which the plot should be saved as

    fig_sets: dict
        A dictionary that contains the settings to be used for the
        figure to be generated, and those settings should include:

        - **figsize** (*tuple*), which is a tuple of the figure settings (e.g. *(12,10)* )
        - **fontsize** (*int*), which is the desired font-size for the figure
        - **savedpi** (*int*), which is the resolution desired for the plot
        - **save_loc** (*string*), which is the path where the plot should be saved

    Returns

    -------
    None

    """
    # assigning dictionary items to local variables for cleaner code
    om_site = om_col_dict["siteid"]
    om_mo_st = om_col_dict["modatestart"]

    my_figsize = fig_sets["figsize"]
    my_fontsize = fig_sets["fontsize"]
    my_savedpi = fig_sets["savedpi"]
    sv_loc = fig_sets["save_loc"]

    # For plot title labels
    if count_var == om_site:
        ttl_key = "Site"
        hue = None
    elif count_var == om_mo_st:
        ttl_key = "Month"
        hue = om_site
    else:
        ttl_key = count_var
        hue = None

    fig = plt.figure(figsize=my_figsize)
    om_df[count_var] = om_df[count_var].astype("category")
    chart = sns.countplot(x=count_var, data=om_df, hue=hue)

    chart.set_xticklabels(
        chart.get_xticklabels(),
        rotation=45,
        horizontalalignment="center",
        fontweight="medium",
        fontsize="large",
    )

    chart.yaxis.set_major_locator(MaxNLocator(integer=True))
    chart.set_xlabel(count_var, fontsize=my_fontsize)
    chart.set_ylabel("Count", fontsize=my_fontsize)
    chart.set_title("Number of Reported Events by " + ttl_key, fontsize=my_fontsize)
    fig.tight_layout()
    fig.savefig(os.path.join(sv_loc, sv_nm), dpi=my_savedpi)
