import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator


def visualize_counts(om_df, om_col_dict, count_var, fig_sets):
    """
    Produces a seaborn countplot of an O&M categorical column using sns.countplot()


    Parameters

    ----------
    om_df: DataFrame
        A data frame corresponding to the O&M data after having been pre-processed
        to address NANs and date consistency, and after applying the ``om_summary_stats`` function.
        This data frame needs at least the columns specified in om_col_dict.

    om_col_dict: dict of {str : str}
        A dictionary that contains the column names relevant for the O&M data

        - **siteid** (*string*), should be assigned to column name for associated site-ID in om_df.
        - **modatestart** (*string*), should be assigned to column name desired for month of event start. This column is calculated by ``om_summary_stats``

    count_var:str
        Column name that contains categorical variable to be plotted

    sv_nm: str
        The name under which the plot should be saved as

    fig_sets: dict
        A dictionary that contains the settings to be used for the
        figure to be generated, and those settings should include:

        - **figsize** (*tuple*), which is a tuple of the figure settings (e.g. *(12,10)* )
        - **fontsize** (*int*), which is the desired font-size for the figure

    Returns

    -------
    None

    """
    # assigning dictionary items to local variables for cleaner code
    om_site = om_col_dict["siteid"]
    om_mo_st = om_col_dict["modatestart"]

    my_figsize = fig_sets["figsize"]
    my_fontsize = fig_sets["fontsize"]

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
    ax = sns.countplot(x=count_var, data=om_df, hue=hue)

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment="center",
        fontweight="medium",
        fontsize=my_fontsize - 2,
    )

    yticks = ax.get_yticks()
    yticks = [int(i) for i in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(
        yticks,
        verticalalignment="center",
        fontweight="medium",
        fontsize=my_fontsize - 2,
    )

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(count_var, fontsize=my_fontsize, fontweight="bold")
    ax.set_ylabel("Count", fontsize=my_fontsize, fontweight="bold")
    ax.set_title("Number of Reported Events by " + ttl_key, fontsize=my_fontsize)
    fig.tight_layout()

    return fig
