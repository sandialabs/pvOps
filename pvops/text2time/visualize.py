"""These functions focus on visualizing the processed O&M and production data"""
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator


def visualize_counts(om_df, om_col_dict, count_var, fig_sets):
    """
    Produces a seaborn countplot of an O&M categorical column using sns.countplot()

    Parameters
    ----------
    om_df : DataFrame
        A data frame corresponding to the O&M data after having been pre-processed
        to address NANs and date consistency, and after applying the ``om_summary_stats`` function.
        This data frame needs at least the columns specified in om_col_dict.
    om_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the O&M data

        - **siteid** (*string*), should be assigned to column name for associated site-ID in om_df.
        - **modatestart** (*string*), should be assigned to column name desired for month of
          event start. This column is calculated by ``om_summary_stats``

    count_var:str
        Column name that contains categorical variable to be plotted
    fig_sets : dict
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


def visualize_categorical_scatter(om_df, om_col_dict, cat_varx, cat_vary, fig_sets):
    """
    Produces a seaborn categorical scatter plot to show 
    the relationship between an O&M numerical column and 
    a categorical column using sns.catplot()

    Parameters
    ----------
    om_df : DataFrame
        A data frame corresponding to the O&M data after having been 
        pre-processed to address NANs and date consistency, and after 
        applying the ``om_summary_stats`` function.
        This data frame needs at least the columns specified 
        in om_col_dict.
    om_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the O&M data

        - **eventdur** (*string*), should be assigned to column name desired for repair duration.
          This column is calculated by ``om_summary_stats``
        - **agedatestart** (*string*), should be assigned to column name desired for age of
          site when event started. This column is calculated by ``om_summary_stats``

    cat_varx : str
        Column name that contains categorical variable to be plotted
    cat_vary : str
        Column name that contains numerical variable to be plotted
    fig_sets : dict
        A dictionary that contains the settings to be used for the
        figure to be generated, and those settings should include:

        - **figsize** (*tuple*), which is a tuple of the figure settings (e.g. *(12,10)* )
        - **fontsize** (*int*), which is the desired font-size for the figure

    Returns
    -------
    None
    """
    # assigning dictionary items to local variables for cleaner code
    om_rep_dur = om_col_dict["eventdur"]
    om_age_st = om_col_dict["agedatestart"]

    my_figsize = fig_sets["figsize"]
    my_fontsize = fig_sets["fontsize"]

    hue = cat_varx

    sns.catplot(x=cat_varx, y=cat_vary, data=om_df, hue=hue)

    ax = plt.gca()
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        xticks,
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

    if cat_vary == om_age_st:
        ttl_key = "Age of System at Event Occurence"
    elif cat_vary == om_rep_dur:
        ttl_key = "Duration of Event"
    else:
        ttl_key = cat_vary

    ax.set_xlabel("Site ID", fontsize=my_fontsize, fontweight="bold")
    ax.set_ylabel("Days", fontsize=my_fontsize, fontweight="bold")
    ax.set_title(ttl_key, fontsize=my_fontsize)

    fig = plt.gcf()
    fig.set_size_inches(my_figsize)
    fig.tight_layout()

    return fig


def visualize_om_prod_overlap(
        prod_df,
        om_df,
        prod_col_dict,
        om_col_dict,
        prod_fldr,
        e_cumu,
        be_cumu,
        samp_freq="H",
        pshift=0.0,
        baselineflag=True
    ):
    """
    Creates Plotly figures of performance data overlaid with coinciding O&M tickets.
    A separate figure for each site in the production data frame (prod_df) is generated.

    Parameters
    ----------
    prod_df : DataFrame
        A data frame corresponding to the performance data after (ideally) having been
        processed by the perf_om_NA_qc and overlappingDFs functions. This data
        frame needs to contain the columns specified in prod_col_dict.
    om_df : DataFrame
        A data frame corresponding to the O&M data after (ideally) having been processed
        by the perf_om_NA_qc and overlappingDFs functions. This data frame needs
        to contain the columns specified in om_col_dict.
    prod_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the production data

        - **siteid** (*string*), should be assigned to associated site-ID column name in
          prod_df
        - **timestamp** (*string*), should be assigned to associated time-stamp column name in
          prod_df
        - **energyprod** (*string*), should be assigned to associated production column name in
          prod_df
        - **irradiance** (*string*), should be assigned to associated irradiance column name in
          prod_df. Data should be in [W/m^2].

    om_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the O&M data

        - **siteid** (*string*), should be assigned to column name for user's site-ID
        - **datestart** (*string*), should be assigned to column name for user's
          O&M event start-date
        - **dateend** (*string*), should be assigned to column name for user's O&M event end-date
        - **workID** (*string*), should be assigned to column name for user's O&M unique event ID
        - **worktype** (*string*), should be assigned to column name for user's
          O&M ticket type (corrective, predictive, etc)
        - **asset** (*string*), should be assigned to column name for affected asset in user's
          O&M ticket

    prod_fldr : str
        Path to directory where plots should be saved.
    e_cumu : bool
        Boolean flag that specifies whether the production (energy output)
        data is input as cumulative information ("True") or on a per time-step basis ("False").
    be_cumu : bool
        Boolean that specifies whether the baseline production data is input as cumulative
        information ("True") or on a per time-step basis ("False").
    samp_freq : str
        Specifies how the performance data should be resampled.
        String value is any frequency that is valid for pandas.DataFrame.resample().
        For example, a value of 'D' will resample on a daily basis, and a
        value of 'H' will resample on an hourly basis.
    pshift : float
        Value that specifies how many hours the performance data
        should be shifted by to help align performance data with O&M data.
        Mostly necessary when resampling frequencies are larger than an hour
    baselineflag : bool
        Boolean that specifies whether or not to display the baseline (i.e.,
        expected production profile) as calculated with the irradiance data
        using the baseline production data. A value of 'True' will display the
        baseline production profile on the generated Plotly figures, and a value of
        'False' will not.

    Returns
    -------
    list
        List of Plotly figure handles generated by function for each site within prod_df.
    """

    # assigning dictionary items to local variables for cleaner code
    om_site = om_col_dict["siteid"]
    om_date_s = om_col_dict["datestart"]
    om_date_e = om_col_dict["dateend"]
    om_wo_id = om_col_dict["workID"]
    om_wtype = om_col_dict["worktype"]
    om_asset = om_col_dict["asset"]

    prod_site = prod_col_dict["siteid"]
    prod_ts = prod_col_dict["timestamp"]
    prod_ener = prod_col_dict["energyprod"]
    prod_baseline = prod_col_dict[
        "baseline"
    ]  # if none is provided, using iec_calc() is recommended

    # creating local dataframes to not modify originals
    prod_df = prod_df.copy()
    om_df = om_df.copy()

    # Setting multi-indices for ease of plotting
    prod_df.set_index([prod_site, prod_ts], inplace=True)
    prod_df.sort_index(inplace=True)
    om_df.set_index([om_site, om_date_s], inplace=True)
    om_df.sort_index(inplace=True)

    figs = []
    for i in prod_df.index.get_level_values(0).unique():
        # Resampling the performance data to obtain daily energy production
        # (different between cumulative and non-cumulative energy output)

        if e_cumu:
            # energy data is cumulative over time, so take difference between
            # largest and smallest value on any given day
            tstep = np.diff(prod_df.loc[i].index)[2] / np.timedelta64(1, "s")
            if samp_freq == "H" and tstep >= 3599:  # 3599 to consider roundoff error
                enrg_site = prod_df.loc[i, prod_ener].diff()
            elif samp_freq == "T" and tstep >= 59.9:
                enrg_site = prod_df.loc[i, prod_ener].diff()
            else:
                enrg_site = (
                    prod_df.loc[i, prod_ener].resample(samp_freq, label="left").max()
                    - prod_df.loc[i, prod_ener].resample(samp_freq, label="left").min()
                )
        else:
            # energy data is given on a per TIMESTEP basis, therefore...
            enrg_site = (
                prod_df.loc[i, prod_ener].resample(samp_freq, label="left").sum()
            )

        # Resampling baselineE and assigning to separate variable to not resample entire data frame.
        if be_cumu:  # baseline energy cumulative over time
            baseline_site = (
                prod_df.loc[i, prod_baseline].resample(samp_freq, label="left").max()
                - prod_df.loc[i, prod_baseline].resample(samp_freq, label="left").min()
            )
        else:  # baseline energy is on a per time-step basis
            baseline_site = (
                prod_df.loc[i, prod_baseline].resample(samp_freq, label="left").sum()
            )

        # shifting time for prod_df and baseline
        baseline_site.index += pd.Timedelta(hours=pshift)
        enrg_site.index += pd.Timedelta(hours=pshift)

        # finding where energy dips:  first by location/index (by integer, not
        # index location => use iloc), and then by converting to dates using original index
        edips_all = find_peaks(enrg_site * -1)[0]
        edips_all_dates = enrg_site.index[edips_all]

        # Finding the corresponding closest dip to each OM om_date_s for
        # each ticket:  first by location/index, and then by converting those
        # indices to dates
        edips_nearom_indices = [
            np.argmin(abs(edips_all_dates - xx)) for xx in om_df.loc[i].index
        ]
        edips_nearom_dates = edips_all_dates[edips_nearom_indices]

        # Adding the nearest performance-dip-date to the OM data frame
        om_df.loc[i, "corr_perfDip"] = edips_nearom_dates

        # Taking largest value of daily output(perf-data) to create a [ficticious]
        # plot-value/column for OM-data => "_h" implies hover text
        om_df.loc[i, "perfval_plotcol"] = enrg_site.max()
        om_start_h = 0.75  # To place StartDate points in visible region
        om_end_h = 0.5  # To place EndDate points below the StartDate points
        om_reg_h = 1.05  # To make the om-region slightly higher than the perf data

        # Correction for om-region if baseline for production data is plotted
        if baselineflag:
            om_reg_hcorr = baseline_site.max() / enrg_site.max()
        else:
            om_reg_hcorr = 1.0

        # initializing plotly-figure
        fig = go.Figure(
            layout_yaxis_range=[-5, enrg_site.max() * om_reg_h * om_reg_hcorr]
        )

        # plotting all Perf data for i-th site (captured in enrg_site)
        if samp_freq == "D":
            perf_name = "Daily Energy"
            baseline_name = "Daily Baseline"
        elif samp_freq == "H":
            perf_name = "Hourly Energy"
            baseline_name = "Hourly Baseline"
        fig.add_trace(go.Scatter(x=enrg_site.index, y=enrg_site.values, 
                                 name=perf_name))
        if baselineflag:
            fig.add_trace(
                go.Scatter(
                    x=baseline_site.index, y=baseline_site.values, name=baseline_name
                )
            )

        # For loop to add shaded regions for each ticket, where left side of the region corresponds
        # to the EventStart (index of om data in this case), and right side of region corresponds
        # to the EventEnd.  These two dates make the edges of the region, x below.
        for j in range(len(om_df.loc[i])):
            fig.add_trace(
                dict(
                    type="scatter",
                    x=[om_df.loc[i].index[j], om_df.loc[i, om_date_e][j]],
                    y=om_df.loc[i, "perfval_plotcol"].values[0:2]
                    * om_reg_h
                    * om_reg_hcorr,
                    mode="markers+lines",
                    line=dict(width=0),
                    marker=dict(size=[0, 0]),
                    fill="tozeroy",
                    fillcolor="rgba(190,0,0,.15)",
                    hoverinfo="none",
                    showlegend=False,
                    name="OM Ticket",
                )
            )

        # Adding EventStart Points with hover-text
        fig.add_trace(
            go.Scatter(
                x=om_df.loc[i].index,
                y=om_df.loc[i, "perfval_plotcol"].values * om_start_h,
                mode="markers",
                hovertemplate="Start: "
                + "%{x} <br>"
                + "WO#: "
                + om_df.loc[i, om_wo_id].astype(str)
                + "<br>"
                + "Type: "
                + om_df.loc[i, om_wtype].astype(str)
                + "<br>"
                + "Asset: "
                + om_df.loc[i, om_asset].fillna("Asset_NA").astype(str)
                + "<br>"
                + "Nearest Prod Dip:  "
                + om_df.loc[i, "corr_perfDip"].dt.strftime("%b %d, %Y"),
                name="OM_start",
            )
        )

        # Adding EventEnd Points with hover-text
        fig.add_trace(
            go.Scatter(
                x=om_df.loc[i, om_date_e],
                y=om_df.loc[i, "perfval_plotcol"].values * om_end_h,
                mode="markers",
                hovertemplate="End: "
                + "%{x} <br>"
                + "WO#: "
                + om_df.loc[i, om_wo_id].astype(str)
                + "<br>"
                + "Type: "
                + om_df.loc[i, om_wtype].astype(str)
                + "<br>"
                + "Asset: "
                + om_df.loc[i, om_asset].fillna("Asset_NA").astype(str),
                name="OM_end",
            )
        )

        # Setting y-axes and title
        fig.update_yaxes(title_text="Energy Delivered (kWh)")
        fig.update_layout(title_text="Site: " + i)

        # appending fig object to figs list
        figs.append(fig)
        # Saving Figure
        fig.write_html(prod_fldr + "/" + i + ".html")

    # Resetting index before completion of function since DFs are mutable
    prod_df.reset_index(inplace=True)
    om_df.reset_index(inplace=True)

    return figs
