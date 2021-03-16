import matplotlib.pyplot as plt
import seaborn as sns
import os

def catscat_fig(om_df, om_col_dict, cat_varx, cat_vary, sv_nm, fig_sets):
    '''
    Produces a seaborn categorical scatter plot to show the relationship between
    an O&M numerical column and a categorical column using sns.catplot()


    Parameters
    
    ----------
    om_df: DataFrame
        A data frame corresponding to the O&M data after having been processed
        by the perf_om_NA_qc, overlappingDFs, and new_om_metrics functions. This data frame needs
        at least the columns specified in om_col_dict.
 
    om_col_dict: dict of {str : str}
        A dictionary that contains the column names relevant for the O&M data
            
        - **repairdur** (*string*), should be assigned to column name desired for repair duration (calculated)
        - **agedatestart** (*string*), should be assigned to column name desired for age of site when event started (calculated) 
    
    cat_varx: str
        Column name that contains categorical variable to be plotted
        
    cat_vary: str
        Column name that contains numerical variable to be plotted
        
    sv_nm: str
        Name under which the plot should be saved as
        
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
    
    '''
    #assigning dictionary items to local variables for cleaner code
    om_rep_dur = om_col_dict['eventdur']
    om_age_st = om_col_dict['agedatestart']

    my_figsize = fig_sets['figsize']
    my_fontsize = fig_sets['fontsize']
    my_savedpi = fig_sets['savedpi']
    sv_loc = fig_sets['save_loc']

    sns.catplot(x=cat_varx, y=cat_vary, data=om_df)

    ax = plt.gca()
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='center',
        fontweight='medium',
        fontsize='large'
    )

    if cat_vary == om_age_st:
        ttl_key = 'Age of System at Event Occurence'
    elif cat_vary == om_rep_dur:
        ttl_key = 'Duration of Event'
    else:
        ttl_key = cat_vary

    ax.set_xlabel('Site ID', fontsize=my_fontsize)
    ax.set_ylabel('Days', fontsize=my_fontsize)
    ax.set_title(ttl_key, fontsize=my_fontsize)

    fig = plt.gcf()
    fig.set_size_inches(my_figsize)
    fig.tight_layout()
    fig.savefig(os.path.join(sv_loc,sv_nm), dpi=my_savedpi)
