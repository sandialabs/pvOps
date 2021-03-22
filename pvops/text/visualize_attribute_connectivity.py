import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_attribute_connectivity(
    om_df,
    om_col_dict,
    figsize=(40, 20),
    attribute_colors=["lightgreen", "cornflowerblue"],
    edge_width_scalar=10,
    graph_aargs={},
):
    """Visualize a knowledge graph which shows the frequency of combinations between attributes
    ``ATTRIBUTE1_COL`` and ``ATTRIBUTE2_COL``

    Parameters

    ----------
    om_df : DataFrame
        A pandas dataframe containing O&M data, which contains columns specified in om_col_dict
    om_col_dict: dict of {str : str}
        A dictionary that contains the column names that describes how remapping is going to be done
        - **attribute1_col** (*string*), should be assigned to associated column name for first attribute of interest in om_df
        - **attribute2_col** (*string*), should be assigned to associated column name for second attribute of interest in om_df
    figsize : tuple
        Figure size
    attribute_colors : list
        List of two strings which designate the colors for Attribute1 and Attribute 2, respectively.
    edge_width_scalar : numeric
        Weight utilized to cause dynamic widths based on number of connections between Attribute 1
        and Attribute 2.
    graph_aargs
        Optional, arguments passed to networkx graph drawer.
        Suggested attributes to pass:
            with_labels=True
            font_weight='bold'
            node_size=19000
            font_size=35
            node_color='darkred'
            font_color='red'

    Returns

    -------
    Matplotlib figure instance,
    networkx EdgeView object
        i.e. [('A', 'X'), ('X', 'B'), ('C', 'Y'), ('C', 'Z')]
    """
    df = om_df.copy()
    ATTRIBUTE1_COL = om_col_dict["attribute1_col"]
    ATTRIBUTE2_COL = om_col_dict["attribute2_col"]

    nx_data = []
    for a in np.unique(df[ATTRIBUTE1_COL].tolist()):
        df_iter = df[df[ATTRIBUTE1_COL] == a]
        for i in np.unique(df_iter[ATTRIBUTE2_COL].tolist()):
            w = len(df_iter[df_iter[ATTRIBUTE2_COL] == i])
            nx_data.append([a, i, w])

    unique_df = pd.DataFrame(nx_data, columns=[ATTRIBUTE1_COL, ATTRIBUTE2_COL, "w"])

    G = nx.from_pandas_edgelist(unique_df, ATTRIBUTE1_COL, ATTRIBUTE2_COL, "w")
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Connectivity between {ATTRIBUTE1_COL} and {ATTRIBUTE2_COL}",
        fontsize=50,
        y=1.08,
        fontweight="bold",
    )

    color_map = []
    for node in G:
        if node in np.unique(df[ATTRIBUTE2_COL].tolist()):
            color_map.append(attribute_colors[1])
        else:
            color_map.append(attribute_colors[0])

    edges = G.edges()
    weights = [G[u][v]["w"] for u, v in edges]
    weights = np.array(weights)
    weights = list(1 + (edge_width_scalar * weights / weights.max()))  # scale 1-11
    nx.draw_shell(G, edges=edges, width=weights, node_color=color_map, **graph_aargs)

    return fig, edges