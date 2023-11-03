# visualizations
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import ConfusionMatrixDisplay
from networkx.algorithms import bipartite

# data structures
import numpy as np
import pandas as pd

# utils
import copy
import datetime
from collections import Counter

# Embedding
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize


def visualize_attribute_connectivity(
    om_df,
    om_col_dict,
    figsize=(20, 10),
    attribute_colors=["lightgreen", "cornflowerblue"],
    edge_width_scalar=10,
    graph_aargs={},
):
    """Visualize a knowledge graph which shows the frequency of combinations between attributes
    ``ATTRIBUTE1_COL`` and ``ATTRIBUTE2_COL``

    NOW USES BIPARTITE LAYOUT
    ATTRIBUTE2_COL is colored using a colormap.

    Parameters
    ----------
    om_df : DataFrame
        A pandas dataframe containing O&M data, which contains columns specified in om_col_dict
    om_col_dict : dict of {str : str}
        A dictionary that contains the column names to be used in
        visualization::

            {
                'attribute1_col' : string,
                'attribute2_col' : string
            }

    figsize : tuple
        Figure size, defaults to (20,10)
    attribute_colors : list[str]
        List of two strings which designate the colors for Attribute1 and Attribute 2, respectively.
    edge_width_scalar : numeric
        Weight utilized to scale widths based on number of connections between Attribute 1
        and Attribute 2. Larger values will produce larger widths, and smaller values will produce smaller widths.
    graph_aargs : dict
        Optional, arguments passed to networkx graph drawer.
        Suggested attributes to pass:

        - with_labels=True
        - font_weight='bold'
        - node_size=19000
        - font_size=35

    Returns
    -------
    Matplotlib axis,
    networkx graph
    """
    # initialize figure
    fig = plt.figure(figsize=figsize, facecolor='w', edgecolor='k')
    ax = plt.gca()

    # attribute column names
    ATTRIBUTE1_COL = om_col_dict["attribute1_col"]
    ATTRIBUTE2_COL = om_col_dict["attribute2_col"]

    ax.set_title(
        f"Connectivity between {ATTRIBUTE2_COL} and {ATTRIBUTE1_COL}",
        fontweight="bold",
    )

    # subset dataframe to relevant columns
    df_mask = (om_df[ATTRIBUTE1_COL].notna() == True) & (om_df[ATTRIBUTE2_COL].notna() == True)
    df = om_df.loc[df_mask].reset_index(drop=True)

    # obtain connectivity weights between attributes
    nx_data = {}
    for attr1 in np.unique(df[ATTRIBUTE1_COL].tolist()):
        df_iter = df[df[ATTRIBUTE1_COL] == attr1]
        for attr2 in np.unique(df_iter[ATTRIBUTE2_COL].tolist()):
            w = len(df_iter[df_iter[ATTRIBUTE2_COL] == attr2])
            nx_data[(attr1, attr2)] = w

    # create graph
    G = nx.Graph()
    G.add_nodes_from(df[ATTRIBUTE1_COL], bipartite=0)
    G.add_nodes_from(df[ATTRIBUTE2_COL], bipartite=1)
    G.add_edges_from(nx_data.keys())

    # rescale weights and add to graph as attribute
    max_weight = max(nx_data.values())
    weights = []
    for node1, node2 in nx_data:
        weight = nx_data[node1, node2]
        rescaled_weight = 1 + (edge_width_scalar * weight / max_weight)  # between 1 and edge_width_scalar+1
        G[node1][node2]["weight"] = rescaled_weight
        weights.append(rescaled_weight)

    # get bipartite positioning
    top_nodes = list(df[ATTRIBUTE2_COL].unique())
    pos = nx.drawing.layout.bipartite_layout(G, top_nodes, align='horizontal')

    # assign colors based on attribute column
    color_map = []
    for node in G:
        if node in np.unique(df[ATTRIBUTE2_COL].tolist()):
            color_map.append(attribute_colors[1])
        else:
            color_map.append(attribute_colors[0])

    nx.draw_networkx(
        G, 
        width=weights, 
        node_color=color_map, 
        pos=pos, 
        **graph_aargs)

    plt.show(block=False)

    return fig, G


def visualize_attribute_timeseries(
    om_df, om_col_dict, date_structure="%Y-%m", figsize=(12, 6), cmap_name="brg"
):
    """Visualize stacked bar chart of attribute frequency over time, where x-axis is time and y-axis is count, displaying separate bars
    for each label within the label column

    Parameters
    ----------
    om_df : DataFrame
        A pandas dataframe of O&M data, which contains columns in om_col_dict
    om_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant for the get_dates fn

        - **label** (*string*), should be assigned to associated column name for the label/attribute of interest in om_df
        - **date** (*string*), should be assigned to associated column name for the dates relating to the documents in om_df

    date_structure : str
        Controls the resolution of the bar chart's timeseries
        Default : "%Y-%m". Can change to include finer resolutions (e.g., by including day, "%Y-%m-%d")
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

    cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(len(asset_set))

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


def visualize_cluster_entropy(
    doc2vec, eval_kmeans, om_df, data_cols, ks, cmap_name="brg"
):
    """Visualize entropy of embedding space parition. Currently only supports doc2vec embedding.

    Parameters
    ----------
    doc2vec : Doc2Vec model instance
        Instance of gensim.models.doc2vec.Doc2Vec
    eval_kmeans : callable
        Callable cluster fit function
        For instance,

        .. code-block:: python

            def eval_kmeans(X,k):
                km = KMeans(n_clusters=k)
                km.fit(X)
                return km

    om_df : DataFrame
        A pandas dataframe containing O&M data, which contains columns specified in om_col_dict
    data_cols : list
        List of column names (str) which have text data.
    ks : list
        List of k parameters required for the clustering mechanic `eval_kmeans`
    cmap_name :
        Optional, color map

    Returns
    -------
    Matplotlib figure instance
    """
    df = om_df.copy()
    cols = data_cols

    fig = plt.figure(figsize=(6, 6))
    cmap = plt.cm.get_cmap(cmap_name, len(cols) * 2)

    for i, col in enumerate(cols):
        X = df[col].tolist()
        X = [x.lower() for x in X]

        tokenized_data = [word_tokenize(x) for x in X]

        doc2vec_data = [
            TaggedDocument(words=x, tags=[str(i)]) for i, x in enumerate(tokenized_data)
        ]
        model = copy.deepcopy(doc2vec)
        model.build_vocab(doc2vec_data)
        model.train(
            doc2vec_data, total_examples=model.corpus_count, epochs=model.epochs
        )
        X_doc2vec = [model.infer_vector(tok_doc) for tok_doc in tokenized_data]

        sse = []
        clusters = []
        for true_k in ks:
            km = eval_kmeans(X_doc2vec, true_k)
            sse.append(km.inertia_)
            clusters.append(km.labels_)
        plt.plot(
            ks, sse, color=cmap(2 * i), marker="o", label=f"Doc2Vec + {col} entropy"
        )

        vectorizer = TfidfVectorizer()
        X_tfidf = vectorizer.fit_transform(X)

        sse = []
        clusters = []
        for true_k in ks:
            km = eval_kmeans(X_tfidf, true_k)
            sse.append(km.inertia_)
            clusters.append(km.labels_)
        plt.plot(
            ks, sse, color=cmap(2 * i + 1), marker="o", label=f"TF-IDF + {col} entropy"
        )

    plt.xlabel(r"Number of clusters *k*")
    plt.ylabel("Sum of squared distance")
    plt.legend()

    return fig


def visualize_document_clusters(cluster_tokens, min_frequency=20):
    """Visualize words most frequently occurring in a cluster. Especially useful when visualizing
    the results of an unsupervised partitioning of documents.

    Parameters
    ----------
    cluster_tokens : list
        List of tokenized documents
    min_frequency : int
        Minimum number of occurrences that a word must have in a cluster for it to be visualized

    Returns
    -------
    Matplotlib figure instance
    """
    # IDEA: instead of using frequency, use importance with other embeddings too
    all_tokens = [item for sublist in cluster_tokens for item in sublist]
    # important_words_freq is [[word1,freq1],[word2,freq2],...]
    total_important_words_freq = Counter(all_tokens).most_common()
    word_freq_df = pd.DataFrame(
        total_important_words_freq, columns=["word", "freq"])

    all_words_of_interest = []
    for tokens in cluster_tokens:
        # important_words_freq is [[word1,freq1],[word2,freq2],...]
        important_words_freq = Counter(tokens).most_common()
        for word, freq in important_words_freq:
            if freq >= min_frequency:
                all_words_of_interest.append(word)

    unique_words = np.unique(all_words_of_interest)

    cluster_list = []
    freq_list = []
    word_list = []
    for wd in unique_words:
        freq = word_freq_df[word_freq_df["word"] == wd]["freq"].tolist()[0]
        clusters_this_wd = [
            idx
            for idx, words_in_cluster in enumerate(all_words_of_interest)
            if wd in words_in_cluster
        ]
        clusters_this_wd = list(map(str, clusters_this_wd))
        cluster_list.append(", ".join(clusters_this_wd))
        freq_list.append(freq)
        word_list.append(wd)

    # fig = plt.figure(figsize=(10,20))

    filter_cluster_list = []
    filter_freq_list = []
    filter_word_list = []
    for fr, cl, wd in sorted(zip(freq_list, cluster_list, word_list)):
        filter_cluster_list.append(cl)
        filter_freq_list.append(fr)
        filter_word_list.append(wd)

    df = pd.DataFrame(index=filter_cluster_list)
    df["freq"] = filter_freq_list
    ax = df["freq"].plot(kind="barh", figsize=(
        20, 14), color="coral", fontsize=13)

    xbias = 0.3
    ybias = 0.0
    for idx, i in enumerate(ax.patches):
        ax.text(
            i.get_width() + xbias,
            i.get_y() + ybias,
            filter_word_list[idx],
            fontsize=15,
            color="dimgrey",
        )

    return ax


def visualize_word_frequency_plot(
    tokenized_words, title="", font_size=16, graph_aargs={}
):
    """Visualize the frequency distribution of words within a set of documents

    Parameters
    ----------
    tokenized_words : list
        List of tokenized words
    title : str
        Optional, title of plot
    font_size : int
        Optional, font size
    aargs :
        Optional, other parameters passed to nltk.FreqDist.plot()

    Returns
    -------
    Matplotlib figure instance
    """
    matplotlib.rcParams.update({"font.size": font_size})
    fd = nltk.FreqDist(tokenized_words)
    fig = plt.figure(figsize=(12, 6))
    fd.plot(30, cumulative=False, title=title, figure=fig, **graph_aargs)
    return fd


def visualize_classification_confusion_matrix(om_df, col_dict, title=''):
    """Visualize confusion matrix comparing known categorical values, and predicted categorical values.

    Parameters
    ----------
    om_df : DataFrame
        A pandas dataframe containing O&M data, which contains columns specified in om_col_dict
    col_dict : dict of {str : str}
        A dictionary that contains the column names needed:

        - data : string, should be assigned to associated column which stores the tokenized text logs
        - attribute_col : string, will be assigned to attribute column and used to create new attribute_col
        - predicted_col : string, will be used to create keyword search label column

    title : str
        Optional, title of plot

    Returns
    -------
    Matplotlib figure instance
    """
    act_col = col_dict['attribute_col']
    pred_col = col_dict['predicted_col']

    # drop any predicted labels with no actual labels in the data, for a cleaner visual
    no_real_values = [cat for cat in om_df[pred_col].unique() if cat not in om_df[act_col].unique()]
    no_real_values_mask = om_df[pred_col].isin(no_real_values)
    om_df = om_df[~no_real_values_mask]
    caption_txt = f'NOTE: Predicted values{no_real_values} had no actual values in the dataset.'

    plt.rcParams.update({'font.size': 8})
    cm_display = ConfusionMatrixDisplay.from_predictions(y_true=om_df[act_col],
                                                         y_pred=om_df[pred_col],
                                                         normalize='true',
                                                         )
    fig = cm_display.plot()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.figtext(0.00, 0.01, caption_txt, wrap=True, fontsize=7)
    plt.title(title)
    return fig
