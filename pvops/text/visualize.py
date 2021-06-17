# visualizations
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

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

    unique_df = pd.DataFrame(
        nx_data, columns=[ATTRIBUTE1_COL, ATTRIBUTE2_COL, "w"])

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
    weights = list(1 + (edge_width_scalar * weights /
                   weights.max()))  # scale 1-11
    nx.draw_shell(G, width=weights, node_color=color_map, **graph_aargs)

    return fig, edges


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
    **aargs :
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
