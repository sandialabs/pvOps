from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    word_freq_df = pd.DataFrame(total_important_words_freq, columns=["word", "freq"])

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
    ax = df["freq"].plot(kind="barh", figsize=(20, 14), color="coral", fontsize=13)

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
