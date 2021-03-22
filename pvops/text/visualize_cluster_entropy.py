import matplotlib.pyplot as plt

# Embedding
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize

import copy


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
