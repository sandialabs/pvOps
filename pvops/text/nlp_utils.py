import nltk

from sklearn.base import BaseEstimator
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import word_tokenize
import scipy
import numpy as np
from gensim.models import Word2Vec

class Doc2VecModel(BaseEstimator):
    """Performs a gensim Doc2Vec transformation of the input documents to create
    embedded representations of the documents. See gensim's
    Doc2Vec model for information regarding the hyperparameters.
    """

    def __init__(
        self,
        vector_size=100,
        dm_mean=None,
        dm=1,
        dbow_words=0,
        dm_concat=0,
        dm_tag_count=1,
        dv=None,
        dv_mapfile=None,
        comment=None,
        trim_rule=None,
        callbacks=(),
        window=5,
        epochs=10,
    ):
        self.d2v_model = None
        self.vector_size = vector_size
        self.dm_mean = dm_mean
        self.dm = dm
        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        self.dv = dv
        self.dv_mapfile = dv_mapfile
        self.comment = comment
        self.trim_rule = trim_rule
        self.callbacks = callbacks
        self.window = window
        self.epochs = epochs

    def fit(self, raw_documents, y=None):
        """Fits the Doc2Vec model."""
        # Initialize model
        self.d2v_model = Doc2Vec(
            vector_size=self.vector_size,
            dm_mean=self.dm_mean,
            dm=self.dm,
            dbow_words=self.dbow_words,
            dm_concat=self.dm_concat,
            dm_tag_count=self.dm_tag_count,
            dv=self.dv,
            dv_mapfile=self.dv_mapfile,
            comment=self.comment,
            trim_rule=self.trim_rule,
            window=self.window,
            epochs=self.epochs,
        )
        # Tag docs
        tagged_documents = [
            TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)])
            for i, _d in enumerate(raw_documents)
        ]
        # Build vocabulary
        self.d2v_model.build_vocab(tagged_documents)
        # Train model
        self.d2v_model.train(
            tagged_documents,
            total_examples=len(tagged_documents),
            epochs=self.d2v_model.epochs,
        )
        return self

    def transform(self, raw_documents):
        """Transforms the documents into Doc2Vec vectors."""
        X = []
        for doc in raw_documents:
            X.append(self.d2v_model.infer_vector(word_tokenize(doc)))
        return X

    def fit_transform(self, raw_documents, y=None):
        """Utilizes the ``fit()`` and ``transform()`` methods in this class."""
        self.fit(raw_documents)
        return self.transform(raw_documents)


class DataDensifier(BaseEstimator):
    """A data structure transformer which converts sparse data to dense data.
    This process is usually incorporated in this library when doing unsupervised machine learning.
    This class is built specifically to work inside a sklearn pipeline.
    Therefore, it uses the default ``transform``, ``fit``, ``fit_transform`` method structure.
    """

    def transform(self, X, y=None):
        """Return a dense array if the input array is sparse.

        Parameters

        ----------
        X : array
            Input data of numerical values. For this package, these values could
            represent embedded representations of documents.

        Returns

        -------
        dense array
        """
        if scipy.sparse.issparse(X):
            return X.toarray()
        else:
            return X.copy()

    def fit(self, X, y=None):
        """Placeholder method to conform to the sklearn class structure.

        Parameters

        ----------
        X : array
            Input data
        y : Not utilized.

        Returns

        -------
        DataDensifier object
        """
        return self

    def fit_transform(self, X, y=None):
        """Performs same action as ``DataDensifier.transform()``,
        which returns a dense array when the input is sparse.

        Parameters

        ----------
        X : array
            Input data
        y : Not utilized.

        Returns

        -------
        dense array
        """
        return self.transform(X=X, y=y)


def create_stopwords(lst_langs=["english"], lst_add_words=[], lst_keep_words=[]):
    """Concatenate a list of stopwords using both words grabbed from nltk and user-specified words.

    Parameters

    ---------
    lst_langs: list
        List of strings designating the languages for a nltk.corpus.stopwords.words query. If empty list is passed, no stopwords will be queried from nltk.
    lst_add_words: list
        List of words(e.g., "road" or "street") to add to stopwords list. If these words are already included in the nltk query, a duplicate will not be added.
    lst_keep_words: list
        List of words(e.g., "before" or "until") to remove from stopwords list. This is usually used to modify default stop words that might be of interest to PV.

    Returns

    -------
    List
        List of alphabetized stopwords
    """
    lst_stopwords = set()
    for lang in lst_langs:
        try:
            stopwords = nltk.corpus.stopwords.words(lang)
        except LookupError:
            nltk.download("stopwords")
            stopwords = nltk.corpus.stopwords.words(lang)
        lst_stopwords = lst_stopwords.union(stopwords)
    lst_stopwords = lst_stopwords.union(lst_add_words)
    lst_stopwords = list(set(lst_stopwords) - set(lst_keep_words))
    return sorted(list(set(lst_stopwords)))

def summarize_text_data(om_df, colname):
    """Display information about a set of documents located in a dataframe, including
    the number of samples, average number of words, vocabulary size, and number of words
    in total.

    Parameters

    ---------
    om_df : DataFrame
        A pandas dataframe containing O&M data, which contains at least the colname of interest
    colname : str
        Column name of column with text

    Returns

    ------
    None
    """
    df = om_df.copy()
    text = df[colname].tolist()

    nonan_text = [x for x in text if (str(x) != "nan" and x is not None)]

    tokenized = [sentence.split() for sentence in nonan_text]
    avg_n_words = np.array([len(tokens) for tokens in tokenized]).mean()
    sum_n_words = np.array([len(tokens) for tokens in tokenized]).sum()
    model = Word2Vec(tokenized, min_count=1)

    # Total vocabulary
    vocab = model.wv

    # Bold title.
    print("\033[1m" + "DETAILS" + "\033[0m")

    info = {
        "n_samples": len(df),
        "n_nan_docs": len(df) - len(nonan_text),
        "n_words_doc_average": avg_n_words,
        "n_unique_words": len(vocab),
        "n_total_words": sum_n_words,
    }

    # Display information.
    print(f'  {info["n_samples"]} samples')
    print(f'  {info["n_nan_docs"]} invalid documents')
    print("  {:.2f} words per sample on average".format(
        info["n_words_doc_average"]))
    print(f'  Number of unique words {info["n_unique_words"]}')
    print("  {:.2f} total words".format(info["n_total_words"]))

    return info
