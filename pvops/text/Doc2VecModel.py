from sklearn.base import BaseEstimator
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import word_tokenize


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
            epochs=self.d2v_model.iter,
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