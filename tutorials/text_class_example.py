# Utilities
from pvops.text import nlp_utils
from pvops.text import utils

# Visualizations
from pvops.text import visualize

# Preprocessing
from pvops.text import preprocess

# Classification
from pvops.text import classify

# Library example definitions
from pvops.text import defaults

# Utility dependencies
import numpy as np
import pandas as pd
import traceback
import nltk

# Embedding
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec

# Clustering
from sklearn.cluster import KMeans

# Scoring
from sklearn.metrics import make_scorer, f1_score, homogeneity_score

class Example:
    def __init__(self, df, LABEL_COLUMN):
        self.LABEL_COLUMN = LABEL_COLUMN
        self.df = df

    def summarize_text_data(self, DATA_COLUMN):
        nlp_utils.summarize_text_data(self.df, DATA_COLUMN)

    def visualize_attribute_timeseries(self, DATE_COLUMN):
        col_dict = {"date": DATE_COLUMN, "label": self.LABEL_COLUMN}
        df = self.df.copy()
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        df = df[df[DATE_COLUMN].notnull()]
        fig = visualize.visualize_attribute_timeseries(
            df, col_dict
        )
        return fig

    def visualize_cluster_entropy(self, cols):
        def eval_kmeans(X, k):
            km = KMeans(n_clusters=k, n_init=10)
            km.fit(X)
            return km

        doc2vec_model = Doc2Vec(vector_size=40, min_count=2, epochs=30)
        ks = np.arange(1, 61, 10)
        fig = visualize.visualize_cluster_entropy(
            doc2vec_model, eval_kmeans, self.df, cols, ks, cmap_name="brg"
        )
        return fig

    def extract_dates(
        self,
        DATA_COLUMN,
        EVENTSTART_COLUMN,
        SAVE_DATA_COLUMN="CleanDesc",
        SAVE_DATE_COLUMN="ExtractedDates",
        print_info=False,
    ):

        col_dict = {
            "data": DATA_COLUMN,
            "eventstart": EVENTSTART_COLUMN,
            "save_data_column": SAVE_DATA_COLUMN,
            "save_date_column": SAVE_DATE_COLUMN,
        }

        try:
            self.df = preprocess.preprocessor(
                self.df, None, col_dict,
                print_info=print_info,
                extract_dates_only=True
            )
        except Exception as e:
            print(e)
            print(traceback.format_exc())

        return self.df[[DATA_COLUMN, SAVE_DATE_COLUMN]]

    def prep_data_for_ML(
        self,
        DATA_COLUMN,
        EVENTSTART_COLUMN,
        SAVE_DATA_COLUMN="CleanDesc",
        SAVE_DATE_COLUMN="ExtractedDates",
        input_lst_add_stopwords=None,
        input_lst_keep_stopwords=None,
    ):
        self.DATA_COLUMN = DATA_COLUMN
        self.EVENTSTART_COLUMN = EVENTSTART_COLUMN
        self.SAVE_DATA_COLUMN = SAVE_DATA_COLUMN
        self.SAVE_DATE_COLUMN = SAVE_DATE_COLUMN

        self.df = self.df.dropna(subset=[self.LABEL_COLUMN])
        self.df = self.df[~self.df[self.DATA_COLUMN].isna()]
        for lbl in set(self.df[self.LABEL_COLUMN].tolist()):
            if len(self.df[self.df[self.LABEL_COLUMN] == lbl].index) <= 1:
                self.df = self.df[self.df[self.LABEL_COLUMN] != lbl]
        self.df = self.df.sample(frac=1)

        lst_add_stopwords = ["dtype", "say", "new",
                             "length", "object", "u", "ha", "wa"]
        lst_keep_words = ["new"]
        if not isinstance(input_lst_add_stopwords, type(None)):
            lst_add_stopwords += input_lst_add_stopwords
        if not isinstance(input_lst_keep_stopwords, type(None)):
            lst_keep_words += input_lst_keep_stopwords

        lst_stopwords = nlp_utils.create_stopwords(
            ["english"],
            lst_add_words=lst_add_stopwords,
            lst_keep_words=lst_keep_words,
        )

        # Save so that can be easily referenced
        self.lst_stopwords = lst_stopwords

        col_dict = {
            "data": DATA_COLUMN,
            "eventstart": EVENTSTART_COLUMN,
            "save_data_column": SAVE_DATA_COLUMN,
            "save_date_column": SAVE_DATE_COLUMN,
        }

        try:
            self.df = preprocess.preprocessor(
                self.df, lst_stopwords, col_dict, print_info=False
            )
        except Exception as e:
            print(e)
            print(traceback.format_exc())

        self.DATA_COLUMN = SAVE_DATA_COLUMN
        return self.df[[DATA_COLUMN, SAVE_DATA_COLUMN]]

    def visualize_freqPlot(self, LBL_CAT=None, DATA_COLUMN="CleanDesc",
                           **graph_aargs):
        if LBL_CAT is None:
            words = " ".join(self.df[DATA_COLUMN].tolist())
            num_rows = len(self.df[DATA_COLUMN].index)
        elif LBL_CAT in set(self.df[self.LABEL_COLUMN].tolist()):
            words = " ".join(
                self.df[self.df[self.LABEL_COLUMN]
                        == LBL_CAT][DATA_COLUMN].tolist()
            )
            num_rows = len(
                self.df[self.df[self.LABEL_COLUMN] == LBL_CAT].index)
        else:
            raise Exception(
                "An invalid label category (LBL_CAT) was passed." +
                "Please pass a value in the following set:\n" +
                f"{set(self.df[self.LABEL_COLUMN].tolist())}"
            )

        tokenized = nltk.word_tokenize(words)
        fig = visualize.visualize_word_frequency_plot(
            tokenized, title=str(LBL_CAT) + f" (Num. Docs: {num_rows})",
            **graph_aargs
        )
        return fig

    def visualize_document_clusters(self, min_frequency=20, DATA_COLUMN="CleanDesc"):
        all_labels = np.unique(self.df[self.LABEL_COLUMN])
        cluster_words = [
            " ".join(
                self.df[self.df[self.LABEL_COLUMN]
                        == LBL_CAT][DATA_COLUMN].tolist()
            )
            for LBL_CAT in all_labels
        ]
        cluster_tokens = [nltk.word_tokenize(words) for words in cluster_words]
        fig = visualize.visualize_document_clusters(
            cluster_tokens, min_frequency=min_frequency
        )
        return fig

    def visualize_attribute_connectivity(self, om_col_dict, **networkxDrawAargs):
        df_filtered = self.df.dropna(subset=list(
            om_col_dict.values()), inplace=False)
        fig, edges = visualize.visualize_attribute_connectivity(
            df_filtered, om_col_dict, **networkxDrawAargs
        )
        return fig, edges

    def _tfidf_classifier_defs(self, search_space):
        """Add TFIDF hyperparamters to grid search protocol"""
        for d in search_space.keys():
            search_space[d]["tfidf__ngram_range"] = [(1, 3)]
            search_space[d]["tfidf__stop_words"] = [None]
        return search_space

    def _doc2vec_classifier_defs(self, search_space):
        """Add TFIDF hyperparamters to grid search protocol"""
        for d in search_space.keys():
            search_space[d]["doc2vec__vector_size"] = [40, 100]
            search_space[d]["doc2vec__window"] = [5, 10]
            # search_space[d]['doc2vec__min_count'] = [2]
            search_space[d]["doc2vec__epochs"] = [30]
        return search_space

    def _classify(
        self,
        embedding,
        pipeline_steps,
        scoring,
        search_space,
        classes,
        n_cv_splits=5,
        verbose=0,
    ):

        try:
            X = self.df[self.DATA_COLUMN].tolist()
            y = self.df[self.LABEL_COLUMN].tolist()

            if embedding == "tfidf":
                print("Starting ML analysis with TF-IDF embeddings")
                search_space = self._tfidf_classifier_defs(search_space)
            elif embedding == "doc2vec":
                print("Starting ML analysis with Doc2Vec embeddings")
                search_space = self._doc2vec_classifier_defs(search_space)

            results_df, best_model = classify.classification_deployer(
                X,
                y,
                n_cv_splits,
                classes,
                search_space,
                pipeline_steps,
                scoring,
                greater_is_better=self.greater_is_better,
                verbose=verbose,
            )
            # organize columns
            cols = [
                "estimator",
                "min_score",
                "mean_score",
                "max_score",
                "std_score",
                "mean_fit_time",
            ]
            cols += [c for c in results_df.columns if c not in cols]
            results_df = results_df[cols]
            # sort values
            results_df = results_df.sort_values(
                ["mean_score"], ascending=not self.greater_is_better
            )
        except Exception as e:
            print(e)
            print(traceback.format_exc())
        return results_df, best_model

    def classify_supervised(
        self,
        embedding="tfidf",
        n_cv_splits=5,
        subset_example_classifiers=None,
        setting="normal",
        user_defined_classes=None,
        user_defined_search_space=None,
        verbose=0,
    ):
        """A wrapper function which evaluates the performance of many
        supervised classifiers

            embedding : str
                Definition of document embedding strategy with "tfidf" and
                "doc2vec" options

            setting : str
                Thoroughness of supervised classification investigation with
                following options:
                'normal': a smaller subset of settings used in grid search
                'detailed': a comprehensive set of settings used in grid search
                For specifics, view the `supervised_classifier_defs.py` file 
                located in `/pvops/text/`. To specify your own classifier pipeline, 
                use `user_defined_classes` and `user_defined_search_space` 
                parameters.


        Two options are permitted when specifying the utilized supervised 
        classifiers.

        First, you can use the classifiers established in this library (found in
        `supervised_classifier_defs`). These classifiers can be subset if wanted by 
        specifying a list of classifier names (strings). For example,

        ```
        subset_example_classifiers = ['LinearSVC', 'AdaBoostClassifier', 
                                      'RidgeClassifier']
        e.classify_supervised(
                        n_cv_splits=5,
                        subset_example_classifiers=subset_example_classifiers
                    )
        ```

        Second, you can pass your own definitions of supervised classifiers,
        just as found in `supervised_classifier_defs.py`. To do this, two objects are 
        required as input: `classes` and `search_space`. The `classes` object is a 
        dictionary with a key as the name of the classifier and a value as the 
        classifier object. The `search_space` specifies the hyperparameters which the 
        grid search protocol will iterate through. For example,

        ```
        user_defined_classes = {
                        'LinearSVC': LinearSVC(),
                        'AdaBoostClassifier': AdaBoostClassifier(),
                        'RidgeClassifier': RidgeClassifier()
        }
        user_defined_search_space = {
                        'LinearSVC': {
                        'clf__C': [1e-2,1e-1],
                        'clf__max_iter':[800,1000],
                        },
                        'AdaBoostClassifier': {
                        'clf__n_estimators': [50,100],
                        'clf__learning_rate':[1.,0.9,0.8],
                        'clf__algorithm': ['SAMME.R']
                        },
                        'RidgeClassifier': {
                        'clf__alpha': [0.,1e-3,1.],
                        'clf__normalize': [False,True]
                        },
        }
        e.classify_supervised(
                        n_cv_splits=5,
                        user_defined_classes=user_defined_classes,
                        user_defined_search_space=user_defined_search_space
                    )
        ```

        """
        if embedding == "tfidf":
            pipeline_steps = [("tfidf", TfidfVectorizer()), ("clf", None)]
        elif embedding == "doc2vec":
            pipeline_steps = [
                ("doc2vec", nlp_utils.Doc2VecModel()), ("clf", None)]

        scoring = make_scorer(f1_score, average="weighted")
        self.greater_is_better = True

        if user_defined_classes is None or user_defined_search_space is None:
            # Utilize a subset of the pre-determined classifiers
            (
                search_space,
                classes,
            ) = defaults.supervised_classifier_defs(setting)

            if not isinstance(subset_example_classifiers, type(None)):
                for clf_str in subset_example_classifiers:
                    if clf_str not in classes:
                        del classes[clf_str]
                        del search_space[clf_str]
                    else:
                        raise Exception(
                            "All components of subset_example_classifiers" +
                            f"must be keys in {classes}"
                        )
        else:
            search_space = user_defined_search_space
            classes = user_defined_classes

        if verbose > 2:
            print('search_space', search_space)
            print('classes', classes)
            print('pipeline_steps', pipeline_steps)
            print('embedding', embedding)
        self.supervised_results, self.supervised_best_model = self._classify(
            embedding,
            pipeline_steps,
            scoring,
            search_space,
            classes,
            n_cv_splits=n_cv_splits,
            verbose=verbose
        )
        return self.supervised_results, self.supervised_best_model

    def classify_unsupervised(
        self,
        embedding="tfidf",
        n_cv_splits=5,
        subset_example_classifiers=None,
        setting="normal",
        user_defined_classes=None,
        user_defined_search_space=None,
        verbose=0,
    ):
        """A wrapper function which evaluates the performance of many unsupervised 
        classifiers

            setting : str
                Thoroughness of classification investigation.
                'normal': a smaller subset of settings used in grid search
                'detailed': a comprehensive set of settings used in grid search

        Two options are permitted when specifying the utilized unsupervised 
        classifiers.

        First, you can use the classifiers established in this library (found in
        `unsupervised_classifier_defs`). These classifiers can be subset if wanted 
        by specifying a list of classifier names (strings). For example,

        ```
        subset_example_classifiers = ['LinearSVC', 'AdaBoostClassifier', 
                                      'RidgeClassifier']
        e.classify_unsupervised(
                        n_cv_splits=5,
                        subset_example_classifiers=subset_example_classifiers
                    )
        ```

        Second, you can pass your own definitions of supervised classifiers, just 
        as found in `supervised_classifier_defs.py`. To do this, two objects are 
        required as input: `classes` and `search_space`. The `classes` object is 
        a dictionary with a key as the name of the classifier and a value as the 
        classifier object. The `search_space` specifies the hyperparameters which 
        the grid search protocol will iterate through. For example,

        ```
        user_defined_classes = {
                        'AffinityPropagation': AffinityPropagation(),
                        'KMeans': KMeans(),
        }
        user_defined_search_space = {
                       'AffinityPropagation': {
                        'clf__damping': [0.5,0.9],
                        'clf__max_iter':[200,600],
                        },
                        'KMeans': {
                        'clf__n_clusters': [n_clusters],
                        'clf__init':['k-means++', 'random'],
                        'clf__n_init': [10,50,100]
                        },
        }
        e.classify_unsupervised(
                        n_cv_splits=5,
                        user_defined_classes=user_defined_classes,
                        user_defined_search_space=user_defined_search_space
                    )

        """
        if embedding == "tfidf":
            pipeline_steps = [
                ("tfidf", TfidfVectorizer()),
                ("to_dense", nlp_utils.DataDensifier()),
                ("clf", None),
            ]
        elif embedding == "doc2vec":
            pipeline_steps = [
                ("doc2vec", nlp_utils.Doc2VecModel()),
                ("to_dense", nlp_utils.DataDensifier()),
                ("clf", None),
            ]
        scoring = make_scorer(homogeneity_score)
        self.greater_is_better = True

        if user_defined_classes is None or user_defined_search_space is None:
            # Utilize a subset of the pre-determined classifiers
            y = self.df[self.LABEL_COLUMN].tolist()
            n_clusters = len(np.unique(y))
            (
                search_space,
                classes,
            ) = defaults.unsupervised_classifier_defs(
                setting, n_clusters
            )

            if not isinstance(subset_example_classifiers, type(None)):
                for clf_str in subset_example_classifiers:
                    if clf_str not in classes:
                        del classes[clf_str]
                        del search_space[clf_str]
                    else:
                        raise Exception(
                            "All components of subset_example_classifiers " +
                            "must be keys in "
                        )
        else:
            search_space = user_defined_search_space
            classes = user_defined_classes

        if verbose > 2:
            print('search_space', search_space)
            print('classes', classes)
            print('pipeline_steps', pipeline_steps)
            print('embedding', embedding)
        self.unsupervised_results, self.unsupervised_best_model = self._classify(
            embedding,
            pipeline_steps,
            scoring,
            search_space,
            classes,
            n_cv_splits=n_cv_splits,
            verbose=verbose
        )
        return self.unsupervised_results, self.unsupervised_best_model

    def predict_best_model(
        self,
        ml_type="supervised",
        eval_func=None,
        PREDICTION_OUTPUT_COL=None,
        *eval_aargs,
    ):

        X = self.df[self.DATA_COLUMN].tolist()
        y = self.df[self.LABEL_COLUMN].tolist()

        if ml_type == "supervised":
            print("Best algorithm found:\n", self.supervised_best_model)
            pred_y = self.supervised_best_model.predict(X)
        elif ml_type == "unsupervised":
            print("Best algorithm found:\n", self.unsupervised_best_model)
            pred_y = self.unsupervised_best_model.predict(X)

        if eval_func is None:
            if ml_type == "supervised":
                score = f1_score(y, pred_y, average="weighted")
                self.greater_is_better = True
                if PREDICTION_OUTPUT_COL is None:
                    output_col = f"Supervised_Pred_{self.LABEL_COLUMN}"
            elif ml_type == "unsupervised":
                score = homogeneity_score(y, pred_y, *eval_aargs)
                self.greater_is_better = True
                if PREDICTION_OUTPUT_COL is None:
                    output_col = f"Unsupervised_Pred_{self.LABEL_COLUMN}"

        self.df[output_col] = pred_y
        print(f"Predictions stored to {output_col} in `df` attribute")

        print(f"Score: {score}")


if __name__ == "__main__":
    DATA_COLUMN = "CompletionDesc"
    LABEL_COLUMN = "Asset"
    DATE_COLUMN = "Date_EventStart"
    folder = "example_data//"
    filename = "example_ML_ticket_data.csv"
    df = pd.read_csv(folder + filename)

    e = Example(df, LABEL_COLUMN)
    e.summarize_text_data(DATA_COLUMN)

    print("\nMessage from pvOps team: See " +
          "`tutorial_textmodule.ipynb` for a" +
          "more in-depth demonstration of the" +
          "text module's functionality.")
