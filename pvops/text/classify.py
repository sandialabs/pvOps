# Classifiers
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from scipy.sparse import issparse

import numpy as np
import pandas as pd
import copy

from pvops.text.preprocess import get_keywords_of_interest

def classification_deployer(
    X,
    y,
    n_splits,
    classifiers,
    search_space,
    pipeline_steps,
    scoring,
    greater_is_better=True,
    verbose=3,
):
    """The classification deployer builds a classifier evaluator with an ingrained hyperparameter fine-tuning grid search protocol.
    The output of this function will be a data frame showing the performance of each classifier when utilizing a specific hyperparameter
    configuration.

    To see an example of this method's application, see ``tutorials//text_class_example.py``

    Parameters
    ----------
    X : list of str
        List of documents (str). The documents will be passed through the pipeline_steps, where they will be transformed into vectors.
    y : list
        List of labels corresponding with the documents in X
    n_splits : int
        Integer defining the number of splits in the cross validation split during training
    classifiers : dict
        Dictionary with key as classifier identifier (str) and value as classifier instance following sklearn's
        base model convention: sklearn_docs.

        .. sklearn_docs: https://scikit-learn.org/stable/modules/generated/sklearn.base.is_classifier.html
        .. code-block:: python

            classifiers = {
                'LinearSVC' : LinearSVC(),
                'AdaBoostClassifier' : AdaBoostClassifier(),
                'RidgeClassifier' : RidgeClassifier()
            }

        See ``supervised_classifier_defs.py`` or ``unsupervised_classifier_defs.py`` for this package's defaults.
    search_space : dict
        Dictionary with classifier identifiers, as used in ``classifiers``, mapped to its hyperparameters.

        .. code-block:: python

            search_space = {
                'LinearSVC' : {
                'clf__C' : [1e-2,1e-1],
                'clf__max_iter':[800,1000],
                },
                'AdaBoostClassifier' : {
                'clf__n_estimators' : [50,100],
                'clf__learning_rate':[1.,0.9,0.8],
                'clf__algorithm' : ['SAMME.R']
                },
                'RidgeClassifier' : {
                'clf__alpha' : [0.,1e-3,1.],
                'clf__normalize' : [False,True]
                }
            }

        See ``supervised_classifier_defs.py`` or ``unsupervised_classifier_defs.py`` for this package's defaults.
    pipeline_steps : list of tuples
        Define embedding and machine learning pipeline. The last tuple must be ``('clf', None)`` so that the output
        of the pipeline is a prediction.
        For supervised classifiers using a TFIDF embedding, one could specify

        .. code-block:: python

            pipeline_steps = [('tfidf', TfidfVectorizer()),
                              ('clf', None)]

        For unsupervised clusterers using a TFIDF embedding, one could specify

        .. code-block:: python

            pipeline_steps = [('tfidf', TfidfVectorizer()),
                              ('to_dense', DataDensifier.DataDensifier()),
                              ('clf', None)]

        A densifier is required from some clusters, which fail if sparse data is passed.
    scoring : sklearn callable scorer (i.e., any statistic that summarizes predictions relative to observations).
        Example scorers include f1_score, accuracy, etc.
        Callable object that returns a scalar score created using sklearn.metrics.make_scorer
        For supervised classifiers, one could specify

        .. code-block:: python

            scoring = make_scorer(f1_score, average = 'weighted')

        For unsupervised classifiers, one could specify

        .. code-block:: python

            scoring = make_scorer(homogeneity_score)

    greater_is_better : bool
        Whether the scoring parameter is better when greater (i.e. accuracy) or not.

    verbose : int
        Control the specificity of the prints. If greater than 1, a print out is shown when a new "best classifier"
        is found while iterating. Additionally, the verbosity during the grid search follows sklearn's definitions.
        The frequency of the messages increase with the verbosity level.

    Returns
    -------
    DataFrame
        Summarization of results from all of the classifiers
    """

    rows = []

    if issparse(X):
        print("Converting passed data to dense array...")
        X = X.toarray()

    # get position of 'clf' in pipeline_steps
    idx_clf_pipeline = [i for i, it in enumerate(
        pipeline_steps) if it[0] == "clf"][0]

    best_gs_instance = None
    if greater_is_better:
        best_model_score = 0.0
    else:
        best_model_score = np.inf
    for iter_idx, key in enumerate(classifiers.keys()):
        clas = classifiers[key]
        space = search_space[key]

        iter_pipeline_steps = copy.deepcopy(pipeline_steps)
        iter_pipeline_steps[idx_clf_pipeline] = ("clf", clas)
        pipe = Pipeline(iter_pipeline_steps)

        gs_clf = GridSearchCV(
            pipe,
            space,
            scoring=scoring,
            cv=n_splits,
            n_jobs=-1,
            return_train_score=True,
            verbose=verbose,
        )
        gs_clf.fit(X, y)
        params = gs_clf.cv_results_["params"]
        scores = []
        for i in range(n_splits):
            r1 = gs_clf.cv_results_[f"split{i}_test_score"]
            scores.append(r1.reshape(len(params), 1))

        r2 = gs_clf.cv_results_["mean_fit_time"]

        all_scores = np.hstack(scores)
        for param, score, time in zip(params, all_scores, r2):
            param["mean_fit_time"] = time
            d = {
                "estimator" : key,
                "min_score" : min(score),
                "max_score" : max(score),
                "mean_score" : np.mean(score),
                "std_score" : np.std(score),
            }
            rows.append((pd.Series({**param, **d})))

        if greater_is_better:
            replacement_logic = gs_clf.best_score_ > best_model_score
        else:
            replacement_logic = gs_clf.best_score_ < best_model_score

        if replacement_logic:
            if verbose > 1:
                print(
                    "Better score ({:.3f}) found on classifier: {}".format(
                        gs_clf.best_score_, key
                    )
                )
            best_model_score = gs_clf.best_score_
            best_gs_instance = gs_clf

    return pd.concat(rows, axis=1).T, best_gs_instance.best_estimator_

def get_attributes_from_keywords(om_df, col_dict, reference_df, reference_col_dict):
    """Find keywords of interest in specified column of dataframe, return as new column value.

    If keywords of interest given in a reference dataframe are in the specified column of the
    dataframe, return the keyword category, or categories.
    For example, if the string 'inverter' is in the list of text, return ['inverter'].

    Parameters
    ----------
    om_df : pd.DataFrame
        Dataframe to search for keywords of interest, must include text_col.
    col_dict : dict of {str : str}
        A dictionary that contains the column names needed:

        - data : string, should be assigned to associated column which stores the tokenized text logs
        - predicted_col : string, will be used to create keyword search label column
    reference_df : DataFrame
        Holds columns that define the reference dictionary to search for keywords of interest,
        Note: This function can currently only handle single words, no n-gram functionality.
    reference_col_dict : dict of {str : str}
        A dictionary that contains the column names that describes how
        referencing is going to be done

        - reference_col_from : string, should be assigned to
          associated column name in reference_df that are possible input reference values
          Example: pd.Series(['inverter', 'invert', 'inv'])
        - reference_col_to : string, should be assigned to
          associated column name in reference_df that are the output reference values
          of interest
          Example: pd.Series(['inverter', 'inverter', 'inverter'])

    Returns
    -------
    om_df: pd.DataFrame
        Input df with new_col added, where each found keyword is its own row, may result in
        duplicate rows if more than one keywords of interest was found in text_col.
    """
    om_df[col_dict['predicted_col']] = om_df[col_dict['data']].apply(get_keywords_of_interest,
                                                                     reference_df=reference_df,
                                                                     reference_col_dict=reference_col_dict)

    # each multi-category now in its own row, some logs have multiple equipment issues
    multiple_keywords_df = om_df[om_df[col_dict['predicted_col']].str.len() > 1]
    om_df = om_df.explode(col_dict['predicted_col'])

    msg = f'{len(multiple_keywords_df)} entries had multiple keywords of interest. Reference: {multiple_keywords_df.index} in original dataframe.'
    print(msg)

    return om_df