from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)

import numpy as np


def supervised_classifier_defs(settings_flag):
    """Esablish supervised classifier definitions which are non-specific to embeddor,
    and therefore, non-specific to the natural language processing application

    Parameters

    ----------
    settings_flag : str
      Either 'light', 'normal' or 'detailed'; a setting which determines the number of hyperparameter combinations
      tested during the grid search. For instance, a dataset of 50 thousand samples may run for hours on the
      'normal' setting but for days on 'detailed'.

    Returns

    -------
    search_space: dict
      Hyperparameter instances for each clusterer
    classifiers: dict
      Contains sklearn classifiers instances
    """
    if settings_flag == "light":
        classifiers = {
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "LogisticRegression": LogisticRegression(),
            "PassiveAggressiveClassifier": PassiveAggressiveClassifier(),
            "RidgeClassifier": RidgeClassifier(),
            "SGDClassifier": SGDClassifier(),
            "ExtraTreesClassifier": ExtraTreesClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "BaggingClassifier": BaggingClassifier(),
            "AdaBoostClassifier": AdaBoostClassifier(),
        }
    else:
        classifiers = {
            "LinearSVC": LinearSVC(),
            "SVC": SVC(),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "MLPClassifier": MLPClassifier(),
            "LogisticRegression": LogisticRegression(),
            "PassiveAggressiveClassifier": PassiveAggressiveClassifier(),
            "RidgeClassifier": RidgeClassifier(),
            "SGDClassifier": SGDClassifier(),
            "ExtraTreesClassifier": ExtraTreesClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "BaggingClassifier": BaggingClassifier(),
            "AdaBoostClassifier": AdaBoostClassifier(),
        }

    if settings_flag == "light":
        search_space = {
            "DecisionTreeClassifier": {
                "clf__criterion": ["gini"],
                "clf__splitter": ["best"],
                "clf__min_samples_split": [2],
                "clf__min_samples_leaf": [1],
            },
            "LogisticRegression": {
                "clf__solver": ["newton-cg", "lbfgs", "sag"],
                "clf__C": np.logspace(0, 4, 10),
            },
            "PassiveAggressiveClassifier": {
                "clf__C": [0.0, 0.01, 0.1, 1.0],
                "clf__loss": ["hinge", "squared_hinge"],
            },
            "RidgeClassifier": {
                "clf__alpha": [0.0, 1e-3, 1.0],
                "clf__normalize": [False, True],
            },
            "SGDClassifier": {
                "clf__loss": ["squared_hinge"],
                "clf__alpha": [1e-3, 1e-2],
            },
            "ExtraTreesClassifier": {
                "clf__n_estimators": [200, 500],
                "clf__criterion": ["gini"],
                "clf__min_samples_split": [2],
                "clf__min_samples_leaf": [1],
            },
            "RandomForestClassifier": {
                "clf__n_estimators": [200, 500],
                "clf__criterion": ["gini"],
                "clf__min_samples_split": [2],
                "clf__min_samples_leaf": [1],
            },
            "BaggingClassifier": {
                "clf__n_estimators": [30, 50, 100],
                "clf__max_samples": [1.0, 0.8],
            },
            "AdaBoostClassifier": {
                "clf__n_estimators": [50, 100],
                "clf__learning_rate": [1.0, 0.9, 0.8],
                "clf__algorithm": ["SAMME.R"],
            },
        }
    elif settings_flag == "normal":
        search_space = {
            "LinearSVC": {
                "clf__C": [1e-2, 1e-1],
                "clf__max_iter": [800, 1000],
            },
            "SVC": {
                "clf__C": [1.0],
                "clf__gamma": [0.5, 0.1, 0.01],
                "clf__kernel": ["rbf"],
            },
            "DecisionTreeClassifier": {
                "clf__criterion": ["gini"],
                "clf__splitter": ["best"],
                "clf__min_samples_split": [2],
                "clf__min_samples_leaf": [1],
            },
            "MLPClassifier": {
                "clf__hidden_layer_sizes": [(100,)],
                "clf__solver": ["adam"],
                "clf__alpha": [1e-2],
                "clf__batch_size": ["auto"],
                "clf__learning_rate": ["adaptive"],
                "clf__max_iter": [1000],
            },
            "LogisticRegression": {
                "clf__solver": ["newton-cg", "lbfgs", "sag"],
                "clf__C": np.logspace(0, 4, 10),
            },
            "PassiveAggressiveClassifier": {
                "clf__C": [0.0, 0.01, 0.1, 1.0],
                "clf__loss": ["hinge", "squared_hinge"],
            },
            "RidgeClassifier": {
                "clf__alpha": [0.0, 1e-3, 1.0],
                "clf__normalize": [False, True],
            },
            "SGDClassifier": {
                "clf__loss": ["squared_hinge"],
                "clf__alpha": [1e-3, 1e-2],
            },
            "ExtraTreesClassifier": {
                "clf__n_estimators": [200, 500],
                "clf__criterion": ["gini"],
                "clf__min_samples_split": [2],
                "clf__min_samples_leaf": [1],
            },
            "RandomForestClassifier": {
                "clf__n_estimators": [200, 500],
                "clf__criterion": ["gini"],
                "clf__min_samples_split": [2],
                "clf__min_samples_leaf": [1],
            },
            "BaggingClassifier": {
                "clf__n_estimators": [30, 50, 100],
                "clf__max_samples": [1.0, 0.8],
            },
            "AdaBoostClassifier": {
                "clf__n_estimators": [50, 100],
                "clf__learning_rate": [1.0, 0.9, 0.8],
                "clf__algorithm": ["SAMME.R"],
            },
        }
    elif settings_flag == "detailed":
        search_space = {
            "LinearSVC": {
                "clf__C": [1e-2, 1e-1, 1, 1e1, 1e2, 1e3],
                "clf__max_iter": [800, 1000, 1200, 1500, 2000],
            },
            "SVC": {
                "clf__C": [1.0, 1e-2, 1e-1, 1, 1e1],
                "clf__gamma": [0.5, 0.1, 0.01, 0.001, 0.0001],
                "clf__kernel": ["rbf", "linear", "sigmoid", "poly"],
            },
            "DecisionTreeClassifier": {
                "clf__criterion": ["gini", "entropy"],
                "clf__splitter": ["best", "random"],
                "clf__min_samples_split": [2, 3, 4],
                "clf__min_samples_leaf": [1, 2, 3],
            },
            "MLPClassifier": {
                "clf__hidden_layer_sizes": [(100,), (100, 64), (100, 64, 16)],
                "clf__solver": ["adam", "lbfgs", "sgd", "adam"],
                "clf__alpha": [1e-2, 1e-3],
                "clf__batch_size": ["auto"],
                "clf__learning_rate": ["adaptive", "invscaling", "constant"],
                "clf__max_iter": [1000],
            },
            "LogisticRegression": {
                "clf__solver": ["newton-cg", "lbfgs", "sag"],
                "clf__C": np.logspace(0, 4, 10),
            },
            "PassiveAggressiveClassifier": {
                "clf__C": [0.0, 0.01, 0.1, 1.0],
                "clf__loss": ["hinge", "squared_hinge"],
            },
            "RidgeClassifier": {
                "clf__alpha": [0.0, 1e-3, 1.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                "clf__normalize": [False, True],
            },
            "SGDClassifier": {
                "clf__loss": ["squared_hinge", "hinge", "log"],
                "clf__alpha": [1e-3, 1e-2],
            },
            "ExtraTreesClassifier": {
                "clf__n_estimators": [200, 500],
                "clf__criterion": ["gini", "entropy"],
                "clf__min_samples_split": [2, 3, 4],
                "clf__min_samples_leaf": [1, 2, 3],
            },
            "RandomForestClassifier": {
                "clf__n_estimators": [200, 500],
                "clf__criterion": ["gini", "entropy"],
                "clf__min_samples_split": [2, 3, 4],
                "clf__min_samples_leaf": [1, 2, 3],
            },
            "BaggingClassifier": {
                "clf__n_estimators": [10, 30, 50, 100, 200],
                "clf__max_samples": [1.0, 0.8, 0.4, 0.2],
            },
            "AdaBoostClassifier": {
                "clf__n_estimators": [30, 50, 100, 150, 300],
                "clf__learning_rate": [1.0, 0.9, 0.8, 0.4],
                "clf__algorithm": ["SAMME.R", "SAMME"],
            },
        }

    return search_space, classifiers
