from sklearn.cluster import (
    AffinityPropagation,
    Birch,
    KMeans,
    MiniBatchKMeans,
    MeanShift,
)
import numpy as np


def unsupervised_classifier_defs(setting_flag, n_clusters):
    """Establish supervised classifier definitions which are non-specific to embeddor,
    and therefore, non-specific to the natural language processing application

    Parameters

    ----------
    setting_flag : str
      Either 'normal' or 'detailed'; a setting which determines the number of hyperparameter combinations
      tested during the grid search. For instance, a dataset of 50,000 samples may run for hours on the
      'normal' setting but for days on 'detailed'.
    n_clusters : int,
      Number of clusters to organize the text data into. Usually set to the number of unique categories within data.

    Returns

    -------
    search_space: dict
      Hyperparameter instances for each clusterer
    clusterers: dict
      Contains sklearn cluster instances
    """

    clusterers = {
        "AffinityPropagation": AffinityPropagation(),
        "Birch": Birch(),
        "KMeans": KMeans(),
        "MiniBatchKMeans": MiniBatchKMeans(),
        "MeanShift": MeanShift(),
    }

    if setting_flag == "normal":
        search_space = {
            "AffinityPropagation": {
                "clf__damping": [0.5, 0.9],
                "clf__max_iter": [200, 600],
            },
            "Birch": {
                "clf__threshold": [0.5, 0.75, 1.0],
                "clf__n_clusters": [n_clusters],
                "clf__branching_factor": [50, 100],
            },
            "KMeans": {
                "clf__n_clusters": [n_clusters],
                "clf__init": ["k-means++", "random"],
                "clf__n_init": [10, 50, 100],
            },
            "MiniBatchKMeans": {
                "clf__n_clusters": [n_clusters],
                "clf__init": ["k-means++", "random"],
                "clf__n_init": [3, 10, 20],
            },
            "MeanShift": {
                "clf__bandwidth": [None],
                "clf__bin_seeding": [False, True],
                "clf__max_iter": [300, 600],
            },
        }
    if setting_flag == "detailed":
        search_space = {
            "AffinityPropagation": {
                "clf__damping": [0.5, 0.75, 0.9],
                "clf__max_iter": [200, 600, 800, 1000, 1200],
            },
            "Birch": {
                "clf__threshold": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "clf__n_clusters": [n_clusters],
                "clf__branching_factor": [25, 50, 100, 200],
            },
            "KMeans": {
                "clf__n_clusters": [n_clusters],
                "clf__init": ["k-means++", "random"],
                "clf__n_init": [10, 50, 100],
                "clf__max_iter": [300, 600],
            },
            "MiniBatchKMeans": {
                "clf__n_clusters": [n_clusters],
                "clf__init": ["k-means++", "random"],
                "clf__n_init": [3, 10, 20],
                "clf__max_iter": [100, 300],
            },
            "MeanShift": {
                "clf__bandwidth": [None],
                "clf__bin_seeding": [False, True],
                "clf__max_iter": [300, 600, 1000],
            },
        }

    return search_space, clusterers