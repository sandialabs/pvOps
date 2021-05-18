from scipy.interpolate import interp1d
import sys
from keras.layers import Bidirectional
from keras.layers import Lambda, dot, concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.layers import InputLayer, Input
from keras.layers import Flatten, Permute, Activation, RepeatVector, Add
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer, f1_score, homogeneity_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from scipy.sparse import issparse
import random
import nltk
import traceback
import pickle
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
import warnings
import copy
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelBinarizer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras import backend as K
from keras.metrics import RootMeanSquaredError
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.callbacks import Callback
from keras.layers import LSTM, Dropout, Dense, Lambda, dot, concatenate, Flatten, Reshape, GlobalAveragePooling2D, GlobalMaxPooling1D
from sklearn.metrics import accuracy_score
# from keract import get_activations
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.engine.input_layer import Input
import keras
import os
from sklearn.model_selection import KFold
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

import os
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin'
keras.backend.clear_session()
# from keract import get_activations

sys.path.append('.')


#############################################################
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#############################################################


# LSTM DeltaInferencer variables


keras.backend.clear_session()
warnings.filterwarnings("ignore")


# Scoring


def get_low_variance_columns(dframe=None, columns=[],
                             skip_columns=[], thresh=0.0,
                             autoremove=True):
    """
    Wrapper for sklearn VarianceThreshold for use on pandas dataframes.
    """
    print("Finding low-variance features.")
    # try:
    # get list of all the original df columns
    all_columns = dframe.columns

    # remove `skip_columns`
    remaining_columns = all_columns.drop(skip_columns)

    # get length of new index
    max_index = len(remaining_columns) - 1

    # get indices for `skip_columns`
    skipped_idx = [all_columns.get_loc(column)
                   for column
                   in skip_columns]

    # adjust insert location by the number of columns removed
    # (for non-zero insertion locations) to keep relative
    # locations intact
    for idx, item in enumerate(skipped_idx):
        if item > max_index:
            diff = item - max_index
            skipped_idx[idx] -= diff
        if item == max_index:
            diff = item - len(skip_columns)
            skipped_idx[idx] -= diff
        if idx == 0:
            skipped_idx[idx] = item

    # get values of `skip_columns`
    skipped_values = dframe.iloc[:, skipped_idx].values

    # get dataframe values
    X = dframe.loc[:, remaining_columns].values

    # instantiate VarianceThreshold object
    vt = VarianceThreshold(threshold=thresh)

    # fit vt to data
    vt.fit(X)

    print(vt.variances_)
    print(remaining_columns)

    # get the indices of the features that are being kept
    feature_indices = vt.get_support(indices=True)

    # remove low-variance columns from index
    feature_names = [remaining_columns[idx]
                     for idx, _
                     in enumerate(remaining_columns)
                     if idx
                     in feature_indices]

    # get the columns to be removed
    removed_features = list(np.setdiff1d(remaining_columns,
                                         feature_names))
    print("Found {0} low-variance columns."
          .format(len(removed_features)))

    # remove the columns
    if autoremove:
        print("Removing low-variance features.")
        # remove the low-variance columns
        X_removed = vt.transform(X)

        print("Reassembling the dataframe (with low-variance "
              "features removed).")
        # re-assemble the dataframe
        dframe = pd.DataFrame(data=X_removed,
                              columns=feature_names)

        # add back the `skip_columns`
        for idx, index in enumerate(skipped_idx):
            dframe.insert(loc=index,
                          column=skip_columns[idx],
                          value=skipped_values[:, idx])
        print("Succesfully removed low-variance columns.")

    # do not remove columns
    else:
        print("No changes have been made to the dataframe.")

    # except Exception as e:
    #     print(e)
    #     print("Could not remove low-variance features. Something "
    #           "went wrong.")
    #     pass

    return dframe, removed_features

# def supervised_classifier_defs(setting):
#     '''Esablish supervised classifier definitions which are non-specific to embeddor,
#        and therefore, non-specific to the NLP application
#     '''
#     classes = {
#             'LinearSVC': LinearSVC(),
#             'SVC': SVC(),
#             'DecisionTreeClassifier': DecisionTreeClassifier(),
#             'MLPClassifier': MLPClassifier(),
#             'LogisticRegression': LogisticRegression(),
#             'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
#             'RidgeClassifier': RidgeClassifier(),
#             'SGDClassifier': SGDClassifier(),
#             'ExtraTreesClassifier': ExtraTreesClassifier(),
#             'RandomForestClassifier': RandomForestClassifier(),
#             'BaggingClassifier': BaggingClassifier(),
#             'AdaBoostClassifier': AdaBoostClassifier(),
#     }

#     if setting == 'normal':
#       search_space = {
#                         'LinearSVC': {
#                         'clf__C': [1e-2,1e-1],
#                         'clf__max_iter':[800,1000],
#                         },
#                         'SVC': {
#                         'clf__C':[1.],
#                         'clf__gamma': [0.5, 0.1, 0.01],
#                         'clf__kernel': ['rbf']
#                         },
#                         'DecisionTreeClassifier': {
#                         'clf__criterion':['gini'],
#                         'clf__splitter': ['best'],
#                         'clf__min_samples_split': [2],
#                         'clf__min_samples_leaf': [1]
#                         },
#                         'MLPClassifier': {
#                         'clf__hidden_layer_sizes': [(100,)],
#                         'clf__solver': ['adam'],
#                         'clf__alpha': [1e-2],
#                         'clf__batch_size': ['auto'],
#                         'clf__learning_rate': ['adaptive'],
#                         'clf__max_iter': [1000]
#                         },
#                         'LogisticRegression': {
#                         'clf__solver': ['newton-cg', 'lbfgs', 'sag'],
#                         'clf__C': np.logspace(0,4,10)
#                         },
#                         'PassiveAggressiveClassifier': {
#                         'clf__C': [0., 0.01, 0.1, 1.],
#                         'clf__loss': ['hinge', 'squared_hinge'],
#                         },
#                         'RidgeClassifier': {
#                         'clf__alpha': [0.,1e-3,1.],
#                         'clf__normalize': [False,True]
#                         },
#                         'SGDClassifier': {
#                         'clf__loss': ['squared_hinge'],
#                         'clf__alpha': [1e-3,1e-2],
#                         },
#                         'ExtraTreesClassifier': {
#                         'clf__n_estimators': [200,500],
#                         'clf__criterion':['gini'],
#                         'clf__min_samples_split': [2],
#                         'clf__min_samples_leaf': [1]
#                         },
#                         'RandomForestClassifier': {
#                         'clf__n_estimators': [200,500],
#                         'clf__criterion':['gini'],
#                         'clf__min_samples_split': [2],
#                         'clf__min_samples_leaf': [1],
#                         },
#                         'BaggingClassifier': {
#                         'clf__n_estimators': [30,50,100],
#                         'clf__max_samples':[1.0,0.8],
#                         },
#                         'AdaBoostClassifier': {
#                         'clf__n_estimators': [50,100],
#                         'clf__learning_rate':[1.,0.9,0.8],
#                         'clf__algorithm': ['SAMME.R']
#                         }
#                     }
#     if setting == 'detailed':
#       search_space = {
#                         'LinearSVC': {
#                         'clf__C': [1e-2,1e-1,1,1e1,1e2,1e3],
#                         'clf__max_iter':[800,1000,1200,1500,2000],
#                         },
#                         'SVC': {
#                         'clf__C':[1.,1e-2,1e-1,1,1e1],
#                         'clf__gamma': [0.5, 0.1, 0.01, 0.001, 0.0001],
#                         'clf__kernel': ['rbf', 'linear', 'sigmoid', 'poly'],
#                         },
#                         'DecisionTreeClassifier': {
#                         'clf__criterion':['gini','entropy'],
#                         'clf__splitter': ['best','random'],
#                         'clf__min_samples_split': [2,3,4],
#                         'clf__min_samples_leaf': [1,2,3],
#                         },
#                         'MLPClassifier': {
#                         'clf__hidden_layer_sizes': [(100,),(100,64),(100,64,16)],
#                         'clf__solver': ['adam','lbfgs', 'sgd', 'adam'],
#                         'clf__alpha': [1e-2,1e-3],
#                         'clf__batch_size': ['auto'],
#                         'clf__learning_rate': ['adaptive', 'invscaling', 'constant'],
#                         'clf__max_iter': [1000]
#                         },
#                         'LogisticRegression': {
#                         'clf__solver': ['newton-cg', 'lbfgs', 'sag'],
#                         'clf__C': np.logspace(0,4,10)
#                         },
#                         'PassiveAggressiveClassifier': {
#                         'clf__C': [0., 0.01, 0.1, 1.],
#                         'clf__loss': ['hinge', 'squared_hinge'],
#                         },
#                         'RidgeClassifier': {
#                         'clf__alpha': [0.,1e-3,1.,1e-4,1e-3,1e-2,1e-1,1.],
#                         'clf__normalize': [False,True]
#                         },
#                         'SGDClassifier': {
#                         'clf__loss': ['squared_hinge', 'hinge', 'log'],
#                         'clf__alpha': [1e-3,1e-2],
#                         },
#                         'ExtraTreesClassifier': {
#                         'clf__n_estimators': [200,500],
#                         'clf__criterion':['gini','entropy'],
#                         'clf__min_samples_split': [2,3,4],
#                         'clf__min_samples_leaf': [1,2,3]
#                         },
#                         'RandomForestClassifier': {
#                         'clf__n_estimators': [200,500],
#                         'clf__criterion':['gini','entropy'],
#                         'clf__min_samples_split': [2,3,4],
#                         'clf__min_samples_leaf': [1,2,3]
#                         },
#                         'BaggingClassifier': {
#                         'clf__n_estimators': [10,30,50,100,200],
#                         'clf__max_samples':[1.,0.8,0.4,0.2],
#                         },
#                         'AdaBoostClassifier': {
#                         'clf__n_estimators': [30,50,100,150,300],
#                         'clf__learning_rate':[1.0,0.9,0.8,0.4],
#                         'clf__algorithm': ['SAMME.R', 'SAMME']
#                         }
#                     }

#     return search_space, classes


# def supervised_regressor_defs(setting):
#     classes = {
#             'SVR': SVR(),
#             'LinearSVR': LinearSVR(),
#             'ARDRegression': ARDRegression(),
#             'LinearRegression': LinearRegression()
#     }

#     search_space = {
#                     'SVR': {
#                         'clf__kernel': ['poly', 'rbf'],
#                         'clf__degree':[2,3],
#                     },
#                     'LinearSVR': {
#                     },
#                     'ARDRegression': {
#                     },
#                     'LinearRegression': {
#                     }
#     }
#     return search_space, classes


# def classification_deployer(X,y, n_splits, classes, search_space, pipeline_steps, scoring, verbose=1):
#     '''Build classifier evaluator with hyperparameter fine-tuning grid search protocol
#        This function is called by evaluate_classifiers.

#     Parameters
#     ----------
#     X : list
#         List of documents (str)
#     y : list
#         List of labels corresponding with the documents in X
#     n_splits : int
#         Integer defining the number of splits in the cross validation split during training

#     classes

#     search_space

#     pipeline_steps

#     scoring

#     verbose : int
#         Verbosity of the print statements

#     Returns
#     -------
#     DataFrame
#         Summarization of results from all of the classifiers
#     '''

#     rows = []

#     if issparse(X):
#         print('Converting passed data to dense array...')
#         X =  X.toarray()

#     # get position of 'clf' in pipeline_steps
#     idx_clf_pipeline = [i for i,it in enumerate(pipeline_steps) if it[0]=='clf'][0]

#     for key in classes.keys():
#         try:
#             clas = classes[key]
#             space = search_space[key]

#             iter_pipeline_steps = copy.deepcopy(pipeline_steps)
#             iter_pipeline_steps[idx_clf_pipeline] = ('clf',clas)
#             pipe = Pipeline(iter_pipeline_steps)

#             gs_clf = GridSearchCV(pipe, space, scoring = scoring, cv = n_splits,
#                                         n_jobs = -1, return_train_score = True, verbose = verbose)
#             gs_clf.fit(X, y)
#             params = gs_clf.cv_results_['params']
#             scores = []
#             for i in range(n_splits):
#                 r1 = gs_clf.cv_results_[f"split{i}_test_score"]
#                 scores.append(r1.reshape(len(params),1))

#             r2 = gs_clf.cv_results_["mean_fit_time"]

#             all_scores = np.hstack(scores)
#             for param, score, time in zip(params, all_scores, r2):
#                 param['mean_fit_time'] = time
#                 d = {
#                     'estimator': key,
#                     'min_score': min(score),
#                     'max_score': max(score),
#                     'mean_score': np.mean(score),
#                     'std_score': np.std(score),
#                     }
#                 print(d)
#                 rows.append((pd.Series({**param,**d})))

#         except Exception as e:
#             print(F'FAILURE: {key}\n{e}')
#             traceback.print_exc()

#     return pd.concat(rows, axis=1).T

# def ML_eval(ml_class, X_train, y_train, X_test, y_test, cv=False):
#     if cv:
#         skf = StratifiedKFold(n_splits=5)
#         folds = list(skf.split(X_train, y_train))
#         for fold_num, (train_index, test_index) in enumerate(folds):
#             Xtr = X_train[train_index]
#             ytr = y_train[train_index]
#             Xte = X_train[test_index]
#             yte = y_train[test_index]
#             ml_class.fit(Xtr, ytr)
#             print(f'\tIteration {fold_num}: Mean accuracy',ml_class.score(Xte,yte))

#     else:
#         ml_class.fit(X_train, y_train)

#     class_predict = ml_class.predict(X_test)
#     class_train_score = ml_class.score(X_train, y_train)
#     class_test_score = ml_class.score(X_test, y_test)
#     print('Accuracy on training set: {:.3f}'.format(class_train_score))
#     print('Accuracy on test set: {:.3f}'.format(class_test_score))

#     class_eval_analysis = [(1 if (class_predict[i] == y_test[i]) else 0) for i in range(len(y_test))]
#     num_correct = class_eval_analysis.count(1)
#     num_incorrect = class_eval_analysis.count(0)
#     total_num = len(class_eval_analysis)

#     print("{} of {} are correct".format(num_correct, total_num))
#     print(confusion_matrix(y_test,class_predict))
#     print(classification_report(y_test,class_predict))
#     print()
#     print()
#     return ml_class, class_predict, round(class_train_score,4), round(class_test_score,4)


# from pvops.iv import physics_utils


def get_diff_array(sample_V, sample_I, pristine_V, pristine_I, debug=False):

    if debug:
        plt.plot(sample_V, sample_I, label='sample')
        plt.plot(pristine_V, pristine_I, label='pristine')
        plt.legend()
        plt.grid()
        plt.show()

    if sample_V[-1] > pristine_V[-1]:
        if debug:
            print('sample ends in higher V')
        x_smaller = pristine_V
        y_smaller = pristine_I
        x_larger = sample_V
        y_larger = sample_I
        option = 1

    else:
        if debug:
            print('pristine ends in higher V')
        x_smaller = sample_V
        y_smaller = sample_I
        x_larger = pristine_V
        y_larger = pristine_I
        option = 2

    f_interp1 = interp1d(np.flipud(pristine_V), np.flipud(pristine_I),
                         kind='linear', fill_value='extrapolate')
    pristine_I_interp = f_interp1(x_larger)
    pristine_I_interp[pristine_I_interp < 0] = 0

    f_interp2 = interp1d(np.flipud(sample_V), np.flipud(sample_I),
                         kind='linear', fill_value='extrapolate')
    sample_I_interp = f_interp2(sample_V)
    sample_I_interp[sample_I_interp < 0] = 0

    all_V = np.sort(np.unique(np.append(sample_V, pristine_V)))
    all_i1 = f_interp1(all_V)
    all_i1[all_i1 < 0] = 0
    all_i2 = f_interp2(all_V)
    all_i2[all_i2 < 0] = 0

    if option == 1:
        diff = all_i2 - all_i1
    if option == 2:
        diff = all_i1 - all_i2

    if debug:
        plt.plot(pristine_V, pristine_I, label='prist')
        plt.plot(sample_V, sample_I, label='sample')
        plt.plot(all_V, diff, label='diff')
        plt.grid()
        plt.legend()
        plt.show()

    return all_V, diff


def feature_generation(bigdf):

    pristine = bigdf[bigdf['mode'] == 'Pristine array']
    pristine.sort_values(by=['E', 'T'], ascending=[False, False], inplace=True)
    sub = bigdf[bigdf['mode'] != 'Pristine array']

    pristine_V = pristine['voltage'].values[0]
    pristine_I = pristine['current'].values[0]

    n = 0
    sample_V = sub['voltage'].values[n]
    sample_I = sub['current'].values[n]

    diffs = []
    for n in range(len(sub)):
        sample_V = sub['voltage'].values[n]
        sample_I = sub['current'].values[n]
        v, i_diff = get_diff_array(sample_V, sample_I, pristine_V, pristine_I)
        diffs.append(i_diff)

    sub['diff'] = diffs

    differential = []
    for ind, row in sub.iterrows():
        Is2 = row['current']
        differential.append(
            np.array([0]+[j-i for i, j in zip(Is2[:-1], Is2[1:])]))

    sub['differential'] = differential
    return sub


def balance_df(df, balance_tactic='truncate', ycol='conf'):

    if balance_tactic is 'gravitate':
        len_rows = len(df.index)
        modes = set(df[ycol].tolist())

        num_modes = len(modes)
        avg_rows_in_mode = int(len_rows / num_modes)

        balanced_df = pd.DataFrame()
        for md in modes:
            df_mode = df[df[ycol] == md]

            if len(df_mode.index) > avg_rows_in_mode:
                # majority class, must downsample

                resmpl_len = int(len(df_mode) * 0.7)

                if resmpl_len < avg_rows_in_mode:
                    # if 0.7 times current value is less than avg rows per mode, utilize ag rows per mode
                    resmpl_len = avg_rows_in_mode

                print('\t[Class {}]: Majority, {} --> {} gravitating to {}'.format(md,
                                                                                   len(df_mode.index), resmpl_len, avg_rows_in_mode))

                df_resampled = resample(df_mode,
                                        replace=True,
                                        n_samples=resmpl_len,
                                        random_state=123)

            elif len(df_mode.index) < avg_rows_in_mode:
                # minority class, must upsample

                resmpl_len = int(len(df_mode) * 1.3)
                if resmpl_len > avg_rows_in_mode:
                    # if 1.3 times current value is greater than avg rows per mode, utilize ag rows per mode
                    resmpl_len = avg_rows_in_mode

                print('\t[Class {}]: Minority, {} --> {} gravitating to {}'.format(md,
                                                                                   len(df_mode.index), resmpl_len, avg_rows_in_mode))

                df_resampled = resample(df_mode,
                                        replace=True,
                                        n_samples=resmpl_len,
                                        random_state=234)
            else:
                # Equal to average, do nothing

                print('\t[Class {}]: Equal, {} = {}'.format(
                    md, len(df_mode.index), avg_rows_in_mode))
                df_resampled = df_mode

            balanced_df = pd.concat([balanced_df, df_resampled])

    elif balance_tactic is 'truncate':
        modes = set(df[ycol].tolist())

        balanced_df = pd.DataFrame()

        lens = []
        for md in modes:
            df_mode = df[df[ycol] == md]
            lens.append(len(df_mode.index))

        minlen = min(lens)

        for md in modes:
            df_mode = df[df[ycol] == md]
            if len(df_mode.index) > minlen:
                print('\t[Class {}]: Resampled, {} --> {}'.format(md,
                                                                  len(df_mode.index), minlen))
                df_resampled = resample(df_mode,
                                        replace=True,
                                        n_samples=minlen,
                                        random_state=123)
            elif minlen == len(df_mode.index):
                print('\t[Class {}]: Resampled, {} == {}'.format(
                    md, len(df_mode.index), minlen))
                df_resampled = df_mode
            balanced_df = pd.concat([balanced_df, df_resampled])

    else:
        raise Exception(
            "Invalid balancedf variable: {}".format(balance_tactic))

    return balanced_df


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def plot_profiles(bigdf, colx, coly, cmap_name='brg'):

    fig = plt.figure(figsize=(10, 5))
    i = 0
    labeled = list(sorted(list(set(bigdf['mode'].tolist()))))
    cmap = plt.cm.get_cmap(cmap_name, len(labeled))
    for mode in labeled:
        df = bigdf[bigdf['mode'] == mode]
        s2is = df[coly].tolist()
        x = range(len(s2is[0]))  # df[colx].tolist()
        s2ismean = np.array(s2is).mean(axis=0)
        s2isstd = np.array(s2is).std(axis=0)
        s2lowerbound = [s2ismean[i] - s2isstd[i] for i in range(len(s2ismean))]
        s2upperbound = [s2ismean[i] + s2isstd[i] for i in range(len(s2ismean))]
        plt.plot(x, s2ismean, color=cmap(i), label='{}'.format(mode))
        plt.fill_between(x,
                         s2lowerbound,
                         s2upperbound,
                         facecolor=cmap(i),
                         alpha=0.2)
        i += 1
    plt.xlabel(colx)
    plt.ylabel(coly)
    plt.legend()
    return fig


# RESTRUCTURING


def _convert_ivdata_to_cnn_structure(df, params):
    data = []
    for ind, row in df.iterrows():
        dat = []
        for i in range(len(row[params[0]])):
            da = []
            for param in params:
                da.append(row[param][i])
            dat.append(da)
        data.append(dat)
    return np.asarray(data)


def _convert_ivdata_to_lstm_multihead_structure(df, params):

    data_features = []
    for param in params:
        data_features.append(bigdf[param].values)

    def list_to_lol(lst, ln):
        # lst: list input
        # len: num vals in each sublist
        maxlen = len(lst)
        i = 0
        l = []
        while i+ln < maxlen:
            x = lst[i:i+ln]
            l.append(x)
            i += ln
        # print(l)
        return l

    n_filters = 10.
    print(f'Making {n_filters} and sample {len(data_features[0][0])}')
    length = int(len(data_features[0][0]) / n_filters)  # lists

    restructured_data_features = [[] for _ in range(len(params))]
    for ividx in range(len(data_features[0])):
        for i in range(len(params)):
            restructured_data_features[i].append(
                list_to_lol(data_features[i][ividx], length))

    for i in range(len(params)):
        restructured_data_features[i] = np.asarray(
            restructured_data_features[i])

    return restructured_data_features


def _grab_structure_lstm_multihead_structure(restructured_data_features, idxs):
    nparms = len(restructured_data_features)
    return [restructured_data_features[i][idxs] for i in range(nparms)]


class NNClassifier:
    def __init__(self):
        pass

    def load_data(self, df, iv_col_dict, balance_tactic='truncate'):
        train_size = 0.9
        self.model_type = 'classifier'
        # Feature generation
        df = feature_generation(df)
        bal_df = balance_df(df, balance_tactic, ycol='mode')
        # Label binarizer
        self.lb = LabelBinarizer()
        ys = bal_df["mode"].tolist()
        ys = self.lb.fit_transform(ys)

        if True:

            def unison_shuffled_copies(a, b):
                assert len(a) == len(b)
                p = np.random.permutation(len(a))
                return a[p], b[p]
            # shuffle
            total_points_in_train = int(train_size*len(ys))
            print(total_points_in_train)
            print(ys.shape)
            print(Xs.shape)
            ys, Xs = unison_shuffled_copies(ys, Xs)
            self.train_ys, self.test_ys = ys[:total_points_in_train], ys[total_points_in_train+1:]
            self.train_Xs, self.test_Xs = Xs[:total_points_in_train], Xs[total_points_in_train+1:]

            # self.train_ys,_, self.train_Xs,_ = train_test_split(self.train_ys, self.train_Xs, test_size=0.0)
            # self.test_ys,_, self.test_Xs,_ = train_test_split(self.test_ys, self.test_Xs, test_size=0.0)
            self.train_ys, self.train_Xs = unison_shuffled_copies(
                self.train_ys, self.train_Xs)
            self.test_ys, self.test_Xs = unison_shuffled_copies(
                self.test_ys, self.test_Xs)

            print(self.train_ys.shape)
            print(self.test_ys.shape)
            print(self.train_Xs.shape)
            print(self.test_Xs.shape)

            ys = None
            Xs = None
            Xs_tech = None

    def prep_data4(self, price_df):

        bigdf, removed_features = get_low_variance_columns(
            dframe=bigdf, columns=[], skip_columns=['date'], thresh=0.05, autoremove=True)

        bigdf.index = bigdf['date']
        dfs = [group[1] for group in bigdf.groupby(bigdf.index.date)]

        return dfs

    def structure(self, dfs, model_type='classifier', model_name='2DCNN', multihead=False, prev_interval_days=15, last_N_mins_of_day=31, multiheaded_subpattern_length=15, smooth_fraction=None, smooth_pad_len=None):

        self.model_type = model_type
        self.model_name = model_name
        self.multihead = multihead

        N = 0
        Xs = []
        ys = []

        for i, _ in enumerate(dfs[prev_interval_days:]):
            # print(i,i+prev_interval_days)
            df_out_iter = pd.concat(dfs[i:i+prev_interval_days])
            del df_out_iter['date']
            X = df_out_iter.values
            df_answer = dfs[i+prev_interval_days]
            y = df_answer.iloc[:last_N_mins_of_day]['pl_perc'].mean()
            # y =
            # print(y)
            # print(df_answer['pl_perc'])

            if np.isnan(y):
                print(X.shape)
                # print(df_answer['close'])
                print(df_answer['close'].isna().sum())
                x = df_answer['close'].isna().tolist()
                print([i for i, a in enumerate(x) if a])
                print(y)

                N += 1
                if N > 5:
                    sys.exit()
            # print(X.shape)
            Xs.append(X)
            ys.append(y)
        Xs = np.asarray(Xs).astype('float32')
        ys = np.asarray(ys).astype('float32')

        print("XS SHAPE:", Xs.shape)

        if self.model_name == '2DCNN':
            Xs = Xs.reshape(Xs.shape[0], Xs.shape[1], Xs.shape[2], 1)
        elif self.model_name == 'LSTM':
            if multihead:
                def divide_chunks(l, n):

                    # looping till length l
                    for i in range(0, len(l), n):
                        yield l[i:i + n]

                multiheaded_Xs = []
                for x in Xs:
                    multiheaded_Xs.append(np.asarray(
                        list(divide_chunks(x, multiheaded_subpattern_length))))

                print(Xs.shape)
                print(np.asarray(multiheaded_Xs).shape)
                Xs = np.asarray(multiheaded_Xs).astype('float32')

            else:
                # no restructuring necessary
                pass

        return Xs, ys

    def data_split(self, train_Xs, train_ys, test_Xs, test_ys):
        def unison_shuffled_copies(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        train_Xs, train_ys = unison_shuffled_copies(train_Xs, train_ys)

        self.train_Xs = train_Xs
        self.train_ys = train_ys
        self.test_Xs = test_Xs
        self.test_ys = test_ys

    def get_model(self, input_sample_shape=None, n_features=None, use_attention_LSTM=False, units=100, dropouts=0.5):
        # name = '2DCNN' or else

        if self.model_name == '2DCNN':
            # self._2D_cnn_multihead()
            self._2D_cnn(input_sample_shape)
            print(self.model.summary())
            return self.model

        if self.model_name == 'LSTM':
            if self.multihead:
                self._lstm_multihead(
                    input_sample_shape, n_features, use_attention_LSTM, units, dropouts)
            else:
                self._lstm_singlehead(input_sample_shape)

        print(self.model.summary())
        return self.model

    def train(self, n_split=5, batch_size=8, max_epochs=100, verbose=0, save_info_folder=''):

        cv = KFold(n_splits=n_split)
        cvscores = []
        for train_idx, test_idx in cv.split(self.train_Xs):

            ytr, yte = self.train_ys[train_idx], self.train_ys[test_idx]

            xtr, xte = self.train_Xs[train_idx], self.train_Xs[test_idx]
            print('ys: ', ytr.shape, yte.shape)
            print('xs: ', xtr.shape, xte.shape)

            if self.multihead:
                xtr = xtr.reshape(
                    xtr.shape[3], xtr.shape[0], xtr.shape[1], xtr.shape[2])
                xte = xte.reshape(
                    xte.shape[3], xte.shape[0], xte.shape[1], xte.shape[2])
                print('reshaped', 'xs: ', xtr.shape, xte.shape)
                # xtr = xtr.tolist()
                # xte = xte.tolist()

            # if self.use_attention:
            #     history = self.model.fit(xtr,ytr, epochs=max_epochs, batch_size=batch_size, verbose = verbose,
            #                             callbacks=[VisualiseAttentionMap(save_info_folder,self.X_test,max_epochs)])
            #     scores = self.model.evaluate(xte, yte, verbose=verbose)
            if True:
                if self.model_type == 'classifier':
                    history = self.model.fit(x=xtr,
                                             # y=dict(zip(self.y_names,ytr)),
                                             y=ytr,
                                             epochs=max_epochs, batch_size=batch_size, verbose=verbose)
                    # scores = self.model.evaluate(xte, dict(zip(self.y_names,yte)), verbose=verbose)
                    scores = self.model.evaluate(xte, yte, verbose=verbose)
                else:

                    print('fit')
                    history = self.model.fit(
                        xtr, ytr, epochs=max_epochs, batch_size=batch_size, verbose=verbose)
                    print('eval')

                    # dataAugmentaion = TimeseriesGenerator(xtr, xtr)
                    scores = self.model.evaluate(xte, yte, verbose=verbose)

            if verbose > 0:
                print(self.model.metrics_names, scores)
                print('\tnum.train {} num.test {}'.format(len(xtr), len(xte)))
            cvscores.append(scores[1])

        self.history = history
        return

    def predict(self):

        yhat = self.model.predict(self.test_Xs, batch_size=8, verbose=1)

        if self.model_type == 'regressor':
            r_squared = r2_score(self.test_ys, yhat)
            mae = mean_absolute_error(self.test_ys, yhat)
            rmse = mean_squared_error(self.test_ys, yhat)
            # correlation_matrix = np.corrcoef(self.test_ys, yhat)
            # correlation_xy = correlation_matrix[0,1]
            # r_squared = correlation_xy**2
            plt.figure()
            plt.plot(self.test_ys, yhat, 'bo')
            plt.xlabel('Truths')
            plt.ylabel('Predicted')
            plt.title(f"R-squared: {r_squared}, MAE: {mae}, RMSE: {rmse}")
            plt.grid()
            plt.show()

            return yhat, self.test_ys

        elif self.model_type == 'classifier':

            ytest = []
            for evaluation in self.test_ys:
                ytest.append(np.argmax(evaluation))

            ypred = []
            for evaluation in yhat:
                ypred.append(np.argmax(evaluation))

            accuracy = accuracy_score(ytest, ypred)
            print(classification_report(ytest, ypred))
            print(confusion_matrix(ytest, ypred))
            print(f"FINAL ACCURACY on TEST: {accuracy}")

            return accuracy, self.test_ys, yhat

    def save_model_and_specs(self, folder):
        if self.model_type == 'classifier':
            metric = 'categorical_accuracy'
        else:
            metric = 'mean_absolute_error'  # 'root_mean_squared_error'

        self.model.save_weights(folder+'weights.h5')

        # plot_model(self.model, show_shapes=False, show_layer_names=False, to_file='D://model_architecture.png')

        # pickle.dump(self.model, open(f'model.h5', 'wb'))
        self.model.save(folder+'model')

        plt.close()
        # print(history.history.keys())
        plt.plot(self.history.history[metric])
        # plt.plot(history.history['val_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel(metric)
        plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(f'{metric}_v_epoch.png')
        plt.close()
        # "Loss"
        plt.plot(self.history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig('validation_v_epoch.png')
        plt.close()
        return

    def load_model(self, folder):
        self.model = load_model(folder+'model')
        print(self.model.summary())


# class VisualiseAttentionMap(Callback):
#     def __init__(self, folder, X_test, max_epochs, type_plot='line'):
#         # type_plot: 'heat' or 'line'
#         self.folder = folder
#         self.X_test = X_test
#         self.max_epochs = max_epochs
#         self.type_plot = type_plot

#     def on_epoch_end(self, epoch, logs=None):
#         attention_map = get_activations(
#             self.model, self.X_test, layer_name='attention_weight')['attention_weight']

#         print('epoch', epoch)
#         print('attention', attention_map)
#         print('attention shape', attention_map.shape,
#               'len xtest', len(self.X_test), len(self.X_test[0]))

#         if self.type_plot == 'heat':
#             # top is attention map.
#             # bottom is ground truth.
#             plt.imshow(attention_map, cmap='hot', aspect='auto')

#             iteration_no = str(epoch+1).zfill(3)
#             plt.axis('off')
#             plt.title(f'Iteration {iteration_no} / {self.max_epochs}')
#             plt.savefig('{}//epoch_{}.png'.format(self.folder, iteration_no))
#             plt.close()
#             plt.clf()

#         if self.type_plot == 'line':

#             x_axis = [f'i-{i}' for i in range(1, len(self.X_test[0])+1)]
#             for x in self.X_test:
#                 plt.plot(x_axis, x, 'k')
#                 plt.scatter(x_axis, x, c=x)

#             plt.title(f'Iteration {iteration_no} / {self.max_epochs}')
#             plt.savefig('{}//epoch_{}.png'.format(self.folder, iteration_no))
#             plt.close()
#             plt.clf()
