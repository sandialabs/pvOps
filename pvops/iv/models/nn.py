from scipy.interpolate import interp1d
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

import keras
from keras.layers import Lambda, concatenate
from keras.layers import LSTM, Flatten, Input, Dropout, Dense, Conv1D
from keras.models import Sequential, Model
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

keras.backend.clear_session()


def get_diff_array(sample_V, sample_I, pristine_V, pristine_I, debug=False):
    """Generate IV current differential between sample and pristine.

    Parameters

    ----------
    sample_V : array
        Voltage array for a sample's IV curve
    sample_I : array
        Current array for a sample's IV curve
    pristine_V : array
        Voltage array for a pristine IV curve
    pristine_I : array
        Current array for a pristine IV curve

    Returns

    -------
    all_V : array
        Combined voltage array
    diff : array
        Current differential calculation
    """

    if sample_V[-1] > pristine_V[-1]:
        x_larger = sample_V
        option = 1

    else:
        x_larger = pristine_V
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

    return all_V, diff


def feature_generation(bigdf, iv_col_dict,
                       pristine_mode_identifier='Pristine array'):
    """Generate features of an IV curve data set including:
    1. Current differential between a sample IV curve and a pristine IV curve
    2. Finite difference of the IV curve along the y-axis, indicating the
    slope of the cuve.

    Parameters

    ----------
    bigdf : dataframe
        Dataframe holding columns from `iv_col_dict`, except for the 
        `derivative` and `current_diff` which are calculated here.
    iv_col_dict : dict
        Dictionary containing definitions for the column names in `df`
        - **current** (*str*): column name for IV current arrays.
        - **voltage** (*str*): column name for IV voltage arrays.
        - **mode** (*str*): column name for failure mode identifier.
        - **irradiance** (*str*): column name for the irradiance definition.
        - **temperature** (*str*): column name for the temperature definition.
        - **derivative** (*str*): column name for the finite derivative, as
          calculated in this function.
        - **current_diff** (*str*): column name for current differential, as
          calculated in `get_diff_array`.
    pristine_mode_identifier : str
        Pristine array identifier. The pristine curve is utilized in
        `get_diff_array`. If multiple rows exist at this
        `pristine_mode_identifier`, the one with the highest irradiance and
        lowest temperature definitions is chosen.

    Returns

    -------
    all_V : array
        Combined voltage array
    diff : array
        Current differential calculation
    """
    current_col = iv_col_dict["current"]
    voltage_col = iv_col_dict["voltage"]
    failure_mode_col = iv_col_dict["mode"]
    irradiance_col = iv_col_dict["irradiance"]
    temperature_col = iv_col_dict["temperature"]
    derivative_col = iv_col_dict["derivative"]
    current_diff_col = iv_col_dict["current_diff"]

    pristine = bigdf[bigdf[failure_mode_col] == pristine_mode_identifier]
    pristine.sort_values(by=[irradiance_col, temperature_col], ascending=[
                         False, True], inplace=True)
    sub = bigdf[bigdf[failure_mode_col] != pristine_mode_identifier]

    pristine_V = pristine[voltage_col].values[0]
    pristine_I = pristine[current_col].values[0]

    n = 0
    sample_V = sub[voltage_col].values[n]
    sample_I = sub[current_col].values[n]

    diffs = []
    for n in range(len(sub)):
        sample_V = sub[voltage_col].values[n]
        sample_I = sub[current_col].values[n]
        v, i_diff = get_diff_array(sample_V, sample_I, pristine_V, pristine_I)
        diffs.append(i_diff)
    sub[current_diff_col] = diffs

    finite_derivatives = []
    for ind, row in sub.iterrows():
        Is2 = row[current_col]
        finite_derivatives.append(
            np.array([0] + [j - i for i, j in zip(Is2[:-1], Is2[1:])]))
    sub[derivative_col] = finite_derivatives
    return sub


def balance_df(df, iv_col_dict, balance_tactic='truncate'):
    """Balance data so that an equal number of samples are found at each
    unique `ycol` definition.

    Parameters

    ----------
    bigdf : dataframe
        Dataframe containing the `ycol` column.
    iv_col_dict : dict
        Dictionary containing at least the following definition:
        - **mode** (*str*), column in `df` which holds the definitions
        which must contain a balanced number of samples for each unique
        definition.
    balance_tactic : str
        mode balancing tactic, either "truncate"
        or "gravitate". Truncate will utilize the exact same number of samples
        for each category. Gravitate will sway the original number of samples
        towards a central target.

    Returns

    -------
    dataframe, balanced according to the `balance_tactic`.
    """

    ycol = iv_col_dict['mode']

    print("Balance data by mode:")
    if balance_tactic == 'gravitate':
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

                print('\t[Class {}]: Majority, {} --> {}'
                      ' gravitating to {}'.format(md,
                                                  len(df_mode.index),
                                                  resmpl_len,
                                                  avg_rows_in_mode))

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

                print('\t[Class {}]: Minority, {} --> {}'
                      'gravitating to {}'.format(md,
                                                 len(df_mode.index),
                                                 resmpl_len,
                                                 avg_rows_in_mode))

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

    elif balance_tactic == 'truncate':
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


def plot_profiles(df, colx, coly, iv_col_dict, cmap_name='brg'):
    """Plot curves also with area colorizations to display the deviation
    in definitions.

    Parameters

    ----------
    df : dataframe
        Dataframe containing the `colx`, `coly`, and iv_col_dict['mode'] column
    colx : str
        Column containing x-axis array of data on each sample.
    coly : str
        Column containing y-axis array of data on each sample.
    iv_col_dict : dict
        Dictionary containing at least the following definition:
        - **mode** (*str*), column in `df` which holds the definitions
        which must contain a balanced number of samples for each unique
        definition.
    cmap_name : str
        Matplotlib colormap.

    Returns

    -------
    matplotlib figure
    """
    mode_col = iv_col_dict["mode"]

    # Check whether the colx has equivalent values in every row
    xs = df[colx].tolist()
    try:
        all_equal = all((x == xs[0]).all() for x in xs)
    except AttributeError:
        all_equal = False
    else:
        all_equal = True

    fig = plt.figure(figsize=(10, 5))
    i = 0
    labeled = list(sorted(list(set(df[mode_col].tolist()))))
    cmap = plt.cm.get_cmap(cmap_name, len(labeled))
    for mode in labeled:
        subdf = df[df[mode_col] == mode]
        s2is = subdf[coly].tolist()
        if all_equal:
            x = subdf[colx].values[0]
        else:
            x = range(len(subdf[colx].values[0]))
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
    if all_equal:
        plt.xlabel(colx)
    else:
        plt.xlabel(colx + " index")
    plt.ylabel(coly)
    plt.legend()
    return fig


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


def _convert_ivdata_to_lstm_multihead_structure(df, params, n_filters=10.):

    data_features = []
    for param in params:
        data_features.append(df[param].values)

    def _list_to_lol(lst, ln):
        # lst: list input
        # len: num vals in each sublist
        maxlen = len(lst)
        i = 0
        lol = []
        while i + ln < maxlen:
            x = lst[i:i + ln]
            lol.append(x)
            i += ln
        return lol

    print(f'Making {n_filters} and sample {len(data_features[0][0])}')
    length = int(len(data_features[0][0]) / n_filters)  # lists

    restructured_data_features = [[] for _ in range(len(params))]
    for ividx in range(len(data_features[0])):
        for i in range(len(params)):
            restructured_data_features[i].append(
                _list_to_lol(data_features[i][ividx], length))

    for i in range(len(params)):
        restructured_data_features[i] = np.asarray(
            restructured_data_features[i])

    return restructured_data_features


def _grab_structure_lstm_multihead_structure(restructured_data_features, idxs):
    nparms = len(restructured_data_features)
    return [restructured_data_features[i][idxs] for i in range(nparms)]


def classify_curves(df, iv_col_dict, nn_config):
    """Build and evaluate an IV trace failure `mode` classifier.

    Parameters

    ----------
    df : dataframe
        Data with columns in `iv_col_dict`
    iv_col_dict : dict
        Dictionary containing definitions for the column names in `df`
        **mode** (*str*): column name for failure mode identifier
    nn_config : dict
        Parameters used for the IV trace classifier. These parameters are 
        disseminated into four categories.

        * Neural network parameters

        - **model_choice** (*str*), model choice, either "1DCNN" or
            "LSTM_multihead"
        - **params** (*list of str*), column names in train & test
            dataframes, used in neural network. Each value in this column
            must be a list.
        - **dropout_pct** (*float*), rate at which to set input units
            to zero.
        - **verbose** (*int*), control the specificity of the prints.

        * Training parameters

        - **train_size** (*float*), split of training data used for training
        - **shuffle_split** (*bool*), shuffle data during test-train split
        - **balance_tactic** (*str*), mode balancing tactic, either "truncate"
            or "gravitate". Truncate will utilize the exact same number of samples
            for each category. Gravitate will sway the original number of samples 
            towards the same number. Default= truncate.

        * LSTM parameters

        - **use_attention_lstm** (*bool*), if True,
            use attention in LSTM network
        - **units** (*int*), number of neurons in initial NN layer

        * 1DCNN parameters

        - **nfilters** (*int*), number of filters in the convolution.
        - **kernel_size** (*int*), length of the convolution window.
    """
    # Balance ys
    bal_df = balance_df(
        df, iv_col_dict, nn_config["balance_tactic"])
    # Label binarizer
    ys = bal_df[iv_col_dict["mode"]].values
    train, test = train_test_split(
        bal_df, train_size=nn_config["train_size"],
        shuffle=nn_config["shuffle_split"],
        stratify=(ys if nn_config["shuffle_split"] else None))

    iv = IVClassifier(nn_config)
    iv.structure(train, test)
    iv.train()
    iv.predict()
    return iv


class IVClassifier:
    def __init__(self, nn_config):
        self.nn_config = nn_config
        self.verbose = nn_config["verbose"]
        self.model_name = nn_config["model_choice"]
        self.params = nn_config["params"]

    def structure(self, train, test):
        """Structure the data according to the chosen network model's input structure.

        Parameters

        ----------
        train : dataframe
            Train data containing IV data and associated features
        test : dataframe
            Test data containing IV data and associated features
        nn_config : dict
            Parameters used for the IV trace classifier. These parameters are
            disseminated into four categories.

            * Neural network parameters

            - **model_choice** (*str*), model choice, either "1DCNN" or
              "LSTM_multihead"
            - **params** (*list of str*), column names in train & test
              dataframes, used in neural network. Each value in this column
              must be a list.
            - **dropout_pct** (*float*), rate at which to set input units
              to zero.
            - **verbose** (*int*), control the specificity of the prints.

            * Training parameters

            - **train_size** (*float*), split of training data used for
              training
            - **shuffle_split** (*bool*), shuffle data during test-train
              split
            - **balance_tactic** (*str*), mode balancing tactic, either
              "truncate" or "gravitate". Truncate will utilize the exact
              same number of samples for each category. Gravitate will sway
              the original number of samples towards the same number.
              Default= truncate.
            - **n_split** (*int*), number of splits in the stratified KFold
              cross validation.
            - **batch_size** (*int*), number of samples per gradient update.
            - **max_epochs** (*int*), maximum number of passes through the
              training process.

            * LSTM parameters

            - **use_attention_lstm** (*bool*), if True,
              use attention in LSTM network
            - **units** (*int*), number of neurons in initial NN layer

            * 1DCNN parameters

            - **nfilters** (*int*), number of filters in the convolution.
            - **kernel_size** (*int*), length of the convolution window.
        """

        num_params = len(self.params)

        # train the label binarizer
        self.lb = LabelBinarizer()
        all_ys = train['mode'].tolist() + test['mode'].tolist()
        self.lb.fit(all_ys)

        self.test_y = test['mode'].values
        self.train_y = train['mode'].values
        self.encoded_length = len(self.lb.classes_)

        self.is_binary = False
        if self.encoded_length == 1:
            raise ValueError("Only one failure mode was passed in dataset. "
                             "Add samples with other failure modes.")
        if self.encoded_length == 2:
            # Binary classification detected
            self.is_binary = True

        self.loss_defn = 'categorical_crossentropy'
        self.metric_defn = 'categorical_accuracy'

        if self.model_name == '1DCNN':
            self.train_x = _convert_ivdata_to_cnn_structure(train, self.params)
            self.test_x = _convert_ivdata_to_cnn_structure(test, self.params)
            self._X_for_cvsplit = self.train_x
            length_arr_in_sample = len(self.test_x[0])
            self._1dcnn((length_arr_in_sample, num_params),
                        nfilters=64,
                        kernel_size=12,
                        dropout_pct=0.5)

        elif self.model_name == 'LSTM_multihead':
            self.train_x = _convert_ivdata_to_lstm_multihead_structure(
                train, self.params)
            self.test_x = _convert_ivdata_to_lstm_multihead_structure(
                test, self.params)
            self._X_for_cvsplit = self.train_x[0]
            n_sequences = np.asarray(self.test_x).shape[2]
            n_samples_in_sequence = np.asarray(self.test_x).shape[3]
            self.test_x = _grab_structure_lstm_multihead_structure(
                self.test_x, np.array(range(np.asarray(self.test_x).shape[1])))
            self._lstm_multihead((n_sequences, n_samples_in_sequence),
                                 num_params,
                                 use_attention_LSTM=self.nn_config[
                                 'use_attention_lstm'],
                                 units=self.nn_config['units'],
                                 dropout_pct=self.nn_config['dropout_pct'])

        self.model.compile(loss=self.loss_defn,
                           optimizer='adam',
                           metrics=[self.metric_defn])

        if self.verbose >= 1:
            print(self.model.summary())

    def train(self):
        """Train neural network with stratified KFold.
        """
        cv = StratifiedKFold(n_splits=self.nn_config["n_CV_splits"])
        cvscores = []
        for train_idx, test_idx in cv.split(self._X_for_cvsplit, self.train_y):

            ytr = self.lb.transform(self.train_y[train_idx])
            yte = self.lb.transform(self.train_y[test_idx])

            if self.is_binary:
                ytr = to_categorical(ytr)
                yte = to_categorical(yte)

            if self.model_name == '1DCNN':
                xtr, xte = self.train_x[train_idx], self.train_x[test_idx]
            elif self.model_name == 'LSTM_multihead':
                xtr = _grab_structure_lstm_multihead_structure(
                    self.train_x, train_idx)
                xte = _grab_structure_lstm_multihead_structure(
                    self.train_x, test_idx)
            self.model.fit(xtr, ytr, epochs=self.nn_config["max_epochs"],
                           batch_size=self.nn_config["batch_size"],
                           verbose=self.verbose - 1)
            scores = self.model.evaluate(xte, yte, verbose=self.verbose)
            if self.verbose >= 1:
                print("%s: %.2f%%" %
                      (self.model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)
        return

    def predict(self, batch_size=8):
        """Predict using the trained model.

        Parameters

        ----------
        batch_size : int
            Number of samples per gradient update
        """
        yhat = self.model.predict(
            self.test_x, batch_size=batch_size, verbose=self.verbose)
        idx = np.argmax(yhat, axis=-1)
        decoded_preds = np.zeros(yhat.shape)
        decoded_preds[np.arange(decoded_preds.shape[0]), idx] = 1
        decoded_preds = self.lb.inverse_transform(decoded_preds)

        self.test_accuracy = accuracy_score(self.test_y, decoded_preds)
        print(classification_report(
            self.test_y, decoded_preds))
        print(confusion_matrix(self.test_y, decoded_preds))
        print('accuracy on test: ', self.test_accuracy)

    def _1dcnn(self, input_shape,
               nfilters=64,
               kernel_size=12,
               dropout_pct=0.5):
        self.model = Sequential()
        self.model.add(Conv1D(filters=nfilters, kernel_size=kernel_size,
                              activation='relu',
                              input_shape=input_shape))
        self.model.add(Dropout(dropout_pct))
        self.model.add(Flatten())
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(self.encoded_length, activation='softmax'))

    def _lstm_multihead(self, input_sample_shape,
                        n_features,
                        use_attention_LSTM,
                        units, dropout_pct):
        inputs = []
        for i in range(n_features):
            inputs.append(Input(shape=input_sample_shape))

        activations = []
        for i in range(n_features):
            activations.append(LSTM(units, return_sequences=False)(inputs[i]))

        dropouts = []
        for i in range(n_features):
            dropouts.append(Dropout(dropout_pct)(activations[i]))

        if use_attention_LSTM:
            hidden_size = []
            for i in range(n_features):
                hidden_size.append(int(dropouts[i].shape[2]))

            hidden_out = []
            for i in range(n_features):
                hidden_out.append(
                    Lambda(lambda x: x[:, -1, :],
                           output_shape=(hidden_size[i],)
                           )(dropouts[i]))

            pre_mlp = concatenate(hidden_out, name='attention_output')

        else:
            pre_mlp = concatenate(dropouts)

        d = Dense(int(((units - self.encoded_length) / 2) + self.encoded_length),
                  activation='relu', kernel_initializer='normal')(pre_mlp)

        activations = Dense(
            self.encoded_length,
            activation='softmax',
            kernel_initializer='normal')(d)

        self.model = Model(inputs=inputs, outputs=[activations])
