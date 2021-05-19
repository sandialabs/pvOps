# Source
import itertools
from sklearn.linear_model import LinearRegression, RANSACRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


def _array_from_df(df, X_parameters):
    return np.vstack([df[xcol].values for xcol in X_parameters]).T


class Model:
    """Linear model kernel
    """
    def __init__(self, estimators=None):
        self.train_index = None
        self.test_index = None
        # Other solvers, like RANSAC or THEIL SEN can be added by user
        self.estimators = estimators or {'OLS':
                                         {'estimator': LinearRegression()},
                                         'RANSAC':
                                         {'estimator': RANSACRegressor()}
                                         }

    def train(self):
        """
        Train the model.
        """
        if self.verbose >= 1:
            print("\nBegin training.")

        for name, info in self.estimators.items():
            info['estimator'].fit(self.train_X, self.train_y)
            self._evaluate(name, info, self.train_X, self.train_y)

    def _evaluate(self, name, info, X, y, data_split='train'):
        """Evaluate the model.
        """
        pred = info['estimator'].predict(X)
        mse = mean_squared_error(y, pred)
        r2 = r2_score(y, pred)

        try:
            coeffs = info['estimator'].coef_
        except AttributeError:
            coeffs = None

        if self.verbose >= 1:
            print(f'[{name}] Mean squared error: %.2f'
                  % mse)
            print(f'[{name}] Coefficient of determination: %.2f'
                  % r2)
            if not isinstance(coeffs, type(None)):
                print(f'[{name}] {len(coeffs)} coefficient trained.')
            else:
                # For RANSAC and others
                pass

        if self.verbose >= 2:
            # The coefficients
            if not isinstance(coeffs, type(None)):
                print(f'[{name}] Coefficients generated: \n', coeffs)
            else:
                # For RANSAC and others
                pass

        if data_split == 'train':
            info['train_index'] = self.train_index
            info['train_prediction'] = pred
            info['train_X'] = X
            info['train_y'] = y
            info['train_eval'] = {'mse': mse, 'r2': r2}
        elif data_split == 'test':
            info['test_index'] = self.test_index
            info['test_prediction'] = pred
            info['test_X'] = X
            info['test_y'] = y
            info['test_eval'] = {'mse': mse, 'r2': r2}

    def predict(self):
        """Predict using the model.
        """
        if self.verbose >= 1:
            print("\nBegin testing.")

        for name, info in self.estimators.items():
            self._evaluate(name, info, self.test_X,
                           self.test_y, data_split='test')


def _map_season(df_index):
    # Derived from https://stackoverflow.com/questions/44526662/group-data-by-season-according-to-the-exact-dates

    spring = range(80, 172)
    summer = range(172, 264)
    fall = range(264, 355)

    def _season(x):
        if x in spring:
            return 1
        if x in summer:
            return 2
        if x in fall:
            return 3
        else:
            return 0

    return df_index.dayofyear.map(_season)


class TimeWeightedProcess:
    """Generate time-oriented dummy variables for linear regression. Available timeframes
    include "month", "season", and "hour".
    """
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.set_time_bins = None

    def time_weight(self, X, time_weighted='season', data_split='train'):

        if time_weighted == 'month':
            if data_split == 'train':
                time_bins = self.train_index.month
            elif data_split == 'test':
                time_bins = self.test_index.month
        if time_weighted == 'season':
            if data_split == 'train':
                time_bins = _map_season(self.train_index)
            elif data_split == 'test':
                time_bins = _map_season(self.test_index)
        elif time_weighted == 'hour':
            if data_split == 'train':
                time_bins = self.train_index.hour
            elif data_split == 'test':
                time_bins = self.test_index.hour

        if data_split == 'train':
            self.set_time_bins = set(time_bins)
        elif data_split == 'test' and not isinstance(self.set_time_bins, set):
            raise Exception(
                "Must construct train before constructing test " +
                "if using the TimeWeightedProcess.")

        if self.verbose >= 1:
            print(data_split, set(time_bins))

        df = pd.DataFrame()
        df['time_bins'] = time_bins

        indices = {}
        for group in self.set_time_bins:
            indices[group] = df[df["time_bins"] == group].index

        for ii in range(X.shape[1]):
            df[f"col_{ii}"] = X[:, ii]
            # add groups
            for group in self.set_time_bins:
                vals = np.zeros(len(df))
                vals[indices[group]] = df.iloc[indices[group]][f"col_{ii}"]
                df[f"col_{ii}_{group}"] = vals
            # remove original data
            del df[f"col_{ii}"]
        del df["time_bins"]

        xs = df.values

        return xs


class DefaultModel(Model, TimeWeightedProcess):
    """Generate a simple model using the input data, without any data transposition.
    """
    def __init__(self, time_weighted=None, estimators=None, verbose=0):
        super().__init__(estimators)
        self.verbose = verbose
        self.time_weighted = time_weighted

    def construct(self, X, y, data_split='train'):

        if not isinstance(self.time_weighted, type(None)):
            X = self.time_weight(
                X, time_weighted=self.time_weighted, data_split=data_split)
        if data_split == 'train':
            self.train_X = X
            self.train_y = y
        elif data_split == 'test':
            self.test_X = X
            self.test_y = y


class PolynomialModel(Model, TimeWeightedProcess):
    """Add all interactions between terms with a degree.
    """
    def __init__(self, degree=2,
                 estimators=None,
                 time_weighted=None,
                 verbose=0):
        super().__init__(estimators)
        self.degree = degree
        self.time_weighted = time_weighted
        self.verbose = verbose

    def construct(self, X, y, data_split='train'):

        num_inputs = X.shape[1]
        # add column of rows in first index of matrix
        xs = np.array(X)

        # construct identity matrix
        iden_matrix = []
        for i in range(num_inputs):
            # create np.array of np.zeros
            row = np.zeros(num_inputs, dtype=int)

            # add 1 to diagonal index
            row[i] = 1
            iden_matrix.append(row)

        # gather list
        combinations = itertools.combinations_with_replacement(
            iden_matrix, self.degree)

        # list of polynomial powers
        poly_powers = []
        for combination in combinations:
            sum_arr = np.zeros(num_inputs, dtype=int)
            sum_arr += sum((np.array(j) for j in combination))
            poly_powers.append(sum_arr)

        self.powers = poly_powers

        # Raise data to specified degree pattern and stack
        A = []
        for power in poly_powers:
            product = (xs**power).prod(1)
            A.append(product.reshape(product.shape + (1,)))
        A = np.hstack(np.array(A))

        if not isinstance(self.time_weighted, type(None)):
            A = self.time_weight(
                A, time_weighted=self.time_weighted, data_split=data_split)

        if data_split == 'train':
            self.train_X = A
            self.train_y = y
        elif data_split == 'test':
            self.test_X = A
            self.test_y = y

        return


class PolynomialLogEModel(Model, TimeWeightedProcess):
    """Add all interactions between terms with degree and add log(Irradiance) term.

    For example, with two covariates and a degree of 2, 
    Y(α , X) = α_0 + α_1 X_1 + α_2 X_2 + α_3 X_1 X_2 + α_4 X_1^2 + α_5 X_2^2
    """
    def __init__(self, degree=3,
                 estimators=None,
                 time_weighted=None,
                 verbose=0):
        super().__init__(estimators)
        self.degree = degree
        self.time_weighted = time_weighted
        self.verbose = verbose

    def construct(self, X, y, data_split='train'):

        # polynomial with included log(POA) parameter
        # Requires POA be first input in xs

        num_inputs = X.shape[1]
        # add column of rows in first index of matrix
        Evals = np.array([row[0] for row in X]) + 1
        xs = np.hstack((X, np.vstack(np.log(Evals))))

        # construct identity matrix
        iden_matrix = []

        for i in range(num_inputs + 1):
            # for i in range(num_inputs+1+num_inputs):
            # create np.array of np.zeros
            row = np.zeros(num_inputs + 1, dtype=int)
            # row = np.zeros(num_inputs+1+num_inputs, dtype=int)
            # add 1 to diagonal index
            row[i] = 1
            iden_matrix.append(row)

        # gather list
        combinations = itertools.combinations_with_replacement(
            iden_matrix, self.degree)

        # list of polynomial powers
        poly_powers = []
        for combination in combinations:
            sum_arr = np.zeros(num_inputs + 1, dtype=int)
            sum_arr += sum((np.array(j) for j in combination))

            poly_powers.append(sum_arr)
        self.powers = poly_powers
        # Raise data to specified degree pattern and stack
        A = []
        for power in poly_powers:
            product = (xs**power).prod(1)
            A.append(product.reshape(product.shape + (1,)))
        A = np.hstack(np.array(A))

        if not isinstance(self.time_weighted, type(None)):
            A = self.time_weight(
                A, time_weighted=self.time_weighted, data_split=data_split)

        if data_split == 'train':
            self.train_X = A
            self.train_y = y
        elif data_split == 'test':
            self.test_X = A
            self.test_y = y

        return


class DiodeInspiredModel(Model, TimeWeightedProcess):
    """Generate a regression kernel derived from the diode model, originally meant to model voltage.

    (static equation):  Y(α , X) = α_0 + α_1 POA + α_2 Temp + α_3 ln(POA) + α_4 ln(Temp)
    """
    def __init__(self,
                 estimators=None,
                 time_weighted=None,
                 verbose=0):
        super().__init__(estimators)
        self.time_weighted = time_weighted
        self.verbose = verbose

    def construct(self, X, y, data_split='train'):
        # Diode Inspired
        # Requires that xs inputs be [POA, Temp], in that order
        xs = np.hstack((X, np.log(X)))

        if not isinstance(self.time_weighted, type(None)):
            X = self.time_weight(
                X, time_weighted=self.time_weighted, data_split=data_split)

        if data_split == 'train':
            self.train_X = xs
            self.train_y = y
        elif data_split == 'test':
            self.test_X = xs
            self.test_y = y
        return


def modeller(prod_df,
             prod_col_dict,
             kernel_type='default',
             time_weighted='month',
             X_parameters=[],
             Y_parameter=None,
             estimators=None,
             test_split=0.2,
             degree=3,
             verbose=0):
    """Wrapper method to conduct the modelling of the timeseries data.

    Parameters

    ----------
    prod_df: DataFrame
        A data frame corresponding to the production data
        used for model development and evaluation. This data frame needs
        at least the columns specified in prod_col_dict.

    prod_col_dict: dict of {str : str}
        A dictionary that contains the column names relevant
        for the production data

        - **siteid** (*string*), should be assigned to
          site-ID column name in prod_df
        - **timestamp** (*string*), should be assigned to
          time-stamp column name in prod_df
        - **irradiance** (*string*), should be assigned to
          irradiance column name in prod_df, where data
          should be in [W/m^2]
        - **baseline** (*string*), should be assigned to
          preferred column name to capture model calculations
          in prod_df
        - **dcsize**, (*string*), should be assigned to
          preferred column name for site capacity in prod_df

    kernel_type : str
        Type of kernel type for the statistical model

        - 'default', establishes a kernel where one component is instantiated
          in the model for each feature.
        - 'polynomial', a paraboiloidal polynomial with a dynamic number of
          covariates (Xs) and degrees (n). For example, with 2 covariates and a
          degree of 2, the formula would be:
          Y(α , X) = α_0 + α_1 X_1 + α_2 X_2 + α_3 X_1 X_2 + α_4 X_1^2 + α_5 X_2^2
        - 'polynomial_log', same as above except with an added log(POA) term.
        - 'diode_inspired', reverse-engineered formula from single diode formula, initially
          intended for modelling voltage.
          (static equation):  Y(α , X) = α_0 + α_1 POA + α_2 Temp + α_3 ln(POA) + α_4 ln(Temp)

    time_weighted : str or None
        Interval for time-based feature generation. For each interval in this time-weight,
        a dummy variable is established in the model prior to training. Options include:

        - if 'hour', establish discrete model components for each hour of day
        - if 'month', establish discrete model components for each month
        - if 'season', establish discrete model components for each season
        - if None, no time-weighted dummy-variable generation is conducted.

    X_parameters : list of str
        List of prod_df column names used in the model

    Y_parameter : str
        Optional, name of the y column. Defaults to prod_col_dict['powerprod'].

    estimators : dict
        Optional, dictionary with key as regressor identifier (str) and value as a
        dictionary with key "estimator" and value the regressor instance following
        sklearn's base model convention: sklearn_docs.

        .. sklearn_docs: https://scikit-learn.org/stable/modules/generated/sklearn.base.is_regressor.html
        .. code-block:: python

            estimators = {'OLS': {'estimator': LinearRegression()},
                          'RANSAC': {'estimator': RANSACRegressor()}
                          }

    test_split : float
        A value between 0 and 1 indicating the proportion of data used for testing.

    degree : int
        Utilized for 'polynomial' and 'polynomial_log' `kernel_type` options, this 
        parameter defines the highest degree utilized in the polynomial kernel.

    verbose : int
        Define the specificity of the print statements during this function's
        execution.

    Returns

    -------
    `model`, which is a `pvops.timeseries.models.linear.Model` object, has a useful attribute
    `estimators`, which allows access to model performance and data splitting information.

    `train_df`, which is the training split of prod_df

    `test_df`, which is the testing split of prod_df
    """
    estimators = estimators or {'OLS': {'estimator': LinearRegression()},
                                'RANSAC': {'estimator': RANSACRegressor()}}

    if Y_parameter is None:
        Y_parameter = prod_col_dict['powerprod']
    if kernel_type == 'polynomial_log':
        try:
            X_parameters.remove(prod_col_dict['irradiance'])
            # Place irradiance in front.
            X_parameters = [prod_col_dict['irradiance']] + X_parameters
        except ValueError:
            raise ValueError(
                "The `prod_col_dict['irradiance']` definition must be in your " +
                "X_parameters input for the `polynomial_log` model.")
    elif kernel_type == 'diode_inspired':
        try:
            X_parameters.remove(prod_col_dict['irradiance'])
            X_parameters.remove(prod_col_dict['temperature'])
            # Place irradiance and temperature in front.
            X_parameters = [prod_col_dict['irradiance'],
                            prod_col_dict['temperature']] + X_parameters
        except ValueError:
            raise ValueError("The `prod_col_dict['irradiance']` and `prod_col_dict['irradiance']`" +
                             "definitions must be in your X_parameters input for the " +
                             "`polynomial_log` model.")

    # Split into test-train
    mask = np.array(range(len(prod_df))) < int(
        len(prod_df) * (1 - test_split))
    train_prod_df = prod_df.iloc[mask]
    test_prod_df = prod_df.iloc[~mask]

    train_y = train_prod_df[Y_parameter].values
    test_y = test_prod_df[Y_parameter].values
    train_X = _array_from_df(train_prod_df, X_parameters)
    test_X = _array_from_df(test_prod_df, X_parameters)

    if kernel_type == 'default':
        model = DefaultModel(time_weighted=time_weighted,
                             estimators=estimators,
                             verbose=verbose)
    elif kernel_type == 'polynomial':
        model = PolynomialModel(
            time_weighted=time_weighted,
            estimators=estimators,
            degree=degree,
            verbose=verbose)
    elif kernel_type == 'polynomial_log':
        model = PolynomialLogEModel(
            time_weighted=time_weighted,
            estimators=estimators,
            degree=degree,
            verbose=verbose)
    elif kernel_type == 'diode_inspired':
        model = DiodeInspiredModel(
            time_weighted=time_weighted,
            estimators=estimators,
            verbose=verbose)

    model.train_index = train_prod_df.index
    model.test_index = test_prod_df.index

    # Always construct train first in case of using time_weighted,
    # which caches the time regions in training to reuse in testing.
    model.construct(train_X, train_y, data_split='train')
    model.construct(test_X, test_y, data_split='test')
    model.train()
    model.predict()
    return model, train_prod_df, test_prod_df
