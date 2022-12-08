# Source
import itertools
from sklearn.linear_model import LinearRegression, RANSACRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


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

        if not isinstance(coeffs, type(None)):
            # @dev: Compare p-value & CI calculations to one of
            # @dev: an open source package as a validation step.
            # @dev: Beware that the evaluation uses OLS and may
            # @dev: be mismatched if user inputs other regressors.
            Xnew = pd.DataFrame(X, columns=self.variate_names)
            Xnew = sm.add_constant(Xnew)
            est = sm.OLS(y, Xnew)
            est2 = est.fit()
            self.statsmodels_est = est2
            statsdf = est2.summary()

            # @dev the below snippet calculates statistical parameters
            # @dev using base numpy. This is commented out because we
            # @dev have opted to use the statsmodels package.
            # params = np.append(info['estimator'].intercept_, coeffs)
            # # Check variable significanace
            # newX = pd.DataFrame({"Constant": np.ones(
            #     len(X)
            #     )}).join(pd.DataFrame(X))
            # MSE = (sum((y-pred)**2))/(len(newX)-len(newX.columns))

            # try:
            #     var_b = MSE * np.linalg.inv(np.dot(newX.T,
            #                                        newX)
            #                                 ).diagonal()
            # except np.linalg.LinAlgError:
            #     # Add random noise to avoid a LinAlg error
            #     newX += 0.00001*np.random.rand(*newX.shape)
            #     var_b = MSE * np.linalg.inv(np.dot(newX.T,
            #                                        newX)
            #                                 ).diagonal()
            # sd_b = np.sqrt(var_b)
            # ts_b = params / sd_b
            # p_values = [2*(1-stats.t.cdf(np.abs(i),
            #                              (len(newX)-newX.shape[1])))
            #             for i in ts_b]
            # vif = [variance_inflation_factor(newX.values, i)
            #        for i in range(newX.shape[1])]
            # lb_ci = params - sd_b * 1.96  # percentile: 0.025
            # ub_ci = params + sd_b * 1.96  # percentile: 0.975

            # # Round
            # sd_b = np.round(sd_b, 3)
            # ts_b = np.round(ts_b, 3)
            # p_values = np.round(p_values, 3)
            # params = np.round(params, 4)
            # lb_ci = np.round(lb_ci, 4)
            # ub_ci = np.round(ub_ci, 4)

            # statsdf = pd.DataFrame()
            # (statsdf["coef"],
            #  statsdf["std err"],
            #  statsdf["t"],
            #  statsdf["P>|t|"],
            #  statsdf["[0.025"],
            #  statsdf["0.975]"],
            #  statsdf["vif"]
            #  ) = [params,
            #       sd_b,
            #       ts_b,
            #       p_values,
            #       lb_ci,
            #       ub_ci,
            #       vif
            #       ]
            # statsdf.index = ["constant"] + list(self.variate_names)
        else:
            statsdf = None

        if self.verbose >= 1:
            print(f'[{name}] Mean squared error: %.2f'
                  % mse)
            print(f'[{name}] Coefficient of determination: %.2f'
                  % r2)

            # Display coefficients
            if not isinstance(coeffs, type(None)):
                print(f'[{name}] {len(coeffs)} coefficient trained.')
            else:
                # For RANSAC and others
                pass

        if self.verbose >= 2:
            # The coefficients
            if not isinstance(coeffs, type(None)):
                if data_split == 'train':
                    print(statsdf)
            else:
                # For RANSAC and others
                pass

        if data_split == 'train':
            info['train_index'] = self.train_index
            info['train_prediction'] = pred
            info['train_X'] = X
            info['train_y'] = y
            info['train_eval'] = {'mse': mse, 'r2': r2, 'statsdf': statsdf}
        elif data_split == 'test':
            info['test_index'] = self.test_index
            info['test_prediction'] = pred
            info['test_X'] = X
            info['test_y'] = y
            info['test_eval'] = {'mse': mse, 'r2': r2, 'statsdf': statsdf}

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
        elif time_weighted == 'capacity':
            if data_split == 'train':
                time_bins = self.train_capacity_bins
            elif data_split == 'test':
                time_bins = self.test_capacity_bins

        if data_split == 'train':
            self.set_time_bins = set(time_bins)
        elif data_split == 'test' and not isinstance(self.set_time_bins, (set,
                                                                          np.ndarray,
                                                                          list)):
            raise Exception(
                "Must construct train before constructing test "
                "if using the TimeWeightedProcess.")

        if self.verbose >= 1:
            print(data_split, set(time_bins))

        df = pd.DataFrame()
        df['time_bins'] = time_bins

        indices = {}
        for group in self.set_time_bins:
            indices[group] = df[df["time_bins"] == group].index

        new_variable_names = []
        for ii, (param,
                 covariate_profile) in enumerate(zip(self.variate_names,
                                                     self.covariate_degree_combinations)):
            df[f"col_{ii}"] = X[:, ii]
            # add groups
            for time_idx, group in enumerate(self.set_time_bins):
                if covariate_profile + [time_idx] in self.exclude_params:
                    continue
                vals = np.zeros(len(df))
                vals[indices[group]] = df.iloc[indices[group]][f"col_{ii}"]
                df[f"col_{ii}_{group}"] = vals
                new_variable_names.append(f"{param} | {time_weighted}:{group}")
            # remove original data
            del df[f"col_{ii}"]
        del df["time_bins"]

        xs = df.values
        self.variate_names = new_variable_names
        return xs


class DefaultModel(Model, TimeWeightedProcess):
    """Generate a simple model using the input data, without
    any data transposition.
    """
    _model_name = 'default'

    def __init__(self, time_weighted=None, estimators=None,
                 verbose=0, X_parameters=[]):
        super().__init__(estimators)
        self.verbose = verbose
        self.time_weighted = time_weighted
        self.X_parameters = X_parameters
        # Set to null as default, this is used in PolynomialModel
        self.exclude_params = []

    def construct(self, X, y, data_split='train'):
        self.variate_names = self.X_parameters
        num_variates = len(self.variate_names)
        self.covariate_degree_combinations = []
        for i in range(num_variates):
            list_sub = [0] * num_variates
            list_sub[i] = 1
            self.covariate_degree_combinations.append(list_sub)

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
    _model_name = "polynomial"

    def __init__(self, degree=2,
                 estimators=None,
                 time_weighted=None,
                 verbose=0,
                 X_parameters=[],
                 exclude_params=[]):
        super().__init__(estimators)
        self.degree = degree
        self.time_weighted = time_weighted
        self.verbose = verbose
        self.X_parameters = X_parameters
        self.exclude_params = exclude_params

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
        all_combinations = []
        for degree in range(1, self.degree + 1):
            all_combinations.append(itertools.combinations_with_replacement(
                iden_matrix, degree))

        # list of polynomial powers
        poly_powers = []
        self.variate_names = []
        self.covariate_degree_combinations = []
        for combinations in all_combinations:
            for combination in combinations:
                covariate_profile = list(np.array(combination).sum(axis=0))
                # Check if this term was to be excluded.
                if isinstance(self.time_weighted, type(None)):
                    if covariate_profile in self.exclude_params:
                        # Skip the params which are meant to be excluded.
                        continue
                else:
                    self.covariate_degree_combinations.append(
                        covariate_profile)

                sum_arr = np.zeros(num_inputs, dtype=int)
                sum_arr += sum((np.array(j) for j in combination))
                s = ""
                has_first_term = False
                for idx, p in enumerate(sum_arr):
                    if p == 0:
                        continue
                    if has_first_term:
                        s += " * "
                    else:
                        has_first_term = True
                    s += r"{}^{}".format(self.X_parameters[idx], p)

                self.variate_names.append(s)
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

        print("Design matrix shape:", A.shape)

        if data_split == 'train':
            self.train_df = pd.DataFrame(A,
                                         columns=self.variate_names,
                                         index=self.train_index)
            self.train_X = A
            self.train_y = y
        elif data_split == 'test':
            self.test_df = pd.DataFrame(A,
                                        columns=self.variate_names,
                                        index=self.test_index)
            self.test_X = A
            self.test_y = y


def _get_params(Y_parameter, X_parameters, prod_col_dict, kernel_type):
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

    return X_parameters, Y_parameter


def modeller(prod_col_dict,
             kernel_type='default',
             time_weighted='month',
             X_parameters=[],
             Y_parameter=None,
             estimators=None,
             prod_df=None,
             test_split=0.2,
             train_df=None,
             test_df=None,
             degree=3,
             exclude_params=[],
             verbose=0):
    """Wrapper method to conduct the modelling of the timeseries data.

    To input the data, there are two options.

    - Option 1: include full production data in `prod_df`
      parameter and `test_split` so that the test split is conducted
    - Option 2: conduct the test-train split prior to calling
      the function and pass in data under `test_df` and `train_df`

    Parameters
    ----------
    prod_col_dict: dict of {str : str}
        A dictionary that contains the column names relevant
        for the production data

        - siteid (*string*), should be assigned to
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
        - **powerprod**, (*string*), should be assigned to
          the column name holding the power or energy production.
          This will be used as the output column if Y_parameter
          is not passed.

    kernel_type : str
        Type of kernel type for the statistical model

        - 'default', establishes a kernel where one component is instantiated
          in the model for each feature.
        - 'polynomial', a paraboiloidal polynomial with a dynamic number of
          covariates (Xs) and degrees (n). For example, with 2 covariates and a
          degree of 2, the formula would be:
          Y(α , X) = α_0 + α_1 X_1 + α_2 X_2 + α_3 X_1 X_2 + α_4 X_1^2 + α_5 X_2^2

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

    prod_df: DataFrame
        A data frame corresponding to the production data
        used for model development and evaluation. This data frame needs
        at least the columns specified in prod_col_dict.

    test_split : float
        A value between 0 and 1 indicating the proportion of data used for testing.
        Only utilized if `prod_df` is specified. If you want to specify your own 
        test-train splits, pass values to `test_df` and `train_df`.

    test_df: DataFrame
        A data frame corresponding to the test-split of the production data.
        Only needed if `prod_df` and `test_split` are not specified.

    train_df: DataFrame
        A data frame corresponding to the test-split of the production data.
        Only needed if `prod_df` and `test_split` are not specified.

    degree : int
        Utilized for 'polynomial' and 'polynomial_log' `kernel_type` options, this 
        parameter defines the highest degree utilized in the polynomial kernel.

    exclude_params : list
        A list of parameter definitions (defined as lists) to be excluded in the model. For 
        example, if want to exclude a parameter in a 4-covariate model that uses 1 degree on first covariate,
        2 degrees on second covariate, and no degrees for 3rd and 4th covariates, you would specify a 
        exclude_params as ``[ [1,2,0,0] ]``. Multiple definitions can be added to list depending
        on how many terms need to be excluded.

        If a time_weighted parameter is selected, a time weighted definition will need to be appended to
        *each* exclusion definition. Continuing the example above, if one wants to exclude "hour 0" for the
        same term, then the exclude_params must be ``[ [1,2,0,0,0] ]``, where the last 0 represents the
        time-weighted partition setting.

    verbose : int
        Define the specificity of the print statements during this function's
        execution.

    Returns
    -------
    model
        which is a ``pvops.timeseries.models.linear.Model`` object, has a useful attribute
    estimators
        which allows access to model performance and data splitting information.
    train_df
        which is the training split of prod_df
    test_df
        which is the testing split of prod_df
    """
    estimators = estimators or {'OLS': {'estimator': LinearRegression()},
                                'RANSAC': {'estimator': RANSACRegressor()}}

    X_parameters, Y_parameter = _get_params(Y_parameter, X_parameters,
                                            prod_col_dict, kernel_type)

    if (not isinstance(prod_df, type(None))) and (not isinstance(test_split, type(None))):
        # Split into test-train
        mask = np.array(range(len(prod_df))) < int(
            len(prod_df) * (1 - test_split))
        train_df = prod_df.iloc[mask]
        test_df = prod_df.iloc[~mask]
    else:
        if isinstance(train_df, type(None)) or isinstance(test_df, type(None)):
            raise ValueError("Because `prod_df` and `test_split` were not specified,"
                             "expected `train_df` and `test_df` to be passed. But, they"
                             "were not specified.")

    train_y = train_df[Y_parameter].values
    test_y = test_df[Y_parameter].values
    train_X = _array_from_df(train_df, X_parameters)
    test_X = _array_from_df(test_df, X_parameters)

    if kernel_type == 'default':
        model = DefaultModel(time_weighted=time_weighted,
                             estimators=estimators,
                             verbose=verbose,
                             X_parameters=X_parameters)
    elif kernel_type == 'polynomial':
        model = PolynomialModel(
            time_weighted=time_weighted,
            estimators=estimators,
            degree=degree,
            verbose=verbose,
            X_parameters=X_parameters,
            exclude_params=exclude_params)

    model.train_index = train_df.index
    model.test_index = test_df.index

    # Support in future versions
    # model.train_capacity_bins = train_capacity_bins
    # model.test_capacity_bins = test_capacity_bins

    # Always construct train first in case of using time_weighted,
    # which caches the time regions in training to reuse in testing.
    model.construct(train_X, train_y, data_split='train')
    model.construct(test_X, test_y, data_split='test')
    model.train()
    model.predict()
    return model, train_df, test_df


def predicter(model, df, Y_parameter, X_parameters, prod_col_dict, verbose=0):
    kernel_type = model._model_name

    X_parameters, Y_parameter = _get_params(Y_parameter, X_parameters,
                                            prod_col_dict, kernel_type)

    test_y = df[Y_parameter].values
    test_X = df[X_parameters].values

    if verbose > 0:
        print(X_parameters)
        print(Y_parameter)
        print(test_y.shape)
        print(test_X.shape)

    model.test_index = df.index
    model.construct(test_X, test_y, data_split='test')
    model.predict()
    return model, test_y, test_X
