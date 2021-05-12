# Source
import itertools
import math
import warnings
from datetime import datetime
import warnings
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Standard
warnings.filterwarnings("ignore")


def _array_from_df(df, X_parameters):
    return np.vstack([df[xcol].values for xcol in X_parameters]).T


class Model:
    def __init__(self, estimators=None):
        self.estimators = estimators or [('OLS',
                                          LinearRegression()),
                                         ('RANSAC',
                                          RANSACRegressor(random_state=42))]

    def train(self, xs, Y, dates):
        """
        Least-squares implementation on multiple covariates
        """

        for name, estimator in self.estimators:
            print(name)
            estimator.fit(self.A, Y)
            # Make predictions using the testing set
            pred = estimator.predict(self.A)

            # The coefficients
            try:
                print('Coefficients: \n', estimator.coef_)
            except:
                # For RANSAC
                pass
            # The mean squared error
            print('Mean squared error: %.2f'
                  % mean_squared_error(Y, pred))
            # The coefficient of determination: 1 is perfect prediction
            print('Coefficient of determination: %.2f'
                  % r2_score(Y, pred))

            import matplotlib.pyplot as plt
            plt.plot(dates[0:50], pred[0:50], label=name)
            plt.plot(dates[0:50], Y[0:50], 'k')
        plt.legend()
        plt.show()


class DefaultModel(Model):
    def __init__(self):
        super().__init__()

    def construct(self, xs):
        self.A = xs


class PolynomialModel(Model):
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree

    def construct(self, xs):

        num_inputs, len_input = xs.shape[1], xs.shape[0]
        # add column of rows in first index of matrix
        xs = np.hstack((np.ones((len_input, 1), dtype=float), xs))

        # construct identity matrix
        iden_matrix = []
        for i in range(num_inputs + 1):
            # create np.array of np.zeros
            row = np.zeros(num_inputs + 1, dtype=int)

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
        self.A = np.hstack(np.array(A))

        return

    # def predict(self, temps):
    #     """Evaluate output with input parameters and polynomial information
    #         temps: temporary inputs
    #     """
    #     fit = 0
    #     for _iter, power in zip(self.a_hat, self.powers):
    #         for index in range(1, len(power)):
    #             _iter *= temps[index - 1] ** power[index]
    #         fit += _iter
    #     return fit


class PolynomialLogEModel(Model):
    def __init__(self, degree=3):
        super().__init__()
        self.degree = degree

    def construct(self, xs):

        # polynomial with included log(POA) parameter
        # Requires POA be first input in xs

        num_inputs, len_input = xs.shape[1], xs.shape[0]
        # add column of rows in first index of matrix
        Evals = np.array([row[0] for row in xs]) + 1
        xs = np.hstack(
            (np.ones((len_input, 1), dtype=float),
                xs, np.vstack(np.log(Evals)))
        )
        # construct identity matrix
        iden_matrix = []

        for i in range(num_inputs + 1 + 1):
            # for i in range(num_inputs+1+num_inputs):
            # create np.array of np.zeros
            row = np.zeros(num_inputs + 1 + 1, dtype=int)
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
            sum_arr = np.zeros(num_inputs + 1 + 1, dtype=int)
            sum_arr += sum((np.array(j) for j in combination))

            poly_powers.append(sum_arr)
        self.powers = poly_powers
        # Raise data to specified degree pattern and stack
        A = []
        for power in poly_powers:
            product = (xs**power).prod(1)
            A.append(product.reshape(product.shape + (1,)))
        self.A = np.hstack(np.array(A))

        return

    # def predict(self, temps):
    #     """Evaluate output with input parameters and polynomial information
    #         temps: temporary inputs
    #     """
    #     fit = 0
    #     for _iter, power in zip(self.a_hat, self.powers):
    #         for index in range(1, len(power)):
    #             _iter *= temps[index - 1] ** power[index]
    #         fit += _iter
    #     return fit


def modeller(prod_df,
             prod_col_dict,
             meta_df, meta_col_dict,
             kernel_type='polynomial_log',
             X_parameters=[],
             Y_parameter=None,
             test_split=0.2,
             degree=3):
    """Wrapper method to conduct the modelling of the timeseries data.

    Parameters

    ----------
    train_prod_df: DataFrame
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

    meta_df: DataFrame
        A data frame corresponding to site metadata.
        At the least, the columns in meta_col_dict be
        present.

    meta_col_dict: dict of {str : str}
        A dictionary that contains the column names relevant
        for the meta-data

        - **siteid** (*string*), should be assigned to site-ID
          column name
        - **dcsize** (*string*), should be assigned to
          column name corresponding to site capacity, where
          data is in [kW]

    kernel_type : str
        Type of kernel type for the statistical model

        - **polynomial**, a paraboiloidal polynomial with a dynamic number of
          covariates (Xs) and degrees (n). For example, with 2 covariates and a 
          degree of 2, the formula would be:
          Y(α , X) = α_0 + α_1 X_1 + α_2 X_2 + α_3 X_1 X_2 + α_4 X_1^2 + α_5 X_2^2
        - **polynomial_log**, same as above except with an added log(POA) term.
        - **diode_inspired**, reverse-engineered formula from single diode formula, initially
          intended for modelling voltage.
          (static equation):  Y(α , X) = α_0 + α_1 POA + α_2 Temp + α_3 ln(POA) + α_4 ln(Temp)

    X_parameters : list of str
        List of prod_df column names used in the model

    Y_parameter : str
        Optional, name of the y column. Defaults to prod_col_dict['powerprod'].
    """
    if Y_parameter is None:
        Y_parameter = prod_col_dict['powerprod']
    if kernel_type == 'polynomial_log':
        try:
            X_parameters.remove(prod_col_dict['irradiance'])
        except ValueError:
            raise ValueError(
                "The `prod_col_dict['irradiance']` definition must be in your " +
                "X_parameters input for the `polynomial_log` model.")
    elif kernel_type == 'diode_inspired':
        try:
            X_parameters.remove(prod_col_dict['irradiance'])
            X_parameters.remove(prod_col_dict['temperature'])
        except ValueError:
            raise ValueError("The `prod_col_dict['irradiance']` and `prod_col_dict['irradiance']`" +
                             "definitions must be in your X_parameters input for the " +
                             "`polynomial_log` model.")

    train, test = test_train_split(
        prod_df, test_size=0.33, random_state=42)

    train_y = train_prod_df[Y_parameter].values
    test_y = test_prod_df[Y_parameter].values

    train_X = _array_from_df(train_prod_df, X_parameters)
    train_X = _array_from_df(test_prod_df, X_parameters)

    if kernel_type == 'default':
        model = DefaultModel()
    elif kernel_type == 'polynomial':
        model = PolynomialModel(degree=degree)
    elif kernel_type == 'polynomial_log':
        model = PolynomialLogEModel(degree=degree)
    elif kernel_type == 'diode_inspired':
        pass

    model.construct(train_X)
    model.train(
        train_X,
        train_y,
        train_prod_df['date'].values
    )
    model.predict(train_X)
    # model.evaluate()

    return
