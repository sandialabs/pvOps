import numpy as np
from numpy.core.fromnumeric import prod
from sklearn.metrics import mean_squared_error, r2_score


class Predictor:
    """
    Predictor class
    """
    def __init__(self):
        super(Predictor, self).__init__()

    def apply_additive_polynomial_model(self, model_terms, Xs):
        """Predict energy using a model derived by pvOps.

        Parameters
        ----------
        df : dataframe
            Data containing columns with the values in
            the `prod_col_dict`

        model_terms : list of tuples
            Contain model coefficients and powers. For example,

            .. code-block:: python

                [(0.29359785963294494, [1, 0]),
                (0.754806343190528, [0, 1]),
                (0.396833207207238, [1, 1]),
                (-0.0588375219110795, [0, 0])]

        prod_col_dict : dict
            Dictionary mapping nicknamed parameters to
            the named parameters in the dataframe `df`.

        Returns
        -------
        Array of predicted energy values
        """
        for idx, (coeff, powers) in enumerate(model_terms):
            for i, (x, n) in enumerate(zip(Xs, powers)):
                if i == 0:
                    term = x**n
                else:
                    term *= x**n
            if idx == 0:
                energy = coeff * term
            else:
                energy += coeff * term
        return energy

    def evaluate(self, real, pred,):
        logrmse = np.log(np.sqrt(mean_squared_error(real, pred)))
        r2 = r2_score(real, pred)
        print(f"The fit has an R-squared of {r2} and a log RMSE of {logrmse}")
        return logrmse, r2


class Processer:
    def __init__(self):
        super(Processer, self).__init__()
        self._col_scaled_prefix = 'stdscaled_'

    def check_data(self, data, prod_col_dict):
        self.do_eval = False
        if 'energyprod' in prod_col_dict:
            if prod_col_dict['energyprod'] in data.columns.tolist():
                self.do_eval = True

        if not self.do_eval:
            print("Because the power production data is not"
                  " passed, the fit will not be evaluated."
                  " Predictions will still be rendered.")

    def _apply_transform(self, data,
                         scaler_info):
        data -= scaler_info["mean"]
        data /= scaler_info["scale"]
        return data

    def _apply_inverse_transform(self, data,
                                 scaler_info):
        data *= scaler_info["scale"]
        data += scaler_info["mean"]
        return data

    def _clean_columns(self, scaler, prod_df, prod_col_dict):
        for k, d in scaler.items():
            del prod_df[self._col_scaled_prefix + prod_col_dict[k]]


# @dev: The 'AIT' class can be one of many models that inherit the
# @dev: Processor and Predictor templates. When adding new models,
# @dev: use the Processor and Predictor classes to hold general
# @dev: functionality while having model-specific nuances in the
# @dev: classes below. The above classes may be placed in a different
# @dev: if it seems fit.
class AIT(Processer, Predictor):
    def __init__(self):
        super(AIT, self).__init__()
        self._load_params()

    def _load_params(self):
        self.scaler_highcap = {"irradiance": {"mean": 571.45952959,
                                              "scale": 324.19905495},
                               "dcsize": {"mean": 14916.2339917,
                                          "scale": 20030.00088265},
                               "energyprod": {"mean": 7449.15184666,
                                              "scale": 12054.52533771}
                               }
        self.model_terms_highcap = [(0.29359785963294494, [1, 0]),
                                    (0.754806343190528, [0, 1]),
                                    (0.396833207207238, [1, 1]),
                                    (-0.0588375219110795, [0, 0])]

        self.scaler_lowcap = {"irradiance": {"mean": 413.53334101,
                                             "scale": 286.11031612},
                              "dcsize": {"mean": 375.91883522,
                                         "scale": 234.15141671},
                              "energyprod": {"mean": 119.00787546,
                                             "scale": 119.82927847}
                              }
        self.model_terms_lowcap = [(0.6866363032474436, [1, 0]),
                                   (0.6473846301807609, [0, 1]),
                                   (0.41926724219597955, [1, 1]),
                                   (0.06624491753542901, [0, 0])]

    def predict_subset(self, prod_df, scaler, model_terms, prod_col_dict):
        prod_df = prod_df.copy()
        self.check_data(prod_df, prod_col_dict)

        """1. Standardize the data using same scales"""
        for k, d in scaler.items():
            data = prod_df[prod_col_dict[k]]
            scaled_data = self._apply_transform(data, d)
            prod_df[self._col_scaled_prefix + prod_col_dict[k]] = scaled_data

        prod_irr = prod_col_dict["irradiance"]
        prod_dcsize = prod_col_dict["dcsize"]

        irr = prod_df[self._col_scaled_prefix + prod_irr].values
        capacity = prod_df[self._col_scaled_prefix + prod_dcsize].values
        Xs = [irr, capacity]

        """2. Predict energy"""
        predicted_energy = self.apply_additive_polynomial_model(model_terms,
                                                                Xs)
        """3. Rescale predictions"""
        predicted_rescaled_energy = self._apply_inverse_transform(predicted_energy,
                                                                  scaler['energyprod'])

        """4. Evaluate"""
        if self.do_eval:
            self.evaluate(prod_df[prod_col_dict["energyprod"]].values,
                          predicted_rescaled_energy)
        return predicted_rescaled_energy

    def predict(self, prod_df, prod_col_dict):

        # High-capacity systems
        high_cap_mask = prod_df[prod_col_dict['dcsize']] > 1000
        if sum(high_cap_mask) > 0:
            predicted = self.predict_subset(prod_df.loc[high_cap_mask, :],
                                            self.scaler_highcap,
                                            self.model_terms_highcap,
                                            prod_col_dict)
            prod_df.loc[high_cap_mask, prod_col_dict["baseline"]] = predicted

        # Low-capacity systems
        low_cap_mask = prod_df[prod_col_dict['dcsize']] <= 1000
        if sum(low_cap_mask) > 0:
            predicted = self.predict_subset(prod_df.loc[low_cap_mask, :],
                                            self.scaler_lowcap,
                                            self.model_terms_lowcap,
                                            prod_col_dict)
            prod_df.loc[low_cap_mask, prod_col_dict["baseline"]] = predicted
        return prod_df


def AIT_calc(prod_df, prod_col_dict):
    """
    Calculates expected energy using measured irradiance
    based on trained regression model from field data.
    Plane-of-array irradiance is recommended when using the pre-trained AIT model.

    Parameters
    ----------
    prod_df : DataFrame
        A data frame corresponding to the production data

    prod_col_dict : dict of {str : str}
        A dictionary that contains the column names relevant
        for the production data

        - **irradiance** (*string*), should be assigned to
          irradiance column name in prod_df, where data
          should be in [W/m^2]
        - **dcsize**, (*string*), should be assigned to
          preferred column name for site capacity in prod_df
        - **energyprod**, (*string*), should be assigned to
          the column name holding the power or energy production.
          If this is passed, an evaluation will be provided.
        - **baseline**, (*string*), should be assigned to
          preferred column name to capture the calculations
          in prod_df

    Example
    -------

    .. code-block:: python

        production_col_dict = {'irradiance': 'irrad_poa_Wm2',
                            'ambient_temperature': 'temp_amb_C',
                            'dcsize': 'capacity_DC_kW',
                            'energyprod': 'energy_generated_kWh',
                            'baseline': 'predicted'
                            }
        data = AIT_calc(data, production_col_dict)


    Returns
    -------
    DataFrame
        A data frame for production data with a new column,
        the predicted energy
    """
    prod_df = prod_df.copy()
    # assigning dictionary items to local variables for cleaner code
    model = AIT()
    prod_df = model.predict(prod_df, prod_col_dict)
    return prod_df
