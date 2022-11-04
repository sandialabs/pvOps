import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from pvops.iv.simulator import Simulator


class IVTimeseriesGenerator(Simulator):

    def __init__(self, **iv_sim_kwargs):
        """Simulate a PV System across time.

        Parameters
        ----------
        iv_sim_kwargs :
            Optional, `simulator.Simulator` inputs
        """
        super().__init__(**iv_sim_kwargs)

    def generate(self, env_df, failures, iv_col_dict,
                 identifier_col, plot_trends=False):
        """Simulate a PV system

        Parameters
        ----------
        env_df : dataframe
            DataFrame containing irradiance ("E") and temperature ("T") columns
        failures : list
            List of timeseries_simulator.TimeseriesFailure objects
        """

        self.specs_df = env_df[[
            identifier_col, iv_col_dict["irradiance"],
            iv_col_dict["temperature"]]].copy()
        for failure in failures:
            # Weigh all failure definitions together
            self.specs_df = failure.add_interpolation(
                self.specs_df, plot_trends)

        self.timeseries_condition_dicts = self._structure_Simulator_inputs(
            self.specs_df, iv_col_dict, identifier_col)
        return self.timeseries_condition_dicts

    def add_time_conditions(self, preset_mod_mapping, nmods=12):
        for condition_dict in self.timeseries_condition_dicts:
            self.add_preset_conditions(preset_mod_mapping, condition_dict,
                                       save_name=f"mod_{condition_dict['identifier']}")
            self.build_strings({f"str_{condition_dict['identifier']}":
                                [f"mod_{condition_dict['identifier']}"] * nmods})

    def _structure_Simulator_inputs(self, specs_df,
                                    iv_col_dict, identifier_col):
        keys = []
        savekeys = []
        spec_df_cols = specs_df.columns
        for key in ['identifier'] + self.acceptible_keys:
            if key == 'identifier':
                savekey = identifier_col
            elif key == 'E':
                savekey = iv_col_dict['irradiance']
            elif key == 'Tc':
                savekey = iv_col_dict['temperature']
            else:
                savekey = key
            if savekey in spec_df_cols:
                keys.append(key)
                savekeys.append(savekey)

        return [dict(zip(keys, vals))
                for vals in specs_df[savekeys].values]


class TimeseriesFailure:
    def __init__(self):
        """Define a failure in terms of the affected diode
        parameters and specify how the failure evolves over
        time (i.e. how quickly does it itensify? how fast is
        it detected? how fast is it fixed?)
        """
        self.longterm_fcn_dict = {}
        self.annual_fcn_dict = {}
        self.daily_fcn_dict = {}

    def trend(self, longterm_fcn_dict=None,
              annual_fcn_dict=None,
              daily_fcn_dict=None,
              **kwargs):
        """Define a failure's trend across intraday (trending
        with time of day) and longterm timeframes.

        Parameters
        ----------
        longterm_fcn_dict : dict
            A dictionary where keys are the diode-multipliers in IVSimulator
            ('Rsh_mult', 'Rs_mult', 'Io_mult', 'Il_mult', 'nnsvth_mult') and
            values are either a function or a string. If a function, the
            function should be a mathematical operation as a `function of the
            number of float years since operation start`, a value on domain
            [0,inf), and outputs the chosen diode-multiplier's values across
            this timeseries. If a string, must use a pre-defined definition:

            * 'degrade' : degrade over time at specified rate.
              Specify rate by passing a definition for
              `degradation_rate`

            For example,

            .. code-block:: python

                # 2 Ways of Doing Same Thing

                # Method 1
                longterm_fcn_dict = {
                    'Rs_mult': lambda x : 1.005 * x
                }
                f = Failure()
                f.trend(longterm_fcn_dict)

                # Method 2
                longterm_fcn_dict = {
                    'Rs_mult': 'degrade'
                }
                f = Failure()
                f.trend(longterm_fcn_dict,
                        degradation_rate=1.005)

        annual_fcn_dict : dict
            A dictionary where keys are the diode-multipliers in IVSimulator
            ('Rsh_mult', 'Rs_mult', 'Io_mult', 'Il_mult', 'nnsvth_mult') and
            values are either a function or a string. If a function, the
            function should be a mathematical operation as a `function of the
            percentage through this year`, a value on domain [0,1], and outputs
            the chosen diode-multiplier's values across this timeseries. If a
            string, must use a pre-defined definition:

        daily_fcn_dict : function or str
            A dictionary where keys are the diode-multipliers in IVSimulator
            ('Rsh_mult', 'Rs_mult', 'Io_mult', 'Il_mult', 'nnsvth_mult') and
            values are either a function or a string. If a function, the
            function should be a mathematical operation as a `function of the
            percentage through this day`, a value on domain [0,1], and outputs
            the chosen diode-multiplier's values across this timeseries. If a
            string, must use a pre-defined definition:
        """

        if not isinstance(longterm_fcn_dict, type(None)):
            self.longterm_fcn_dict = longterm_fcn_dict

            for param, fcn in longterm_fcn_dict.items():
                if isinstance(fcn, str):
                    self._predefined_trend(param, longterm_fcn=fcn, **kwargs)

        if not isinstance(annual_fcn_dict, type(None)):
            self.annual_fcn_dict = annual_fcn_dict

            for param, fcn in annual_fcn_dict.items():
                if isinstance(fcn, str):
                    self._predefined_trend(param, annual_fcn=fcn, **kwargs)

        if not isinstance(daily_fcn_dict, type(None)):
            self.daily_fcn_dict = daily_fcn_dict

            for param, fcn in daily_fcn_dict.items():
                if isinstance(fcn, str):
                    self._predefined_trend(param, daily_fcn=fcn, **kwargs)

    def _predefined_trend(self, param, longterm_fcn='degrade',
                          annual_fcn='', daily_fcn='uniform',
                          **kwargs):

        if longterm_fcn == 'degrade':
            try:
                degr_rate = kwargs['degradation_rate']
            except KeyError:
                raise KeyError("TimeseriesFailure.trend requires a "
                               "passed parameter `degradation_rate`"
                               "if using `degrade` longterm_fcn definition.")
            self.longterm_fcn_dict[param] = lambda x: degr_rate * x

    def _combine(self, arr, specs_df, param):
        if param not in specs_df.columns:
            specs_df[param] = np.ones(len(specs_df))

        if param in ["Rsh_mult", "Io_mult", "Il_mult"]:
            specs_df[param] -= arr

        elif param in ["Rs_mult", "nnsvth_mult"]:
            specs_df[param] += arr

    def add_interpolation(self, specs_df, plot_trends=False):
        """Add failure properties to specs_df
        """

        # Degradation since start
        float_years = np.array(
            (specs_df.index - specs_df.index[0]) / timedelta(days=365.25))
        for param, fcn in self.longterm_fcn_dict.items():
            vals = fcn(float_years)
            self._combine(vals, specs_df, param)
            if plot_trends:
                plt.plot(specs_df.index, vals, 'o--', alpha=0.8, label=param)
        if plot_trends:
            if len(self.longterm_fcn_dict.keys()):
                plt.legend()
                plt.title("Longterm")
                plt.show()

        # Degradation cyclic per year
        pct_of_year = np.array(specs_df.index.dayofyear) / 365
        for param, fcn in self.annual_fcn_dict.items():
            vals = fcn(pct_of_year)
            self._combine(vals, specs_df, param)
            if plot_trends:
                plt.plot(specs_df.index, vals, 'o--', alpha=0.8, label=param)
        if plot_trends:
            if len(self.annual_fcn_dict.keys()):
                plt.legend()
                plt.title("Annual")
                plt.show()

        # Degradation per day
        pct_of_day = np.array(specs_df.index.hour) / 24
        for param, fcn in self.daily_fcn_dict.items():
            vals = fcn(pct_of_day)
            self._combine(vals, specs_df, param)
            if plot_trends:
                plt.plot(specs_df.index, vals, 'o--', alpha=0.8, label=param)
        if plot_trends:
            if len(self.annual_fcn_dict.keys()):
                plt.legend()
                plt.title("Daily")
                plt.show()

        return specs_df
