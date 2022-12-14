"""
Derive the effective diode parameters from a set of input curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
from pvops.iv.simulator import Simulator
import time
from pvops.iv.physics_utils import iv_cutoff, T_to_tcell, \
    calculate_IVparams, smooth_curve


class BruteForceExtractor():
    '''Process measured IV curves to extract diode parameters.
    Requires a set of curves to create Isc vs Irr and Voc vs Temp vs Isc(Irr)

    Parameters
    ----------
    input_df : DataFrame
        Contains IV curves with a datetime index
    current_col : string
        Indicates column where current values in IV curve are located;
        each cell is an array of current values in a single IV curve
    voltage_col : string
        Indicates column where voltage values in IV curve are located;
        each cell is an array of voltage values in a single IV curve
    irradiance_col : string
        Indicates column where irradiance value (W/m2)
    temperature_col : string
        Indicates column where temperature value (C)
    T_type : string
        Describe input temperature, either 'ambient' or 'module' or 'cell'
    '''

    def __init__(
        self,
        input_df,
        current_col,
        voltage_col,
        irradiance_col,
        temperature_col,
        T_type,
        windspeed_col=None,
        Simulator_mod_specs=None,
        Simulator_pristine_condition=None):

        self.Simulator_mod_specs = Simulator_mod_specs
        self.Simulator_pristine_condition = Simulator_pristine_condition

        self.tstamps = input_df.index.tolist()
        self.Is = input_df[current_col].tolist()
        self.Vs = input_df[voltage_col].tolist()
        self.Irrs = input_df[irradiance_col].tolist()
        self.Temps = input_df[temperature_col].tolist()
        self.T_type = T_type
        self.Tcs = []

        if self.T_type == 'ambient' and windspeed_col is None:
            raise Exception(
                "Wind speed must be specified if passing ambient temperature so that the cell temperature can be derived.")

        if windspeed_col is not None:
            self.WSs = input_df[windspeed_col].tolist()
            if self.T_type == 'ambient':
                for irr, temp, ws in zip(self.Irrs, self.Temps, self.WSs):
                    Tc = T_to_tcell(irr, temp, ws, self.T_type)
                    self.Tcs.append(Tc)

        if self.T_type == 'module':
            for irr, temp in zip(self.Irrs, self.Temps):
                Tc = T_to_tcell(irr, temp, [], self.T_type)
                self.Tcs.append(Tc)

        self.measured_info = []
        for i in range(len(self.Is)):
            Varray = self.Vs[i]
            Iarray = self.Is[i]
            Irr = self.Irrs[i]
            T = self.Temps[i]
            self.measured_info.append({"V": Varray, "I": Iarray, "E": Irr, "T": T})

        self.n_samples = len(input_df.index)

        self.params = {}

    def create_string_object(self, iph, io, rs, rsh, nnsvth):
        # TODO write docstring
        kwargs = {}
        if self.Simulator_mod_specs is not None:
            kwargs.update({'mod_specs': self.Simulator_mod_specs})
        if self.Simulator_pristine_condition is not None:
            kwargs.update(
                {'pristine_condition': self.Simulator_pristine_condition})
        kwargs.update({'replacement_5params': {'I_L_ref': iph,
                                               'I_o_ref': io,
                                               'R_s': rs,
                                               'R_sh_ref': rsh,
                                               'a_ref': nnsvth}
                       })

        sim = Simulator(**kwargs)

        # set new defaults
        for sample_i, sample in enumerate(self.measured_info):

            condition = {'identifier': f'case_{self.counter}_{sample_i}',
                         'E': sample['E'],
                         'Tc': sample['T']
                         }

            sim.add_preset_conditions(
                'complete', condition, save_name=f'mod_case_{self.counter}_{sample_i}')

            if isinstance(self.n_mods, int):
                if self.n_mods > 1:
                    sim.build_strings({f'str_case_{self.counter}_{sample_i}': [
                                      f'mod_case_{self.counter}_{sample_i}'] * self.n_mods})

                elif self.n_mods != 1:
                    raise Exception(
                        f"Input a valid number of modules, n_mods. You inputted {self.n_mods}")
            # elif isinstance(self.n_mods, (tuple, list, np.ndarray)):
            #     sim.build_strings({f'str_case_{self.counter}_{sample_i}': [
            #                       f'mod_case_{self.counter}_{sample_i}']*self.n_mods[0] + ['pristine'] * (self.n_mods[1]-self.n_mods[0])})
            else:
                raise ValueError(
                    f"Expected n_mods to be a integer. Got: {type(self.n_mods)}")

        start_t = time.time()
        sim.simulate()

        if self.verbose >= 2:
            print(
                f'\tSimulations completed after {round(time.time()-start_t,2)} seconds')

        return sim

    def f_multiple_samples(self, params):
        # TODO write docstring
        iph, io, rs, rsh, nnsvth = params

        if self.user_func is None:
            sim = self.create_string_object(self, iph, io, rs, rsh, nnsvth)
        else:
            sim = self.user_func(self, iph, io, rs, rsh, nnsvth)

        msse_tot = 0

        if self.verbose >= 2:
            perc_diff = 100 * \
                (np.array(params) - np.array(self.start_conds)) / \
                np.array(self.start_conds)

            meas_Iscs = []
            meas_Vocs = []
            meas_Pmps = []
            sim_Iscs = []
            sim_Vocs = []
            sim_Pmps = []

        for sample_i, sample in enumerate(self.measured_info):

            if self.n_mods > 1:
                Varr = sim.multilevel_ivdata['string'][f'str_case_{self.counter}_{sample_i}']['V'][0]
                Iarr = sim.multilevel_ivdata['string'][f'str_case_{self.counter}_{sample_i}']['I'][0]
            elif self.n_mods == 1:
                Varr = sim.multilevel_ivdata['module'][f'mod_case_{self.counter}_{sample_i}']['V'][0]
                Iarr = sim.multilevel_ivdata['module'][f'mod_case_{self.counter}_{sample_i}']['I'][0]

            # resample to same voltage domain as measured
            simI_interp = np.interp(sample['V'], Varr, Iarr)

            msse = sklearn.metrics.mean_squared_error(sample['I'], simI_interp)
            msse_tot += msse

            if self.verbose >= 2:

                Vco, Ico = iv_cutoff(Varr, Iarr, 0)
                sim_params = calculate_IVparams(Vco, Ico)
                meas_params = calculate_IVparams(sample['V'], sample['I'])

                meas_Iscs.append(meas_params['isc'])
                meas_Vocs.append(meas_params['voc'])
                meas_Pmps.append(meas_params['pmp'])
                sim_Iscs.append(sim_params['isc'])
                sim_Vocs.append(sim_params['voc'])
                sim_Pmps.append(sim_params['pmp'])

        if self.verbose >= 2:

            minpmps_m = min(min(meas_Pmps), min(sim_Pmps))
            maxpmps_m = max(max(meas_Pmps), max(sim_Pmps))
            plt.plot(meas_Pmps, sim_Pmps, 'go')
            plt.plot(list(range(int(minpmps_m - 10), int(maxpmps_m + 10 + 1))),
                     list(range(int(minpmps_m - 10), int(maxpmps_m + 10 + 1))), 'b--')
            plt.title('Measured v. Simulated Pmpp')
            plt.xlabel('Measured (W)')
            plt.ylabel('Simulated (W)')
            plt.xlim(minpmps_m - 5, maxpmps_m + 5)
            plt.ylim(minpmps_m - 5, maxpmps_m + 5)
            plt.show()

            minvocs_m = min(min(meas_Vocs), min(sim_Vocs))
            maxvocs_m = max(max(meas_Vocs), max(sim_Vocs))
            plt.plot(meas_Vocs, sim_Vocs, 'ro')
            plt.plot(list(range(int(minvocs_m - 10), int(maxvocs_m + 10 + 1))),
                     list(range(int(minvocs_m - 10), int(maxvocs_m + 10 + 1))), 'b--')
            plt.title('Measured v. Simulated Voc')
            plt.xlabel('Measured (V)')
            plt.ylabel('Simulated (V)')
            plt.xlim(minvocs_m - 5, maxvocs_m + 5)
            plt.ylim(minvocs_m - 5, maxvocs_m + 5)
            plt.show()

            miniscs_m = min(min(meas_Iscs), min(sim_Iscs))
            maxiscs_m = max(max(meas_Iscs), max(sim_Iscs))
            plt.plot(meas_Iscs, sim_Iscs, 'ko')
            plt.plot(list(range(int(miniscs_m - 0.5), int(maxiscs_m + 0.5 + 2))),
                     list(range(int(miniscs_m - 0.5), int(maxiscs_m + 0.5 + 2))), 'b--')
            plt.title('Measured v. Simulated Isc')
            plt.xlabel('Measured (A)')
            plt.ylabel('Simulated (A)')
            plt.xlim(miniscs_m - 0.5, maxiscs_m + 0.5)
            plt.ylim(miniscs_m - 0.5, maxiscs_m + 0.5)
            plt.show()

            plt.plot(sample['V'], simI_interp, 'r', label='Simulated')
            plt.title("SIMULATED")
            plt.show()

            plt.plot(sample['V'], simI_interp, 'r', label='Simulated')
            plt.plot(sample['V'], sample['I'], 'k', label='Measured')
            plt.legend()
            plt.xlabel('Voltage (V)')
            plt.ylabel('Current (A)')
            plt.title(
                f'One example: case {self.counter} with % Diff.: {perc_diff}')
            plt.show()

            print('Params used in ^ iteration: ', params)

        self.counter += 1
        self.msses.append(msse_tot)
        return msse_tot

    def fit_params(self, cell_parameters, n_mods, bounds_func, user_func=None, verbose=0):
        """
        Fit diode parameters from a set of IV curves.

        Parameters
        ----------
        cell_parameters : dict
            Cell-level parameters, usually extracted from the CEC
            database, which will be used as the
            initial guesses in the optimization process.
        n_mods : int
            if int, defines the number of modules in a
            string(1=simulate a single module)
        bounds_func : function
            Function to establish the bounded search space
            See below for an example:

            .. code-block:: python

                def bounds_func(iph,io,rs,rsh,nnsvth,perc_adjust=0.5):
                    return ((iph - 0.5*iph*perc_adjust, iph + 2*iph*perc_adjust),
                            (io - 40*io*perc_adjust, io + 40*io*perc_adjust),
                            (rs - 20*rs*perc_adjust, rs + 20*rs*perc_adjust),
                            (rsh - 150*rsh*perc_adjust, rsh + 150*rsh*perc_adjust),
                            (nnsvth - 10*nnsvth*perc_adjust, nnsvth + 10*nnsvth*perc_adjust))

        user_func : function
            Optional, a function similar to `self.create_string_object`
            which has the following inputs:
            `self, iph, io, rs, rsh, nnsvth`. This can be used to
            extract unique failure parameterization.
        verbose : int
            if verbose >= 1, print information about fitting
            if verbose >= 2, plot information about each iteration
        """

        self.user_func = user_func
        self.verbose = verbose
        self.n_mods = n_mods
        self.g = 1000
        self.t = 25

        self.cell_parameters = cell_parameters

        self.counter = 0
        self.msses = []

        iph = cell_parameters['I_L_ref']
        io = cell_parameters['I_o_ref']
        rs = cell_parameters['R_s']
        rsh = cell_parameters['R_sh_ref']
        nnsvth = cell_parameters['a_ref']

        self.start_conds = (iph, io, rs, rsh, nnsvth)

        bounds = bounds_func(*self.start_conds)

        if self.verbose >= 1:
            print('Given 5params:', iph, io, rs, rsh, nnsvth)
        converged_solution = scipy.optimize.minimize(self.f_multiple_samples,
                                                     (iph, io, rs, rsh, nnsvth),
                                                     bounds=bounds,
                                                     method='TNC')

        if self.verbose >= 1:
            print('bounds', bounds)
            print('initial: ', (iph, io, rs, rsh, nnsvth))
            print('solution: ', converged_solution)

        return converged_solution['x']
