"""
Derive the effective diode parameters from a set of input curves.
"""

from sklearn import linear_model
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


class IVProcessor():
    '''Process measured IV curves
        Requires a set of curves to create Isc vs Irr and Voc vs Temp vs Isc(Irr)
    '''

    def __init__(self, input_df, current_col, voltage_col, irradiance_col, temperature_col, T_type, windspeed_col=None,
                 Simulator_mod_specs=None, Simulator_pristine_condition=None):
        '''
        Parameters
        ----------
        input_df, df
            Contains IV curves with a datetime index
        current_col, str
            Indicates column where current values in IV curve are located; each cell is an array of current values in a single IV curve
        voltage_col, str
            Indicates column where voltage values in IV curve are located; each cell is an array of voltage values in a single IV curve
        irradiance_col, str
            Indicates column where irradiance value (W/m2)
        temperature_col, str
            Indicates column where temperature value (C)
        T_type: string,
            Describe input temperature, either 'ambient' or 'module' or 'cell'
        '''
        # if Simulator_mod_specs is not None and Simulator_pristine_condition is not None:
        #    Simulator.__init__(self, mod_specs = Simulator_mod_specs, pristine_condition = Simulator_pristine_condition)
        self.Simulator_mod_specs = Simulator_mod_specs
        self.Simulator_pristine_condition = Simulator_pristine_condition

        self.tstamps = input_df.index.tolist()
        self.Is = input_df[current_col].tolist()
        self.Vs = input_df[voltage_col].tolist()
        self.Irrs = input_df[irradiance_col].tolist()
        self.Temps = input_df[temperature_col].tolist()
        self.T_type = T_type
        self.Tcs = []

        if self.T_type is 'ambient' and windspeed_col is None:
            raise Exception(
                "Wind speed must be specified if passing ambient temperature so that the cell temperature can be derived.")

        if windspeed_col is not None:
            self.WSs = input_df[windspeed_col].tolist()
            if self.T_type is 'ambient':
                for irr, temp, ws in zip(self.Irrs, self.Temps, self.WSs):
                    Tc = T_to_tcell(irr, temp, ws, self.T_type)
                    self.Tcs.append(Tc)

        if self.T_type is 'module':
            for irr, temp in zip(self.Irrs, self.Temps):
                Tc = T_to_tcell(irr, temp, [], self.T_type)
                self.Tcs.append(Tc)

        self.measured_info = []
        for i in range(len(self.Is)):
            V = self.Vs[i]
            I = self.Is[i]
            Irr = self.Irrs[i]
            T = self.Temps[i]
            self.measured_info.append({"V": V, "I": I, "E": Irr, "T": T})

        self.n_samples = len(input_df.index)

        self.params = {}

    def smooth_curve(self):

        lowess = sm.nonparametric.lowess

        xs_string, ys_string = [], []
        for i in range(self.curve_number):

            x = v[i]
            y = c[i]

            # pyqt-fit
            #xx = np.linspace(np.min(x),np.max(x),npts)
            #yhat = smooth.NonParamRegression(x, y, method=npr_methods.LocalPolynomialKernel(q=5))
            # yhat.fit()
            #yh = yhat(xx)

            # Lowess trial
            #xx = x
            #yh = lowess(y, x, frac= 0.08, it=0)
            #yh = yh[:,1]

            # numpy poly

            #x = x[10:]
            #y = y[10:]
            xx = np.linspace(1, np.max(x), 50)
            yhat = np.poly1d(np.polyfit(x, y, 12))
            yh = yhat(xx)

            #xy = np.hstack([np.array([v]).T,np.array([yhat]).T])
            #xy = xy[(xy[:,1]>=0)]
            #x = xy[:, 0]
            #y = xy[:, 1]

            xs_string.append(xx)
            ys_string.append(yh)

        self.xs = xs_string
        self.ys = ys_string

        return xs_string, ys_string

    def calculate_all_params(self):
        ''' Calculate all common IV parameters
        '''
        for it in range(self.n_samples):
            # self.smooth_curve()  ?
            self._calculate_params(self.Vs[it], self.Is[it], self.tstamps[it])

    def _calculate_params(self, v, c):
        ''' Calculate the common IV parameters

            v: array
            i: array

            save to self.params dict
        '''
        return calculate_params(v, c)

    def create_string_object(self, iph, io, rs, rsh, nnsvth):
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
                                      f'mod_case_{self.counter}_{sample_i}']*self.n_mods})

                elif self.n_mods != 1:
                    raise Exception(
                        f"Input a valid number of modules, n_mods. You inputted {self.n_mods}")
            elif isinstance(self.n_mods, (tuple, list, np.ndarray)):
                sim.build_strings({f'str_case_{self.counter}_{sample_i}': [
                                  f'mod_case_{self.counter}_{sample_i}']*self.n_mods})

        print('starting simulation')
        start_t = time.time()
        sim.simulate()
        print(
            f'\tSimulations completed after {round(time.time()-start_t,2)} seconds')

        return sim

    def f_multiple_samples(self, params):

        iph, io, rs, rsh, nnsvth = params

        if self.user_func == None:
            sim = self.create_string_object(iph, io, rs, rsh, nnsvth)
        else:
            sim = self.user_func(self, iph, io, rs, rsh, nnsvth)

        perc_diff = 100 * \
            (np.array(params) - np.array(self.start_conds)) / \
            np.array(self.start_conds)
        msse_tot = 0

        meas_Iscs = []
        meas_Vocs = []
        meas_Pmps = []
        sim_Iscs = []
        sim_Vocs = []
        sim_Pmps = []

        for sample_i, sample in enumerate(self.measured_info):

            if self.n_mods > 1:
                V = sim.multilevel_ivdata['string'][f'str_case_{self.counter}_{sample_i}']['V'][0]
                I = sim.multilevel_ivdata['string'][f'str_case_{self.counter}_{sample_i}']['I'][0]
            elif self.n_mods == 1:
                V = sim.multilevel_ivdata['module'][f'mod_case_{self.counter}_{sample_i}']['V'][0]
                I = sim.multilevel_ivdata['module'][f'mod_case_{self.counter}_{sample_i}']['I'][0]

            # resample to same voltage domain as measured
            simI_interp = np.interp(sample['V'], V, I)

            # print('meas:',sample['I'])
            # print('sim:',simI_interp)

            msse = mean_squared_error(sample['I'], simI_interp)
            msse_tot += msse

            Vco, Ico = iv_cutoff(V, I, 0)
            sim_params = self._calculate_params(Vco, Ico)
            meas_params = self._calculate_params(sample['V'], sample['I'])

            meas_Iscs.append(meas_params['isc'])
            meas_Vocs.append(meas_params['voc'])
            meas_Pmps.append(meas_params['pmp'])
            sim_Iscs.append(sim_params['isc'])
            sim_Vocs.append(sim_params['voc'])
            sim_Pmps.append(sim_params['pmp'])

        minpmps_m = min(min(meas_Pmps), min(sim_Pmps))
        maxpmps_m = max(max(meas_Pmps), max(sim_Pmps))
        plt.plot(meas_Pmps, sim_Pmps, 'go')
        plt.plot(list(range(int(minpmps_m-10), int(maxpmps_m+10+1))),
                 list(range(int(minpmps_m-10), int(maxpmps_m+10+1))), 'b--')
        plt.title('Measured v. Simulated Pmpp')
        plt.xlabel('Measured (W)')
        plt.ylabel('Simulated (W)')
        plt.xlim(minpmps_m-5, maxpmps_m+5)
        plt.ylim(minpmps_m-5, maxpmps_m+5)
        plt.show()

        minvocs_m = min(min(meas_Vocs), min(sim_Vocs))
        maxvocs_m = max(max(meas_Vocs), max(sim_Vocs))
        plt.plot(meas_Vocs, sim_Vocs, 'ro')
        plt.plot(list(range(int(minvocs_m-10), int(maxvocs_m+10+1))),
                 list(range(int(minvocs_m-10), int(maxvocs_m+10+1))), 'b--')
        plt.title('Measured v. Simulated Voc')
        plt.xlabel('Measured (V)')
        plt.ylabel('Simulated (V)')
        plt.xlim(minvocs_m-5, maxvocs_m+5)
        plt.ylim(minvocs_m-5, maxvocs_m+5)
        plt.show()

        miniscs_m = min(min(meas_Iscs), min(sim_Iscs))
        maxiscs_m = max(max(meas_Iscs), max(sim_Iscs))
        plt.plot(meas_Iscs, sim_Iscs, 'ko')
        plt.plot(list(range(int(miniscs_m-0.5), int(maxiscs_m+0.5+2))),
                 list(range(int(miniscs_m-0.5), int(maxiscs_m+0.5+2))), 'b--')
        plt.title('Measured v. Simulated Isc')
        plt.xlabel('Measured (A)')
        plt.ylabel('Simulated (A)')
        plt.xlim(miniscs_m-0.5, maxiscs_m+0.5)
        plt.ylim(miniscs_m-0.5, maxiscs_m+0.5)
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

    # , meas_I, meas_V, g, tc):
    def fit_params(self, cell_parameters, module_parameters, n_mods, user_func=None, verbose=0):
        #     #parameters = sim.fit_params(i, v, g,t)
        # n_mods: if int, defines number of modules in string
        #         if tuple, define (n_mods_affected, total_mods)
        # if verbose >= 1, print information about fitting
        # if verbose >= 2, plot information about each iteration

        self.user_func = user_func

        self.n_mods = n_mods
        self.g = 1000
        self.t = 25

        self.module_parameters = module_parameters
        self.cell_parameters = cell_parameters

        self.counter = 0
        self.msses = []

        iph = cell_parameters['I_L_ref']
        io = cell_parameters['I_o_ref']
        rs = cell_parameters['R_s']
        rsh = cell_parameters['R_sh_ref']
        Tref_K = 25 + 273.15
        Tcell_K = 25 + 273.15
        nnsvth = cell_parameters['a_ref']  # * (Tcell_K / Tref_K)

        perc_adjust = 0.5
        self.start_conds = (iph, io, rs, rsh, nnsvth)

        if self.verbose >= 1:
            print('Given 5params:', iph, io, rs, rsh, nnsvth)
        converged_solution = minimize(self.f_multiple_samples, (iph, io, rs, rsh, nnsvth), bounds=((0, 2*iph),
                                                                                                   (io - 40*io*perc_adjust, io + 40*io*perc_adjust),
                                                                                                   (rs - 20*rs*perc_adjust, rs + 20*rs*perc_adjust),
                                                                                                   (rsh - 150*rsh*perc_adjust, rsh + 150*rsh*perc_adjust),
                                                                                                   (nnsvth - 10*nnsvth*perc_adjust, nnsvth + 10*nnsvth*perc_adjust)),
                                      method='TNC')

        if self.verbose >= 1:
            print('bounds')
            print(((iph - iph*perc_adjust, iph + iph*perc_adjust),
                   (io - io*perc_adjust, io + io*perc_adjust),
                   (rs - rs*perc_adjust, rs + rs*perc_adjust),
                   (rsh - rsh*perc_adjust, rsh + rsh*perc_adjust),
                   (nnsvth - nnsvth*perc_adjust, nnsvth + nnsvth*perc_adjust)))

            print('initial: ', (iph, io, rs, rsh, nnsvth))
            print('solution: ', converged_solution)

        return converged_solution['x']

    def f_failure_multiple_samples(self, params):

        iph, io, rs, rsh, nnsvth = params

        sim = self.simulator_failure(iph, io, rs, rsh, nnsvth)

        perc_diff = 100 * \
            (np.array(params) - np.array(self.start_conds)) / \
            np.array(self.start_conds)
        msse_tot = 0

        meas_Iscs = []
        meas_Vocs = []
        meas_Pmps = []
        sim_Iscs = []
        sim_Vocs = []
        sim_Pmps = []

        for sample_i, sample in enumerate(self.measured_info):

            if self.n_mods > 1:
                V = sim.multilevel_ivdata['string'][f'str_case_{self.counter}_{sample_i}']['V'][0]
                I = sim.multilevel_ivdata['string'][f'str_case_{self.counter}_{sample_i}']['I'][0]
            elif self.n_mods == 1:
                V = sim.multilevel_ivdata['module'][f'mod_case_{self.counter}_{sample_i}']['V'][0]
                I = sim.multilevel_ivdata['module'][f'mod_case_{self.counter}_{sample_i}']['I'][0]

            # resample to same voltage domain as measured
            simI_interp = np.interp(sample['V'], V, I)

            msse = mean_squared_error(sample['I'], simI_interp)
            msse_tot += msse

            if self.verbose >= 2:
                Vco, Ico = iv_cutoff(V, I, 0)
                sim_params = self._calculate_params(Vco, Ico)
                meas_params = self._calculate_params(sample['V'], sample['I'])

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
            plt.plot(list(range(int(minpmps_m-10), int(maxpmps_m+10+1))),
                     list(range(int(minpmps_m-10), int(maxpmps_m+10+1))), 'b--')
            plt.title('Measured v. Simulated Pmpp')
            plt.xlabel('Measured (W)')
            plt.ylabel('Simulated (W)')
            plt.xlim(minpmps_m-5, maxpmps_m+5)
            plt.ylim(minpmps_m-5, maxpmps_m+5)
            plt.show()

            minvocs_m = min(min(meas_Vocs), min(sim_Vocs))
            maxvocs_m = max(max(meas_Vocs), max(sim_Vocs))
            plt.plot(meas_Vocs, sim_Vocs, 'ro')
            plt.plot(list(range(int(minvocs_m-10), int(maxvocs_m+10+1))),
                     list(range(int(minvocs_m-10), int(maxvocs_m+10+1))), 'b--')
            plt.title('Measured v. Simulated Voc')
            plt.xlabel('Measured (V)')
            plt.ylabel('Simulated (V)')
            plt.xlim(minvocs_m-5, maxvocs_m+5)
            plt.ylim(minvocs_m-5, maxvocs_m+5)
            plt.show()

            miniscs_m = min(min(meas_Iscs), min(sim_Iscs))
            maxiscs_m = max(max(meas_Iscs), max(sim_Iscs))
            plt.plot(meas_Iscs, sim_Iscs, 'ko')
            plt.plot(list(range(int(miniscs_m-0.5), int(maxiscs_m+0.5+2))),
                     list(range(int(miniscs_m-0.5), int(maxiscs_m+0.5+2))), 'b--')
            plt.title('Measured v. Simulated Isc')
            plt.xlabel('Measured (A)')
            plt.ylabel('Simulated (A)')
            plt.xlim(miniscs_m-0.5, maxiscs_m+0.5)
            plt.ylim(miniscs_m-0.5, maxiscs_m+0.5)
            plt.show()

            plt.plot(sample['V'], simI_interp, 'r', label='Simulated')
            plt.plot(sample['V'], sample['I'], 'k', label='Measured')
            plt.legend()
            plt.xlabel('Voltage (V)')
            plt.ylabel('Current (A)')
            plt.title(
                f'One example: case {self.counter} with % Diff.: {perc_diff}')
            plt.show()

        self.counter += 1
        self.msses.append(msse_tot)
        return msse_tot
