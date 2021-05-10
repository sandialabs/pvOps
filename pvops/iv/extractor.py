
class IVProcessor():
    '''Process measured IV curves
        Requires a set of curves to create Isc vs Irr and Voc vs Temp vs Isc(Irr)
    '''

    def __init__(self, pristine_curves, current_col, voltage_col, irradiance_col, temperature_col, T_type, windspeed_col = None,
                 Simulator_mod_specs = None, Simulator_pristine_condition = None):
        '''
        Parameters
        ----------
        pristine_curves, df
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
        #if Simulator_mod_specs is not None and Simulator_pristine_condition is not None:
        #    Simulator.__init__(self, mod_specs = Simulator_mod_specs, pristine_condition = Simulator_pristine_condition)
        self.Simulator_mod_specs = Simulator_mod_specs
        self.Simulator_pristine_condition = Simulator_pristine_condition

        self.tstamps = pristine_curves.index.tolist()
        self.Is = pristine_curves[current_col].tolist()
        self.Vs = pristine_curves[voltage_col].tolist()
        self.Irrs = pristine_curves[irradiance_col].tolist()
        self.Temps = pristine_curves[temperature_col].tolist()
        self.T_type = T_type
        self.Tcs = []

        if self.T_type is 'ambient' and windspeed_col is None:
            raise Exception("Wind speed must be specified if passing ambient temperature so that the cell temperature can be derived.")

        if windspeed_col is not None:
            self.WSs = pristine_curves[windspeed_col].tolist()
            if self.T_type is 'ambient':
                for irr,temp,ws in zip(self.Irrs, self.Temps, self.WSs):
                    Tc = T_to_tcell(irr,temp,ws,self.T_type)
                    self.Tcs.append(Tc)

        if self.T_type is 'module':
            for irr,temp in zip(self.Irrs, self.Temps):
                Tc = T_to_tcell(irr,temp,[],self.T_type)
                self.Tcs.append(Tc)

        self.measured_info = []
        for i in range(len(self.Is)):
            V = self.Vs[i]
            I = self.Is[i]
            Irr = self.Irrs[i]
            T = self.Temps[i]
            self.measured_info.append({"V":V,"I":I,"E":Irr,"T":T})
        
        self.n_samples = len(pristine_curves.index)

        self.params = {}
        
    def smooth_curve(self):
    
        lowess = sm.nonparametric.lowess
    
        xs_string,ys_string = [],[]
        for i in range(self.curve_number): 
        
            x = v[i]
            y = c[i]

            # pyqt-fit
            #xx = np.linspace(np.min(x),np.max(x),npts)
            #yhat = smooth.NonParamRegression(x, y, method=npr_methods.LocalPolynomialKernel(q=5))
            #yhat.fit()
            #yh = yhat(xx)
            
            # Lowess trial
            #xx = x
            #yh = lowess(y, x, frac= 0.08, it=0)
            #yh = yh[:,1]
            
            # numpy poly
            
            
            #x = x[10:]
            #y = y[10:]
            xx = np.linspace(1,np.max(x),50)
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

            WAIT FOR FUNCTION FROM JEN IN FINDING LINEAR REGION

            save to self.params dict
        '''
        return calculate_params(v,c)
        
    def train_met_fit(self, simulation_function):
        '''Assuming PRISTINE IV_measured

            Vsim_prist, Isim_prist

            Note:
            -----
            Possible alternative to requiring technician: make this automated
                    Make assumption of slope, 
                    Isc vs. Irradiance
        '''

        Gstc, Tstc = 1000., 50.

        import matplotlib.pyplot as plt

        meas_Iscs = []
        sim_Iscs = []
        meas_Vocs = []
        sim_Vocs = []
        Tcs = []
        for v,c,irr,Tc in zip(self.Vs, self.Is, self.Irrs, self.Tcs):
            
            Vsim, Isim = simulation_function(irr, Tc)
            Vsimcutoff, Isimcutoff = iv_cutoff(Vsim, Isim, 0)
            sim_prist_param = self._calculate_params(Vsimcutoff, Isimcutoff)
            param = self._calculate_params(v, c)

            print('isc', param['isc'], sim_prist_param['isc'])
            print('voc', param['voc'], sim_prist_param['voc'])

            meas_Iscs.append(param['isc'])
            sim_Iscs.append(sim_prist_param['isc'])
            meas_Vocs.append(param['voc'])
            sim_Vocs.append(sim_prist_param['voc'])
            Tcs.append(Tc)

        Tcs = np.array(Tcs)
        self.Irrs = np.array(self.Irrs)
        meas_Iscs = np.array(meas_Iscs)
        self.sim_Iscs = np.array(sim_Iscs)
        meas_Vocs = np.array(meas_Vocs)
        self.sim_Vocs = np.array(sim_Vocs)

        self.isc_meas_lm = linear_model.LinearRegression().fit(meas_Iscs.reshape(-1, 1), self.Irrs.reshape(-1, 1))
        self.isc_sim_lm =  linear_model.LinearRegression().fit(self.sim_Iscs.reshape(-1, 1), self.Irrs.reshape(-1, 1))
        self.voc_meas_lm = linear_model.LinearRegression().fit(meas_Vocs.reshape(-1, 1),Tcs.reshape(-1, 1))
        self.voc_sim_lm =  linear_model.LinearRegression().fit(self.sim_Vocs.reshape(-1, 1),Tcs.reshape(-1, 1))
        
        # print(list(meas_Iscs))
        # iscpredictE = self.isc_meas_lm.predict(meas_Iscs.reshape(-1, 1))
        # print(iscpredictE)
        # plt.plot(meas_Iscs, self.Irrs, 'bo', markersize=2, label='Measured Isc')
        # plt.plot(meas_Iscs, iscpredictE, 'b')
        # plt.plot(sim_Iscs, self.Irrs, 'go', markersize=2, label='Simulated Isc')
        # plt.plot(sim_Iscs, self.isc_sim_lm.predict(sim_Iscs.reshape(-1, 1)), 'g')
        # plt.legend()
        # plt.show()

        # plt.plot(Tcs, meas_Vocs, 'bo', markersize=2, label='Measured Voc')
        # plt.plot(Tcs, self.voc_meas_lm.predict(Tcs.reshape(-1, 1)), 'b')
        # plt.plot(Tcs, sim_Vocs, 'go', markersize=2, label='Simulated Voc')
        # plt.plot(Tcs, self.voc_sim_lm.predict(Tcs.reshape(-1, 1)), 'g')
        # plt.legend()
        # plt.show()

        return Tcs, self.Irrs, meas_Iscs, self.sim_Iscs, meas_Vocs, self.sim_Vocs

    def met_fit(self, Vsim, Isim, Irr, Tc, meas_Isc, meas_Voc):
        '''Correct simulated IV curve to look like a measured
            Model built in train_met_fit

            For Isc, get E at correct Isc, not at simulated Isc

            **Need method like Jens which determines params with linfit first
        '''
        param = self._calculate_params(Vsim, Isim)

        # if sim models have no associated data in 

        #E_meas_pred = self.isc_meas_lm.predict(np.asarray([param['isc']]).reshape(-1,1))[0][0]
        E_sim_pred = self.isc_sim_lm.predict(np.asarray([meas_Isc]).reshape(-1,1))[0][0]

        #isc_sim_pred = self.isc_sim_lm.predict(np.asarray([Irr]).reshape(-1,1))[0][0]
        #deltaI = isc_sim_pred - isc_meas_pred
        #print('met fit')
        #print(isc_sim_pred, isc_meas_pred)
        #print('isc:',param['isc'])
        #print(f'E: {Irr} -> {E_meas_pred}')
        #Isim -= deltaI

        #voc_meas_pred = self.voc_meas_lm.predict(np.asarray([Tc]).reshape(-1,1))[0][0]
        #voc_sim_pred = self.voc_sim_lm.predict(np.asarray([Tc]).reshape(-1,1))[0][0]
        #deltaV = voc_sim_pred - voc_meas_pred
        #print(voc_sim_pred, voc_meas_pred)
        #Vsim -= deltaV
        #print('voc:',param['voc'])
        #tc_meas_pred = self.voc_meas_lm.predict(np.asarray([param['voc']]).reshape(-1,1))[0][0]
        tc_sim_pred = self.voc_sim_lm.predict(np.asarray([meas_Voc]).reshape(-1,1))[0][0]
        #print(f'Tc: {Tc} -> {tc_meas_pred}')

        #print('MET CHOICES: ', E_meas_pred-Irr, tc_meas_pred-Tc)
        return E_sim_pred, tc_sim_pred
        
        #Irr - (E_meas_pred-Irr), Tc - (tc_meas_pred-Tc)

    # vref = v * (1 - (beta)*(tact-tref))
    # iref = i * (gref/gact)*(1-(alpha)*(tact-tref))
        

        #I, V, E, Ts = self.Is, self.Vs, self.Irrs, self.Temps
        #Isc = I.apply
        
        
        #PSEUDOCODE
        # linear fit Isc / Irradiance
        # if RMSE of a new point (IV_measured Isc/Irr) < thresh. (2-5%) if larger than 10%, return error
        #    #prompt a correction
        #    delta = computed irradiance f(Isc)  -   Irradiance
        #  BIN: (800,1000:-> 2)
        #
        #
        # Same for temp correction but Voc vs temp vs isc
        #
        #
        # Rsh: 0-20% of Voc trend line and match slopes
        # Rs: knee-100% of Voc ? and samehting
        #       detect linear portion of curve near Voc
        #           ddiv()

    def f(self, params):
        #iph, io, rs, rsh, nnsvth = params
        # voc_est = pvlib.singlediode.estimate_voc(iph, io, nnsvth)
        # v = voltage_pts(200, voc_est, self.module_parameters['breakdown_voltage'])
        
        # i = pvlib.singlediode.bishop88_i_from_v(v, iph, io, rs, rsh, nnsvth,
        #                                         breakdown_factor=self.module_parameters['breakdown_factor'],
        #                                         breakdown_voltage=self.module_parameters['breakdown_voltage'],
        #                                         breakdown_exp=self.module_parameters['breakdown_exp'])

        # simulate this to string according
        #TODO: ensure Simulator_mod_specs and Simulator_pristine_condition are populated
        kwargs = {}
        if self.Simulator_mod_specs is not None:
            kwargs.update({'mod_specs':self.Simulator_mod_specs})
        if self.Simulator_pristine_condition is not None:
            kwargs.update({'pristine_condition':self.Simulator_pristine_condition})

        sim = Simulator(**kwargs)
        # set new defaults
        sim.define_5params(params)

        #**** # comment out if want only module-level (indoor test)
        condition = {'identifier':'case',
                        'E':self.g,
                        'Tc':self.t,
                        }
        sim.add_preset_conditions('complete', condition, save_name = f'mod_case_{self.counter}')
        sim.build_strings({f'str_case_{self.counter}': [f'mod_case_{self.counter}']*12})

        sim.simulate()

        #****
        # 1. FSEC string level
        V = sim.multilevel_ivdata['string'][f'str_case_{self.counter}']['V'][0]
        I = sim.multilevel_ivdata['string'][f'str_case_{self.counter}']['I'][0]
        # 2. Sandia indoor module level
        # V = sim.multilevel_ivdata['module'][f'pristine']['V'][0]
        # I = sim.multilevel_ivdata['module'][f'pristine']['I'][0]

        # resample to same voltage domain as measured
        simI_interp = np.interp(self.meas_V, V, I)

        perc_diff = 100 * (np.array(params) - np.array(self.start_conds)) / np.array(self.start_conds)

        plt.plot(self.meas_V, simI_interp, label='simulated')
        plt.plot(self.meas_V, self.meas_I, label='measured')
        plt.title(f'Case {self.counter} with % Diff.: {perc_diff}')
        plt.legend()
        plt.show()

        self.counter += 1

        msse = mean_squared_error(self.meas_I, simI_interp)
        #sims = np.column_stack([V,I])
        #msse = mean_squared_error(self.meases_2d, sims)

        print(params)
        print(msse)
        return msse

    def create_string_object(self, iph, io, rs, rsh, nnsvth):
        kwargs = {}
        if self.Simulator_mod_specs is not None:
            kwargs.update({'mod_specs':self.Simulator_mod_specs})
        if self.Simulator_pristine_condition is not None:
            kwargs.update({'pristine_condition':self.Simulator_pristine_condition})
        kwargs.update({'replacement_5params':{'I_L_ref': iph,
                                            'I_o_ref': io,
                                            'R_s': rs,
                                            'R_sh_ref': rsh,
                                            'a_ref': nnsvth}
                                            })

        sim = Simulator(**kwargs)

        # set new defaults
        for sample_i, sample in enumerate(self.measured_info):

            condition = {'identifier':f'case_{self.counter}_{sample_i}',
                        'E':sample['E'],
                        'Tc':sample['T']
                        }

            sim.add_preset_conditions('complete', condition, save_name = f'mod_case_{self.counter}_{sample_i}')

            if isinstance(self.n_mods, int):
                if self.n_mods > 1:
                    sim.build_strings({f'str_case_{self.counter}_{sample_i}': [f'mod_case_{self.counter}_{sample_i}']*self.n_mods})

                elif self.n_mods != 1:
                    raise Exception(f"Input a valid number of modules, n_mods. You inputted {self.n_mods}")
            elif isinstance(self.n_mods, (tuple, list, np.ndarray)):
                sim.build_strings({f'str_case_{self.counter}_{sample_i}': [f'mod_case_{self.counter}_{sample_i}']*self.n_mods})

        print('starting simulation')
        start_t = time.time()
        sim.simulate()
        print(f'\tSimulations completed after {round(time.time()-start_t,2)} seconds')

        return sim

    def f_multiple_samples(self, params):
        
        # voc_est = pvlib.singlediode.estimate_voc(iph, io, nnsvth)
        # v = voltage_pts(200, voc_est, self.module_parameters['breakdown_voltage'])
        
        # i = pvlib.singlediode.bishop88_i_from_v(v, iph, io, rs, rsh, nnsvth,
        #                                         breakdown_factor=self.module_parameters['breakdown_factor'],
        #                                         breakdown_voltage=self.module_parameters['breakdown_voltage'],
        #                                         breakdown_exp=self.module_parameters['breakdown_exp'])

        # simulate this to string according
        #TODO: ensure Simulator_mod_specs and Simulator_pristine_condition are populated

        iph, io, rs, rsh, nnsvth = params

        if self.user_func == None:
            sim = self.create_string_object(iph, io, rs, rsh, nnsvth)
        else:
            sim = self.user_func(self, iph, io, rs, rsh, nnsvth)

        perc_diff = 100 * (np.array(params) - np.array(self.start_conds)) / np.array(self.start_conds)
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

            #print('meas:',sample['I'])
            #print('sim:',simI_interp)

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
        plt.plot(list(range(int(minpmps_m-10), int(maxpmps_m+10+1))), list(range(int(minpmps_m-10), int(maxpmps_m+10+1))), 'b--')
        plt.title('Measured v. Simulated Pmpp')
        plt.xlabel('Measured (W)')
        plt.ylabel('Simulated (W)')
        plt.xlim(minpmps_m-5, maxpmps_m+5)
        plt.ylim(minpmps_m-5, maxpmps_m+5)
        plt.show()

        minvocs_m = min(min(meas_Vocs), min(sim_Vocs))
        maxvocs_m = max(max(meas_Vocs), max(sim_Vocs))  
        plt.plot(meas_Vocs, sim_Vocs, 'ro')
        plt.plot(list(range(int(minvocs_m-10), int(maxvocs_m+10+1))), list(range(int(minvocs_m-10), int(maxvocs_m+10+1))), 'b--')
        plt.title('Measured v. Simulated Voc')
        plt.xlabel('Measured (V)')
        plt.ylabel('Simulated (V)')
        plt.xlim(minvocs_m-5, maxvocs_m+5)
        plt.ylim(minvocs_m-5, maxvocs_m+5)
        plt.show()

        miniscs_m = min(min(meas_Iscs), min(sim_Iscs))
        maxiscs_m = max(max(meas_Iscs), max(sim_Iscs))
        plt.plot(meas_Iscs, sim_Iscs, 'ko')
        plt.plot(list(range(int(miniscs_m-0.5), int(maxiscs_m+0.5+2))), list(range(int(miniscs_m-0.5), int(maxiscs_m+0.5+2))), 'b--')
        plt.title('Measured v. Simulated Isc')
        plt.xlabel('Measured (A)')
        plt.ylabel('Simulated (A)')
        plt.xlim(miniscs_m-0.5, maxiscs_m+0.5)
        plt.ylim(miniscs_m-0.5, maxiscs_m+0.5)
        plt.show()

        plt.plot(sample['V'], simI_interp,'r', label='Simulated')
        plt.plot(sample['V'], sample['I'],'k', label='Measured')
        plt.legend()
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.title(f'One example: case {self.counter} with % Diff.: {perc_diff}')
        plt.show()

        print('Params used in ^ iteration: ',params)

        self.counter += 1
        self.msses.append(msse_tot)
        return msse_tot

    def fit_params(self, cell_parameters, module_parameters, n_mods, user_func = None): #, meas_I, meas_V, g, tc):
        #     #parameters = sim.fit_params(i, v, g,t)
        # n_mods: if int, defines number of modules in string
        #         if tuple, define (n_mods_affected, total_mods)

        self.user_func = user_func
        
        self.n_mods = n_mods
        #self.meas_I = meas_I
        #self.meas_V = meas_V
        #self.meases_2d = np.column_stack([self.meas_V, self.meas_I])
        self.g = 1000#g
        self.t = 25#tc

        self.module_parameters = module_parameters
        self.cell_parameters = cell_parameters 

        self.counter = 0
        self.msses = []

        # iph, io, rs, rsh, nnsvth = pvlib.pvsystem.calcparams_cec(effective_irradiance=self.g, temp_cell=self.t,
        #                                                         alpha_sc=cell_parameters['alpha_sc'],
        #                                                         a_ref=cell_parameters['a_ref'],
        #                                                         I_L_ref=cell_parameters['I_L_ref'],
        #                                                         I_o_ref=cell_parameters['I_o_ref'],
        #                                                         R_sh_ref=cell_parameters['R_sh_ref'],
        #                                                         R_s=cell_parameters['R_s'],
        #                                                         Adjust=cell_parameters['Adjust'])
        
        iph = cell_parameters['I_L_ref']
        io = cell_parameters['I_o_ref']
        rs = cell_parameters['R_s']
        rsh = cell_parameters['R_sh_ref']
        Tref_K = 25 + 273.15
        Tcell_K = 25 + 273.15
        nnsvth = cell_parameters['a_ref'] # * (Tcell_K / Tref_K)

        perc_adjust = 0.5
        self.start_conds = (iph, io, rs, rsh, nnsvth)
        #print('start conds',self.start_conds)
        print('Given 5params:', iph, io, rs, rsh, nnsvth)
        converged_solution = minimize(self.f_multiple_samples, (iph, io, rs, rsh, nnsvth), bounds = ((0,2*iph),#(iph - iph*perc_adjust, iph + iph*perc_adjust),
                                                                                    (io - 40*io*perc_adjust, io + 40*io*perc_adjust),
                                                                                    (rs - 20*rs*perc_adjust, rs + 20*rs*perc_adjust),
                                                                                    (rsh - 150*rsh*perc_adjust, rsh + 150*rsh*perc_adjust),
                                                                                    (nnsvth - 10*nnsvth*perc_adjust, nnsvth + 10*nnsvth*perc_adjust)),
                                                                           method='TNC') #'TNC', 'SLSQP'

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
        
        # voc_est = pvlib.singlediode.estimate_voc(iph, io, nnsvth)
        # v = voltage_pts(200, voc_est, self.module_parameters['breakdown_voltage'])
        
        # i = pvlib.singlediode.bishop88_i_from_v(v, iph, io, rs, rsh, nnsvth,
        #                                         breakdown_factor=self.module_parameters['breakdown_factor'],
        #                                         breakdown_voltage=self.module_parameters['breakdown_voltage'],
        #                                         breakdown_exp=self.module_parameters['breakdown_exp'])

        # simulate this to string according
        #TODO: ensure Simulator_mod_specs and Simulator_pristine_condition are populated

        iph, io, rs, rsh, nnsvth = params

        sim = self.simulator_failure(iph, io, rs, rsh, nnsvth)

        perc_diff = 100 * (np.array(params) - np.array(self.start_conds)) / np.array(self.start_conds)
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

            #print('meas:',sample['I'])
            #print('sim:',simI_interp)

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
        plt.plot(list(range(int(minpmps_m-10), int(maxpmps_m+10+1))), list(range(int(minpmps_m-10), int(maxpmps_m+10+1))), 'b--')
        plt.title('Measured v. Simulated Pmpp')
        plt.xlabel('Measured (W)')
        plt.ylabel('Simulated (W)')
        plt.xlim(minpmps_m-5, maxpmps_m+5)
        plt.ylim(minpmps_m-5, maxpmps_m+5)
        plt.show()

        minvocs_m = min(min(meas_Vocs), min(sim_Vocs))
        maxvocs_m = max(max(meas_Vocs), max(sim_Vocs))  
        plt.plot(meas_Vocs, sim_Vocs, 'ro')
        plt.plot(list(range(int(minvocs_m-10), int(maxvocs_m+10+1))), list(range(int(minvocs_m-10), int(maxvocs_m+10+1))), 'b--')
        plt.title('Measured v. Simulated Voc')
        plt.xlabel('Measured (V)')
        plt.ylabel('Simulated (V)')
        plt.xlim(minvocs_m-5, maxvocs_m+5)
        plt.ylim(minvocs_m-5, maxvocs_m+5)
        plt.show()

        miniscs_m = min(min(meas_Iscs), min(sim_Iscs))
        maxiscs_m = max(max(meas_Iscs), max(sim_Iscs))
        plt.plot(meas_Iscs, sim_Iscs, 'ko')
        plt.plot(list(range(int(miniscs_m-0.5), int(maxiscs_m+0.5+2))), list(range(int(miniscs_m-0.5), int(maxiscs_m+0.5+2))), 'b--')
        plt.title('Measured v. Simulated Isc')
        plt.xlabel('Measured (A)')
        plt.ylabel('Simulated (A)')
        plt.xlim(miniscs_m-0.5, maxiscs_m+0.5)
        plt.ylim(miniscs_m-0.5, maxiscs_m+0.5)
        plt.show()

        plt.plot(sample['V'], simI_interp,'r', label='Simulated')
        plt.plot(sample['V'], sample['I'],'k', label='Measured')
        plt.legend()
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.title(f'One example: case {self.counter} with % Diff.: {perc_diff}')
        plt.show()

        self.counter += 1
        self.msses.append(msse_tot)
        return msse_tot

    def fit_failure_condition(self, simulator_failure, starter):
        #TODO

        self.start_conds = starter
        #print('start conds',self.start_conds)
        print('Given 5params:', iph, io, rs, rsh, nnsvth)
        converged_solution = minimize(simulator_failure, (condition_failure), bounds = ((0.05,0.95)),
                                                                           method='TNC') #'TNC', 'SLSQP'

        print('initial: ', (start_cond))
        print('solution: ', converged_solution)


