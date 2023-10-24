import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy
import pyDOE
import itertools
import copy
import random
from tqdm import tqdm
import pvlib
from pvops.iv.utils import get_CEC_params
from pvops.iv.physics_utils import voltage_pts, add_series, bypass, \
    intersection, iv_cutoff, gt_correction


class Simulator():
    """An object which simulates Photovoltaic (PV) current-voltage (IV) curves with failures

    Parameters
    ----------
    mod_specs : dict
        Define the module and some definitions of that module which are not included 
        in the the CEC database. The `key` in this dictionary is the name of the 
        module in the CEC database. The `values` are `ncols`, which is the number of
        columns in the module, and `nsubstrings`, which is the number of substrings.
    pristine_condition : dict
        Defines the pristine condition. 
        A full condition is defined as a dictionary with the
        following key/value pairs:

        .. code-block:: python

            {
                'identifier': IDENTIFIER_NAME, # (str) Name used to define condition
                'E': IRRADIANCE, # (numeric) Value of irradiance (Watts per meter-squared)
                'Tc': CELL_TEMPERATURE, # (numeric) Multiplier usually less than 1 
                                        # to simulate a drop in Rsh
                'Rsh_mult': RSH_MULTIPLIER, # (numeric) Multiplier usually less than 1 
                                            # to simulate a drop in RSH
                'Rs_mult': RS_MULTIPLIER, # (numeric) Multiplier usually less than 1 
                                          # to simulate an increase in RS
                'Io_mult': IO_MULTIPLIER, # (numeric) Multiplier usually less than 1 
                                          # to simulate a drop in IO
                'Il_mult': IL_MULTIPLIER, # (numeric) Multiplier usually less than 1 
                                          # to simulate a drop in IL
                'nnsvth_mult': NNSVTH_MULTIPLIER, # (numeric) Multiplier usually less 
                                                  # than 1 to simulate a drop in NNSVTH, and therefore a_ref
                'modname': MODULE_NAME_IN_CECDB # (str) Module name in CEC database 
                                                # (e.g. Jinko_Solar_Co___Ltd_JKMS260P_60)
            }

    replacement_5params : dict
        Optional, replace the definitions of the five electrical parameters, which normally 
        are extracted from the CEC database. These parameters can be determined by 
        the :py:class:`IVProcessor` class

        Key/value pairs:

        .. code-block:: python

            {
                'I_L_ref': None,
                'I_o_ref': None,
                'R_s': None,
                'R_sh_ref': None,
                'a_ref': None
            }

    simulation_method : int
        Module simulation method (1 or 2)

        1) Avalanche breakdown model, as hypothesized in Ref. [1]_

        2) : Add-on to method 1, includes a rebalancing of the $I_sc$ prior to adding in series

        .. [1] "Computer simulation of the effects of electrical mismatches in photovoltaic cell 
            interconnection circuits" JW Bishop, Solar Cell (1988) DOI: 10.1016/0379-6787(88)90059-2

    Attributes
    ----------
    multilevel_ivdata : dict
        Dictionary containing the simulated IV curves

        - For nth-definition of string curves, 
          ``multilevel_ivdata['string']['STRING IDENTIFIER'][n]``
        - For nth-definition of module curves,
          ``multilevel_ivdata['module']['MODULE IDENTIFIER'][n]``
        - For nth-definition of substring (substr_id = 1,2,3,...) curves,
          ``multilevel_ivdata['module']['MODULE IDENTIFIER']['substr{sbstr_id}'][n]``

    pristine_condition : dict
        Dictionary of conditions defining the pristine case
    module_parameters : dict
        Dictionary of module-level parameters
    cell_parameters : dict
        Dictionary of cell-level parameters
    """

    def __init__(self,
                 mod_specs={
                     'Jinko_Solar_Co___Ltd_JKM270PP_60': {'ncols': 6,
                                                          'nsubstrings': 3
                                                          }
                 },
                 pristine_condition={
                     'identifier': 'pristine',
                     'E': 1000,
                     'Tc': 50,
                     'Rsh_mult': 1,
                     'Rs_mult': 1,
                     'Io_mult': 1,
                     'Il_mult': 1,
                     'nnsvth_mult': 1,
                 },
                 replacement_5params={'I_L_ref': None,
                                      'I_o_ref': None,
                                      'R_s': None,
                                      'R_sh_ref': None,
                                      'a_ref': None},
                 num_points_in_IV=200,
                 simulation_method=2):
        self.num_points_in_IV = num_points_in_IV
        self.modcells = dict()
        self.condition_dict = dict()
        self.simulation_method = simulation_method

        self.module_parameters = {}
        self.cell_parameters = {}

        mod_name = list(mod_specs.keys())[0]
        self.module_parameters, self.cell_parameters = get_CEC_params(
            mod_name, mod_specs[mod_name])
        self.module_parameters['v_bypass'] = 0.5

        # For all non-NULL values in given replacement parameters, *
        # replace in module_parameters and cell_parameters
        for k, v in replacement_5params.items():
            if v is not None:

                if k in ['R_sh_ref']:
                    # rsh_premultiply
                    self.module_parameters[k] = v * \
                        self.module_parameters["N_s"]
                elif k in ['a_ref', 'R_s']:
                    self.module_parameters[k] = v * \
                        self.module_parameters["N_s"]
                else:
                    self.module_parameters[k] = v

                if k == 'R_sh_ref':
                    self.cell_parameters[k] = v
                else:
                    self.cell_parameters[k] = v

        self.pristine_condition = pristine_condition
        self._add_pristine_condition()

        self.acceptible_keys = ['E', 'Tc', 'Rsh_mult',
                                'Rs_mult', 'Io_mult', 'Il_mult', 'nnsvth_mult']

        # Store substring, module, and string-level IV data
        # the substring data is stored within the module data
        self.multilevel_ivdata = {'string': {}, 'module': {}}
        self.string_cond = {}
        self.specific_cells_plotted = 0

    def _add_pristine_condition(self):
        """Save pristine case to modcells and condition_dict
        """

        self.modcells['pristine'] = [np.zeros(self.module_parameters['N_s'])]

        self.pristine_condition['V'] = list()
        self.pristine_condition['I'] = list()

        self.condition_dict[0] = [self.pristine_condition]

    def add_preset_conditions(self, fault_name, fault_condition, save_name=None, **kwargs):
        """Create cell-level fault conditions from presets defined by authors

        Parameters
        ----------
        fault_name: str
            Options:

            - 'complete': entire module has fault_condition (e.g. Full module shading)
              Requires no other specifications
              e.g. add_preset_conditions('complete', fault_condition)
            - 'landscape': entire rows are affected by fault_condition (e.g. interrow shading)
              Requires specification of rows_aff
              e.g. add_preset_conditions('landscape', fault_condition, rows_aff = 2)
            - 'portrait': entire columns are affected by fault_condition (e.g. vegetation growth shading)
              Requires specification of cols_aff

                - e.g. add_preset_conditions('portrait', fault_condition, cols_aff = 2)

            - 'pole': Place pole shadow over module
              Requires specification of width (integer), which designates the width of main shadow and \\
              requires light_shading fault_condition specification which specifies less intense shading \\
              on edges of shadow

                - Optional: pos = (left, right) designates the start and end of the pole shading,
                  where left is number in the first column and right is number in last column
                  if pos not specified, the positions are chosen randomly
                  e.g. add_preset_conditions('pole', fault_condition, light_shading = light_fault_condition, width = 2, pos = (5, 56))

            - 'bird_droppings': Random positions are chosen for bird_dropping simulations

              - Optional specification is n_droppings. If not specified, chosen as random number between 
                1 and the number of cells in a column
                e.g. add_preset_conditions('bird_droppings', fault_condition, n_droppings = 3)

        fault_location: dict
            Same dict as one shown in __init__.

        kwargs: variables dependent on which fault_name you choose, see above

        Tip
        ---
        For a wider spectrum of cases, run all of these multiple times. Each time it's run, the case is saved 
        """
        acceptible_fault_names = [
            'complete', 'landscape', 'portrait', 'pole', 'bird_droppings']

        # check if fault_condition has modname as key
        # if not, make it same as pristine module
        kwargs = locals()['kwargs']
        if fault_name in acceptible_fault_names:
            modcell, new_id, savename = self._simulate_soiling_cases(
                fault_name, kwargs)
            svname = save_name or savename
            if fault_name == 'pole':
                self._add_conditions({svname: modcell},
                                     {new_id: fault_condition,
                                      new_id + 1: kwargs['light_shading']})
            else:
                self._add_conditions({svname: modcell}, {
                                     new_id: fault_condition})

    def _simulate_soiling_cases(self, case, vardict):

        # Define our new key as max val of current keys plus 1
        new_id = max(self.condition_dict.keys()) + 1

        # Case 1: Complete shading
        if case == 'complete':
            return np.array([new_id] * self.module_parameters['N_s']), new_id, 'complete'

        modcell = np.zeros(self.module_parameters['N_s'])

        # Case 2: Landscape shading, shade by row (e.g. Interrow shading )
        if case == 'landscape':
            idx = self._simulate_landscape(vardict['rows_aff'])
            modcell[idx] = new_id
            savename = f'landscape_{vardict["rows_aff"]}rows'

        # Case 3: Portrait shading, shade by column (e.g. Vegetation shading)
        if case == 'portrait':
            idx = self._simulate_portrait(vardict['cols_aff'])
            modcell[idx] = new_id
            savename = f'portrait_{vardict["cols_aff"]}cols'

        # Case 4: Pole shading
        if case == 'pole':
            dark_idx, light_idx = self._simulate_pole_shading(vardict['width'])
            modcell[np.array(dark_idx)] = new_id
            modcell[np.array(light_idx)] = new_id + 1
            savename = f'pole_{vardict["width"]}width'

        # Case 5: Bird droppings
        if case == 'bird_droppings':
            if 'n_droppings' in vardict.keys():

                idx, n_droppings = self._simulate_bird_droppings(
                    vardict['n_droppings'])
            else:
                idx, n_droppings = self._simulate_bird_droppings(None)
            savename = f'bird_{n_droppings}droppings'
            modcell[idx] = new_id

        return modcell, new_id, savename

    def add_manual_conditions(self, modcell, condition_dict):
        """Create cell-level fault conditions manually

        Parameters
        ----------
        modcell : dict
            Key: name of the condition
            Value: list,

                - 1D list: Give a single situation for this condition
                - 2D list: Give multiple situations for this condition
                - A list where each value signifies a cell's condition. 

                If key is same as an existing key, the list is appended to list of scenarios \\
                which that key owns
        condition_dict: dict
            Define the numerical value written in modcell

            .. note::

               If the variable is not defined, values will default to those specified \\
               in the pristine condition, defined in __init__.

            A full condition is defined as:

            .. code-block:: python

                {ID: {'identifier': IDENTIFIER_NAME, # (str) Name used to define condition
                      'E': IRRADIANCE, # (numeric) Value of irradiance (Watts per meter-squared)
                      'Tc': CELL_TEMPERATURE, # (numeric) Value of cell temperature (Celcius)
                      'Rsh_mult': RSH_MULTIPLIER, # (numeric) Multiplier usually less than 1 
                                                  # to simulate a drop in Rsh
                      'Rs_mult': RS_MULTIPLIER, # (numeric) Multiplier usually greater than 1 
                                                # to simulate increase in Rs
                      'Io_mult': IO_MULTIPLIER, # (numeric) Multiplier usually less than 1 
                                                # to simulate a drop in IO
                      'Il_mult': IL_MULTIPLIER, # (numeric) Multiplier usually less than 1 
                                                # to simulate a drop in IL
                      'nnsvth_mult': NNSVTH_MULTIPLIER # (numeric) Multiplier usually less than 1 to 
                                                       # simulate a drop in NNSVTH, and therefore a_ref
                    }
                }      
        """
        self._add_conditions(modcell, condition_dict)

    def _add_val_to_dict(self, d, k, v):
        """Utility function to conglomerate a dictionary with 2d lists as value

        Parameters
        ----------
        d : dict
        k : a key in `d` dictionary
        v : value to update at key

        Returns
        -------
        Dictionary with updated v value
        """
        if k in list(d):
            d[k] += v
        else:
            d[k] = [v]
        return d

    def _get_key_set(self, d, key):
        """Get all values at key for every ID in dict
        Dictionary[k] = key value

        Parameters
        ----------
        d : dict
        key : a key in `d` dictionary

        Returns
        -------
        A reformatted dictionary
        """
        dct = {}
        for k in list(d):
            dct[k] = d[k][0][key]
        return dct

    def _add_conditions(self, modcells, condition_dict):
        """Organize all failure conditions into objects
        See `add_manual_conditions` for parameter definitions.
        """

        rename_dict = {}

        # Fill in gaps with info from pristine
        for ID in list(condition_dict):
            iter_cond_dict = condition_dict[ID].copy()

            ID_verdict = ID

            found_empty_dict_flag = False
            found_duplicate_flag = False
            found_same_ID_flag = False

            # If no keys defined, set this condition as pristine
            # or if only the 'identifier' is specified and it is 'pristine'

            if not set(iter_cond_dict):
                found_empty_dict_flag = True
            else:
                if 'identifier' in list(iter_cond_dict):
                    if iter_cond_dict['identifier'] == 'pristine':
                        found_empty_dict_flag = True

            if found_empty_dict_flag:
                # If ID is zero, just delete this case because pristine is added already (in __init__)
                # If ID not zero, need to also change all IDs in modcell to 0
                if ID != 0:
                    ID_verdict = 0
                    rename_dict = self._add_val_to_dict(
                        rename_dict, ID_verdict, ID)
                continue

            else:
                valid_msk = [(k not in set(self.pristine_condition))
                             for k in list(iter_cond_dict)]
                if any(valid_msk):
                    raise Exception(
                        f'Invalid key(s) passed in condition_dict: {list(iter_cond_dict)[valid_msk]}\nValid keys are {list(self.pristine_condition)}')

                # Get keys which are not defined
                undefined_keys = set(
                    self.pristine_condition) - set(iter_cond_dict)

                # Define undefined keys as the conditions in pristine_condition
                for undef_key in undefined_keys:
                    iter_cond_dict[undef_key] = self.pristine_condition[undef_key]

                # Check if there are keys in current condition_dict
                if list(self.condition_dict.keys()):

                    # get list of current 'identifier'
                    identifiers_dict = self._get_key_set(
                        self.condition_dict, 'identifier')

                    # if inputted identifier is in current identifiers, append sample to list
                    if iter_cond_dict['identifier'] in identifiers_dict.values():

                        # Check for identifier in dict keys, return dict key (cell_num)
                        ID_found = [key for (key, value) in identifiers_dict.items(
                        ) if value == iter_cond_dict['identifier']]

                        if (len(ID_found) == 0) or (len(ID_found) > 1):
                            raise Exception(
                                f"Debugging: {len(ID_found)} identifiers found even though it should have been one: {ID_found}")

                        elif len(ID_found) == 1:
                            ID_found = ID_found[0]

                            # Replace ID with ID_found and append condition_dict ONLY if \
                            # it is different.

                            if ID != ID_found:
                                rename_dict = self._add_val_to_dict(
                                    rename_dict, ID_found, ID)
                                ID_verdict = ID_found

                                # check if condition is already defined
                                for dct in self.condition_dict[ID_found]:
                                    if dct == iter_cond_dict:
                                        # If match found, delete current ID
                                        found_duplicate_flag = True
                                        continue

                            if not found_duplicate_flag:
                                # duplicate not found, append to dict
                                if len(self.condition_dict[ID_found]) == 0:
                                    # if none in list
                                    self.condition_dict[ID_found] = [
                                        iter_cond_dict]
                                else:
                                    # if conditions in list but none are this condition_dict
                                    self.condition_dict[ID_found].append(
                                        iter_cond_dict)

                    else:
                        # identifier was not found in self.condition_dict
                        # so, must create case

                        # Organize input IDs with all existing keys
                        # If current condition_dict has matching key with inputted, find next numerical key available and rename in condition_dict and modcells
                        if ID_verdict in list(self.condition_dict.keys()):
                            new_key = max(list(self.condition_dict.keys())) + 1
                            rename_dict = self._add_val_to_dict(
                                rename_dict, new_key, ID_verdict)
                            ID_verdict = new_key
                            found_same_ID_flag = True

                        if not found_same_ID_flag:
                            # An original condition is found
                            # Ensure that when adding a new condition, the ID is sequentially larger
                            # Used to organize ID systematically
                            expected_id = max(
                                list(self.condition_dict.keys())) + 1
                            if ID_verdict > expected_id:
                                rename_dict = self._add_val_to_dict(
                                    rename_dict, expected_id, ID_verdict)
                                ID_verdict = expected_id

                        self.condition_dict[ID_verdict] = [iter_cond_dict]
                else:
                    # self.condition_dict is blank
                    raise Exception(
                        "Debugging: this should never happen because pristine case is defined in __init__ which populates self.condition_dict.")

        # REPLACE IDs
        all_keys = list(modcells.keys())
        for idx in range(len(all_keys)):
            module_identifier = all_keys[idx]
            if (module_identifier == 'V') or (module_identifier == 'I'):
                raise ValueError(
                    f"Invalid module identifier, {module_identifier}. It cannot be 'I' or 'V'")

            modcell_list = modcells[module_identifier]

            if isinstance(modcell_list, (tuple, list)):
                modcell_list = np.array(modcell_list)
            elif not isinstance(modcell_list, np.ndarray):
                raise TypeError(
                    f"Invalid object ({type(modcell_list).__name__}) was passed to modcell[{module_identifier}]. Please define a list, tuple, or array as the value.")

            if len(modcell_list.shape) == 2:
                # 2d list found, therefore must process multiple modcells
                process_modcells = modcell_list
            elif len(modcell_list.shape) == 1:
                # 1d list found, put into a list for processing format
                process_modcells = [modcell_list]
            else:
                raise TypeError(
                    f"Invalid array shape ({modcell_list.shape}) passed to modcell is {len(modcell_list.shape)}D. Expected 1D or 2D iterable object.\nHere is object:\n{modcell_list}")

            required_n_cells = self.module_parameters["ncells_substring"] * \
                self.module_parameters["nsubstrings"]
            for modcell_arr in process_modcells:
                if len(modcell_arr) != required_n_cells:
                    raise Exception(
                        f"An input modcell has an invalid length. The input definition has {len(modcell_arr)} when {required_n_cells} is required.")

            comparator = copy.deepcopy(process_modcells)
            for replacement_id in list(rename_dict):
                for current_id in rename_dict[replacement_id]:
                    # for modcell_iter in process_modcells:
                    for idx in range(len(process_modcells)):

                        indexes = []
                        for i in range(len(comparator[idx])):
                            if comparator[idx][i] == current_id:
                                indexes.append(i)

                        for index in indexes:
                            process_modcells[idx][index] = replacement_id

            # Append modcells and condition_dict
            # Check if modcell key already exists
            if module_identifier in self.modcells.keys():
                for mdcel in process_modcells:
                    if mdcel not in np.array(self.modcells[module_identifier]).astype(int):
                        self.modcells[module_identifier] += process_modcells
            else:
                self.modcells[module_identifier] = process_modcells

    def reset_conditions(self):
        """Reset failure conditions
        """
        self.modcells = []
        self.condition_dict = dict()
        self.string_cond = dict()

    def print_info(self):
        """Display information about the established definitions
        """
        print('Condition list: (Cell definitions)')
        if len(list(self.condition_dict.keys())) > 0:
            for ID in list(self.condition_dict.keys()):
                ident = self.condition_dict[ID][0]['identifier']
                print(
                    f'\t[{ident}]: {len(self.condition_dict[ID])} definition(s)')
        else:
            print('\tNo instances.')
        print()
        print('Modcell types: (Cell mappings on module)')
        if len(list(self.modcells.keys())) > 0:
            for ident in list(self.modcells.keys()):
                print(f'\t[{ident}]: {len(self.modcells[ident])} definition(s)')
        else:
            print('\tNo instances.')
        print()
        print('String definitions (Series of modcells)')
        passed = True
        if len(list(self.string_cond.keys())) > 0:
            for str_key in self.string_cond:
                try:
                    print(
                        f"\t[{str_key}]: {len(self.multilevel_ivdata['string'][str_key]['V'])} definition(s)")
                except:
                    passed = False
                    continue
            if not passed:
                print('String definitions are defined by deducing the combination of module definitions. So, for an accurate display of the string-level definitions, call this module after enacting .simulate()')
        else:
            print('\tNo instances.')
        print()

    def sims_to_df(self, focus=['string', 'module'], cutoff=False):
        """Return the failure definitions as a dataframe.

        Parameters
        ----------
        focus : list of string
            Subset the definitions to a level of the system
            Currently available: 'substring', 'module', 'string'
        cutoff : bool
            Cutoff curves to only return on positive voltage domain

        Returns
        -------
        Dataframe with columns:

            - 'current': IV trace current
            - 'voltage': IV trace voltage
            - 'E': Average irradiance for all samples used to build this array
            - 'T': Average cell temperature for all samples used to build this array
            - 'mode': failure name
            - 'level': level of system (i.e. module, string), as defined by the input `focus` parameter

        #TODO: create focus for cell. For now, one can do it manually themselves.
        """
        Vs = []
        Is = []
        temps = []
        irrs = []
        mode = []
        level = []

        if 'substring' in focus:
            if len(self.multilevel_ivdata['module'].keys()) > 0:
                for mod_key in self.multilevel_ivdata['module'].keys():
                    for substr_id in range(1, 4):
                        v_s = self.multilevel_ivdata['module'][
                            mod_key][f'substr{substr_id}']['V']
                        i_s = self.multilevel_ivdata['module'][
                            mod_key][f'substr{substr_id}']['I']
                        e_s = self.multilevel_ivdata['module'][
                            mod_key][f'substr{substr_id}']['E']
                        t_s = self.multilevel_ivdata['module'][
                            mod_key][f'substr{substr_id}']['T']

                        Vs += v_s
                        Is += i_s
                        irrs += e_s
                        temps += t_s
                        level += ['substring'] * len(v_s)
                        mode += [mod_key] * len(v_s)

        if 'module' in focus:
            if len(self.multilevel_ivdata['module'].keys()) > 0:
                # Module definitions
                for mod_key in self.multilevel_ivdata['module'].keys():
                    v_s = self.multilevel_ivdata['module'][mod_key]['V']
                    i_s = self.multilevel_ivdata['module'][mod_key]['I']
                    e_s = self.multilevel_ivdata['module'][mod_key]['E']
                    t_s = self.multilevel_ivdata['module'][mod_key]['T']

                    Vs += v_s
                    Is += i_s
                    irrs += e_s
                    temps += t_s
                    level += ['module'] * len(v_s)
                    mode += [mod_key] * len(v_s)

        if 'string' in focus:
            if len(list(self.string_cond.keys())) > 0:
                # String definitions
                for str_key in self.string_cond:
                    v_s = self.multilevel_ivdata['string'][str_key]['V']
                    i_s = self.multilevel_ivdata['string'][str_key]['I']
                    e_s = self.multilevel_ivdata['string'][str_key]['E']
                    t_s = self.multilevel_ivdata['string'][str_key]['T']

                    Vs += v_s
                    Is += i_s
                    irrs += e_s
                    temps += t_s
                    level += ['string'] * len(v_s)
                    mode += [str_key] * len(v_s)

        if cutoff:
            cut_Vs = []
            cut_Is = []
            for V, I in zip(Vs, Is):
                v_, i_ = iv_cutoff(V, I, 0)
                cut_Vs.append(v_)
                cut_Is.append(i_)
            return pd.DataFrame({'current': cut_Is,
                                 'voltage': cut_Vs,
                                 'E': irrs,
                                 'T': temps,
                                 'mode': mode,
                                 'level': level})

        else:
            return pd.DataFrame({'current': Is,
                                 'voltage': Vs,
                                 'E': irrs,
                                 'T': temps,
                                 'mode': mode,
                                 'level': level})

    def simulate(self, sample_limit=None):
        """Simulate the cell, substring, module, and string-level IV curves using the defined conditions

        Parameters
        ----------
        sample_limit : int
            Optional, used when want to restrict number of combinations of failures at the string level.
        """

        self._simulate_all_cells()  # correct_gt = False)

        all_mods_sampled = []
        # Initializing structure
        for str_key in tqdm(self.string_cond, desc='Adding up simulations'):
            self.multilevel_ivdata['string'][str_key] = {
                'V': list(), 'I': list(), 'E': list(), 'T': list()}
            # if not isinstance(self.string_cond[str_key], list()): #THIS should be added (TODO)
            mods_set = set(self.string_cond[str_key])
            for mod_ident in mods_set:
                self.multilevel_ivdata['module'][mod_ident] = {
                    'V': list(), 'I': list(), 'E': list(), 'T': list()}
                for sbstr_id in range(1, self.module_parameters['nsubstrings'] + 1):
                    self.multilevel_ivdata['module'][mod_ident][f'substr{sbstr_id}'] = {
                        'V': list(), 'I': list(), 'E': list(), 'T': list()}
            all_mods_sampled += mods_set
            self._simulate_string(str_key, sample_limit=sample_limit)

        # simulate other modules not in strings
        all_mods_set = list(set(all_mods_sampled))

        for mod in tqdm(list(self.modcells.keys()), desc='Adding up other definitions'):
            if mod not in all_mods_set:
                self.multilevel_ivdata['module'][mod] = {
                    'V': list(), 'I': list(), 'E': list(), 'T': list()}
                for sbstr_id in range(1, self.module_parameters['nsubstrings'] + 1):
                    self.multilevel_ivdata['module'][mod][f'substr{sbstr_id}'] = {
                        'V': list(), 'I': list(), 'E': list(), 'T': list()}

                self.simulate_module(mod)
        return

    def simulate_module(self, mod_key):
        """Wrapper method which simulates a module depending on the defined simulation_method.

        Parameters
        ----------
        mod_key : str
            Module name as defined in condiction_dict and modcells
        """
        if self.simulation_method == 1:
            self.BISHOP88_simulate_module(mod_key)
        elif self.simulation_method == 2:
            self.PVOPS_simulate_module(mod_key)
        else:
            raise ValueError(
                "Invalid value passed to `simulation_method`. Must be either 1 or 2.")

    def simulate_modules(self):
        """Simulates all instantiated modules
        """
        for discrete_mod in list(self.modcells.keys()):
            # print(discrete_mod)
            # print('in simulate_modules, iterating to ', discrete_mod)
            self.simulate_module(discrete_mod)

    def BISHOP88_simulate_module(self, mod_key):
        cells_per_substring = self.module_parameters['N_s'] // self.module_parameters['nsubstrings']
        cell_id = [[j * cells_per_substring + i for i in range(
            0, cells_per_substring)] for j in range(0, self.module_parameters['nsubstrings'])]

        for md_idx, modset in enumerate(self.modcells[mod_key]):
            # get cell numbers in mod_key
            # put all descriptions of those numbers into a list
            # the following code (up to "combs") could be before this for loop \
            # but the different modcells[mod_key] combinations could have \
            # different cells inside.
            cell_in_modset = set(modset)

            cell_defs = []
            for cell_num in cell_in_modset:
                lkeys = []
                for sub_list_cell_num in self.condition_dict[cell_num]:
                    lkeys.append(
                        {cell_num: {k: sub_list_cell_num[k] for k in ['V', 'I', 'E', 'Tc']}})
                cell_defs.append(lkeys)

            combs = list(itertools.product(*cell_defs))
            for instance in combs:
                # Iterate through all combinations of cell conditions in a modcell combination

                # create lookup index: In tuple, index j corresponds with cond k_cond
                # using itertools, we created combs but the dictionaries are nested in tuples,
                # so need a lookup table for locations of cell conditions
                lookup = {}

                for j, inst in enumerate(instance):
                    k_cond = list(inst.keys())[0]
                    lookup[k_cond] = j

                # Get pristine I and V
                for key in self.condition_dict:
                    ident = self.condition_dict[key][0]['identifier']
                    if ident is self.pristine_condition['identifier']:
                        id_save = key
                        break
                pristineIV = self.condition_dict[id_save][0]

                mod_v, mod_i = None, None
                modEs, modTs = list(), list()
                # module: loop through substrings, cells in substring
                for s in range(self.module_parameters['nsubstrings']):
                    substr_v, substr_i = None, None
                    prist_substr_v, prist_substr_i = None, None
                    temps = list()
                    irrs = list()
                    for c in cell_id[s]:
                        cond = lookup[modset[c]]
                        substr_v, substr_i = add_series(instance[cond][modset[c]]['V'],
                                                        instance[cond][modset[c]]['I'],
                                                        substr_v, substr_i)
                        temps.append(instance[cond][modset[c]]['Tc'])
                        irrs.append(instance[cond][modset[c]]['E'])

                        prist_substr_v, prist_substr_i = add_series(pristineIV['V'], pristineIV['I'],
                                                                    prist_substr_v, prist_substr_i)

                    substr_v = bypass(
                        substr_v, self.module_parameters['v_bypass'])
                    self.multilevel_ivdata['module'][mod_key][f'substr{s+1}']['V'].append(
                        substr_v)
                    self.multilevel_ivdata['module'][mod_key][f'substr{s+1}']['I'].append(
                        substr_i)
                    self.multilevel_ivdata['module'][mod_key][f'substr{s+1}']['T'].append(
                        sum(temps) / len(temps))
                    self.multilevel_ivdata['module'][mod_key][f'substr{s+1}']['E'].append(
                        sum(irrs) / len(irrs))

                    mod_v, mod_i = add_series(substr_v, substr_i, mod_v, mod_i)
                    modEs += irrs
                    modTs += temps
                mod_v = bypass(mod_v, self.module_parameters['v_bypass'])
                self.multilevel_ivdata['module'][mod_key]['V'].append(mod_v)
                self.multilevel_ivdata['module'][mod_key]['I'].append(mod_i)
                self.multilevel_ivdata['module'][mod_key]['E'].append(
                    sum(modEs) / len(modEs))
                self.multilevel_ivdata['module'][mod_key]['T'].append(
                    sum(modTs) / len(modTs))

    def PVOPS_simulate_module(self, mod_key):
        cells_per_substring = self.module_parameters['N_s'] // self.module_parameters['nsubstrings']
        cell_id = [[j * cells_per_substring + i for i in range(
            0, cells_per_substring)] for j in range(0, self.module_parameters['nsubstrings'])]

        show_debugging_plots = False

        for md_idx, modset in enumerate(self.modcells[mod_key]):

            # Map all definitions to every possible definition-map combination
            cell_defs = []
            for cell_num in set(modset):
                lkeys = []
                for iiii, sub_list_cell_num in enumerate(self.condition_dict[cell_num]):
                    lkeys.append({cell_num: {k: sub_list_cell_num[k] for k in [
                                 'V', 'I', 'E', 'Tc', 'identifier']}})
                cell_defs.append(lkeys)

            # cell_defs structure: list[celltype][definition_num][celltype_ID][K E Y S]
            combs = list(itertools.product(*cell_defs))
            for instance in combs:
                # Iterate through all combinations of cell conditions in a modcell combination
                # create lookup index: In tuple, index j corresponds with cond k_cond
                # using itertools, we created combs but the dictionaries are nested in tuples,
                # so need a lookup table for locations of cell conditions
                lookup = {}
                # lookup[CELL_CONDITION_ID] = index_in_instance
                for j, inst in enumerate(instance):
                    k_cond = list(inst.keys())[0]
                    lookup[k_cond] = j

                # Get pristine I and V
                for key in self.condition_dict:
                    ident = self.condition_dict[key][0]['identifier']
                    if ident is self.pristine_condition['identifier']:
                        id_save = key
                        break
                pristineIV = self.condition_dict[id_save][0]

                mod_v, mod_i = None, None
                modEs, modTs = list(), list()
                # module: loop through substrings, cells in substring

                for s in range(self.module_parameters['nsubstrings']):

                    ivs = {}

                    celltypes = []
                    for celltype in cell_id[s]:
                        try:
                            celltypes.append(modset[celltype])
                        except:
                            print(modset)
                            print(celltype)
                            print(cell_id[s])
                    celltypes_set = set(celltypes)

                    # for each cell type,
                    for celltype in celltypes_set:
                        celltypeindex = celltypes.index(
                            celltype) + (s * cells_per_substring)
                        celltype = int(celltype)
                        # create location to store results
                        ivs[celltype] = {'V': None, 'I': None}
                        # get ncells with this cell type
                        ncell_this_type = celltypes.count(celltype)
                        # get cell IV curve
                        cond = lookup[modset[celltypeindex]]
                        cur = instance[cond][modset[celltypeindex]]['I']
                        vol = instance[cond][modset[celltypeindex]]['V']
                        # add series the num cells
                        for _ in range(ncell_this_type):
                            ivs[celltype]['V'], ivs[celltype]['I'] = add_series(
                                vol, cur, ivs[celltype]['V'], ivs[celltype]['I'])

                    # initialize substr IV
                    substr_v, substr_i = None, None
                    prist_substr_v, prist_substr_i = None, None
                    # for each cell type series
                    irrs, temps = list(), list()
                    for idx, cellseries in enumerate(list(ivs.keys())):
                        celltypeindex = celltypes.index(cellseries)
                        cond = lookup[modset[celltypeindex]]
                        iter_V = ivs[cellseries]['V']
                        iter_I = ivs[cellseries]['I']
                        temps.append(
                            instance[cond][modset[celltypeindex]]['Tc'])
                        irrs.append(instance[cond][modset[celltypeindex]]['E'])
                        if idx == 0:
                            substr_v, substr_i = add_series(
                                ivs[cellseries]['V'], ivs[cellseries]['I'], substr_v, substr_i)

                            if show_debugging_plots:
                                # ONLY CALCULATING FOR VISUAL PURPOSES
                                for c in cell_id[s]:
                                    prist_substr_v, prist_substr_i = add_series(pristineIV['V'], pristineIV['I'],
                                                                                prist_substr_v, prist_substr_i)
                        else:
                            # observe higher value, for now -10
                            def find_nearest(array, value):
                                array = np.asarray(array)
                                idx = (np.abs(array - value)).argmin()
                                return idx
                            idx_left_substr = find_nearest(substr_v, 0)
                            idx_left_iter = find_nearest(iter_V, 0)

                            # get effective Isc which is intersection in revere bias region
                            # Correct higher curve to effective Isc
                            if substr_i[idx_left_substr] > iter_I[idx_left_iter]:
                                substr_v_cutoff, substr_i_cutoff = substr_v.copy(
                                ), substr_i.copy()
                                realisc = substr_i[find_nearest(
                                    substr_v_cutoff, 0)]

                                # Reflect the iter curve
                                effective_Isc = intersection(
                                    list(-substr_v_cutoff), list(substr_i_cutoff), list(iter_V), list(iter_I))
                                # Essentially Isc minus effectiveISC
                                delta = realisc - effective_Isc[1][0]
                                substr_i -= delta
                            elif substr_i[idx_left_substr] < iter_I[idx_left_iter]:
                                iter_V_cutoff, iter_I_cutoff = iter_V.copy(
                                ), iter_I.copy()

                                realisc = iter_I[find_nearest(
                                    iter_V_cutoff, 0)]

                                # Reflect the substr curve
                                effective_Isc = intersection(
                                    list(-iter_V_cutoff), list(iter_I_cutoff), substr_v, substr_i)
                                # Essentially Isc minus effectiveISC
                                delta = realisc - effective_Isc[1][0]
                                iter_I -= delta

                            else:
                                # Equal! Doing nothing.
                                pass

                            substr_v, substr_i = add_series(
                                substr_v, substr_i, iter_V, iter_I)

                    substr_v = bypass(
                        substr_v, self.module_parameters['v_bypass'])
                    self.multilevel_ivdata['module'][mod_key][f'substr{s+1}']['V'].append(
                        substr_v)
                    self.multilevel_ivdata['module'][mod_key][f'substr{s+1}']['I'].append(
                        substr_i)

                    if show_debugging_plots:
                        plt.plot(prist_substr_v, prist_substr_i,
                                 'bo', markersize=2, label='pristine')
                        plt.plot(substr_v, substr_i, 'ro',
                                 markersize=2, label='Potential failure')
                        plt.legend()
                        plt.xlabel('V (Volts)')
                        plt.ylabel('I (Amps)')
                        plt.ylim(0, 9.5)
                        plt.xlim(-13.5, max(substr_v) + 2.)
                        plt.show()

                        plt.plot(substr_v, substr_i, label=f'{cellseries}')
                        plt.legend()
                        plt.title('Final substring plot')
                        plt.xlabel('V (Volts)')
                        plt.ylabel('I (Amps)')
                        plt.ylim(0, 9.5)
                        plt.grid()
                        plt.show()

                    modEs += irrs
                    modTs += temps

                    mod_v, mod_i = add_series(
                        substr_v, substr_i, mod_v, mod_i)

                mod_v = bypass(mod_v, self.module_parameters['v_bypass'])
                self.multilevel_ivdata['module'][mod_key]['V'].append(mod_v)
                self.multilevel_ivdata['module'][mod_key]['I'].append(mod_i)
                self.multilevel_ivdata['module'][mod_key]['E'].append(
                    sum(modEs) / len(modEs))
                self.multilevel_ivdata['module'][mod_key]['T'].append(
                    sum(modTs) / len(modTs))

    def _simulate_string(self, str_key, sample_limit=None):

        # Step 1. Create cell IVs from conditions by pushing condition list through calcparams_cec,  \
        # *  which calculates the params needed for single diode equation

        # STEP 2: Construct substring, module, and string level estimates
        # *  add forward turn-on voltage for bypass diode

        # list of strings, strings are modcells
        modstring = self.string_cond[str_key]
        module_set = set(modstring)
        # cell: condition_dict[ID][n]

        # only simulate the discrete modules
        # get conditions
        lengths = []
        for discrete_mod in module_set:

            # if no data for discrete_mod, simulate it
            if len(self.multilevel_ivdata['module'][discrete_mod]['V']) == 0:
                self.simulate_module(discrete_mod)

            # get all combinations of all discete modules in string (of all definitions of modules -- now stored in )
            lengths.append(
                len(self.multilevel_ivdata['module'][discrete_mod]['V']))

        rng = [list(range(ii)) for ii in lengths]
        combination_indices = list(itertools.product(*rng))

        if sample_limit is not None:
            combination_indices = combination_indices[:sample_limit]

        # iterate through all cominations of modules (defined earlier)
        for comb_idx in combination_indices:
            string_v, string_i = None, None
            temps, irrs = list(), list()
            for idx, discretemod_idx in enumerate(comb_idx):
                cur_module_name = list(module_set)[idx]
                # num. of discretemod in string
                num_instring = sum(
                    [True for modi in modstring if modi is cur_module_name])
                for _ in range(num_instring):
                    mod_v = self.multilevel_ivdata['module'][cur_module_name]['V'][discretemod_idx]
                    mod_i = self.multilevel_ivdata['module'][cur_module_name]['I'][discretemod_idx]
                    mod_E = self.multilevel_ivdata['module'][cur_module_name]['E'][discretemod_idx]
                    mod_T = self.multilevel_ivdata['module'][cur_module_name]['T'][discretemod_idx]

                    # String level sum
                    string_v, string_i = add_series(
                        mod_v, mod_i, string_v, string_i)
                    temps.append(mod_T)
                    irrs.append(mod_E)
            avgE = sum(irrs) / len(irrs)
            avgT = sum(temps) / len(temps)

            self.multilevel_ivdata['string'][str_key]['V'].append(string_v)
            self.multilevel_ivdata['string'][str_key]['I'].append(string_i)
            self.multilevel_ivdata['string'][str_key]['E'].append(avgE)
            self.multilevel_ivdata['string'][str_key]['T'].append(avgT)

        return

    def generate_many_samples(self, identifier, N, distributions=None, default_sample=None):
        # If specify 'low' and 'upp', use a truncnorm
        # If not, use a norm
        """For cell `identifier`, create `N` more samples by randomly sampling a gaussian or truncated gaussian distribution.

        Parameters
        ----------
        identifier : str
            Cell identifier to upsample
        N : int
            Number of samples to generate
        distributions : dict
            Dictionary of distribution definitions, either gaussian or truncated gaussian. 
            Each definition must note a 'mean' and 'std', however if 'low' and 'upp' thresholds 
            are also included, then a truncated gaussian distribution will be generated.

            One does not need to define distributions for all parameters, only those that you want altered.

            .. code-block:: python

                distributions = {
                    'Rsh_mult':{'mean':None,
                                'std': None,
                                'low': None,
                                'upp': None},
                    'Rs_mult': {'mean':None,
                                'std': None,
                                'low': None,
                                'upp': None},
                        ...
                        # All keys in self.acceptible_keys
                    }

        default_sample : 
            If provided, use this sample to replace the parameters which do not have distributions specified. Else, uses
            the pristine condition as a reference.
        """

        dicts = {'E': {'mean': 800,
                       'std': 500,
                       'low': 200,
                       'upp': 1250
                       },
                 'Tc': {'mean': 35,
                        'std': 10,
                        },
                 'Rsh_mult': {'mean': 0.9,
                              'std': 0.3,
                              'low': 0.1,
                              'upp': 1.25
                              },
                 'Rs_mult': {'mean': 1.4,
                             'std': 1.,
                             'low': 0.9,
                             'upp': 3.
                             },
                 'Il_mult': {'mean': 0.8,
                             'std': 0.5,
                             'low': 0.6,
                             'upp': 1.25
                             },
                 'Io_mult': {'mean': 0.8,
                             'std': 0.5,
                             'low': 0.4,
                             'upp': 1.25
                             },
                 'nnsvth_mult': {'mean': 0.9,
                                 'std': 0.5,
                                 'low': 0.6,
                                 'upp': 1.1
                                 }
                 }

        distribs = distributions or dicts

        validated_keys = list(
            set(distribs).intersection(set(self.acceptible_keys)))
        missing_keys = [
            m_k for m_k in self.acceptible_keys if m_k not in validated_keys]

        if default_sample is None:
            replacer = self.pristine_condition
        else:
            replacer = default_sample

            if set(replacer.keys()) != set(self.pristine_condition.keys()):
                raise Exception(
                    f"Inputted default_sample dictionary must have following keys: {self.pristine_condition.keys()}")

        n_features = len(validated_keys)
        design = []
        design = pyDOE.lhs(n_features, samples=N)

        for idx, k in enumerate(validated_keys):
            dict_iter = distribs[k]
            if ('low' in dict_iter.keys()) and ('upp' in dict_iter.keys()):
                # use truncnorm distribution
                mean, std, low, upp = dict_iter['mean'], dict_iter['std'], dict_iter['low'], dict_iter['upp']
                design[:, idx] = scipy.stats.truncnorm(
                    (low - mean) / std,
                    (upp - mean) / std,
                    loc=mean,
                    scale=std
                ).ppf(design[:, idx])
            else:
                # use normal distribution
                mean, std = dict_iter['mean'], dict_iter['std']
                design[:, idx] = scipy.stats.norm(
                    loc=mean,
                    scale=std,
                ).ppf(design[:, idx])

        found = False
        for key in self.condition_dict:
            ident = self.condition_dict[key][0]['identifier']
            if ident == identifier:
                id_save = key
                found = True
                break
        if not found:
            raise Exception(
                f"Passed 'identifier' must be an existing cell condition. You passed '{identifier}'.")

        # TODO: make this more efficient
        for row_idx in range(len(design)):
            d_iter = {}
            for i, param in enumerate(validated_keys):
                d_iter[param] = design[row_idx][i]
            for param in missing_keys:
                d_iter[param] = replacer[param]

            d_iter['identifier'] = identifier
            self.condition_dict[id_save].append(d_iter)

    def _addition_soiling(self, args):
        # TODO
        # Not pure addition. But, combining to soiling cases does make intensity increase
        for vals in locals()['aargs']:
            pass
        return

    def _combine_independent_failures(self, *aargs, delete_combined=False):
        # TODO
        # "condition_dict[num]['identifier'] + blah"
        # placeholder for linter
        condition_list = {}
        all_ = list(locals()['aargs'])
        combined = []
        for idx in range(len(all_[0])):
            args = []
            for inp in all_:
                args.append(condition_list[inp[idx]][0])
            combined.append(self._addition_soiling(args))
        return combined

    def _simulate_all_cells(self):
        """Simulates the set of unique conditions on PV cells
        """
        for ID in tqdm(self.condition_dict, desc='Simulating cells'):
            for n in range(len(self.condition_dict[ID])):
                cond_dict = self.condition_dict[ID][n]
                g, tc, rsh_mult, rs_mult, Io_mult, Il_mult, nnsvth_mult = cond_dict['E'], cond_dict['Tc'], cond_dict[
                    'Rsh_mult'], cond_dict['Rs_mult'], cond_dict['Io_mult'], cond_dict['Il_mult'], cond_dict['nnsvth_mult']
                # calculate the 5 parameters for each set of cell conditions

                # Eventually, replace this with derived 5-parameters
                iph, io, rs, rsh, nnsvth = pvlib.pvsystem.calcparams_cec(effective_irradiance=g, temp_cell=tc,
                                                                         alpha_sc=self.cell_parameters['alpha_sc'],
                                                                         a_ref=self.cell_parameters['a_ref'],
                                                                         I_L_ref=self.cell_parameters['I_L_ref'],
                                                                         I_o_ref=self.cell_parameters['I_o_ref'],
                                                                         R_sh_ref=self.cell_parameters['R_sh_ref'],
                                                                         R_s=self.cell_parameters['R_s'],
                                                                         Adjust=self.cell_parameters['Adjust'])
                rs, rsh, io, iph, nnsvth = rs * rs_mult, rsh * \
                    rsh_mult, io * Io_mult, iph * Il_mult, nnsvth * nnsvth_mult

                # calculate cell IV curves by condition, rather than by cell index
                voc_est = pvlib.singlediode.estimate_voc(iph, io, nnsvth)
                v = voltage_pts(self.num_points_in_IV, voc_est,
                                self.module_parameters['breakdown_voltage'])
                i = pvlib.singlediode.bishop88_i_from_v(v, iph, io, rs, rsh, nnsvth,
                                                        breakdown_factor=self.module_parameters['breakdown_factor'],
                                                        breakdown_voltage=self.module_parameters[
                                                            'breakdown_voltage'],
                                                        breakdown_exp=self.module_parameters['breakdown_exp'])

                # @dev: Uncomment if debugging pvlib bishop88 simulation results
                # plt.plot(v,i)
                # plt.xlim(-5,v[-1])
                # plt.ylim(0,iph+1)
                # plt.title(f"{ID}: {n} :: {rs},"
                #           f"{rsh}, {io}, {iph}, {nnsvth}")
                # plt.show()

                self.condition_dict[ID][n]['V'] = v
                self.condition_dict[ID][n]['I'] = i
                self.condition_dict[ID][n]['E'] = g
                self.condition_dict[ID][n]['Tc'] = tc
        return

    def build_strings(self, config_dict):
        """Pass a dictionary into object memory

        e.g. For 6 modules faulted with modcell specification 'complete'

        .. code-block:: python

            config_dict = {
                'faulting_bottom_mods': [
                    'pristine', 'pristine', 'pristine', 
                    'pristine', 'pristine', 'pristine', 
                    'complete', 'complete', 'complete', 
                    'complete', 'complete', 'complete'
                    ]
                }
        """

        # print(config_dict)
        self.string_cond.update(config_dict)

    def _histograms(self):

        vals = []
        cell_ids = []
        params = []
        number_gp = len(self.acceptible_keys)

        # iterate through cells
        for c_id in list(self.condition_dict.keys()):
            cell_ids += [c_id] * (len(self.condition_dict[c_id]) * number_gp)
            for dct in self.condition_dict[c_id]:
                for parm in self.acceptible_keys:
                    params.append(parm)
                    vals.append(dct[parm])

        df_gp = pd.DataFrame()
        df_gp['param_names'] = params
        df_gp['cell_id'] = cell_ids
        df_gp['value'] = vals

        colors = sns.color_palette("hls", len(cell_ids))

        # freq = the percentage for each age group, and there’re 7 age groups.
        def ax_settings(ax, var_name, x_min, x_max):
            ax.set_xlim(x_min, x_max)
            ax.set_yticks([])

            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            ax.spines['bottom'].set_edgecolor('#444444')
            ax.spines['bottom'].set_linewidth(2)

            ax.text(0.02, 0.05, var_name, fontsize=17,
                    fontweight="bold", transform=ax.transAxes)
            return None
        # Manipulate each axes object in the left. Try to tune some parameters and you'll know how each command works.
        fig = plt.figure(figsize=(12, 7))
        gs = matplotlib.gridspec.GridSpec(nrows=number_gp,
                                          ncols=1,
                                          figure=fig,
                                          width_ratios=[1],
                                          height_ratios=[1] * number_gp,
                                          wspace=0.2, hspace=0.05
                                          )
        ax = [None] * (number_gp + 1)

        # Create a figure, partition the figure into 7*2 boxes, set up an ax array to store axes objects, and create a list of age group names.
        for i in range(number_gp):
            ax[i] = fig.add_subplot(gs[i, 0])

            ax_settings(ax[i], 'Variable: ' +
                        self.acceptible_keys[i], -1000, 20000)

            cell_idx = 0
            for cellid_iter in list(self.condition_dict.keys()):
                # print('id', cellid_iter)
                sns.kdeplot(data=df_gp[(df_gp.cell_id == cellid_iter) & (df_gp.param_names == self.acceptible_keys[i])].value,
                            ax=ax[i], shade=True, color=colors[cell_idx], bw=300, legend=False)
                cell_idx += 1

            if i < (number_gp - 1):
                ax[i].set_xticks([])
        # this 'for loop' is to create a bunch of axes objects, and link them to GridSpec boxes. Then, we manipulate them with sns.kdeplot() and ax_settings() we just defined.
        ax[0].legend(self.acceptible_keys, facecolor='w')
        # adding legends on the top axes object
        ax[number_gp] = fig.add_subplot(gs[:, 0])
        ax[number_gp].spines['right'].set_visible(False)
        ax[number_gp].spines['top'].set_visible(False)
        ax[number_gp].set_xlim(0, 100)
        ax[number_gp].invert_yaxis()
        ax[number_gp].text(1.09, -0.04, '(%)', fontsize=10,
                           transform=ax[number_gp].transAxes)
        ax[number_gp].tick_params(axis='y', labelsize=14)
        # manipulate the bar plot on the right. Try to comment out some of the commands to see what they actually do to the bar plot.
        plt.show()
        return

    def visualize(self, lim=False):
        """Run visualization suite to visualize information about the simulated curves.
        """
        d = {}
        for c_id in list(self.condition_dict.keys()):
            iden = self.condition_dict[c_id][0]['identifier']
            d[iden] = {}
            for k in self.acceptible_keys:
                d[iden][k] = []

        dict_keys = {}
        maxIdent = ''
        maxL = 0
        for c_id in list(self.condition_dict.keys()):
            iden = self.condition_dict[c_id][0]['identifier']
            keys = []
            for dct in self.condition_dict[c_id]:
                for k in self.acceptible_keys:
                    keys.append(k)
                    d[iden][k].append(dct[k])
            dict_keys[iden] = keys
            if len(keys) > maxL:
                maxIdent = iden
                maxL = len(keys)

        # Saved for testing purposes
        self.maxL = maxL
        self.maxIdent = maxIdent

        # Get variables which were actually changed
        dynamic_vars = []
        for idx, k in enumerate(self.acceptible_keys):
            for c_id in list(self.condition_dict.keys()):
                iden = self.condition_dict[c_id][0]['identifier']
                data = d[iden][k]
                if np.array(data).std() != 0:
                    dynamic_vars.append(self.acceptible_keys[idx])
        dynamic_vars = list(set(dynamic_vars))
        for idx, k in enumerate(dynamic_vars):
            fig, axs = plt.subplots()
            for c_id in list(self.condition_dict.keys()):
                iden = self.condition_dict[c_id][0]['identifier']
                data = d[iden][k]

                if np.array(data).std() == 0:
                    # If no variance in this cell sample (likely 'pristine' case)
                    axs.axvline(x=data[0], label=iden, lw=5)

                else:
                    # Circumventing error where only one subplot needs to be defined.
                    if idx == 0:
                        axs = sns.distplot(
                            data, hist=False, rug=True, label=iden)
                    else:
                        axs = sns.distplot(
                            data, hist=False, rug=True, ax=axs, label=iden)

            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            axs.set_xlabel(k)
        plt.show()

        for idx, ident in enumerate(list(self.modcells.keys())):
            if idx == 0:
                ax = self.visualize_specific_iv(
                    string_identifier=None, module_identifier=ident, substring_identifier=None)
            else:
                ax = self.visualize_specific_iv(
                    ax=ax, string_identifier=None, module_identifier=ident, substring_identifier=None)
        plt.title('Module IV curves')

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if lim:
            plt.xlim(xmin=0)
            plt.ylim(ymin=0)
        plt.show()

        if len(self.string_cond.keys()) > 0:
            for idx, str_key in enumerate(self.string_cond):
                if idx == 0:
                    ax = self.visualize_specific_iv(
                        string_identifier=str_key, module_identifier=None, substring_identifier=None)
                else:
                    ax = self.visualize_specific_iv(
                        ax=ax, string_identifier=str_key, module_identifier=None, substring_identifier=None)
            plt.title('String IV curves')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            if lim:
                plt.xlim(xmin=0)
                plt.ylim(ymin=0)
            plt.show()

    def visualize_specific_iv(self, ax=None, string_identifier=None, module_identifier=None, substring_identifier=None, cutoff=True, correct_gt=False):
        """Visualize a string, module, or substring IV curve.
        If the object has multiple definitions, all definitions will be plotted

        Parameters
        ----------
        ax : matplotlib axes
            Optional, pass an axes to add visualization
        string_identifier : str
            Optional, Identification of string definition
        module_identifier : str
            Optional, Identification of module definition
        substring_identifier : str
            Optional, Identification of module definition
        cutoff : bool
            If True, only visualize IV curves in positive voltage domain
        correct_gt : bool
            If True, correct curves according to irradiance and temperature
            Here, cutoff must also be True.

        Returns
        -------
        matplotlib axes
        """

        color_wheel = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
        color = color_wheel[self.specific_cells_plotted % len(color_wheel)]
        self.specific_cells_plotted += 1

        if ax is None:
            fig, ax = plt.subplots()

        if string_identifier is not None:
            Vs = self.multilevel_ivdata['string'][string_identifier]['V']
            Is = self.multilevel_ivdata['string'][string_identifier]['I']
            Es = self.multilevel_ivdata['string'][string_identifier]['E']
            Ts = self.multilevel_ivdata['string'][string_identifier]['T']
            label = string_identifier

        if module_identifier is not None:
            if substring_identifier is None:
                Vs = self.multilevel_ivdata['module'][module_identifier]['V']
                Is = self.multilevel_ivdata['module'][module_identifier]['I']
                Es = self.multilevel_ivdata['module'][module_identifier]['E']
                Ts = self.multilevel_ivdata['module'][module_identifier]['T']
                label = module_identifier

            if substring_identifier is not None:
                Vs = self.multilevel_ivdata['module'][module_identifier][substring_identifier]['V']
                Is = self.multilevel_ivdata['module'][module_identifier][substring_identifier]['I']
                Es = self.multilevel_ivdata['module'][module_identifier][substring_identifier]['E']
                Ts = self.multilevel_ivdata['module'][module_identifier][substring_identifier]['T']
                label = f'module: {module_identifier}, substring: {substring_identifier}'

        for idx in range(len(Vs)):
            varr = Vs[idx]
            iarr = Is[idx]
            Eval = Es[idx]
            Tval = Ts[idx]

            if cutoff:
                varr, iarr = iv_cutoff(varr, iarr, 0)

            if correct_gt:
                if cutoff:
                    # get average g&tc for
                    varr, iarr = gt_correction(
                        varr, iarr, Eval, Tval, self.cell_parameters)

                else:
                    # raise issue
                    raise ValueError(
                        "If pass `correct_gt = True` in `visualize_specific_iv`, must also have `cutoff = True`.")

            parr = (varr * iarr).tolist()
            maxidx = parr.index(max(parr))
            imax = iarr.tolist()[maxidx]
            vmax = varr.tolist()[maxidx]

            # pmpp_defected = (varr * iarr).max()

            ax.plot(vmax, imax, 'ko')
            ax.plot(varr, iarr, color=color)
            ax.set_xlabel('V (Volts)')
            ax.set_ylabel('I (Amps)')

        ax.plot([], [], color=color, label=label)
        return ax

    def visualize_multiple_cells_traces(self, list_cell_identifiers, cutoff=True):
        """Visualize multiple cell traces

        Parameters
        ----------
        list_cell_identifiers : list
            list of cell identifiers. call `self.print_info()` for full list.
        cutoff : bool
            If True, only visualize IV curves in positive voltage domain

        Returns
        -------
        matplotlib axes
        """

        colors = sns.color_palette("hls", len(list_cell_identifiers))

        for i, cell_identity in enumerate(list_cell_identifiers):
            # print(cell_identity)
            if i == 0:
                axs = self._vis_cell_trace(
                    cell_identity, colors[i], cutoff=cutoff)
            else:
                axs = self._vis_cell_trace(
                    cell_identity, colors[i], cutoff=cutoff, axs=axs)

        axs.set_xlabel('V (Volts)')
        axs.set_ylabel('I (Amps)')
        axs.legend()
        return axs

    def _vis_cell_trace(self, cell_identifier, color, cutoff=True, axs=None):
        for k in self.condition_dict.keys():
            if self.condition_dict[k][0]['identifier'] == cell_identifier:
                cell_id = k
                found_flag = True
                continue

        if not found_flag:
            self.print_info()
            raise Exception(
                f"Invalid cell_identifier, '{cell_identifier}', given. Use print_info() to see list of conditions available.")

        if axs is None:
            fig, axs = plt.subplots()

        formatted_conds = []
        for cond_dict in self.condition_dict[cell_id]:
            g, tc, rsh_mult, rs_mult, Il_mult, Io_mult, nnsvth_mult = cond_dict['E'], cond_dict['Tc'], cond_dict[
                'Rsh_mult'], cond_dict['Rs_mult'], cond_dict['Io_mult'], cond_dict['Il_mult'], cond_dict['nnsvth_mult']

            v, i = cond_dict['V'], cond_dict['I']

            if cutoff:
                v, i = iv_cutoff(v, i, 0)

            p = (v * i).tolist()
            maxidx = p.index(max(p))
            imax = i.tolist()[maxidx]
            vmax = v.tolist()[maxidx]

            # pmpp_defected = (v * i).max()

            axs.plot(vmax, imax, 'ko', markersize=4)
            axs.plot(v, i, color, label=cell_identifier)

            formatted_conds.append([round(obj, 2) for obj in [
                                   g, tc, rsh_mult, rs_mult, Il_mult, Io_mult, nnsvth_mult]])
        return axs

    def visualize_cell_level_traces(self, cell_identifier, cutoff=True, table=True, axs=None):
        """Visualize IV curves for cell_identifier and tabulate the definitions.

        Parameters
        ----------
        cell_identifier : str
            Cell identifier. Call `self.print_info()` for full list.
        cutoff : bool
            If True, only visualize IV curves in positive voltage domain
        table : bool
            If True, append table to bottom of figure
        axs : maplotlib axes
            Matplotli subplots axes

        Returns
        -------
        matplotlib axes

        """
        found_flag = False
        # find cell_identifier
        for k in self.condition_dict.keys():
            if self.condition_dict[k][0]['identifier'] == cell_identifier:
                cell_id = k
                found_flag = True
                continue

        if not found_flag:
            self.print_info()
            raise Exception(
                f"Invalid cell_identifier, '{cell_identifier}', given. Use print_info() to see list of conditions available.")

        if (not table) or (len(self.condition_dict[cell_id]) > 20):
            plotting_table = False
            if axs is None:
                fig, axs = plt.subplots()
        else:
            plotting_table = True
            fig, axs = plt.subplots(2, 1, figsize=(10, 15))

        formatted_conds = []
        for cond_dict in self.condition_dict[cell_id]:
            g, tc, rsh_mult, rs_mult, Il_mult, Io_mult, nnsvth_mult = cond_dict['E'], cond_dict['Tc'], cond_dict[
                'Rsh_mult'], cond_dict['Rs_mult'], cond_dict['Io_mult'], cond_dict['Il_mult'], cond_dict['nnsvth_mult']

            v, i = cond_dict['V'], cond_dict['I']

            if cutoff:
                v, i = iv_cutoff(v, i, 0)

            p = (v * i).tolist()
            maxidx = p.index(max(p))
            imax = i.tolist()[maxidx]
            vmax = v.tolist()[maxidx]

            # pmpp_defected = (v * i).max()

            if plotting_table:
                axs[0].plot(vmax, imax, 'ko', markersize=4)
                axs[0].plot(v, i)

            else:
                axs.plot(vmax, imax, 'ko', markersize=4)
                axs.plot(v, i)
                axs.set_xlabel('V (Volts)')
                axs.set_ylabel('I (Amps)')

            formatted_conds.append([round(obj, 2) for obj in [
                                   g, tc, rsh_mult, rs_mult, Il_mult, Io_mult, nnsvth_mult]])

        if plotting_table:
            axs[0].set_xlabel('V (Volts)')
            axs[0].set_ylabel('I (Amps)')
            axs[1].axis('off')

            rowlabels = [
                f'condition{i+1}' for i in range(len(formatted_conds))]
            our_colors = sns.color_palette("hls", len(rowlabels))
            collabels = self.acceptible_keys

            axs[1].table(cellText=formatted_conds, rowLoc='right',
                         rowColours=our_colors, rowLabels=rowlabels,
                         colLabels=collabels,
                         colLoc='center', loc='center')
            fig.subplots_adjust(hspace=0.01)
            plt.suptitle(f'Cell conditions: {cell_identifier}')
        else:
            plt.title(f'Cell conditions: {cell_identifier}')

        # axs[0].set_xlim(-2, 3)
        # axs[0].set_ylim(-2, 10)

        return axs

    def _normalize_voltage_domain(self, faulted_ivcurves, pristine_ivcurve, n_pts=100):
        pristine_V = np.array(pristine_ivcurve['string']['V'])
        pristine_I = np.array(pristine_ivcurve['string']['I'])

        vmax = pristine_V.max()
        vnot = pristine_V.min()
        resol = (vmax - vnot) / n_pts

        v_interps = np.arange(vnot, vmax, resol)

        pristine_I_interps = np.interp(v_interps, pristine_V, pristine_I)
        pristine_out_ivs = {'V': v_interps, 'I': pristine_I_interps}

        outivs = []
        for ivcurve in faulted_ivcurves:
            faulted_V = np.array(ivcurve['string']['V'])
            faulted_I = np.array(ivcurve['string']['I'])

            resamp_I = np.interp(v_interps, faulted_V, faulted_I)
            outivcurve = {'V': v_interps,
                          'I': resamp_I}
            outivs.append(outivcurve)

        return outivs, pristine_out_ivs

    def visualize_module_configurations(self, module_identifier, title=None, n_plots_atonce=3):
        """Visualize failure locations on a module.

        Parameters
        ----------
        module_identifier : int
            Module identifier. Call `self.print_info()` for full list.
        title : str
            Optional, add this title to figure.
        n_plots_atonce : int
            Number of plots to render in a single figure.

        Returns
        -------
        matplotlib axes

        TODO: MAKE COLOR either 1) same as condition color from other tables
                                2) colored by intensity of param definition, given a param (e.g. 'E')

        """

        if n_plots_atonce is None:
            # n_samples = min(n_plots_atonce,len(self.modcells[module_identifier]))
            n_samples = len(self.modcells[module_identifier])
            n_iter_samples = [n_samples]

            if isinstance(title, (list, tuple, np.ndarray)):
                if len(title) != n_samples:
                    raise Exception(
                        "Debugging: If inputting array of titles for all figures to have own title, and if n_samples is None, make sure that the title array is the same length as the min(n_plots_atonce, n_modcells_in_identifier).")
        else:
            n_samples = len(self.modcells[module_identifier])
            if (n_samples < n_plots_atonce):
                n_iter_samples = [n_samples]
            else:
                if (n_samples // n_plots_atonce) != (n_samples / n_plots_atonce):
                    n_iter_samples = [n_plots_atonce] * \
                        (n_samples // n_plots_atonce)
                    n_iter_samples += [n_samples % n_plots_atonce]
                else:
                    n_iter_samples = [n_plots_atonce] * \
                        (n_samples // n_plots_atonce)
            if isinstance(title, (list, tuple, np.ndarray)):
                if len(title) != n_samples:
                    raise Exception(
                        f"If inputting array of titles for all figures to have own title, make sure that the title array ({len(title)}) is the same length as the n_samples ({n_samples}).")

        # print(np.arange(0.5,1,(1-0.5)/max(map(max, self.modcells[module_identifier]))))
        # TODO: replace this with colors used in cell IV curve visualization
        allcells_conds = [item for sublist in self.modcells[module_identifier]
                          for item in sublist]
        # print(allcells_conds)
        set_conds = set(allcells_conds)
        # print(set_conds)
        # colors = ['white'] + [str(i) for i in np.arange(0.5, 1, (1 - 0.5) / max(map(max, self.modcells[module_identifier])))]
        clrs = ['white'] + [str(i)
                            for i in np.arange(0.5, 1, (1 - 0.5) / len(set_conds))]
        clrs = clrs[::-1]
        colors = {}
        for cond, clr in zip(set_conds, clrs):
            colors[cond] = clr

        sample_pos = 0
        counter = 0
        for idx, nsample_iter in enumerate(n_iter_samples):
            modcell_samples = self.modcells[module_identifier][sample_pos:sample_pos + nsample_iter]
            # nsample_iter = min(nsample_iter, len(modcell_samples))
            sample_pos += nsample_iter

            fig = plt.figure()
            fig.set_size_inches(3.1 * nsample_iter, 5)
            for m in range(len(modcell_samples)):
                ax = fig.add_subplot(1, len(modcell_samples), m + 1)
                ax.set_xlim(0, 8)
                ax.set_ylim(0, 12)
                num = 0
                for n in range(len(modcell_samples[m])):
                    i = int(n / self.module_parameters['nrows'])
                    j = int(n % self.module_parameters['nrows'])
                    modtype = int(modcell_samples[m][n])
                    rect = matplotlib.patches.Rectangle(
                        (i + 1, j + 1), 1, 1, linewidth=1, edgecolor='red', facecolor=colors[modtype])
                    ax.add_patch(rect)
                    ax.text(i + 1.2, j + 1.2, str(num))
                    num = num + 1

                if (title is not None) and (isinstance(title, list)):
                    ax.set_title(title[counter])
                ax.axis('off')
                counter += 1

            if (title is not None) and (isinstance(title, str)):
                fig.suptitle(title)
        return fig

    def _position_inrow(self, num, n_cells_row):
        # num: cell number
        return num % n_cells_row

    def _simulate_pole_shading(self, width, pos=None):
        # pos: (number in first column, number in last column)

        # n: number of rows
        # m: number of cols
        n = self.module_parameters['ncols']
        m = self.module_parameters['nrows']

        # n_cells_row: number cells in row
        # n_cells_col: number cells in column
        n_cells_row = int(self.module_parameters['N_s'] / n)
        n_cells_col = int(self.module_parameters['N_s'] / m)

        if pos is None:
            bottom_point = random.randint(0, n_cells_row - 1)
            top_point = random.randint(
                (n_cells_col - 1) * n_cells_row, (n_cells_col * n_cells_row) - 1)
        else:
            bottom_point, top_point = pos

        m = (self._position_inrow(top_point, n_cells_row) -
             self._position_inrow(bottom_point, n_cells_row)) / n

        direction = random.randint(0, 1)

        dark = []
        light = []
        for row_id in range(int(n)):
            chosen_cell = int((m * row_id) + bottom_point)
            chosen_cell_modschemed = chosen_cell + n_cells_row * row_id

            permitted_set = set(
                range(row_id * n_cells_row, (row_id + 1) * n_cells_row))
            if direction:
                # Grow downwards
                subdark = list(np.arange(chosen_cell_modschemed -
                                         width + 1, chosen_cell_modschemed + 1))
                sublight = [chosen_cell_modschemed -
                            width, chosen_cell_modschemed + 1]

                dark += list(set(subdark).intersection(permitted_set))
                light += list(set(sublight).intersection(permitted_set))

            else:
                # Grow upwards
                subdark = list(np.arange(chosen_cell_modschemed,
                                         chosen_cell_modschemed + width))
                sublight = [chosen_cell_modschemed -
                            1, chosen_cell_modschemed + width]

                dark += list(set(subdark).intersection(permitted_set))
                light += list(set(sublight).intersection(permitted_set))

        return dark, light

    def _simulate_bird_droppings(self, n_droppings):
        n_cells = self.module_parameters['N_s']
        if n_droppings is None:
            n_cells_cols = int(n_cells / self.module_parameters['ncols'])
            n_droppings_randomly = random.randint(1, n_cells_cols)
            idx = [random.randint(1, n_cells - 1)
                   for i in range(n_droppings_randomly)]
            return idx, n_droppings_randomly
        else:
            idx = [random.randint(1, n_cells - 1) for i in range(n_droppings)]
            return idx, n_droppings

    def _simulate_portrait(self, cols_aff):
        # n_cells_row: number cells in row
        # n_cells_col: number cells in column
        n_cells_row = int(
            self.module_parameters['N_s'] / self.module_parameters['ncols'])
        n_cells_col = int(
            self.module_parameters['N_s'] / self.module_parameters['nrows'])
        return [c + (n_cells_row * i) for c in range(cols_aff) for i in range(n_cells_col)]

    def _simulate_landscape(self, rows_aff):
        # n_cells_row: number cells in row
        n_cells_row = int(
            self.module_parameters['N_s'] / self.module_parameters['ncols'])
        return np.arange(0, rows_aff * n_cells_row, 1)


def create_df(Varr, Iarr, POA, T, mode):
    """Builds a dataframe from the given parameters

    Parameters
    ----------
    Varr
    Iarr
    POA
    T
    mode

    Returns
    -------
    df : DataFrame
    """
    df = pd.DataFrame()
    df['voltage'] = Varr
    df['current'] = Iarr
    df['E'] = POA
    df['T'] = T
    df['mode'] = mode
    return df
