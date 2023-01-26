import pvlib
import copy


def get_CEC_params(name, mod_spec):
    '''Query module-level parameters from CEC database and 
    derive cell-level parameters.

    Utilizing methods from pvsystem.retrieve_sam('CECMod')

    Parameters
    ----------
    name : string
        Representing module name in CEC database

    mod_specs : dict
        Provide 'ncols' and 'nsubstrings'

    Returns
    -------
    module_parameters (dict), cell_parameters (dict)
    '''

    moddb = pvlib.pvsystem.retrieve_sam('CECMod')
    module_parameters = moddb[name].to_dict()

    # add reverse bias parameters
    module_parameters['breakdown_factor'] = 1.e-4
    module_parameters['breakdown_voltage'] = -30.  # -5.5
    module_parameters['breakdown_exp'] = 3.28
    module_parameters['ncols'] = mod_spec['ncols']
    module_parameters['nsubstrings'] = mod_spec['nsubstrings']
    module_parameters['ncells_substring'] = module_parameters['N_s'] / \
        mod_spec['nsubstrings']
    module_parameters['nrows'] = module_parameters['N_s'] / \
        module_parameters['ncols']
    # module_parameters['R_sh_ref'] *= rsh_premultiply # What should this value be? Dynamic.
    # TODO: Adjust Io smaller

    # set up cell-level parameters
    cell_parameters = copy.copy(module_parameters)
    cell_parameters['a_ref'] = module_parameters['a_ref'] / \
        module_parameters['N_s']
    cell_parameters['R_sh_ref'] = module_parameters['R_sh_ref'] / \
        module_parameters['N_s']
    cell_parameters['R_s'] = module_parameters['R_s'] / \
        module_parameters['N_s']
    return module_parameters, cell_parameters
