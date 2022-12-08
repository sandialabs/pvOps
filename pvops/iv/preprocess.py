import numpy as np
import pandas as pd
from pvops.iv.physics_utils import gt_correction


def preprocess(input_df, resmpl_resolution, iv_col_dict, resmpl_cutoff=0.03,
               correct_gt=False, normalize_y=True, CECmodule_parameters=None,
               n_mods=None, gt_correct_option=3):
    """IV processing function which supports irradiance & temperature correction

    Parameters
    ----------
    input_df : DataFrame
    resmpl_resolution : float
    iv_col_dict : dict
    resmpl_cutoff : float
    correct_gt : bool
    normalize_y : bool
    CECmodule_parameters : None
    n_mods : int
    gt_correct_option : int

    Returns
    -------
    df : DataFrame
    """

    current_col = iv_col_dict["current"]
    voltage_col = iv_col_dict["voltage"]
    power_col = iv_col_dict["power"]
    failure_mode_col = iv_col_dict["mode"]
    irradiance_col = iv_col_dict["irradiance"]
    temperature_col = iv_col_dict["temperature"]

    # Correct for irradiance and temperature
    if correct_gt:
        Vs, Is = [], []
        for ind, row in input_df.iterrows():
            if CECmodule_parameters is None or n_mods is None:
                raise ValueError(
                    "You must specify CECmodule_parameters and n_mods if you want to correct the IV curves for irradiance and temperature.")
            Vt, It = gt_correction(row[voltage_col], row[current_col], row[irradiance_col], row[temperature_col],
                                   cecparams=CECmodule_parameters, n_units=n_mods, option=gt_correct_option)
            Vs.append(Vt)
            Is.append(It)
    else:
        Is = input_df[current_col].tolist()
        Vs = input_df[voltage_col].tolist()

    v_interps = np.arange(
        resmpl_cutoff, 1, resmpl_resolution)
    v_interps = np.append(v_interps, 1.0)

    procVs = []
    procIs = []
    # Resample IV curve to static voltage domain
    for iii in range(len(Vs)):
        Voc = max(Vs[iii])
        Vnorm = Vs[iii] / Voc
        procVs.append(v_interps)
        interpolated_I = np.interp(v_interps, Vnorm, Is[iii])

        if normalize_y:
            isc_iter = interpolated_I.max()
            procIs.append(interpolated_I / isc_iter)

        else:
            procIs.append(interpolated_I)

    df = pd.DataFrame()
    df[failure_mode_col] = input_df[failure_mode_col]

    procIs = np.array(procIs)
    procVs = np.array(procVs)
    procPs = procIs * procVs

    df[current_col] = list(procIs)
    df[voltage_col] = list(procVs)
    df[power_col] = list(procPs)
    df[irradiance_col] = input_df[irradiance_col].tolist()
    df[temperature_col] = input_df[temperature_col].tolist()

    return df
