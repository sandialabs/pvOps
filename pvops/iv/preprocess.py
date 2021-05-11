import numpy as np
import pandas as pd
from physics_utils import gt_correction


def preprocess(input_df, resmpl_resolution, resmpl_cutoff=0.03, correct_gt=True, normalize=True, CECmodule_parameters=None, n_mods=None):
    """IV processing function which supports irradiance & temperature correction
    """

    # Correct for irradiance and temperature
    if correct_gt:
        Vs, Is = [], []
        for ind, row in input_df.iterrows():
            if CECmodule_parameters is None or n_mods is None:
                raise ValueError(
                    "You must specify CECmodule_parameters and n_mods if you want to correct the IV curves for irradiance and temperature.")
            Vt, It = gt_correction(row['voltage'], row['current'], row['E'], row['T'],
                                   cecparams=CECmodule_parameters, n_units=n_mods)
            Vs.append(Vt)
            Is.append(It)
    else:
        Is = input_df['current'].tolist()
        Vs = input_df['voltage'].tolist()

    v_interps = np.arange(
        resmpl_cutoff, 1 + resmpl_resolution, resmpl_resolution)

    procVs = []
    procIs = []
    # Resample IV curve to static voltage domain
    for iii in range(len(Vs)):
        Voc = max(Vs[iii])
        Vnorm = Vs[iii] / Voc
        procVs.append(v_interps)

        if normalize:
            interpolated_I = np.interp(v_interps, Vnorm, Is[iii])
            isc_iter = interpolated_I.max()
            procIs.append(interpolated_I / isc_iter)

        else:
            interpolated_I = np.interp(v_interps, Vs[iii], Is[iii])
            procIs.append(interpolated_I)

    df = pd.DataFrame()
    df['mode'] = input_df['mode']

    procIs = np.array(procIs)
    procVs = np.array(procVs)
    procPs = procIs * procVs

    df['current'] = list(procIs)
    df['voltage'] = list(procVs)
    df['power'] = list(procPs)
    df['E'] = input_df['E'].tolist()
    df['T'] = input_df['T'].tolist()

    return df
