import numpy as np
import scipy
import math
from sklearn.linear_model import LinearRegression
import copy


def calculate_IVparams(v, c):
    """Calculate parameters of IV curve.

    This needs to be reworked: extrapolate parameters from linear region instead of
    hardcoded regions.

    Parameters
    ----------
    x : numpy array
        X-axis data
    y : numpy array
        Y-axis data
    npts : int
        Optional, number of points to resample curve
    deg : int
        Optional, polyfit degree

    Returns
    -------
    Dictionary of IV curve parameters
    """
    isc_lim = 0.1
    voc_lim = 0.01

    # maximum power point
    pmax = np.max((v * c))
    mpp_idx = np.argmax((v * c))
    vpmax = v[mpp_idx]
    ipmax = c[mpp_idx]

    # for snippet_idx in range(len(v[::5])):
    # isc and rsh
    if isinstance(isc_lim, float):
        isc_size = int(len(c) * isc_lim)
    else:
        isc_size = isc_lim
    isc_lm = LinearRegression().fit(
        v[:isc_size].reshape(-1, 1), c[:isc_size].reshape(-1, 1))
    isc = isc_lm.predict(np.asarray([0]).reshape(-1, 1))[0][0]
    rsh = 1 / (isc_lm.coef_[0][0] * -1)

    # voc and rs
    if isinstance(voc_lim, float):
        voc_size = int(len(v) * voc_lim)
    else:
        voc_size = voc_lim

    voc_lm = LinearRegression().fit(c[::-1][:voc_size].reshape(-1, 1),
                                    v[::-1][:voc_size].reshape(-1, 1))
    voc = voc_lm.predict(np.asarray([0]).reshape(-1, 1))[0][0]
    rs = voc_lm.coef_[0][0] * -1

    # fill factor
    ff = (ipmax * vpmax) / (isc * voc)

    return {
        'pmp': pmax,
        'vmp': vpmax,
        'imp': ipmax,
        'voc': voc,
        'isc': isc,
        'rs': rs,
        'rsh': rsh,
        'ff': ff,
    }


def smooth_curve(x, y, npts=50, deg=12):
    """Smooth curve using a polyfit

    Parameters
    ----------
    x : numpy array
        X-axis data
    y : numpy array
        Y-axis data
    npts : int
        Optional, number of points to resample curve
    deg : int
        Optional, polyfit degree

    Returns
    -------
    smoothed x array
    smoothed y array
    """
    xx = np.linspace(1, np.max(x), npts)
    yhat = np.poly1d(np.polyfit(x, y, deg))
    yh = yhat(xx)
    return xx, yh


def iv_cutoff(Varr, Iarr, val):
    """Cut IV curve greater than voltage `val` (usually 0)

    Parameters
    ----------
    V: numpy array
        Voltage array
    I: numpy array
        Current array
    val: numeric
        Filter threshold

    Returns
    -------
    V_cutoff, I_cutoff
    """
    msk = Varr > val
    return Varr[msk], Iarr[msk]


def intersection(x1, y1, x2, y2):
    """Compute intersection of curves, y1=f(x1) and y2=f(x2).
    Adapted from https://stackoverflow.com/a/5462917

    Parameters
    ----------
    x1: numpy array
        X-axis data for curve 1
    y1: numpy array
        Y-axis data for curve 1
    x2: numpy array
        X-axis data for curve 2
    y2: numpy array
        Y-axis data for curve 2

    Returns
    -------
    intersection coordinates
    """
    x1 = copy.copy(np.asarray(x1))
    x2 = copy.copy(np.asarray(x2))
    y1 = copy.copy(np.asarray(y1))
    y2 = copy.copy(np.asarray(y2))

    def _upsample_curve(Varr, Iarr, n_pts=1000):
        vmax = Varr.max()
        vnot = Varr.min()
        resol = (vmax - vnot) / n_pts
        v_interps = np.arange(vnot, vmax, resol)
        return v_interps, np.interp(v_interps, Varr, Iarr)

    x1, y1 = _upsample_curve(x1, y1)
    x2, y2 = _upsample_curve(x2, y2)

    def _rect_inter_inner(x1, x2):
        n1 = x1.shape[0] - 1
        n2 = x2.shape[0] - 1
        X1 = np.c_[x1[:-1], x1[1:]]
        X2 = np.c_[x2[:-1], x2[1:]]
        S1 = np.tile(X1.min(axis=1), (n2, 1)).T
        S2 = np.tile(X2.max(axis=1), (n1, 1))
        S3 = np.tile(X1.max(axis=1), (n2, 1)).T
        S4 = np.tile(X2.min(axis=1), (n1, 1))
        return S1, S2, S3, S4

    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T

    if not len(xy0[:, 1]):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(13, 8))
        plt.plot(x1, y1,
                 'bo', markersize=2, label='1')
        plt.plot(x2, y2, 'ro',
                 markersize=2, label='2')
        plt.legend()
        plt.xlabel('V (Volts)')
        plt.ylabel('I (Amps)')
        # plt.ylim(0, max(max(y2),max(y1)) + 2.)
        plt.ylim(-4, 20)
        plt.xlim(-13.5, max(max(x2), max(x1)) + 2.)
        plt.show()
        print("x1 = ", list(x1))
        print("x2 = ", list(x2))
        print("y1 = ", list(y1))
        print("y2 = ", list(y2))
    return xy0[:, 0], xy0[:, 1]


def T_to_tcell(POA, T, WS, T_type, a=-3.56, b=-0.0750, delTcnd=3):
    ''' Ambient temperature to cell temperature according to NREL 
    weather-correction. See :cite:t:`osti_1078057`.

    Parameters
    ----------
    Tamb: numerical,
        Ambient temperature, in Celcius
    WS: numerical,
        Wind speed at height of 10 meters, in m/s
    a, b, delTcnd: numerical,
        Page 12 in :cite:p:`osti_1078057`
    T_type: string,
        Describe input temperature, either 'ambient' or 'module'

    Returns
    -------
    numerical
        Cell temperature, in Celcius
    '''
    Gstc = 1000

    if T_type == 'ambient':
        Tm = POA * np.exp(a + b * WS) + T
        Tcell = Tm + (POA / Gstc) * delTcnd
    elif T_type == 'module':
        Tcell = T + (POA / Gstc) * delTcnd
    return Tcell


def _aggregate_vectors(current_1, current_2):
    if current_2 is not None:
        return np.flipud(np.sort(np.unique(np.append(current_1, current_2))))
    else:
        return current_1


def bypass(voltage, v_bypass):
    ''' Limits voltage to greater than -v_bypass.

    Parameters
    ----------
    voltage : numeric
        Voltage for IV curve [V]
    v_bypass : float or None, default None
        Forward (positive) turn-on voltage of bypass diode, e.g., 0.5V [V]

    Returns
    -------
    voltage : numeric
        Voltage clipped to greater than -v-bpass
    '''
    return voltage.clip(min=-v_bypass)


def add_series(voltage_1, current_1, voltage_2=None, current_2=None, v_bypass=None):
    ''' Adds two IV curves in series.

    Parameters
    ----------
    voltage_1 : numeric
        Voltage for first IV curve [V]
    current_1 : numeric
        Current for first IV curve [A]
    voltage_2 : numeric or None, default None
        Voltage for second IV curve [V]
    current_1 : numeric or None, default None
        Voltage for second IV curve [A]
    v_bypass : float or None, default None
        Forward (positive) turn-on voltage of bypass diode, e.g., 0.5V [V]

    Returns
    -------
    voltage : numeric
        Voltage for combined IV curve [V]
    current : numeric
        Current for combined IV curve [V]

    Note
    ----
    Current for the combined IV curve is the sorted union of the current of the
    two input IV curves. At current values in the other IV curve, voltage is
    determined by linear interpolation. Voltage at current values outside an
    IV curve's range is determined by linear extrapolation.

    If `voltage_2` and `current_2` are None, returns `(voltage_1, current_1)`
    to facilitate starting a loop over IV curves.
    '''
    if (voltage_2 is None) and (current_2 is None):
        all_v, all_i = voltage_1, current_1
    else:
        all_i = _aggregate_vectors(current_1, current_2)
        all_v = np.zeros_like(all_i)
        f_interp1 = scipy.interpolate.interp1d(np.flipud(current_1), np.flipud(voltage_1),
                                               kind='linear', fill_value='extrapolate')
        all_v += f_interp1(all_i)
        f_interp2 = scipy.interpolate.interp1d(np.flipud(current_2), np.flipud(voltage_2),
                                               kind='linear', fill_value='extrapolate')
        all_v += f_interp2(all_i)
    if v_bypass:
        all_v = bypass(all_v, v_bypass)
    return all_v, all_i


def voltage_pts(npts, v_oc, v_rbd):
    '''Provide voltage points for an IV curve.

    Points range from v_brd to v_oc, with denser spacing at both limits.
    v=0 is included as the midpoint.

    Based on method PVConstants.npts from pvmismatch

    Parameters
    ----------
    npts : integer
        Number of points in voltage array.
    v_oc : float
        Open circuit voltage [V]
    v_rbd : float
        Reverse bias diode voltage (negative value expected) [V]

    Returns
    -------
    array [V]
    '''

    npts_pos = npts // 2
    if npts % 2:
        npts_pos += 1
    npts_neg = npts - npts_pos
    # point spacing from 0 to 1, denser approaching 1
    # decrease point spacing as voltage approaches Voc by using logspace
    pts_pos = (11. - np.logspace(np.log10(11.), 0., npts_pos)) / 10.
    pts_pos[0] = 0
    pts_pos *= v_oc
    pts_neg = (11. - np.logspace(np.log10(11.), 0., npts_neg)) / 10.
    pts_neg = np.flipud(pts_neg)
    pts_neg[0] -= 0.1 * (pts_neg[0] - pts_neg[1])
    pts_neg *= v_rbd
    pts = np.concatenate((pts_neg, pts_pos))
    return pts


def gt_correction(v, i, gact, tact, cecparams, n_units=1, option=3):
    """IV Trace Correction using irradiance and temperature.
    Three correction options are provided, two of which are from an IEC standard.

    Parameters
    ----------
    v : numpy array
        Voltage array
    i : numpy array
        Current array
    gact : float
        Irradiance upon measurement of IV trace
    tact : float
        Temperature (C or K) upon measuremnt of IV trace
    cecparams : dict
        CEC database parameters, as extracted by `pvops.iv.utils.get_CEC_params`.
    n_units : int
        Number of units (cells or modules) in string, default 1
    option : int
        Correction method choice. See method for specifics.

    Returns
    -------
    vref
        Corrected voltage array
    iref
        Corrected current array
    """
    beta = cecparams['beta_oc']  # Voc temperature coefficient of Voc
    alpha = cecparams['alpha_sc']  # Isc temperature coefficient of Isc

    gref = 1000  # Reference Irradiance
    tref = 50  # Reference temperature

    beta *= n_units

    if option in [1, 2]:
        params = calculate_IVparams(v, i)
        isc = params['isc']
        voc = params['voc']
        rs = params['rs']

        # curve correction factor, k, must be derived
        k1 = 0
        k2 = 0

    if option == 1:
        # IEC60891 Procedure 1
        iref = i + isc * ((gref / gact) - 1) + alpha * (tref - tact)
        vref = v - rs * (iref - i) - k1 * iref * \
            (tref - tact) + beta * (tref - tact)

    elif option == 2:
        # IEC60891 Procedure 2
        iref = i * (1 + alpha * (tref - tact)) * (gref / gact)
        vref = v + voc * (beta * (tref - tact) + alpha * math.log(gref / gact)
                          ) - rs * (iref - i) - k2 * iref * (tref - tact)

    elif option == 3:
        vref = (v * (math.log10(gref) / math.log10(gact)) -
                (beta * (tact - tref)))
        iref = (i * (gref / gact)) - (alpha * (tact - tref))

    return vref, iref
