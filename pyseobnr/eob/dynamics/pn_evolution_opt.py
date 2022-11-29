from math import log
from typing import Callable

import numpy as np
import pygsl.errno as errno
import pygsl.odeiv2 as odeiv2
from numba import *
from numba import jit, types
from pyseobnr.auxiliary.interpolate.vector_spline import VectorSpline
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from math import cos, sin

from ..fits.fits_Hamiltonian import dSO as dSO_poly_fit
from ..hamiltonian import Hamiltonian
from .initial_conditions_aligned_precessing import computeIC_augm
from ..utils.math_ops_opt import *
from ..utils.utils_precession_opt import project_spins_augment_dynamics_opt

# Test cythonization of PN equations
from pyseobnr.eob.dynamics.pn_EoMs_opt import prec_eqns_20102022_cython_opt1
from pyseobnr.eob.utils.containers_coeffs_PN import PNCoeffs
from .rhs_precessing import get_rhs_prec

step = odeiv2.pygsl_odeiv2_step
_control = odeiv2.pygsl_odeiv2_control
evolve = odeiv2.pygsl_odeiv2_evolve


def strictly_decreasing(
    L: list,
) -> bool:
    return all(x > y for x, y in zip(L, L[1:]))


class control_y_new(_control):
    def __init__(self, eps_abs, eps_rel):
        a_y = 1
        a_dydt = 1
        _control.__init__(self, eps_abs, eps_rel, a_y, a_dydt, None)


@jit(
    float64(
        float64,
        float64,
        float64,
        float64,
        float64,
        float64,
        float64[:],
        float64[:],
        float64[:],
    ),
    cache=True,
    nopython=True,
)
def domega_T4_new(omega, nu, m_1, m_2, X1, X2, Lh, S1, S2):

    flagtailSSvdot, flagNLOSSvdot, flag4PNvdot = 1.0, 1.0, 1.0

    # print(f"Lh = {Lh}, S1 = {S1}, S2 = {S2}, e1_vec = {e1_vec}")
    # print(f"len(S1) = {len(S1)}, len(S2) = {len(S2)}, len(Lh) = {len(Lh)}, len(e1_vec) = {len(e1_vec)}")
    v = omega ** (1.0 / 3)
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v
    v7 = v6 * v
    v8 = v7 * v
    v9 = v8 * v
    v10 = v9 * v
    v11 = v10 * v
    logv = np.log(v)

    dm = m_1 - m_2
    lNS1 = my_dot(S1, Lh)
    lNS2 = my_dot(S2, Lh)
    lNS1_2 = lNS1 * lNS1
    lNS2_2 = lNS2 * lNS2

    nu2 = nu * nu
    nu3 = nu2 * nu
    pi2 = np.pi * np.pi
    log2 = log(2)

    # Precession frequencies of the spins
    S1sq = my_dot(S1, S1)
    S2sq = my_dot(S2, S2)
    S1S2 = my_dot(S1, S2)

    omega_dot = (
        -96
        * nu
        * v11
        * (
            1
            + (-2.2113095238095237 - (11 * nu) / 4.0) * v2
            + v4
            * (
                (721 * lNS1 * lNS2) / (48.0 * nu)
                + (34103 + 122949 * nu + 59472 * nu2) / 18144.0
                - (247 * S1S2) / (48.0 * nu)
                + (719 * lNS2_2 * (-nu + X1)) / (96.0 * nu2)
                - (233 * S2sq * (-nu + X1)) / (96.0 * nu2)
                + (719 * lNS1_2 * (-nu + X2)) / (96.0 * nu2)
                - (233 * S1sq * (-nu + X2)) / (96.0 * nu2)
            )
            + v5
            * (
                -0.001488095238095238 * ((4159 + 15876 * nu) * np.pi)
                + lNS2
                * (
                    -21.439484126984127
                    + (79 * nu) / 6.0
                    + (35.125 - 809 / (84.0 * nu)) * X1
                )
                + lNS1
                * (
                    -21.439484126984127
                    + (79 * nu) / 6.0
                    + (35.125 - 809 / (84.0 * nu)) * X2
                )
            )
            + v3
            * (
                4 * np.pi
                + lNS2 * (-3.1666666666666665 - (25 * X1) / (4.0 * nu))
                + lNS1 * (-3.1666666666666665 - (25 * X2) / (4.0 * nu))
            )
            + flag4PNvdot
            * v8
            * (
                (13 * lNS1 * lNS2 * (1615505 - 780799 * nu + 17857 * nu2))
                / (24192.0 * nu)
                - ((9355721 + 15851457 * nu + 3413361 * nu2) * S1S2) / (72576.0 * nu)
                + (
                    lNS2
                    * np.pi
                    * (2 * nu * (-93914 + 102909 * nu) + 9 * (-13320 + 50483 * nu) * X1)
                )
                / (2016.0 * nu)
                + (
                    lNS2_2
                    * (
                        nu * (-11888267 + 14283281 * nu - 6086003 * nu2)
                        + (11888267 + 12189903 * nu - 16745701 * nu2) * X1
                    )
                )
                / (48384.0 * nu2)
                + (
                    S2sq
                    * (
                        nu * (8207303 + 19211067 * nu + 2904783 * nu2)
                        + (-8207303 - 21933531 * nu + 5264553 * nu2) * X1
                    )
                )
                / (145152.0 * nu2)
                + (
                    lNS1
                    * np.pi
                    * (2 * nu * (-93914 + 102909 * nu) + 9 * (-13320 + 50483 * nu) * X2)
                )
                / (2016.0 * nu)
                + (
                    lNS1_2
                    * (
                        nu * (-11888267 + 14283281 * nu - 6086003 * nu2)
                        + (11888267 + 12189903 * nu - 16745701 * nu2) * X2
                    )
                )
                / (48384.0 * nu2)
                + (
                    S1sq
                    * (
                        nu * (8207303 + 19211067 * nu + 2904783 * nu2)
                        + (-8207303 - 21933531 * nu + 5264553 * nu2) * X2
                    )
                )
                / (145152.0 * nu2)
            )
            + v7
            * (
                (5 * (-2649 + 143470 * nu + 146392 * nu2) * np.pi) / 12096.0
                + (
                    lNS2
                    * (
                        nu * (-1932041 + 2538207 * nu - 454398 * nu2)
                        + (-1195759 + 4626414 * nu - 1646001 * nu2) * X1
                    )
                )
                / (18144.0 * nu)
                + (
                    lNS1
                    * (
                        nu * (-1932041 + 2538207 * nu - 454398 * nu2)
                        + (-1195759 + 4626414 * nu - 1646001 * nu2) * X2
                    )
                )
                / (18144.0 * nu)
                + flagtailSSvdot
                * (
                    (207 * lNS1 * lNS2 * np.pi) / (4.0 * nu)
                    - (12 * np.pi * S1S2) / nu
                    + (209 * lNS2_2 * np.pi * (-nu + X1)) / (8.0 * nu2)
                    - (6 * np.pi * S2sq * (-nu + X1)) / nu2
                    + (209 * lNS1_2 * np.pi * (-nu + X2)) / (8.0 * nu2)
                    - (6 * np.pi * S1sq * (-nu + X2)) / nu2
                )
            )
            + v6
            * (
                117.72574285227559
                - (1712 * np.euler_gamma) / 105.0
                - (3424 * log2) / 105.0
                - (1712 * logv) / 105.0
                - (56198689 * nu) / 217728.0
                + (541 * nu2) / 896.0
                - (5605 * nu3) / 2592.0
                + (16 * pi2) / 3.0
                + (451 * nu * pi2) / 48.0
                - (lNS2 * np.pi * (74 * nu + 151 * X1)) / (6.0 * nu)
                - (lNS1 * np.pi * (74 * nu + 151 * X2)) / (6.0 * nu)
                + flagNLOSSvdot
                * (
                    lNS1 * lNS2 * (-40.89930555555556 + 14433 / (224.0 * nu))
                    + (22.12847222222222 + 16255 / (672.0 * nu)) * S1S2
                    + (S2sq * (-(nu * (76527 + 42077 * nu)) + (76527 - 8239 * nu) * X1))
                    / (4032.0 * nu2)
                    + (
                        lNS2_2
                        * (nu * (19665 + 177611 * nu) + (-19665 + 135961 * nu) * X1)
                    )
                    / (4032.0 * nu2)
                    + (S1sq * (-(nu * (76527 + 42077 * nu)) + (76527 - 8239 * nu) * X2))
                    / (4032.0 * nu2)
                    + (
                        lNS1_2
                        * (nu * (19665 + 177611 * nu) + (-19665 + 135961 * nu) * X2)
                    )
                    / (4032.0 * nu2)
                )
            )
        )
    ) / 5.0

    return -omega_dot


@jit(
    nopython=True,
    cache=True,
)
def prec_eqns_20102022(t, z, nu, m_1, m_2, X1, X2):
    """Post-Newtonian precession equations, as
    well as the evolution of the orbital frequency

    Args:
        t (float): Time
        z (np.array): Vector of unknowns
        m_1 (float): Mass of primary
        m_2 (float): Mass of secondary
        tp (str): What Taylor approximant to use
        coeff (float): Additional coeff in Taylor approx
        PN_order : 0-12 PN order omegadot equation
        TPL: 0/1 do not include/include the TPL terms

    Raises:
        NotImplementedError: If Taylor approximant is unknown

    Returns:
        np.array: RHS of equations
    """
    # print(z)
    Lh = z[:3]
    S1 = z[3:6]
    S2 = z[6:9]
    omega = z[-1]

    flagNLOSSpn = 1.0
    flagNNLOSSpn = 1.0
    flagNNLOSOpn = 1.0

    Lh_norm = my_norm(Lh)
    Lh /= Lh_norm

    # print(f"Lh = {Lh}, S1 = {S1}, S2 = {S2}, e1_vec = {e1_vec}")
    # print(f"len(S1) = {len(S1)}, len(S2) = {len(S2)}, len(Lh) = {len(Lh)}, len(e1_vec) = {len(e1_vec)}")
    v = omega ** (1.0 / 3)
    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v
    v7 = v6 * v
    v8 = v7 * v
    v9 = v8 * v
    v10 = v9 * v
    v11 = v10 * v
    logv = np.log(v)

    dm = m_1 - m_2

    q = m_1 / m_2
    nu2 = nu * nu
    nu3 = nu2 * nu
    pi2 = np.pi * np.pi
    log2 = log(2)
    # flagNLOSSpn = 1.
    # flagNNLOSOpn = 1.

    # Precession frequencies of the spins
    lNxS1 = my_cross(Lh, S1)
    lNxS2 = my_cross(Lh, S2)
    S1xS2 = my_cross(S1, S2)
    lNS1xS2 = my_dot(Lh, S1xS2)
    lNS1 = my_dot(S1, Lh)
    lNS2 = my_dot(S2, Lh)
    lNS1_2 = lNS1 * lNS1
    lNS2_2 = lNS2 * lNS2

    # chi1_v = S1/(m_1*m_1)
    # chi2_v = S2/(m_2*m_2)

    # chi1_LN = my_dot(chi1_v, Lh)
    # chi2_LN = my_dot(chi2_v, Lh)
    # ap = chi1_LN * X1 + chi2_LN * X2
    # am = chi1_LN * X1 - chi2_LN * X2

    # dSO = dSO_poly_fit(nu, ap, am)

    S1dot_v9 = (
        lNxS1 * (0.75 - (3 * dm) / 4.0 + nu / 2.0) * v5
        + (lNxS1 * ((-3 * lNS2) / 2.0 - (3 * lNS1) / (2.0 * q)) - S1xS2 / 2.0) * v6
        + lNxS1
        * (0.5625 + dm * (-0.5625 + (5 * nu) / 8.0) + (5 * nu) / 4.0 - nu2 / 24.0)
        * v7
        + flagNNLOSOpn
        * lNxS1
        * (
            0.84375
            + (3 * nu) / 16.0
            + dm * (-0.84375 + (39 * nu) / 8.0 - (5 * nu2) / 32.0)
            - (105 * nu2) / 32.0
            - nu3 / 48.0
        )
        * v9
        + flagNLOSSpn
        * v8
        * (
            (nu * S1xS2) / 4.0
            + lNxS1
            * (
                lNS2 * (-0.5 + nu / 12.0)
                + lNS1 * ((-17 * nu) / 12.0 + 9 / (4.0 * q) - (15 * X2) / 4.0)
            )
        )
        + flagNNLOSSpn
        * v10
        * (
            -((0.375 + (49 * nu) / 16.0 + nu2 / 48.0) * S1xS2)
            + lNxS1
            * (
                lNS2 * (-2.25 + (139 * nu) / 48.0 + (103 * nu2) / 144.0)
                + lNS1
                * (
                    (-91 * nu) / 16.0
                    + (121 * nu2) / 144.0
                    + 27 / (16.0 * q)
                    + (-6.0625 + (385 * nu) / 48.0) * X2
                )
            )
        )
    )

    S2dot_v9 = (
        lNxS2 * (0.75 + (3 * dm) / 4.0 + nu / 2.0) * v5
        + (lNxS2 * ((-3 * lNS1) / 2.0 - (3 * lNS2 * q) / 2.0) + S1xS2 / 2.0) * v6
        + lNxS2
        * (0.5625 + dm * (0.5625 - (5 * nu) / 8.0) + (5 * nu) / 4.0 - nu2 / 24.0)
        * v7
        + flagNNLOSOpn
        * lNxS2
        * (
            0.84375
            + (3 * nu) / 16.0
            + dm * (0.84375 - (39 * nu) / 8.0 + (5 * nu2) / 32.0)
            - (105 * nu2) / 32.0
            - nu3 / 48.0
        )
        * v9
        + flagNLOSSpn
        * v8
        * (
            -0.25 * (nu * S1xS2)
            + lNxS2
            * (
                lNS1 * (-0.5 + nu / 12.0)
                + lNS2 * ((-17 * nu) / 12.0 + (9 * q) / 4.0 - (15 * X1) / 4.0)
            )
        )
        + flagNNLOSSpn
        * v10
        * (
            (0.375 + (49 * nu) / 16.0 + nu2 / 48.0) * S1xS2
            + lNxS2
            * (
                lNS1 * (-2.25 + (139 * nu) / 48.0 + (103 * nu2) / 144.0)
                + lNS2
                * (
                    (-91 * nu) / 16.0
                    + (121 * nu2) / 144.0
                    + (27 * q) / 16.0
                    + (-6.0625 + (385 * nu) / 48.0) * X1
                )
            )
        )
    )

    # Remove SS terms from the LNhatdot equation as they introduce perpendicular components to LNhat,
    # which can cause some
    # flagNLOSSpn = 1.
    # flagNNLOSSpn = 1.
    # lNS1xS2 = 0.

    # Expression PN-expanding the numerator of LNhatdot
    LNhatdot_v10 = (
        flagNNLOSOpn
        * (
            (
                lNxS2
                * (
                    81
                    + 27 * nu
                    + 423 * nu2
                    - 9 * dm * (-9 - 21 * nu + 7 * nu2)
                    - 2 * nu3
                )
            )
            / (96.0 * nu)
            + (
                lNxS1
                * (
                    81
                    + 27 * nu
                    + 423 * nu2
                    + 9 * dm * (-9 - 21 * nu + 7 * nu2)
                    - 2 * nu3
                )
            )
            / (96.0 * nu)
        )
        * v10
        + flagNNLOSSpn
        * (
            (dm * Lh * lNS1xS2 * (-27 + 89 * nu)) / (96.0 * nu)
            + lNxS2
            * (
                (lNS1 * (207 + 1413 * nu + 3 * dm * (9 + 71 * nu) - 32 * nu2))
                / (192.0 * nu)
                + (
                    lNS2
                    * (
                        279
                        - 366 * nu
                        + dm * (279 + 192 * nu - 439 * nu2)
                        - 1381 * nu2
                        - 138 * nu3
                    )
                )
                / (96.0 * nu2)
            )
            + lNxS1
            * (
                (lNS2 * (207 + 1413 * nu - 3 * dm * (9 + 71 * nu) - 32 * nu2))
                / (192.0 * nu)
                + (
                    lNS1
                    * (
                        279
                        - 366 * nu
                        - 1381 * nu2
                        + dm * (-279 - 192 * nu + 439 * nu2)
                        - 138 * nu3
                    )
                )
                / (96.0 * nu2)
            )
            - (9 * dm * (1 + 2 * nu) * S1xS2) / (32.0 * nu)
        )
        * v11
        + (
            -0.25 * (lNxS1 * (3 - 3 * dm + 2 * nu)) / nu
            - (lNxS2 * (3 + 3 * dm + 2 * nu)) / (4.0 * nu)
        )
        * v6
        + (
            lNxS2
            * ((3 * lNS2 * (1 + dm - 2 * nu)) / (4.0 * nu2) + (3 * lNS1) / (2.0 * nu))
            + lNxS1
            * ((3 * lNS2) / (2.0 * nu) - (3 * lNS1 * (-1 + dm + 2 * nu)) / (4.0 * nu2))
        )
        * v7
        + (
            (lNxS1 * (9 - 9 * nu - 9 * dm * (1 + nu) + 2 * nu2)) / (8.0 * nu)
            + (lNxS2 * (9 - 9 * nu + 9 * dm * (1 + nu) + 2 * nu2)) / (8.0 * nu)
        )
        * v8
        + flagNLOSSpn
        * (
            (-5 * dm * Lh * lNS1xS2) / (8.0 * nu)
            + lNxS1
            * (
                -0.0625 * (lNS2 * (99 - 15 * dm + 20 * nu)) / nu
                + (lNS1 * (-9 - 3 * dm * (-3 + nu) + 21 * nu + 4 * nu2)) / (2.0 * nu2)
            )
            + lNxS2
            * (
                -0.0625 * (lNS1 * (99 + 15 * dm + 20 * nu)) / nu
                + (lNS2 * (-9 + 3 * dm * (-3 + nu) + 21 * nu + 4 * nu2)) / (2.0 * nu2)
            )
            + (3 * dm * S1xS2) / (8.0 * nu)
        )
        * v9
    )

    # omega_dot = domega_T4_new(omega, nu, m_1, m_2,  X1, X2, Lh, S1,S2)

    # Precession frequencies of the spins
    S1sq = my_dot(S1, S1)
    S2sq = my_dot(S2, S2)
    S1S2 = my_dot(S1, S2)

    flagtailSSvdot, flagNLOSSvdot, flag4PNvdot = 1.0, 1.0, 1.0

    omega_dot = (
        -(
            -96
            * nu
            * v11
            * (
                1
                + (-2.2113095238095237 - (11 * nu) / 4.0) * v2
                + v4
                * (
                    (721 * lNS1 * lNS2) / (48.0 * nu)
                    + (34103 + 122949 * nu + 59472 * nu2) / 18144.0
                    - (247 * S1S2) / (48.0 * nu)
                    + (719 * lNS2_2 * (-nu + X1)) / (96.0 * nu2)
                    - (233 * S2sq * (-nu + X1)) / (96.0 * nu2)
                    + (719 * lNS1_2 * (-nu + X2)) / (96.0 * nu2)
                    - (233 * S1sq * (-nu + X2)) / (96.0 * nu2)
                )
                + v5
                * (
                    -0.001488095238095238 * ((4159 + 15876 * nu) * np.pi)
                    + lNS2
                    * (
                        -21.439484126984127
                        + (79 * nu) / 6.0
                        + (35.125 - 809 / (84.0 * nu)) * X1
                    )
                    + lNS1
                    * (
                        -21.439484126984127
                        + (79 * nu) / 6.0
                        + (35.125 - 809 / (84.0 * nu)) * X2
                    )
                )
                + v3
                * (
                    4 * np.pi
                    + lNS2 * (-3.1666666666666665 - (25 * X1) / (4.0 * nu))
                    + lNS1 * (-3.1666666666666665 - (25 * X2) / (4.0 * nu))
                )
                + flag4PNvdot
                * v8
                * (
                    (13 * lNS1 * lNS2 * (1615505 - 780799 * nu + 17857 * nu2))
                    / (24192.0 * nu)
                    - ((9355721 + 15851457 * nu + 3413361 * nu2) * S1S2)
                    / (72576.0 * nu)
                    + (
                        lNS2
                        * np.pi
                        * (
                            2 * nu * (-93914 + 102909 * nu)
                            + 9 * (-13320 + 50483 * nu) * X1
                        )
                    )
                    / (2016.0 * nu)
                    + (
                        lNS2_2
                        * (
                            nu * (-11888267 + 14283281 * nu - 6086003 * nu2)
                            + (11888267 + 12189903 * nu - 16745701 * nu2) * X1
                        )
                    )
                    / (48384.0 * nu2)
                    + (
                        S2sq
                        * (
                            nu * (8207303 + 19211067 * nu + 2904783 * nu2)
                            + (-8207303 - 21933531 * nu + 5264553 * nu2) * X1
                        )
                    )
                    / (145152.0 * nu2)
                    + (
                        lNS1
                        * np.pi
                        * (
                            2 * nu * (-93914 + 102909 * nu)
                            + 9 * (-13320 + 50483 * nu) * X2
                        )
                    )
                    / (2016.0 * nu)
                    + (
                        lNS1_2
                        * (
                            nu * (-11888267 + 14283281 * nu - 6086003 * nu2)
                            + (11888267 + 12189903 * nu - 16745701 * nu2) * X2
                        )
                    )
                    / (48384.0 * nu2)
                    + (
                        S1sq
                        * (
                            nu * (8207303 + 19211067 * nu + 2904783 * nu2)
                            + (-8207303 - 21933531 * nu + 5264553 * nu2) * X2
                        )
                    )
                    / (145152.0 * nu2)
                )
                + v7
                * (
                    (5 * (-2649 + 143470 * nu + 146392 * nu2) * np.pi) / 12096.0
                    + (
                        lNS2
                        * (
                            nu * (-1932041 + 2538207 * nu - 454398 * nu2)
                            + (-1195759 + 4626414 * nu - 1646001 * nu2) * X1
                        )
                    )
                    / (18144.0 * nu)
                    + (
                        lNS1
                        * (
                            nu * (-1932041 + 2538207 * nu - 454398 * nu2)
                            + (-1195759 + 4626414 * nu - 1646001 * nu2) * X2
                        )
                    )
                    / (18144.0 * nu)
                    + flagtailSSvdot
                    * (
                        (207 * lNS1 * lNS2 * np.pi) / (4.0 * nu)
                        - (12 * np.pi * S1S2) / nu
                        + (209 * lNS2_2 * np.pi * (-nu + X1)) / (8.0 * nu2)
                        - (6 * np.pi * S2sq * (-nu + X1)) / nu2
                        + (209 * lNS1_2 * np.pi * (-nu + X2)) / (8.0 * nu2)
                        - (6 * np.pi * S1sq * (-nu + X2)) / nu2
                    )
                )
                + v6
                * (
                    117.72574285227559
                    - (1712 * np.euler_gamma) / 105.0
                    - (3424 * log2) / 105.0
                    - (1712 * logv) / 105.0
                    - (56198689 * nu) / 217728.0
                    + (541 * nu2) / 896.0
                    - (5605 * nu3) / 2592.0
                    + (16 * pi2) / 3.0
                    + (451 * nu * pi2) / 48.0
                    - (lNS2 * np.pi * (74 * nu + 151 * X1)) / (6.0 * nu)
                    - (lNS1 * np.pi * (74 * nu + 151 * X2)) / (6.0 * nu)
                    + flagNLOSSvdot
                    * (
                        lNS1 * lNS2 * (-40.89930555555556 + 14433 / (224.0 * nu))
                        + (22.12847222222222 + 16255 / (672.0 * nu)) * S1S2
                        + (
                            S2sq
                            * (-(nu * (76527 + 42077 * nu)) + (76527 - 8239 * nu) * X1)
                        )
                        / (4032.0 * nu2)
                        + (
                            lNS2_2
                            * (nu * (19665 + 177611 * nu) + (-19665 + 135961 * nu) * X1)
                        )
                        / (4032.0 * nu2)
                        + (
                            S1sq
                            * (-(nu * (76527 + 42077 * nu)) + (76527 - 8239 * nu) * X2)
                        )
                        / (4032.0 * nu2)
                        + (
                            lNS1_2
                            * (nu * (19665 + 177611 * nu) + (-19665 + 135961 * nu) * X2)
                        )
                        / (4032.0 * nu2)
                    )
                )
            )
        )
        / 5.0
    )

    derivs = [
        LNhatdot_v10[0],  # LNhat
        LNhatdot_v10[1],
        LNhatdot_v10[2],
        S1dot_v9[0],
        S1dot_v9[1],
        S1dot_v9[2],
        S2dot_v9[0],
        S2dot_v9[1],
        S2dot_v9[2],
        omega_dot,
    ]

    return derivs


####################################################################################


def ODE_system_RHS_omega_opt(
    t: float,
    z: np.ndarray,
    H,
    RR,
    m_1: float,
    m_2: float,
    params,
) -> np.ndarray:
    """Return the dynamics equations for aligned-spin systems

    Args:
        t (float): The current time
        z (np.array): The dynamics variables, stored as (q,p)
        H (Hamiltonian): The Hamiltonian object to use
        RR (function): The RR force to use. Must have same signature as Hamiltonian
        chi_1 (float): z-component of the primary spin
        chi_2 (float): z-component of the secondary spin
        m_1 (float): mass of the primary
        m_2 (float): mass of the secondary

    Returns:
        np.array: The dynamics equations, including RR
    """
    # print(z)
    # N = len(z)
    # n = N // 2
    n = 2
    q = z[:2]
    p = z[2:]

    # if q[0]<1.41:
    #    print(f"q = {np.round(q,5)}, p = {np.round(p,4)}, chi1_LN = {np.round(params.p_params.chi_1,4)}, chi2_LN = {np.round(params.p_params.chi_2,4)}, dSO = {np.round(H.calibration_coeffs['dSO'],5)}, omega_circ = {np.round(params.p_params.omega_circ,5)}")

    chi1_LN = params.p_params.chi_1
    chi2_LN = params.p_params.chi_2

    chi1_L = params.p_params.chi1_L
    chi2_L = params.p_params.chi2_L

    # chi1_v = np.array([params.p_params.chi_1x, params.p_params.chi_1y, params.p_params.chi_1z])
    # chi2_v = np.array([params.p_params.chi_2x, params.p_params.chi_2y, params.p_params.chi_2z])

    # chi1_v = [params.p_params.chi_1x, params.p_params.chi_1y, params.p_params.chi_1z]
    # chi2_v = [params.p_params.chi_2x, params.p_params.chi_2y, params.p_params.chi_2z]

    # params.p_params.update_spins(chi1_LN, chi2_LN)
    dynamics = H.dynamics(
        q,
        p,
        params.p_params.chi1_v,
        params.p_params.chi2_v,
        m_1,
        m_2,
        chi1_LN,
        chi2_LN,
        chi1_L,
        chi2_L,
    )
    omega = dynamics[3]
    H_val = dynamics[4]
    csi = dynamics[5]
    params.dynamics.p_circ[1] = p[1]

    # omega_circ = H.omega(q, params.dynamics.p_circ, chi1, chi2, m_1, m_2)
    omega_circ = H.omega(
        q,
        params.dynamics.p_circ,
        params.p_params.chi1_v,
        params.p_params.chi2_v,
        m_1,
        m_2,
        chi1_LN,
        chi2_LN,
        chi1_L,
        chi2_L,
    )

    params.p_params.omega_circ = omega_circ
    # if q[0]<1.41:
    #   dSO_new = H.calibration_coeffs['dSO']
    #   print(f"r = {np.round(q[0],5)}, dSO = {np.round(dSO_new,5)}, chi1_LN = {np.round(chi1_LN,5)}, chi2_LN = {np.round(chi2_LN,5)}, chi1_L = {np.round(chi1_L,5)}, chi2_L = {np.round(chi2_L,5)}, omega_RHS = {np.round(omega,5)}, omega_circ = {np.round(omega_circ,5)}")
    #   print(f"r = {np.round(q[0],5)}, dSO = {np.round(dSO_new,5)}, a1 = {np.round(np.linalg.norm(params.p_params.chi1_v),5)}, a2 = {np.round(np.linalg.norm(params.p_params.chi2_v),5)}, omega_RHS = {np.round(omega,5)}, omega_circ = {np.round(omega_circ,5)}, q = {np.round(q,3)}, p_circ = {np.round(params.dynamics.p_circ,3)}")

    # params.p_params.omega = omega

    RR_f = RR.RR(q, p, omega, omega_circ, H_val, params)

    deriv = [
        csi * dynamics[2],
        dynamics[3],
        -dynamics[0] * csi + RR_f[0],
        -dynamics[1] + RR_f[1],
    ]

    return deriv


def compute_omega_orb(
    t: float,
    z: np.ndarray,
    H,
    RR,
    m_1: float,
    m_2: float,
    params,
) -> np.ndarray:
    """Return the dynamics equations for aligned-spin systems

    Args:
        t (float): The current time
        z (np.array): The dynamics variables, stored as (q,p)
        H (Hamiltonian): The Hamiltonian object to use
        RR (function): The RR force to use. Must have same signature as Hamiltonian
        chi_1 (float): z-component of the primary spin
        chi_2 (float): z-component of the secondary spin
        m_1 (float): mass of the primary
        m_2 (float): mass of the secondary

    Returns:
        np.array: The dynamics equations, including RR
    """

    n = 2
    q = z[:n]
    p = z[n:]
    # chi1_LN = params.p_params.chi_1
    # chi2_LN = params.p_params.chi_2

    # chi1_L = params.p_params.chi1_L
    # chi2_L = params.p_params.chi2_L
    # omega = H.omega(q, p, params.p_params.chi1_v, params.p_params.chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L)

    omega = H.omega(
        q,
        p,
        params.p_params.chi1_v,
        params.p_params.chi2_v,
        m_1,
        m_2,
        params.p_params.chi_1,
        params.p_params.chi_2,
        params.p_params.chi1_L,
        params.p_params.chi2_L,
    )

    return omega


@jit(nopython=True)
def h_max(r):
    return 1


def rhs_wrapper_old(t, z, args):
    return ODE_system_RHS_omega_opt(t, z, *args)


def rhs_wrapper(t, z, args):
    return get_rhs_prec(t, z, *args)




def compute_quasiprecessing_PNdynamics_opt(
    omega0: float,
    omega_start: float,
    m_1: float,
    m_2: float,
    chi_1: np.ndarray,
    chi_2: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-10,
):
    """
    Compute the dynamics starting from omega_start, with spins
    defined at omega0.

    First, PN evolution equations are integrated (including backwards in time)
    to get spin and orbital angular momentum. From that we construct splines
    either in time or orbital frequency for the PN quantities. Given the splines
    we now integrate aligned-spin EOB dynamics where at every step the projections
    of the spins onto orbital angular momentum is computed via the splines.

    Args:
        omega0 (float): Reference frequency
        omega_start (float): Starting frequency
        H (Hamiltonian): Hamiltonian to use
        RR (Callable): RR force to use
        m_1 (float): Mass of primary
        m_2 (float): Mass of secondary
        chi_1 (np.ndarray): Dimensionless spin of the primary
        chi_2 (np.ndarray): Dimensionless spin of the secondary
        ODE_system_RHS (Callable): Which system of equations to use
        rtol (float, optional): Relative tolerance for EOB integration. Defaults to 1e-12.
        atol (float, optional): Absolute tolerance for EOB integration. Defaults to 1e-12.
        backend (str, optional): The backend to use for ODE integration. Defaults to "solve_ivp".
        params (Dict[Any,Any], optional): Dictionary of additional inputs. Defaults to None.
        step_back (float, optional): Amount of time to step back for fine integration. Defaults to 10.
        max_step (float, optional): Max step allowed for fine stepping. Defaults to 0.05.
        min_step (float, optional): Min step allowed. Defaults to 1.0e-9. Currently not used
        tp (str, optional): Whether to use time or orbital frequency splines. Defaults to "time".

    Raises:
        NotImplementedError: If a type of splines is not supported

    Returns:
        tuple: Aligned-spin EOB dynamics, PN time, PN dynamics, PN splines
    """

    # Step 1: Figure out if we need to integrate backwards in time
    # Initial conditions, always defined at omega0
    Lhat0 = [0.0, 0.0, 1.0]  # Convention
    S_1 = chi_1 * m_1**2
    S_2 = chi_2 * m_2**2

    z0 = np.array([*Lhat0, *S_1, *S_2, omega0])

    # FIXME: check that the initial omega0 is not too high

    mt = m_1 + m_2
    nu = m_1 * m_2 / mt / mt
    X1 = m_1 / mt
    X2 = m_2 / mt
    delta = m_1 - m_2
    pn_coeffs = PNCoeffs(nu, delta)

    def spinTaylor_stoppingTest(t, y, *args):
        """/* We are testing if the orbital energy increases with \f$\omega\f$.
        * We should be losing energy to GW flux, so if E increases
        * we stop integration because the dynamics are becoming unphysical.
        * 'test' is the PN expansion of \f$dE/d\omega\f$ without the prefactor,
        * i.e. \f$dE/d\omega = dE/dv * dv/d\omega = - (M^2*eta/6) * test\f$
        * Therefore, the energy is increasing with \f$\omega\f$ iff. test < 0.
        */
        """
        # LNhx, LNhy, LNhz, S1x, S1y, S1z, S2x, S2y, S2z, e1x, e1y, e1z, omega = y
        LNhx, LNhy, LNhz, S1x, S1y, S1z, S2x, S2y, S2z, omega = y

        r = omega ** (-2.0 / 3)
        # if omega > spinTaylor_stoppingTest.omegas[-1]:
        #    spinTaylor_stoppingTest.omegaPeaks += 1

        omegaEnd = 0.0
        omegaStart = omega0

        omegadiff = omega - spinTaylor_stoppingTest.omegas[-1]
        # rdiff = r - spinTaylor_stoppingTest.rs[-1]

        # // Check d^2omega/dt^2 > 0 (true iff current_domega - prev_domega > 0)
        ddomega = omegadiff
        # // ...but we can integrate forward or backward, so beware of the sign!
        if (
            omegaEnd < omegaStart
            and omegaEnd != 0.0
            and spinTaylor_stoppingTest.omegas[-1] != 0.0
        ):
            ddomega *= -1

        spinTaylor_stoppingTest.omegas.append(omega)
        # spinTaylor_stoppingTest.rs.append(r)

        # spinTaylor_stoppingTest.rs.append(r)
        # paramsT4 = args['paramsT4']
        v = np.cbrt(omega)

        # return y[-1] - 0.8

        # print(f"t = {t}, omega = {omega}, y = {y}")
        # print(omegadiff=-)
        # Check d^2omega/dt^2 > 0 (true iff current_domega - prev_domega > 0)
        # ddomega = dvalues[1] - paramsT4.prev_domega

        # ...but we can integrate forward or backward, so beware of the sign!
        # if paramsT4.fEnd < paramsT4.fStart and paramsT4.fEnd !=0. and  paramsT4.prev_domega != 0. :
        #    ddomega *= 1

        # Copy current value of domega to prev. value of domega for next call
        # paramsT4.prev_domega = dvalues[1]
        LAL_REAL4_MANT = 24  # bits in REAL4 mantissa
        LAL_REAL4_EPS = 2 ** (1 - LAL_REAL4_MANT)

        # Test LAL stopping conditions
        # if( fabs(omegaEnd) > LAL_REAL4_EPS && omegaEnd > omegaStart && omega > omegaEnd) # /* freq. above bound */
        # return LALSIMINSPIRAL_ST_TEST_FREQBOUND;
        if (
            abs(omegaEnd) > LAL_REAL4_EPS and omegaEnd < omegaStart and omega < omegaEnd
        ):  # /* freq. below bound */
            yout = 0
        # elif (test < 0.0): # energy test fails!
        #    yout = 0
        elif np.isnan(omega):  # return LALSIMINSPIRAL_ST_TEST_ENERGY;
            # print("omegaPN is nan!")
            yout = 0
        elif v >= 1.0:  #  v/c >= 1!
            # print(f"vPN>1 : v  = {v}")
            yout = 0
        elif ddomega <= 0.0:  #  // d^2omega/dt^2 <= 0!
            # print(f" omegaPN_diff = {ddomega}:  (omegaPN[i] = {omega}) - (omegaPN[i-1] = {spinTaylor_stoppingTest.omegas[-2]})")
            yout = 0
        elif omega > 0.35:
            # print(f"omegaPN = {omega}> 0.35")
            yout = 0
        else:  # /* Step successful, continue integrating */

            yout = 1

        return yout

    spinTaylor_stoppingTest.omegas = [omega0]
    # spinTaylor_stoppingTest.rs = [omega0 ** (-2 / 3.0)]
    spinTaylor_stoppingTest.omegaPeaks = 0

    spinTaylor_stoppingTest.terminal = True
    spinTaylor_stoppingTest.direction = -1

    # print(f"z0 = {z0}")
    # print(rhs_type)
    # rhs_type='TaylorT1'
    # Using numba test # Currently(17/11/2022) faster
    res_forward = solve_ivp(
        prec_eqns_20102022,
        (0, 1e7),
        z0,
        events=spinTaylor_stoppingTest,
        args=(nu, m_1, m_2, X1, X2),
        rtol=1e-10,
        atol=1e-12,
        method="DOP853",  # "RK45"
    )

    # Using cython # Currently(17/11/2022) ~ twice slower than numba
    # res_forward = solve_ivp(
    #    prec_eqns_20102022_cython_opt1,
    #    (0, 1e7),
    #    z0,
    #    events=spinTaylor_stoppingTest,
    #    args=(nu, pn_coeffs,),
    #    rtol=1e-10,
    #    atol=1e-12,
    #    method="DOP853",  # "RK45"
    # )

    if omega_start < omega0:
        # We want to start at a lower frequency thean omega0
        # Thus we first integrate backwards in time
        def term_backwards(t, y, *args):
            return y[-1] - omega_start

        term_backwards.terminal = True

        res_back = solve_ivp(
            prec_eqns_20102022,
            (0, -1e7),
            z0,
            events=term_backwards,
            args=(nu, m_1, m_2, X1, X2),
            rtol=1e-10,
            atol=1e-12,
            method="DOP853",  # "RK45"
        )

        # Combine the forwards and backward pieces
        combined_t = np.concatenate((res_back.t[::-1][:-1], res_forward.t[1:]))
        combined_y = np.vstack((res_back.y.T[::-1][:-1], res_forward.y.T[1:, :]))

        # Set t=0 at the lowest frequency
        combined_t -= combined_t[0]
    else:

        combined_t = res_forward.t[:]
        combined_y = res_forward.y.T[:]

    # Use this to avoid interpolation errors due to repeating the last point

    # print(f"omega_pn_final = {res_forward.y.T[:,-1][-2:]}")
    # print(f"omega_pn_final = {combined_y.T[-1,-2:]}, absdiff ={abs(combined_y[-1,-1]-combined_y[-2,-1])}")
    tdiff_abs = abs(combined_t[-1] - combined_t[-2])
    if tdiff_abs < 1e-10:
        # if abs(combined_y[-1,-1]-combined_y[-2,-1])<1e-6:
        combined_t = combined_t[:-1]
        combined_y = combined_y[:-1]

    omega_diff = combined_y[-1, -1] - combined_y[-2, -1]
    if omega_diff < 0:
        combined_t = combined_t[:-1]
        combined_y = combined_y[:-1]

    return combined_t, combined_y


def build_splines_PN_opt(
    combined_t: np.ndarray,
    combined_y: np.ndarray,
    m_1: float,
    m_2: float,
    omega_start: float,
):

    # Build the splines for PN quantities
    splines = {}
    s1_vec = combined_y[:, 3:6]
    s2_vec = combined_y[:, 6:9]
    chi1_v = s1_vec / m_1**2
    chi2_v = s2_vec / m_2**2
    # e1_v = combined_y[:, 9:12]
    omega = combined_y[:, -1]
    LN_hat = combined_y[:, :3]

    # tmp_LN = combined_y[:, :3]
    LN_hat_norm = np.sqrt(np.einsum("ij,ij->i", LN_hat, LN_hat))
    LN_hat = (LN_hat.T / LN_hat_norm).T
    # combined_y[:,3] = tmp_LN

    mtot = m_1 + m_2
    nu = m_1 * m_2 / mtot / mtot
    X1 = m_1 / mtot
    X2 = m_2 / mtot

    v = omega ** (1.0 / 3.0)

    s1_lN = np.einsum("ij,ij->i", s1_vec, LN_hat)
    s2_lN = np.einsum("ij,ij->i", s2_vec, LN_hat)

    chi1_LN = np.einsum("ij,ij->i", chi1_v, LN_hat)
    chi2_LN = np.einsum("ij,ij->i", chi2_v, LN_hat)

    s1s2 = np.einsum("ij,ij->i", s1_vec, s2_vec)
    s1sq = np.einsum("ij,ij->i", s1_vec, s1_vec)
    s2sq = np.einsum("ij,ij->i", s2_vec, s2_vec)

    q = m_1 / m_2
    # print(len(v),len(dSO_vec))
    L_vec = compute_Lvec_35PN_vec_opt(
        q, nu, X1, X2, v, s1_vec, s2_vec, LN_hat, s1sq, s2sq, s1s2, s1_lN, s2_lN
    )

    L_vec_norm = np.sqrt(np.einsum("ij,ij->i", L_vec, L_vec))
    L_hat = (L_vec.T / L_vec_norm).T

    chi1_L = np.einsum("ij,ij->i", chi1_v, L_hat)
    chi2_L = np.einsum("ij,ij->i", chi2_v, L_hat)

    # Create spline dictionary

    omega = combined_y[:, -1]
    splines["chi1"] = VectorSpline(omega, chi1_v)
    splines["chi2"] = VectorSpline(omega, chi2_v)
    splines["L_N"] = VectorSpline(omega, LN_hat)
    splines["Lvec"] = VectorSpline(omega, L_vec)
    # splines["e1_v"] = VectorSpline(omega, e1_v)

    splines["chi1_L"] = CubicSpline(omega, chi1_L)
    splines["chi2_L"] = CubicSpline(omega, chi2_L)

    splines["chi1_LN"] = CubicSpline(omega, chi1_LN)
    splines["chi2_LN"] = CubicSpline(omega, chi2_LN)
    all_array = np.c_[chi1_LN, chi2_LN, chi1_L, chi2_L, chi1_v, chi2_v]
    splines["everything"] = CubicSpline(omega, all_array)
    # The values at the starting time.
    chi1_spline = splines["chi1"]
    chi2_spline = splines["chi2"]
    LN_spline = splines["L_N"]
    chi1LN_spline = splines["chi1_LN"]
    chi2LN_spline = splines["chi2_LN"]
    chi1L_spline = splines["chi1_L"]
    chi2L_spline = splines["chi2_L"]
    # Note t=0 is *always* the time when EOB integration would start
    # even if we have integrated PN backwards

    chi1_v = chi1_spline(omega_start)
    chi2_v = chi2_spline(omega_start)
    LN_hat = LN_spline(omega_start)
    LN_hat /= my_norm(LN_hat)

    chi_1 = my_dot(chi1_v, LN_hat)
    chi_2 = my_dot(chi2_v, LN_hat)

    return (
        splines,
        chi_1,
        chi_2,
        chi1L_spline,
        chi2L_spline,
        chi1LN_spline,
        chi2LN_spline,
    )


#############################################

#############################################


def build_splines_PN_opt_v1(
    combined_t: np.ndarray,
    combined_y: np.ndarray,
    m_1: float,
    m_2: float,
    a1: float,
    a2: float,
    omega_start: float,
):

    # Build the splines for PN quantities
    splines = {}
    s1_vec = combined_y[:, 3:6]
    s2_vec = combined_y[:, 6:9]
    chi1_v = s1_vec / m_1**2
    chi2_v = s2_vec / m_2**2
    # e1_v = combined_y[:, 9:12]
    omega = combined_y[:, -1]
    LN_hat = combined_y[:, :3]

    chi1_perp = np.sqrt(chi1_v[:, 0] ** 2.0 + chi1_v[:, 1] ** 2.0)
    phi1 = np.arctan2(chi1_v[:, 1], chi1_v[:, 0])
    theta1 = np.arctan2(chi1_perp, chi1_v[:, 2])

    chi2_perp = np.sqrt(chi2_v[:, 0] ** 2.0 + chi2_v[:, 1] ** 2.0)
    phi2 = np.arctan2(chi2_v[:, 1], chi2_v[:, 0])
    theta2 = np.arctan2(chi2_perp, chi2_v[:, 2])

    # tmp_LN = combined_y[:, :3]
    LN_hat_norm = np.sqrt(np.einsum("ij,ij->i", LN_hat, LN_hat))
    LN_hat = (LN_hat.T / LN_hat_norm).T

    LN_perp = np.sqrt(LN_hat[:, 0] ** 2.0 + LN_hat[:, 1] ** 2.0)
    phi_LN = np.arctan2(LN_hat[:, 1], LN_hat[:, 0])
    theta_LN = np.arctan2(LN_perp, LN_hat[:, 2])

    # combined_y[:,3] = tmp_LN

    mtot = m_1 + m_2
    nu = m_1 * m_2 / mtot / mtot
    X1 = m_1 / mtot
    X2 = m_2 / mtot

    v = omega ** (1.0 / 3.0)

    s1_lN = np.einsum("ij,ij->i", s1_vec, LN_hat)
    s2_lN = np.einsum("ij,ij->i", s2_vec, LN_hat)

    chi1_LN = np.einsum("ij,ij->i", chi1_v, LN_hat)
    chi2_LN = np.einsum("ij,ij->i", chi2_v, LN_hat)

    s1s2 = np.einsum("ij,ij->i", s1_vec, s2_vec)
    s1sq = np.einsum("ij,ij->i", s1_vec, s1_vec)
    s2sq = np.einsum("ij,ij->i", s2_vec, s2_vec)

    q = m_1 / m_2
    # print(len(v),len(dSO_vec))
    L_vec = compute_Lvec_35PN_vec_opt(
        q, nu, X1, X2, v, s1_vec, s2_vec, LN_hat, s1sq, s2sq, s1s2, s1_lN, s2_lN
    )

    L_vec_norm = np.sqrt(np.einsum("ij,ij->i", L_vec, L_vec))
    L_hat = (L_vec.T / L_vec_norm).T

    L_perp = np.sqrt(L_hat[:, 0] ** 2.0 + L_hat[:, 1] ** 2.0)
    phi_L = np.arctan2(L_hat[:, 1], L_hat[:, 0])
    theta_L = np.arctan2(L_perp, L_hat[:, 2])

    # Create spline dictionary

    omega = combined_y[:, -1]
    all_array = np.c_[
        phi1, theta1, phi2, theta2, phi_LN, theta_LN, phi_L, theta_L, L_vec_norm
    ]
    splines["everything"] = CubicSpline(omega, all_array)

    # The values at the starting time.
    # Note t=0 is *always* the time when EOB integration would start
    # even if we have integrated PN backwards
    tmp = splines["everything"](omega_start)
    phi1_start = tmp[0]
    theta1_start = tmp[1]
    phi2_start = tmp[2]
    theta2_start = tmp[3]
    phiLN_start = tmp[4]
    thetaLN_start = tmp[5]

    chi_1 = my_projection(a1, phi1_start, theta1_start, 1.0, phiLN_start, thetaLN_start)
    chi_2 = my_projection(a2, phi2_start, theta2_start, 1.0, phiLN_start, thetaLN_start)

    return (
        splines,
        chi_1,
        chi_2,
    )


############################################################# Lvec formula from Mohammed


@jit(
    float64[:](
        float64,
        float64,
        float64,
        float64,
        float64,
        float64[:],
        float64[:],
        float64[:],
        float64,
        float64,
        float64,
        float64,
        float64,
    ),
    cache=True,
    # nopython=True,
)
def compute_Lvec_35PN_opt(q, nu, X1, X2, v, S1, S2, Lh, S1sq, S2sq, S1S2, lNS1, lNS2):

    flagNLOSSpn = 1.0
    flagNNLOSOpn = 1.0
    flagN3LOSOpn = 0.0

    dSO = 0.0
    flagNNLOSSpn = 1.0

    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v
    v7 = v6 * v
    v8 = v7 * v
    logv = log(v)

    nu2 = nu * nu
    nu3 = nu2 * nu
    nu4 = nu3 * nu

    lNS1_2 = lNS1 * lNS1
    lNS2_2 = lNS2 * lNS2

    pi2 = np.pi * np.pi
    log2 = log(2.0)

    # Non-spinning term
    """
    newtonian_factor =  Lh*nu/v
    L0vec_factorized = 1 + ((9 + nu)*v2)/6. + ((81 - 57*nu + nu2)*v4)/24. + ((10935 + 1674*nu2 + 7*nu3 + 9*nu*(-6889 + 246*pi2))*v6)/1296.

    vecL0 = newtonian_factor*L0vec_factorized

    # SO term
    vecLSO_v2  = v2 * ( (S2*((-45 + nu)*nu + 3*(-9 + 10*nu)*X1))/48. + (S1*((-45 + nu)*nu + 3*(-9 + 10*nu)*X2))/48. + (11*Lh*(lNS1*(-45 + nu)*nu + lNS2*(-45 + nu)*nu + 3*lNS2*(-9 + 10*nu)*X1 + 3*lNS1*(-9 + 10*nu)*X2))/144. )
    vecLSO_v4  = v4 * ( flagNNLOSOpn*((S2*(nu*(-243 + 165*nu + nu2) - 3*(27 - 156*nu + 5*nu2)*X1))/96. + (S1*(nu*(-243 + 165*nu + nu2) - 3*(27 - 156*nu + 5*nu2)*X2))/96. + (5*Lh*(lNS1*nu*(-243 + 165*nu + nu2) + lNS2*nu*(-243 + 165*nu + nu2) - 3*lNS2*(27 - 156*nu + 5*nu2)*X1 - 3*lNS1*(27 - 156*nu + 5*nu2)*X2))/96.) )
    vecLSO_v6  = v6 * ( (S2*(-nu - 3*X1))/4. + (S1*(-nu - 3*X2))/4. - (7*Lh*(lNS2*(nu + 3*X1) + lNS1*(nu + 3*X2)))/12. )
    vecLSO = vecLSO_v2 + vecLSO_v4 + vecLSO_v6
    # SS term
    vecLSS_v3  = v3 * ((lNS2/2. + lNS1/(2.*q))*S1 + (lNS1/2. + (lNS2*q)/2.)*S2 + Lh*(2*lNS1*lNS2 + lNS1_2/q + lNS2_2*q - nu2*S1S2 + S2sq*(nu2/2. - (nu*X1)/2.) + S1sq*(nu2/2. - (nu*X2)/2.)))
    vecLSS_v5  = v5 * (flagNLOSSpn*(S2*((5*lNS1)/4. + ((-7*lNS1 + 5*lNS2)*nu)/24. + (11*lNS2*q)/8. - (lNS2*X1)/2.) + S1*(((5*lNS1 - 7*lNS2)*nu)/24. + (11*lNS1)/(8.*q) + (5*lNS2 - 2*lNS1*X2)/4.) +
     Lh*(((121*lNS1_2 + 26*lNS1*lNS2 + 121*lNS2_2)*nu)/72. - (35*lNS1_2)/(8.*q) - (35*lNS2_2*q)/8. + (2*nu3*S1S2)/3. + (11*lNS2_2*X1)/2. + S2sq*(-nu2 - nu3/3. + (nu - (5*nu2)/3.)*X1) + (lNS1*(-7*lNS2 + 33*lNS1*X2))/6. + S1sq*(-0.3333333333333333*nu3 + nu2*(-1 - (5*X2)/3.) + nu*X2))))
    vecLSS = vecLSS_v3 + vecLSS_v5
    vecL = vecL0 + vecLSO + vecLSS
    """

    absVecL0 = nu * (
        1 / v
        + ((9 + nu) * v) / 6.0
        + ((81 - 57 * nu + nu2) * v3) / 24.0
        + ((10935 + 1674 * nu2 + 7 * nu3 + 9 * nu * (-6889 + 246 * pi2)) * v5) / 1296.0
        + v7
        * (
            (
                13778100
                - 77400 * nu3
                - 1100 * nu4
                - 900 * nu2 * (-71207 + 2706 * pi2)
                - 27
                * nu
                * (-395476 + 983040 * np.euler_gamma + 96825 * pi2 + 1966080 * log2)
            )
            / 622080.0
            - (128 * nu * logv) / 3.0
        )
    )

    vecL0 = absVecL0 * Lh

    vecLSO = (
        v2
        * (
            (S2 * (-nu - 3 * X1)) / 4.0
            + (S1 * (-nu - 3 * X2)) / 4.0
            + Lh
            * ((-7 * lNS2 * (nu + 3 * X1)) / 12.0 - (7 * lNS1 * (nu + 3 * X2)) / 12.0)
        )
        + v4
        * (
            (S2 * ((-45 + nu) * nu + 3 * (-9 + 10 * nu) * X1)) / 48.0
            + (S1 * ((-45 + nu) * nu + 3 * (-9 + 10 * nu) * X2)) / 48.0
            + Lh
            * (
                (11 * lNS2 * ((-45 + nu) * nu + 3 * (-9 + 10 * nu) * X1)) / 144.0
                + (11 * lNS1 * ((-45 + nu) * nu + 3 * (-9 + 10 * nu) * X2)) / 144.0
            )
        )
        + flagNNLOSOpn
        * v6
        * (
            (S2 * (nu * (-243 + 165 * nu + nu2) - 3 * (27 - 156 * nu + 5 * nu2) * X1))
            / 96.0
            + (S1 * (nu * (-243 + 165 * nu + nu2) - 3 * (27 - 156 * nu + 5 * nu2) * X2))
            / 96.0
            + Lh
            * (
                (
                    5
                    * lNS2
                    * (
                        nu * (-243 + 165 * nu + nu2)
                        - 3 * (27 - 156 * nu + 5 * nu2) * X1
                    )
                )
                / 96.0
                + (
                    5
                    * lNS1
                    * (
                        nu * (-243 + 165 * nu + nu2)
                        - 3 * (27 - 156 * nu + 5 * nu2) * X2
                    )
                )
                / 96.0
            )
        )
        + flagN3LOSOpn
        * v8
        * (
            (
                S2
                * (
                    nu
                    * (-131463 - 30753 * nu2 + 440 * nu3 + 36 * nu * (2428 + 123 * pi2))
                    - 3
                    * (
                        14499
                        + 41823 * nu2
                        + 424 * nu3
                        - 36 * nu * (-587 - 96 * dSO + 123 * pi2)
                    )
                    * X1
                )
            )
            / 20736.0
            + (
                S1
                * (
                    nu
                    * (-131463 - 30753 * nu2 + 440 * nu3 + 36 * nu * (2428 + 123 * pi2))
                    - 3
                    * (
                        14499
                        + 41823 * nu2
                        + 424 * nu3
                        - 36 * nu * (-587 - 96 * dSO + 123 * pi2)
                    )
                    * X2
                )
            )
            / 20736.0
            + Lh
            * (
                (
                    19
                    * lNS2
                    * (
                        nu
                        * (
                            -131463
                            - 30753 * nu2
                            + 440 * nu3
                            + 36 * nu * (2428 + 123 * pi2)
                        )
                        - 3
                        * (
                            14499
                            + 41823 * nu2
                            + 424 * nu3
                            - 36 * nu * (-587 - 96 * dSO + 123 * pi2)
                        )
                        * X1
                    )
                )
                / 62208.0
                - (
                    19
                    * lNS1
                    * (
                        nu
                        * (
                            131463
                            + 30753 * nu2
                            - 440 * nu3
                            - 36 * nu * (2428 + 123 * pi2)
                        )
                        + 3
                        * (
                            14499
                            + 41823 * nu2
                            + 424 * nu3
                            - 36 * nu * (-587 - 96 * dSO + 123 * pi2)
                        )
                        * X2
                    )
                )
                / 62208.0
            )
        )
    )

    vecLSS = (
        (
            (lNS2 / 2.0 + lNS1 / (2.0 * q)) * S1
            + (lNS1 / 2.0 + (lNS2 * q) / 2.0) * S2
            + Lh
            * (
                2 * lNS1 * lNS2
                + lNS1_2 / q
                + lNS2_2 * q
                - S1S2
                - S1sq / (2.0 * q)
                - (q * S2sq) / 2.0
            )
        )
        * v3
        + flagNLOSSpn
        * v5
        * (
            S2
            * (
                lNS1 * (1.25 - (7 * nu) / 24.0)
                + (lNS2 * (5 * nu + 33 * q - 12 * X1)) / 24.0
            )
            + Lh
            * (
                lNS1 * lNS2 * (-1.1666666666666667 + (13 * nu) / 36.0)
                + (2 * nu * S1S2) / 3.0
                + S2sq * (-0.3333333333333333 * nu + q - (5 * X1) / 3.0)
                + (lNS2_2 * (121 * nu - 315 * q + 396 * X1)) / 72.0
                + S1sq * (-0.3333333333333333 * nu + 1 / q - (5 * X2) / 3.0)
                + lNS1_2 * ((121 * nu) / 72.0 - 35 / (8.0 * q) + (11 * X2) / 2.0)
            )
            + S1
            * (
                lNS2 * (1.25 - (7 * nu) / 24.0)
                + (lNS1 * (33 + 5 * nu * q - 12 * q * X2)) / (24.0 * q)
            )
        )
        + flagNNLOSSpn
        * v7
        * (
            S2
            * (
                (lNS1 * (1080 - 3141 * nu - 446 * nu2)) / 576.0
                + (
                    lNS2
                    * ((1689 - 235 * nu) * nu + 189 * q - 3 * (224 + 339 * nu) * X1)
                )
                / 288.0
            )
            + S1
            * (
                (lNS2 * (1080 - 3141 * nu - 446 * nu2)) / 576.0
                - (
                    lNS1
                    * (
                        -189
                        + nu * (-1689 + 235 * nu) * q
                        + 3 * (224 + 339 * nu) * q * X2
                    )
                )
                / (288.0 * q)
            )
            + Lh
            * (
                (lNS1 * lNS2 * (3240 + 1083 * nu - 722 * nu2)) / 864.0
                - (5 * (18 + 147 * nu + nu2) * S1S2) / 72.0
                + (
                    lNS2_2
                    * ((3123 - 505 * nu) * nu - 2997 * q + (10746 - 8499 * nu) * X1)
                )
                / 864.0
                + (5 * S2sq * (nu * (165 + nu) + 27 * q + (-273 + 59 * nu) * X1))
                / 144.0
                + (5 * S1sq * (27 + nu * (165 + nu) * q + (-273 + 59 * nu) * q * X2))
                / (144.0 * q)
                - (
                    lNS1_2
                    * (
                        2997
                        + nu * (-3123 + 505 * nu) * q
                        + 3 * (-3582 + 2833 * nu) * q * X2
                    )
                )
                / (864.0 * q)
            )
        )
    )

    vecL = vecL0 + vecLSO + vecLSS

    return vecL


def compute_Lvec_35PN_vec_opt(
    q, nu, X1, X2, v, S1, S2, Lh, S1sq, S2sq, S1S2, lNS1, lNS2
):

    flagNLOSSpn = 1.0
    flagNNLOSOpn = 1.0
    flagN3LOSOpn = 0.0

    dSO = 0.0
    flagNNLOSSpn = 1.0

    v2 = v * v
    v3 = v2 * v
    v4 = v3 * v
    v5 = v4 * v
    v6 = v5 * v
    v7 = v6 * v
    v8 = v7 * v
    logv = np.log(v)

    nu2 = nu * nu
    nu3 = nu2 * nu
    nu4 = nu3 * nu

    lNS1_2 = lNS1 * lNS1
    lNS2_2 = lNS2 * lNS2

    pi2 = np.pi * np.pi
    log2 = log(2.0)

    LhT = Lh.T
    S1T = S1.T
    S2T = S2.T

    # Non-spinning term

    absVecL0 = nu * (
        1 / v
        + ((9 + nu) * v) / 6.0
        + ((81 - 57 * nu + nu2) * v3) / 24.0
        + ((10935 + 1674 * nu2 + 7 * nu3 + 9 * nu * (-6889 + 246 * pi2)) * v5) / 1296.0
        + v7
        * (
            (
                13778100
                - 77400 * nu3
                - 1100 * nu4
                - 900 * nu2 * (-71207 + 2706 * pi2)
                - 27
                * nu
                * (-395476 + 983040 * np.euler_gamma + 96825 * pi2 + 1966080 * log2)
            )
            / 622080.0
            - (128 * nu * logv) / 3.0
        )
    )

    vecL0 = absVecL0 * LhT

    vecLSO = (
        v2
        * (
            (S2T * (-nu - 3 * X1)) / 4.0
            + (S1T * (-nu - 3 * X2)) / 4.0
            + LhT
            * ((-7 * lNS2 * (nu + 3 * X1)) / 12.0 - (7 * lNS1 * (nu + 3 * X2)) / 12.0)
        )
        + v4
        * (
            (S2T * ((-45 + nu) * nu + 3 * (-9 + 10 * nu) * X1)) / 48.0
            + (S1T * ((-45 + nu) * nu + 3 * (-9 + 10 * nu) * X2)) / 48.0
            + LhT
            * (
                (11 * lNS2 * ((-45 + nu) * nu + 3 * (-9 + 10 * nu) * X1)) / 144.0
                + (11 * lNS1 * ((-45 + nu) * nu + 3 * (-9 + 10 * nu) * X2)) / 144.0
            )
        )
        + flagNNLOSOpn
        * v6
        * (
            (S2T * (nu * (-243 + 165 * nu + nu2) - 3 * (27 - 156 * nu + 5 * nu2) * X1))
            / 96.0
            + (
                S1T
                * (nu * (-243 + 165 * nu + nu2) - 3 * (27 - 156 * nu + 5 * nu2) * X2)
            )
            / 96.0
            + LhT
            * (
                (
                    5
                    * lNS2
                    * (
                        nu * (-243 + 165 * nu + nu2)
                        - 3 * (27 - 156 * nu + 5 * nu2) * X1
                    )
                )
                / 96.0
                + (
                    5
                    * lNS1
                    * (
                        nu * (-243 + 165 * nu + nu2)
                        - 3 * (27 - 156 * nu + 5 * nu2) * X2
                    )
                )
                / 96.0
            )
        )
        + flagN3LOSOpn
        * v8
        * (
            (
                S2T
                * (
                    nu
                    * (-131463 - 30753 * nu2 + 440 * nu3 + 36 * nu * (2428 + 123 * pi2))
                    - 3
                    * (
                        14499
                        + 41823 * nu2
                        + 424 * nu3
                        - 36 * nu * (-587 - 96 * dSO + 123 * pi2)
                    )
                    * X1
                )
            )
            / 20736.0
            + (
                S1T
                * (
                    nu
                    * (-131463 - 30753 * nu2 + 440 * nu3 + 36 * nu * (2428 + 123 * pi2))
                    - 3
                    * (
                        14499
                        + 41823 * nu2
                        + 424 * nu3
                        - 36 * nu * (-587 - 96 * dSO + 123 * pi2)
                    )
                    * X2
                )
            )
            / 20736.0
            + LhT
            * (
                (
                    19
                    * lNS2
                    * (
                        nu
                        * (
                            -131463
                            - 30753 * nu2
                            + 440 * nu3
                            + 36 * nu * (2428 + 123 * pi2)
                        )
                        - 3
                        * (
                            14499
                            + 41823 * nu2
                            + 424 * nu3
                            - 36 * nu * (-587 - 96 * dSO + 123 * pi2)
                        )
                        * X1
                    )
                )
                / 62208.0
                - (
                    19
                    * lNS1
                    * (
                        nu
                        * (
                            131463
                            + 30753 * nu2
                            - 440 * nu3
                            - 36 * nu * (2428 + 123 * pi2)
                        )
                        + 3
                        * (
                            14499
                            + 41823 * nu2
                            + 424 * nu3
                            - 36 * nu * (-587 - 96 * dSO + 123 * pi2)
                        )
                        * X2
                    )
                )
                / 62208.0
            )
        )
    )

    vecLSS = (
        (
            (lNS2 / 2.0 + lNS1 / (2.0 * q)) * S1T
            + (lNS1 / 2.0 + (lNS2 * q) / 2.0) * S2T
            + LhT
            * (
                2 * lNS1 * lNS2
                + lNS1_2 / q
                + lNS2_2 * q
                - S1S2
                - S1sq / (2.0 * q)
                - (q * S2sq) / 2.0
            )
        )
        * v3
        + flagNLOSSpn
        * v5
        * (
            S2T
            * (
                lNS1 * (1.25 - (7 * nu) / 24.0)
                + (lNS2 * (5 * nu + 33 * q - 12 * X1)) / 24.0
            )
            + LhT
            * (
                lNS1 * lNS2 * (-1.1666666666666667 + (13 * nu) / 36.0)
                + (2 * nu * S1S2) / 3.0
                + S2sq * (-0.3333333333333333 * nu + q - (5 * X1) / 3.0)
                + (lNS2_2 * (121 * nu - 315 * q + 396 * X1)) / 72.0
                + S1sq * (-0.3333333333333333 * nu + 1 / q - (5 * X2) / 3.0)
                + lNS1_2 * ((121 * nu) / 72.0 - 35 / (8.0 * q) + (11 * X2) / 2.0)
            )
            + S1T
            * (
                lNS2 * (1.25 - (7 * nu) / 24.0)
                + (lNS1 * (33 + 5 * nu * q - 12 * q * X2)) / (24.0 * q)
            )
        )
        + flagNNLOSSpn
        * v7
        * (
            S2T
            * (
                (lNS1 * (1080 - 3141 * nu - 446 * nu2)) / 576.0
                + (
                    lNS2
                    * ((1689 - 235 * nu) * nu + 189 * q - 3 * (224 + 339 * nu) * X1)
                )
                / 288.0
            )
            + S1T
            * (
                (lNS2 * (1080 - 3141 * nu - 446 * nu2)) / 576.0
                - (
                    lNS1
                    * (
                        -189
                        + nu * (-1689 + 235 * nu) * q
                        + 3 * (224 + 339 * nu) * q * X2
                    )
                )
                / (288.0 * q)
            )
            + LhT
            * (
                (lNS1 * lNS2 * (3240 + 1083 * nu - 722 * nu2)) / 864.0
                - (5 * (18 + 147 * nu + nu2) * S1S2) / 72.0
                + (
                    lNS2_2
                    * ((3123 - 505 * nu) * nu - 2997 * q + (10746 - 8499 * nu) * X1)
                )
                / 864.0
                + (5 * S2sq * (nu * (165 + nu) + 27 * q + (-273 + 59 * nu) * X1))
                / 144.0
                + (5 * S1sq * (27 + nu * (165 + nu) * q + (-273 + 59 * nu) * q * X2))
                / (144.0 * q)
                - (
                    lNS1_2
                    * (
                        2997
                        + nu * (-3123 + 505 * nu) * q
                        + 3 * (-3582 + 2833 * nu) * q * X2
                    )
                )
                / (864.0 * q)
            )
        )
    )

    vecL = vecL0 + vecLSO + vecLSS

    return vecL.T
