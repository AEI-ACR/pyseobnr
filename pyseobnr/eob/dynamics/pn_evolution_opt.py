"""
Contains the PN precession equations and their integration, as well as some auxiliary functions
"""

from math import log
from typing import Callable

import numpy as np

from numba import *
from numba import jit, types
from pyseobnr.auxiliary.interpolate.vector_spline import VectorSpline
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from ..hamiltonian import Hamiltonian
from .initial_conditions_aligned_precessing import computeIC_augm
from ..utils.math_ops_opt import *

# Test cythonization of PN equations
from .rhs_precessing import get_rhs_prec


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
        X1 (float): m_1/(m1+m_2)
        X2 (float): m_2/(m1+m_2)

    Returns:
        np.array: RHS of equations
    """
    # print(z)
    Lh = z[:3]
    S1 = z[3:6]
    S2 = z[6:9]
    omega = z[-1]

    Lh_norm = my_norm(Lh)
    Lh /= Lh_norm

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
    logv = log(v)

    dm = m_1 - m_2

    q = m_1 / m_2
    nu2 = nu * nu
    nu3 = nu2 * nu
    pi2 = np.pi * np.pi
    log2 = log(2)

    # Precession frequencies of the spins
    lNxS1 = my_cross(Lh, S1)
    lNxS2 = my_cross(Lh, S2)
    S1xS2 = my_cross(S1, S2)
    lNS1xS2 = my_dot(Lh, S1xS2)
    lNS1 = my_dot(S1, Lh)
    lNS2 = my_dot(S2, Lh)
    lNS1_2 = lNS1 * lNS1
    lNS2_2 = lNS2 * lNS2

    S1dot_v9 = (
        lNxS1 * (0.75 - (3 * dm) / 4.0 + nu / 2.0) * v5
        + (lNxS1 * ((-3 * lNS2) / 2.0 - (3 * lNS1) / (2.0 * q)) - S1xS2 / 2.0) * v6
        + lNxS1
        * (0.5625 + dm * (-0.5625 + (5 * nu) / 8.0) + (5 * nu) / 4.0 - nu2 / 24.0)
        * v7
        + lNxS1
        * (
            0.84375
            + (3 * nu) / 16.0
            + dm * (-0.84375 + (39 * nu) / 8.0 - (5 * nu2) / 32.0)
            - (105 * nu2) / 32.0
            - nu3 / 48.0
        )
        * v9
        + v8
        * (
            (nu * S1xS2) / 4.0
            + lNxS1
            * (
                lNS2 * (-0.5 + nu / 12.0)
                + lNS1 * ((-17 * nu) / 12.0 + 9 / (4.0 * q) - (15 * X2) / 4.0)
            )
        )
        + v10
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
        + lNxS2
        * (
            0.84375
            + (3 * nu) / 16.0
            + dm * (0.84375 - (39 * nu) / 8.0 + (5 * nu2) / 32.0)
            - (105 * nu2) / 32.0
            - nu3 / 48.0
        )
        * v9
        + v8
        * (
            -0.25 * (nu * S1xS2)
            + lNxS2
            * (
                lNS1 * (-0.5 + nu / 12.0)
                + lNS2 * ((-17 * nu) / 12.0 + (9 * q) / 4.0 - (15 * X1) / 4.0)
            )
        )
        + v10
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

    # Expression PN-expanding the numerator of LNhatdot
    LNhatdot_v10 = (
        (
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
        + (
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
        + (
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
        LNhatdot_v10[0],
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

def compute_omega_orb(
    t: float,
    z: np.ndarray,
    H: Hamiltonian,
    RR: Callable,
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
        m_1 (float): mass of the primary
        m_2 (float): mass of the secondary
        params (EOBParams): Container of additional inputs

    Returns:
        np.array: The dynamics equations, including RR
    """

    n = 2
    q = z[:n]
    p = z[n:]

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


def rhs_wrapper(t, z, args):
    """
    Wrapper to call the cython version of the function which evaluates the
    right hand sides for the non-precessing EOB evolution

    Args:
        t (float): The current time
        z (np.array): The dynamics variables, stored as (q,p)

    Returns:
        tuple: Solution of the rhs (drdt,dphidt,dprdt,dpphidt)
    """

    return get_rhs_prec(t, z, *args)



def compute_quasiprecessing_PNdynamics_opt(
    omega_ref: float,
    omega_start: float,
    m_1: float,
    m_2: float,
    chi_1: np.ndarray,
    chi_2: np.ndarray,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    backend: str="DOP853",
):
    """
    Compute the dynamics starting from omega_start, with spins
    defined at omega_ref.

    First, PN evolution equations are integrated (including backwards in time)
    to get spin and orbital angular momentum. From that we construct splines
    either in time or orbital frequency for the PN quantities. Given the splines
    we now integrate aligned-spin EOB dynamics where at every step the projections
    of the spins onto orbital angular momentum is computed via the splines.

    Args:
        omega_ref (float): Reference frequency
        omega_start (float): Starting frequency
        m_1 (float): Mass of primary
        m_2 (float): Mass of secondary
        chi_1 (np.ndarray): Dimensionless spin of the primary
        chi_2 (np.ndarray): Dimensionless spin of the secondary
        rtol (float, optional): Relative tolerance for EOB integration. Defaults to 1e-12.
        atol (float, optional): Absolute tolerance for EOB integration. Defaults to 1e-12.
        backend (str, optional): The backend to use for ODE integration. Defaults to "DOP853".

    Returns:
        tuple: Aligned-spin EOB dynamics, PN time, PN dynamics, PN splines
    """

    # Step 1,1: Initial conditions, always defined at omega_ref
    Lhat0 = [0.0, 0.0, 1.0]
    S_1 = chi_1 * m_1**2
    S_2 = chi_2 * m_2**2

    z0 = np.array([*Lhat0, *S_1, *S_2, omega_ref])

    # FIXME: check that the initial omega_ref is not too high

    mt = m_1 + m_2
    nu = m_1 * m_2 / mt / mt
    X1 = m_1 / mt
    X2 = m_2 / mt
    delta = m_1 - m_2

    def spinTaylor_stoppingTest(t, y, *args):
        """
            Function to evaluate stopping conditions of the spin-precessing PN evolution

            Args:
                t (float): Time
                y (np.ndarray): Solution of the evolution (LNhx, LNhy, LNhz, S1x, S1y, S1z, S2x, S2y, S2z, omega)

            Returns:
                (int) If 0 stop the evolution, otherwise continue
        """

        # Read the solution
        LNhx, LNhy, LNhz, S1x, S1y, S1z, S2x, S2y, S2z, omega = y

        r = omega ** (-2.0 / 3)

        omegaEnd = 0.0
        omegaStart = omega_ref

        omegadiff = omega - spinTaylor_stoppingTest.omegas[-1]

        # Check d^2omega/dt^2 > 0 (true iff current_domega - prev_domega > 0)
        ddomega = omegadiff

        spinTaylor_stoppingTest.omegas.append(omega)

        v = (omega)**(1./3.)

        if np.isnan(omega):
            yout = 0

        elif v >= 1.0:  #  v/c >= 1!
            yout = 0

        elif ddomega <= 0.0 or abs(ddomega) < 1e-9:  #  // d^2omega/dt^2 <= 0!
            yout = 0

        # Empirical bound on the PN integration (typically it ends earlier)
        elif omega > 0.35:
            yout = 0

        else:
            yout = 1

        return yout

    spinTaylor_stoppingTest.omegas = [omega_ref]
    spinTaylor_stoppingTest.omegaPeaks = 0

    spinTaylor_stoppingTest.terminal = True
    spinTaylor_stoppingTest.direction = -1

    res_forward = solve_ivp(
        prec_eqns_20102022,
        (0, 2e9),
        z0,
        events=spinTaylor_stoppingTest,
        args=(nu, m_1, m_2, X1, X2),
        rtol=rtol,
        atol=atol,
        method=backend,
    )


    if omega_start < omega_ref:
        # We want to start at a lower frequency than omega_ref
        # Thus we first integrate backwards in time
        def term_backwards(t, y, *args):
            return y[-1] - omega_start

        term_backwards.terminal = True

        res_back = solve_ivp(
            prec_eqns_20102022,
            (0, -2e9),
            z0,
            events=term_backwards,
            args=(nu, m_1, m_2, X1, X2),
            rtol=rtol,
            atol=atol,
            method=backend,
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
    tdiff_abs = abs(combined_t[-1] - combined_t[-2])
    if tdiff_abs < 1e-10:
        combined_t = combined_t[:-1]
        combined_y = combined_y[:-1]

    omega_diff = combined_y[-1, -1] - combined_y[-2, -1]
    if omega_diff < 0:
        combined_t = combined_t[:-1]
        combined_y = combined_y[:-1]

    return combined_t, combined_y

def build_splines_PN(
    combined_t: np.ndarray,
    combined_y: np.ndarray,
    m_1: float,
    m_2: float,
    omega_start: float,
):
    """
    Compute splines of the spin-precessing PN dynamics

    Args:
        combined_t (np.ndarray): Time array
        combined_y (np.ndarray): Solution of the spin-precessing PN evolution (LN_hat,S1,S2) vectors
        m_1 (float): Mass of primary
        m_2 (float): Mass of secondary
        omega_start (float): Starting orbital frequency

    Returns:
        (dict): Dictionary containing the splines in orbital frequency of the vector components of
                the spins, LN and L as well as the spin projections onto LN and L
    """

    # Build the splines for PN quantities
    splines = {}
    s1_vec = combined_y[:, 3:6]
    s2_vec = combined_y[:, 6:9]
    chi1_v = s1_vec / m_1**2
    chi2_v = s2_vec / m_2**2

    omega = combined_y[:, -1]
    LN_hat = combined_y[:, :3]

    # Renormalize to account for numerical error
    LN_hat_norm = np.sqrt(np.einsum("ij,ij->i", LN_hat, LN_hat))
    LN_hat = (LN_hat.T / LN_hat_norm).T

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

    # Compute the orbital angular momentum from LN_hat, S1 and S2 up to 4PN
    L_vec = compute_Lvec_35PN_vec_opt(
        q, nu, X1, X2, v, s1_vec, s2_vec, LN_hat, s1sq, s2sq, s1s2, s1_lN, s2_lN
    )

    # Compute L_hat unit vector
    L_vec_norm = np.sqrt(np.einsum("ij,ij->i", L_vec, L_vec))
    L_hat = (L_vec.T / L_vec_norm).T

    # Compute the projections of the dimensionless spins onto L_hat
    chi1_L = np.einsum("ij,ij->i", chi1_v, L_hat)
    chi2_L = np.einsum("ij,ij->i", chi2_v, L_hat)

    # Create spline dictionary
    all_array = np.c_[chi1_LN, chi2_LN, chi1_L, chi2_L, chi1_v, chi2_v, LN_hat, L_vec]
    splines["everything"] = CubicSpline(omega, all_array)

    return splines


def compute_Lvec_35PN_vec_opt(
                              q:float,
                              nu:float,
                              X1:float,
                              X2:float,
                              v: np.ndarray,
                              S1: np.ndarray,
                              S2: np.ndarray,
                              Lh: np.ndarray,
                              S1sq: np.ndarray,
                              S2sq: np.ndarray,
                              S1S2: np.ndarray,
                              lNS1: np.ndarray,
                              lNS2: np.ndarray
):
    """
    Compute orbital angular momentum vector, L, up to 4PN using the Newtonian orbital angular momentum
    unit vector and the spin vectors

    Args:
        q (float): mass ratio > 1
        nu (float): symmetric mass ratio m1/m2> 1
        X1 (float): Mass of primary divided total mass m1/(m1+m2)
        X2 (float): Mass of secondary divided total mass m2/(m1+m2)
        v (float): Post-Newtonian velocity parameter, v= (orbital frequency)^(1/3)
        X2 (float): Mass of secondary
        S1 (np.ndarray): Dimensionful spin vector of the primary
        S2 (np.ndarray): Dimensionful spin vector of the secondary
        Lh (np.ndarray): Newtonian orbital angular momentum unit vector
        S1sq (np.ndarray): Dot product of the spin vector of the primary
        S2sq (np.ndarray): Dot product of the spin vector of the secondary
        S1S2 (np.ndarray): Dot product of the spin vectors of the primary and the secondary
        lNS1 (np.ndarray): Dot product of the Newtonian orbital angular momentum unit vector and the spin vectors of the primary
        lNS2 (np.ndarray): Dot product of the Newtonian orbital angular momentum unit vector and the spin vectors of the secondary

    Returns:
        (dict): Dictionary containing the splines in orbital frequency of the vector components of
                the spins, LN and L as well as the spin projections onto LN and L
    """

    # Compute necessary powers of v
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

    # SO term up to NNLO
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
        + v6
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
    )

    # SS term up to NNLO
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
        + v5
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
        + v7
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
