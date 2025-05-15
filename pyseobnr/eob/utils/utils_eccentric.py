"""
Additional utility functions to manipulate various aspects of the eccentric
dynamics.
"""

from __future__ import annotations

import logging

import numpy as np
from pygsl_lite import spline
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar

from ..utils.containers import EOBParams

logger = logging.getLogger(__name__)


def interpolate_dynamics_ecc(
    dyn_fine: np.ndarray,
    dt: float = 0.1,
    peak_omega: float | None = None,
    step_back: float = 250.0,
    step_back_total: bool = False,
) -> np.ndarray:
    """
    Interpolate the dynamics to a finer grid in the case of the eccentric model.

    The interpolation is performed:

    * If ``peak_omega`` is defined: between ``peak_omega - step_back`` and ``peak_omega``
      if the former is within the time range of the provided array
    * On the full array if ``peak_omega`` is not defined

    Args:
        dyn_fine: Dynamics array
        dt: Time step to which to interpolate
        peak_omega: Position of the peak (stopping condition for the dynamics)
        step_back: Step back relative to the end of the dynamics
        step_back_total: If ``True``, the returned interpolated array will be
            the interpolated section only. If ``False``, the non-interpolated and
            interpolated sections will be joined at ``peak_omega-step_back``.

    Returns:
        Interpolated dynamics array

    .. seealso::

        :py:func:`~.utils.interpolate_dynamics`
    """

    if peak_omega:
        start = max(peak_omega - step_back, dyn_fine[0, 0])
        t_new = np.arange(start, peak_omega, dt)

    else:
        t_new = np.arange(dyn_fine[0, 0], dyn_fine[-1, 0], dt)

    intrp = CubicSpline(dyn_fine[:, 0], dyn_fine[:, 1:])
    res = intrp(t_new)

    if step_back_total or t_new[0] == dyn_fine[0, 0]:
        return np.c_[t_new, res]

    else:
        last_index_fine = np.flatnonzero(dyn_fine[:, 0] < start)[-1]
        assert (
            dyn_fine[: last_index_fine + 1][-1, 0] < t_new[0]
        ), "Problem with the interpolation."
        return np.concatenate((dyn_fine[: last_index_fine + 1], np.c_[t_new, res]))


def compute_ref_values(
    *,
    dynamics_fine: np.ndarray,
    r_final_qc: float,
) -> tuple[float, float]:
    """
    Compute the reference values of time and separation at which the
    eccentric and background QC dynamics are aligned.

    Args:
        dynamics_fine: Dynamical variables
            (t, r, phi, pr, pphi, eccentricity, relativistic anomaly,
            x-parameter, Hamiltonian, omega)
            corresponding to the fine dynamics
        r_final_qc: Final value of the separation of the background
            QC dynamics

    Returns:
        tuple: (t_ref, r_ref). Reference values used for alignment.
    """

    if r_final_qc < dynamics_fine[-1, 1]:
        t_ref = dynamics_fine[-1, 0]
        r_ref = dynamics_fine[-1, 1]

    else:
        r_ref = r_final_qc
        # Find the time at which r_eccentric = r_ref
        # First, search in the second half of the fine dynamics
        sp = 0.001
        n = int(len(dynamics_fine[:, 0]) / 2)
        intrp_r = spline.cspline(n)
        N = int((dynamics_fine[-1, 0] - dynamics_fine[n, 0]) / sp)
        if len(dynamics_fine[:, 0]) % 2 == 1:
            n = n + 1
        zoom = np.linspace(dynamics_fine[n, 0], dynamics_fine[-1, 0], N)
        intrp_r.init(dynamics_fine[n:, 0], dynamics_fine[n:, 1])
        r_zoomed_in = intrp_r.eval_e_vector(zoom)
        idx = (np.abs(r_zoomed_in - r_ref)).argmin()
        t_ref = zoom[idx]
        # If the root is not found, then search in the first half
        if abs(r_zoomed_in[idx] - r_ref) > 1e-3:
            intrp_r = spline.cspline(n)
            zoom = np.linspace(dynamics_fine[0, 0], dynamics_fine[n - 1, 0], N)
            intrp_r.init(dynamics_fine[:n, 0], dynamics_fine[:n, 1])
            r_zoomed_in = intrp_r.eval_e_vector(zoom)
            idx = (np.abs(r_zoomed_in - r_ref)).argmin()
            t_ref = zoom[idx]

    return t_ref, r_ref


def compute_attachment_time_qc(
    *,
    r_ref: float,
    dynamics_qc: np.ndarray,
    dynamics_fine_qc: np.ndarray,
    params: EOBParams,
) -> tuple[np.ndarray, float, float]:
    """
    Computation of the merger-ringdown attachment time for the QC
    evolution. Additionally, computation of the QC time array aligned at
    a reference time.

    Args:
        r_ref (float): Reference separation at which the QC and eccentric
            dynamics are aligned
        dynamics_qc (double[:,:]): Dynamical variables (t, r, phi, pr, pphi, omega)
            of the background QC dynamics
        dynamics_fine_qc (double[:,:]): Dynamical variables (t, r, phi, pr, pphi, omega)
            of the background QC fine dynamics
        params (EOBParams): EOB parameters of the system

    Returns:
        tuple: (t_qc_aligned, delta_t_attach, delta_t_ISCO).
        Time array aligned at a reference time, and the relative values of
        t_attach and t_ISCO relative to the reference time
    """

    # Calculate t_ISCO for the QC dynamics
    r_ISCO = params.ecc_params.r_ISCO
    t_fine_qc = dynamics_fine_qc[:, 0]
    r_fine_qc = dynamics_fine_qc[:, 1]

    if r_ISCO < r_fine_qc[-1]:
        # In some corners of parameter space r_ISCO can be *after*
        # the end of the dynamics. In those cases just use the last
        # point of the dynamics as the reference point
        t_ISCO_qc = t_fine_qc[-1]
        logger.debug("Kerr ISCO after the last r in the dynamics.")

    else:
        # Find a time corresponding to r_ISCO for the QC dynamics
        sp = 0.001
        N = int((t_fine_qc[-1] - t_fine_qc[0]) / sp)
        zoom = np.linspace(t_fine_qc[0], t_fine_qc[-1], N)
        n = len(t_fine_qc)
        intrp_r = spline.cspline(n)
        intrp_r.init(t_fine_qc, r_fine_qc)
        r_zoomed_in = intrp_r.eval_e_vector(zoom)
        idx = (np.abs(r_zoomed_in - r_ISCO)).argmin()
        t_ISCO_qc = zoom[idx]

    # Define the attachment with respect to t_ISCO
    params.ecc_params.t_ISCO_qc = t_ISCO_qc
    t_attach_qc = t_ISCO_qc - params.ecc_params.NR_deltaT
    params.ecc_params.t_attach_qc_predicted = t_attach_qc

    # If the fit for NR_deltaT is too negative and overshoots the end of the
    # dynamics, attach the MR at the last point
    params.ecc_params.attachment_check_qc = 0.0
    if t_attach_qc > t_fine_qc[-1]:
        params.ecc_params.attachment_check_qc = 1.0
        t_attach_qc = t_fine_qc[-1]
        logger.debug(
            "For the background QC dynamics, the NR_deltaT is too negative. "
            "Attaching the MR at the last point of the dynamics, careful!"
        )
    params.ecc_params.t_attach_qc = t_attach_qc

    # Compute the reference time at which the QC dynamics is aligned with
    # the eccentric dynamics
    if dynamics_fine_qc[-1, 1] == r_ref:
        t_ref = dynamics_fine_qc[-1, 0]
    else:
        sp = 0.001
        N = int((dynamics_fine_qc[-1, 0] - dynamics_fine_qc[0, 0]) / sp)
        zoom = np.linspace(dynamics_fine_qc[0, 0], dynamics_fine_qc[-1, 0], N)
        n = len(dynamics_fine_qc[:, 0])
        intrp_r = spline.cspline(n)
        intrp_r.init(dynamics_fine_qc[:, 0], dynamics_fine_qc[:, 1])
        r_zoomed_in = intrp_r.eval_e_vector(zoom)
        idx = (np.abs(r_zoomed_in - r_ref)).argmin()
        t_ref = zoom[idx]

    t_qc_aligned = dynamics_qc[:, 0] - t_ref
    delta_t_attach = t_attach_qc - t_ref
    delta_t_ISCO = t_ISCO_qc - t_ref

    return t_qc_aligned, delta_t_attach, delta_t_ISCO


def interpolate_background_qc_dynamics(
    *,
    t_ecc_low_aligned: np.ndarray,
    t_ecc_fine_aligned: np.ndarray,
    t_qc_aligned: np.ndarray,
    dynamics_low: np.ndarray,
    dynamics_fine: np.ndarray,
    dynamics: np.ndarray,
    dyn_qc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate the background QC dynamics into the eccentric grid.

    Args:
        t_ecc_low_aligned: Time array corresponding to the low
            eccentric dynamics, aligned at the eccentric attachment
            time
        t_ecc_fine_aligned: Time array corresponding to the fine
            eccentric dynamics, aligned at the eccentric attachment time
        t_qc_aligned: Time array corresponding to the full
            eccentric dynamics, aligned at the eccentric attachment time
        dynamics_low: Dynamical variables
            (r, phi, pr, pphi, eccentricity, relativistic anomaly,
            x-parameter, Hamiltonian, omega)
            corresponding to the coarse (low) dynamics
        dynamics_fine: Dynamical variables
            (r, phi, pr, pphi, eccentricity, relativistic anomaly,
            x-parameter, Hamiltonian, omega)
            corresponding to the fine dynamics
        dynamics: Dynamical variables
            (r, phi, pr, pphi, eccentricity, relativistic anomaly,
            x-parameter, Hamiltonian, omega)
            corresponding to the full dynamics
        dyn_qc: Dynamical variables
            (r, phi, pr, pphi, omega)
            corresponding to the full dynamics of the QC evolution

    Returns:
        tuple: (dynamics_low, dynamics_fine, dynamics, dynamics_low_qc, dynamics_fine_qc)

        Low, fine and low + fine arrays for the eccentric dynamics, and also the low
        and fine arrays for the QC dynamics.
    """

    # After the alignment, if the aligned QC dynamics ends before the
    # eccentric dynamics, then cut the eccentric dynamics to avoid
    # interpolations of the QC dynamics in the next steps
    if t_qc_aligned[-1] < t_ecc_fine_aligned[-1]:
        idx_qc_ecc = np.argmin(np.abs(t_ecc_fine_aligned - t_qc_aligned[-1]))
        if t_qc_aligned[-1] < t_ecc_fine_aligned[idx_qc_ecc]:
            idx_qc_ecc -= 1
        t_ecc_fine_aligned = t_ecc_fine_aligned[: idx_qc_ecc + 1]
        dynamics_fine = dynamics_fine[: idx_qc_ecc + 1, :]
        dynamics = np.vstack((dynamics_low, dynamics_fine))

    # Remove not used variables from the dyn_qc array.
    # This is because only the values of (r, pr, omega) are employed
    # for the computation of the NQCs
    dyn_qc = np.delete(dyn_qc, [2, 4, 5, 7], 1)

    # Interpolate the QC dynamics (r, pr, omega)
    dyn_qc_interp = CubicSpline(t_qc_aligned, dyn_qc[:, 1:])

    # Evaluate the QC dynamics in the time-shifted eccentric dynamics
    dynamics_low_qc = dyn_qc_interp(t_ecc_low_aligned)
    dynamics_fine_qc = dyn_qc_interp(t_ecc_fine_aligned)

    # Construct the QC dynamics in the original eccentric dynamics time
    dynamics_low_qc = np.c_[dynamics_low[:, 0], dynamics_low_qc]
    dynamics_fine_qc = np.c_[dynamics_fine[:, 0], dynamics_fine_qc]

    return (
        dynamics_low,
        dynamics_fine,
        dynamics,
        dynamics_low_qc,
        dynamics_fine_qc,
    )


def root_r(t, r_stop, ezx_interp, params):
    """
    Wrapper for the root solving procedure employed in the function
    'compute_starting_values'.

    Args:
        t (float): Time
        r_stop (float): Target separation
        ezx_interp (scipy.CubicSpline): Interpolating function for the
            eccentricity, relativistic anomaly, and dimensionless
            orbit-averaged orbital frequency x = (M omega_avg)^{2/3}
        params (EOBParams): EOB parameters of the system

    Returns:
        float: Value of r - r_stop
    """

    flags_ecc_diss = params.ecc_params.flags_ecc.copy()
    flags_ecc_diss["flagPA"] = 1

    e, z, x = ezx_interp(t)

    r = r_omega_avg_e_z(
        x**1.5,
        e,
        z,
        params.p_params.nu,
        params.p_params.delta,
        params.p_params.chi_A,
        params.p_params.chi_S,
        flags_ecc_diss,
    )

    return r - r_stop


def compute_starting_values(
    r_stop: float,
    dynamics: np.ndarray,
    threshold: bool,
    params: EOBParams,
) -> tuple[float, float, float, float]:
    """
    Compute the exact values of starting time, eccentricity,
    relativistic anomaly, and dimensionless starting orbit-averaged
    orbital frequency that correspond to r = r_stop.

    Args:
        r_stop: Value of the separation at which to compute the
            starting values
        dynamics: Dynamical variables
            (t, eccentricity, rel_anomaly, omega_avg^{2/3})
        threshold: If False, then perform a root solving procedure
            to find the time at which r = r_stop
        params: EOB parameters of the system

    Returns:
        tuple: (t_start, eccentricity, rel_anomaly, omega_avg).
            Starting values at r = r_stop.
    """

    if threshold:
        t_start, eccentricity, rel_anomaly, omega_avg = (
            dynamics[0, 0],
            dynamics[0, 1],
            float(dynamics[0, 2]) % (2 * np.pi),
            dynamics[0, 3] ** 1.5,
        )

    else:
        ezx_interp = CubicSpline(dynamics[:, 0], dynamics[:, 1:4])
        sol = root_scalar(
            root_r,
            bracket=[dynamics[0, 0], 0.0],
            args=(
                r_stop,
                ezx_interp,
                params,
            ),
            xtol=1e-12,
            rtol=1e-10,
        )
        if not sol.converged:
            error_message = (
                "Internal function call failed: Input domain error. "
                "The computation of starting values of eccentricity, "
                "rel_anomaly, and omega_avg at r = {r_target} failed. "
                "Please review the physical sense of the input parameters."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        t_start = sol.root
        e_interp, z_interp, x_interp = ezx_interp(t_start)
        eccentricity = float(e_interp)
        rel_anomaly = float(z_interp) % (2 * np.pi)
        omega_avg = float(x_interp**1.5)

    return t_start, eccentricity, rel_anomaly, omega_avg


def dot_phi_omega_avg_e_z(
    omega_avg,
    e,
    z,
    nu,
    delta,
    chi_A,
    chi_S,
    flags_ecc: dict | None = None,
):
    """
    Instantaneous orbital angular velocity 'dot_phi' as a function of the
    orbit-averaged orbital angular frequency 'omega_avg', eccentricity 'e',
    and relativistic anomaly 'z', with post-adiabatic (PA) contributions.

    The PA contributions are set to zero by default.
    The name of the corresponding expression in the supplementary material
    "docs/mathematica/dynamical_variables.m" is "phidotomegaAvgez".
    """

    if flags_ecc is None:
        flags_ecc = {}

    flagPN1 = flags_ecc.get("flagPN1", 1)
    flagPN32 = flags_ecc.get("flagPN32", 1)
    flagPN2 = flags_ecc.get("flagPN2", 1)
    flagPN52 = flags_ecc.get("flagPN52", 1)
    flagPN3 = flags_ecc.get("flagPN3", 1)
    flagPA = flags_ecc.get("flagPA", 0)

    dot_phi = (
        -e
        * flagPN1
        * omega_avg ** (5 / 3)
        * (3 * e + 2 * np.cos(z))
        * (e * np.cos(z) + 1) ** 2
        / (1 - e**2) ** (5 / 2)
        + e
        * flagPN32
        * (e + np.cos(z))
        * (2 * chi_A * delta + chi_S * (2 - nu))
        * (e * omega_avg * np.cos(z) + omega_avg) ** 2
        / (1 - e**2) ** 3
        - 1
        / 12
        * flagPN2
        * omega_avg ** (7 / 3)
        * (e * np.cos(z) + 1) ** 2
        * (
            12 * e**4 * (nu - 6)
            + 8 * e**3 * (nu - 15) * np.cos(z)
            + e**2
            * (
                -6 * chi_A**2 * (4 * nu - 1) * (np.cos(2 * z) + 2)
                + 12 * chi_A * chi_S * delta * (np.cos(2 * z) + 2)
                + 6 * chi_S**2 * (np.cos(2 * z) + 2)
                - 42 * nu
                - 51
            )
            + e
            * (
                24 * chi_A**2 * (1 - 4 * nu) * np.cos(z)
                + 48 * chi_A * chi_S * delta * np.cos(z)
                + 24 * chi_S**2 * np.cos(z)
                - 8 * (nu - 6) * np.cos(z)
            )
            - 36 * nu
            + (1 - e**2) ** (3 / 2) * (36 * nu - 90)
            + 90
        )
        / (1 - e**2) ** (7 / 2)
        + (1 / 384)
        * flagPN3
        * omega_avg**3
        * (e * np.cos(z) + 1) ** 2
        * (
            -384 * chi_A**2 * (6 * nu**2 - 46 * nu + 11)
            + 768 * chi_A * chi_S * delta * (9 * nu - 11)
            - 384 * chi_S**2 * (4 * nu**2 - 16 * nu + 11)
            + 288 * e**6 * (7 * nu - 12)
            + 2880 * e**5 * (nu - 4) * np.cos(z)
            + e**4
            * (
                64
                * chi_A**2
                * (4 * nu - 1)
                * (4 * nu + (2 * nu - 21) * np.cos(2 * z) - 84)
                - 128
                * chi_A
                * chi_S
                * delta
                * (28 * nu + (2 * nu - 21) * np.cos(2 * z) - 84)
                + 64
                * chi_S**2
                * (12 * nu**2 - 52 * nu + (21 - 2 * nu) * np.cos(2 * z) + 84)
                + 208 * nu**2
                - 7936 * nu
                - 7392
            )
            + e**3
            * (
                128 * chi_A**2 * (16 * nu**2 - 352 * nu + 87) * np.cos(z)
                + 256 * chi_A * chi_S * delta * (87 - 22 * nu) * np.cos(z)
                + 128 * chi_S**2 * (9 * nu**2 - 40 * nu + 87) * np.cos(z)
                + 64 * (6 * nu * np.cos(2 * z) - 106 * nu + 99) * np.cos(z)
            )
            + e**2
            * (
                -64
                * chi_A**2
                * (70 * nu**2 - 37 * nu + (8 * nu**2 - 26 * nu + 6) * np.cos(2 * z) + 3)
                + 128
                * chi_A
                * chi_S
                * delta
                * (82 * nu + 2 * (nu - 3) * np.cos(2 * z) - 3)
                + 64
                * chi_S**2
                * (-36 * nu**2 + 139 * nu + 2 * (nu - 3) * np.cos(2 * z) - 3)
                - 3328 * nu**2
                + 1152 * nu * np.cos(2 * z)
                + nu * (22112 - 615 * np.pi**2)
                + 15168
            )
            + e
            * (
                -256 * chi_A**2 * (8 * nu**2 - 14 * nu + 3) * np.cos(z)
                - 512 * chi_A * chi_S * delta * (nu + 3) * np.cos(z)
                + 128 * chi_S**2 * (3 * nu**2 - 8 * nu - 6) * np.cos(z)
                + 64 * (102 - 23 * nu) * np.cos(z)
            )
            - 1920 * nu**2
            - 2 * nu * (-10880 + 123 * np.pi**2)
            + (1 - e**2) ** (3 / 2)
            * (
                384 * chi_A**2 * (6 * nu**2 - 46 * nu + 11)
                + 768 * chi_A * chi_S * delta * (11 - 9 * nu)
                + 384 * chi_S**2 * (4 * nu**2 - 16 * nu + 11)
                + 192 * e**2 * (5 * nu**2 + 34 * nu - 90)
                + 1920 * e * (2 * nu - 5) * np.cos(z)
                + 1920 * nu**2
                + nu * (-21760 + 246 * np.pi**2)
                + 5760
            )
            - 5760
        )
        / (1 - e**2) ** (9 / 2)
        + (1 / 2016000)
        * flagPN52
        * omega_avg ** (8 / 3)
        * (
            -8064000 * chi_A * delta * (nu - 3)
            + 4032000 * chi_S * (nu**2 - 8 * nu + 6)
            + e**6
            * (
                -126000 * chi_A * delta * (11 * nu + 192) * np.cos(z) ** 2
                + 126000 * chi_S * (10 * nu**2 + 55 * nu - 192) * np.cos(z) ** 2
            )
            + e**5
            * (
                -15750
                * chi_A
                * delta
                * (
                    355 * nu
                    + 4 * (65 * nu + 288) * np.cos(2 * z)
                    + (81 * nu - 32) * np.cos(4 * z)
                    + 4256
                )
                * np.cos(z)
                + 15750
                * chi_S
                * (
                    290 * nu**2
                    + 1023 * nu
                    + 4 * (46 * nu**2 - 11 * nu - 288) * np.cos(2 * z)
                    + (54 * nu**2 - 187 * nu + 32) * np.cos(4 * z)
                    - 4256
                )
                * np.cos(z)
            )
            + e**4
            * (
                -63000
                * chi_A
                * delta
                * (
                    290 * nu
                    + (97 * nu - 40) * np.cos(4 * z)
                    + (365 * nu + 512) * np.cos(2 * z)
                    + 936
                )
                + 63000
                * chi_S
                * (
                    172 * nu**2
                    - 266 * nu
                    + (62 * nu**2 - 231 * nu + 40) * np.cos(4 * z)
                    + (214 * nu**2 - 607 * nu - 512) * np.cos(2 * z)
                    - 936
                )
            )
            + e**3
            * (
                -42000
                * chi_A
                * delta
                * (1143 * nu + (1137 * nu - 544) * np.cos(2 * z) + 608)
                * np.cos(z)
                + 42000
                * chi_S
                * (
                    558 * nu**2
                    - 2477 * nu
                    + (690 * nu**2 - 2795 * nu + 544) * np.cos(2 * z)
                    - 608
                )
                * np.cos(z)
            )
            + e**2
            * (
                -42000
                * chi_A
                * delta
                * (1137 * nu + 2 * (555 * nu - 424) * np.cos(2 * z) - 752)
                + 42000
                * chi_S
                * (
                    582 * nu**2
                    - 3043 * nu
                    + (636 * nu**2 - 2938 * nu + 848) * np.cos(2 * z)
                    + 752
                )
            )
            + e
            * (
                -42000 * chi_A * delta * (1059 * nu - 1568) * np.cos(z)
                + 42000 * chi_S * (570 * nu**2 - 3265 * nu + 1568) * np.cos(z)
            )
            + flagPA
            * nu
            * (1 - e**2) ** 4
            * (
                e**7
                * (
                    46317133100 * np.sin(z)
                    + 8694569548 * np.sin(3 * z)
                    + 43540 * np.sin(5 * z)
                    - 1268 * np.sin(7 * z)
                )
                + 56
                * e**6
                * (
                    321428775 * np.sin(2 * z)
                    + 5687282 * np.sin(4 * z)
                    + 33 * np.sin(6 * z)
                )
                + 1680
                * e**5
                * (8026530 * np.sin(z) + 1133545 * np.sin(3 * z) + 13 * np.sin(5 * z))
                + 33600 * e**4 * (115007 * np.sin(2 * z) + 1065 * np.sin(4 * z))
                + 22400 * e**3 * (121992 * np.sin(z) + 9535 * np.sin(3 * z))
                + 422105600 * e**2 * np.sin(2 * z)
                + 276326400 * e * np.sin(z)
            )
            + (1 - e**2) ** (3 / 2)
            * (
                8064000 * chi_A * delta * (nu - 3)
                - 4032000 * chi_S * (nu**2 - 8 * nu + 6)
                + e**2
                * (
                    8064000 * chi_A * delta * (nu - 3) * np.cos(z) ** 2
                    - 4032000 * chi_S * (nu**2 - 8 * nu + 6) * np.cos(z) ** 2
                )
                + e
                * (
                    16128000 * chi_A * delta * (nu - 3) * np.cos(z)
                    - 8064000 * chi_S * (nu**2 - 8 * nu + 6) * np.cos(z)
                )
            )
        )
        / (1 - e**2) ** 4
        + omega_avg * (e * np.cos(z) + 1) ** 2 / (1 - e**2) ** (3 / 2)
    )

    return dot_phi


def r_omega_avg_e_z(
    omega_avg,
    e,
    z,
    nu,
    delta,
    chi_A,
    chi_S,
    flags_ecc: dict | None = None,
):
    """
    Relative separation 'r' as a function of the orbit-averaged orbital angular
    frequency 'omega_avg', eccentricity 'e', and relativistic anomaly 'z',
    with post-adiabatic (PA) contributions. The PA contributions are set to
    zero by default.

    The name of the corresponding expression in the supplementary material
    "docs/mathematica/dynamical_variables.m" is "romegaAvgez".
    """

    if flags_ecc is None:
        flags_ecc = {}

    flagPN1 = flags_ecc.get("flagPN1", 1)
    flagPN32 = flags_ecc.get("flagPN32", 1)
    flagPN2 = flags_ecc.get("flagPN2", 1)
    flagPN52 = flags_ecc.get("flagPN52", 1)
    flagPN3 = flags_ecc.get("flagPN3", 1)
    flagPA = flags_ecc.get("flagPA", 0)

    r = (
        flagPN1 * (e**2 * (6 - nu) + nu) / (3 * e * np.cos(z) + 3)
        + flagPN2
        * omega_avg ** (2 / 3)
        * (
            e**4 * (4 * nu**2 + 15 * nu + 36)
            + e**2
            * (
                chi_A**2 * (36 - 144 * nu)
                + 72 * chi_A * chi_S * delta
                + 36 * chi_S**2
                - 8 * nu**2
                - 78 * nu
                - 54
            )
            + 4 * nu**2
            - 117 * nu
            + 36 * (1 - e**2) ** (3 / 2) * (2 * nu - 5)
            + 180
        )
        / (-36 * e**3 * np.cos(z) - 36 * e**2 + 36 * e * np.cos(z) + 36)
        + (1 / 5184)
        * flagPN3
        * omega_avg ** (4 / 3)
        * (
            576 * chi_A**2 * (48 * nu**2 - 281 * nu + 65)
            + 5760 * chi_A * chi_S * delta * (13 - 17 * nu)
            + 576 * chi_S**2 * (41 * nu**2 - 149 * nu + 65)
            - 8 * e**6 * (16 * nu**3 + 90 * nu**2 - 81 * nu + 432)
            + e**4
            * (
                -1728 * chi_A**2 * (4 * nu**2 - 61 * nu + 15)
                + 3456 * chi_A * chi_S * delta * (7 * nu - 15)
                - 1728 * chi_S**2 * (3 * nu**2 - 13 * nu + 15)
                + 384 * nu**3
                - 288 * nu**2
                + 10152 * nu
                - 20736
            )
            + e**2
            * (
                1728 * chi_A**2 * (18 * nu**2 - 13 * nu + 1)
                - 3456 * chi_A * chi_S * delta * (32 * nu - 1)
                + 1728 * chi_S**2 * (16 * nu**2 - 55 * nu + 1)
                - 384 * nu**3
                + 24336 * nu**2
                + 81 * nu * (-4328 + 123 * np.pi**2)
                + 20736
            )
            + 128 * nu**3
            + 22032 * nu**2
            + 2214 * nu * (-140 + 3 * np.pi**2)
            + (1 - e**2) ** (3 / 2)
            * (
                -3456 * chi_A**2 * (6 * nu**2 - 46 * nu + 11)
                + 6912 * chi_A * chi_S * delta * (9 * nu - 11)
                - 3456 * chi_S**2 * (4 * nu**2 - 16 * nu + 11)
                - 1728 * e**2 * (5 * nu**2 + 4 * nu - 15)
                - 17280 * nu**2
                - 18 * nu * (-10880 + 123 * np.pi**2)
                - 51840
            )
            + 51840
        )
        / ((1 - e**2) ** 2 * (e * np.cos(z) + 1))
        - 1
        / 3
        * flagPN32
        * omega_avg ** (1 / 3)
        * (3 * e**2 + 1)
        * (2 * chi_A * delta + chi_S * (2 - nu))
        / (np.sqrt(1 - e**2) * (e * np.cos(z) + 1))
        - 1
        / 1008000
        * flagPN52
        * omega_avg
        * (
            -21000 * chi_A * delta * (115 * nu - 384)
            + 21000 * chi_S * (26 * nu**2 - 481 * nu + 384)
            + e**4
            * (
                -21000 * chi_A * delta * (65 * nu + 96)
                + 21000 * chi_S * (46 * nu**2 - 107 * nu - 96)
            )
            + e**2
            * (
                -252000 * chi_A * delta * (13 * nu - 8)
                + 252000 * chi_S * (2 * nu**2 - 51 * nu + 8)
            )
            + (1 - e**2) ** (3 / 2)
            * (
                2688000 * chi_A * delta * (nu - 3)
                - 1344000 * chi_S * (nu**2 - 8 * nu + 6)
                + flagPA
                * nu
                * (
                    5
                    * e**8
                    * (
                        342591431 * np.sin(2 * z)
                        + 4401978 * np.sin(4 * z)
                        + 719778 * np.sin(6 * z)
                        + 42031 * np.sin(8 * z)
                    )
                    + e**7
                    * (
                        3363598000 * np.sin(z)
                        + 6809334 * np.sin(3 * z)
                        + 1041950 * np.sin(5 * z)
                        + 43766 * np.sin(7 * z)
                    )
                    - 14
                    * e**6
                    * (
                        -50250275 * np.sin(2 * z)
                        + 140608 * np.sin(4 * z)
                        + 6207 * np.sin(6 * z)
                    )
                    + 140
                    * e**5
                    * (
                        10561380 * np.sin(z)
                        + 17595 * np.sin(3 * z)
                        + 1237 * np.sin(5 * z)
                    )
                    + 2800 * e**4 * (77759 * np.sin(2 * z) - 125 * np.sin(4 * z))
                    + 5600 * e**3 * (85292 * np.sin(z) - 21 * np.sin(3 * z))
                    + 31539200 * e**2 * np.sin(2 * z)
                    + 79296000 * e * np.sin(z)
                )
            )
        )
        / ((1 - e**2) ** (3 / 2) * (e * np.cos(z) + 1))
        + (1 - e**2) / (omega_avg ** (2 / 3) * (e * np.cos(z) + 1))
    )

    return r


def pphi_omega_avg_e_z(
    omega_avg,
    e,
    z,
    nu,
    delta,
    chi_A,
    chi_S,
    flags_ecc: dict | None = None,
):
    """
    Angular momentum 'pphi' as a function of the orbit-averaged orbital angular
    frequency 'omega_avg', eccentricity 'e', and relativistic anomaly 'z',
    with post-adiabatic (PA) contributions. The PA contributions are set to
    zero by default.

    The name of the corresponding expression in the supplementary material
    "docs/mathematica/dynamical_variables.m" is "pphiomegaAvgez".
    """
    if flags_ecc is None:
        flags_ecc = {}

    flagPN1 = flags_ecc.get("flagPN1", 1)
    flagPN32 = flags_ecc.get("flagPN32", 1)
    flagPN2 = flags_ecc.get("flagPN2", 1)
    flagPN52 = flags_ecc.get("flagPN52", 1)
    flagPN3 = flags_ecc.get("flagPN3", 1)
    flagPA = flags_ecc.get("flagPA", 0)

    pphi = (
        (1 / 6)
        * flagPN1
        * omega_avg ** (1 / 3)
        * (e**2 * (9 - nu) + nu + 9)
        / np.sqrt(1 - e**2)
        + (1 / 24)
        * flagPN2
        * omega_avg
        * (
            chi_A**2 * (24 - 96 * nu)
            + 48 * chi_A * chi_S * delta
            + 24 * chi_S**2
            + e**4 * (nu**2 + 11 * nu - 3)
            + e**2
            * (
                chi_A**2 * (36 - 144 * nu)
                + 72 * chi_A * chi_S * delta
                + 36 * chi_S**2
                - 2 * nu * (nu + 19)
            )
            + nu**2
            - 81 * nu
            + (1 - e**2) ** (3 / 2) * (24 * nu - 60)
            + 141
        )
        / (1 - e**2) ** (3 / 2)
        + (1 / 10368)
        * flagPN3
        * omega_avg ** (5 / 3)
        * (
            1152 * chi_A**2 * (42 * nu**2 - 260 * nu + 59)
            - 2304 * chi_A * chi_S * delta * (85 * nu - 59)
            + 1152 * chi_S**2 * (44 * nu**2 - 146 * nu + 59)
            - 8 * e**6 * (7 * nu**3 + 72 * nu**2 + 621 * nu + 837)
            + e**4
            * (
                -7776 * chi_A**2 * (4 * nu**2 - 37 * nu + 9)
                + 5184 * chi_A * chi_S * delta * (11 * nu - 27)
                - 2592 * chi_S**2 * (4 * nu**2 - 19 * nu + 27)
                + 168 * nu**3
                - 4896 * nu**2
                + 36504 * nu
                - 34344
            )
            + e**2
            * (
                864 * chi_A**2 * (40 * nu**2 + 237 * nu - 67)
                - 1728 * chi_A * chi_S * delta * (77 * nu + 67)
                + 864 * chi_S**2 * (44 * nu**2 - 123 * nu - 67)
                - 168 * nu**3
                + 18432 * nu**2
                + 27 * nu * (-23512 + 861 * np.pi**2)
                - 23976
            )
            + 56 * nu**3
            + 32400 * nu**2
            + 54 * nu * (-12604 + 369 * np.pi**2)
            + (1 - e**2) ** (3 / 2)
            * (
                -3456 * chi_A**2 * (6 * nu**2 - 46 * nu + 11)
                + 6912 * chi_A * chi_S * delta * (9 * nu - 11)
                - 3456 * chi_S**2 * (4 * nu**2 - 16 * nu + 11)
                - 864 * e**2 * (8 * nu**2 + 31 * nu - 75)
                - 19008 * nu**2
                - 18 * nu * (-10256 + 123 * np.pi**2)
                - 12960
            )
            + 100440
        )
        / (1 - e**2) ** (5 / 2)
        - flagPN32
        * omega_avg ** (2 / 3)
        * (3 * e**2 + 5)
        * (2 * chi_A * delta + chi_S * (2 - nu))
        / (3 - 3 * e**2)
        + (1 / 25200)
        * flagPN52
        * omega_avg ** (4 / 3)
        * (
            350 * chi_A * delta * (313 * nu - 792)
            - 350 * chi_S * (62 * nu**2 - 1231 * nu + 792)
            + e**4
            * (
                525 * chi_A * delta * (49 * nu + 144)
                - 525 * chi_S * (38 * nu**2 - 67 * nu - 144)
            )
            + e**2
            * (186025 * chi_A * delta * nu + 175 * chi_S * nu * (3133 - 410 * nu))
            + flagPA
            * nu
            * (1 - e**2) ** 2
            * (
                -24
                * e**7
                * (
                    119980 * np.sin(z)
                    + 8015 * np.sin(3 * z)
                    + 1869 * np.sin(5 * z)
                    + 120 * np.sin(7 * z)
                )
                + 210
                * e**6
                * (4824 * np.sin(2 * z) + 417 * np.sin(4 * z) + 28 * np.sin(6 * z))
                - 1344
                * e**5
                * (1555 * np.sin(z) + 65 * np.sin(3 * z) + 9 * np.sin(5 * z))
                + 1680 * e**4 * (367 * np.sin(2 * z) + 15 * np.sin(4 * z))
                - 10080 * e**3 * (129 * np.sin(z) + np.sin(3 * z))
                + 245280 * e**2 * np.sin(2 * z)
                - 510720 * e * np.sin(z)
            )
            + (1 - e**2) ** (3 / 2)
            * (-33600 * chi_A * delta * (nu - 3) + 16800 * chi_S * (nu**2 - 8 * nu + 6))
        )
        / (1 - e**2) ** 2
        + np.sqrt(1 - e**2) / omega_avg ** (1 / 3)
    )

    return pphi


def r_diss_omega_avg_e_z(omega_avg, e, z, nu, flags_ecc: dict | None = None):
    """
    Dissipative contribution to the relative separation 'r' as a function of
    the orbit-averaged orbital angular frequency 'omega_avg', eccentricity 'e',
    and relativistic anomaly 'z'.

    The name of the corresponding expression in the supplementary material
    "docs/mathematica/dynamical_variables.m" is "rdissomegaAvgez".
    """
    if flags_ecc is None:
        flags_ecc = {}

    flagPN52 = flags_ecc.get("flagPN52", 1)
    flagPA = flags_ecc.get("flagPA", 1)

    r_diss = (
        -1
        / 504000
        * e
        * flagPN52
        * flagPA
        * nu
        * omega_avg
        * (
            7197780 * e**6 * np.cos(4 * z)
            + 420310 * e**6 * np.cos(6 * z)
            + 1716556045 * e**6
            - 11471040 * e**5 * np.cos(3 * z)
            - 753088 * e**5 * np.cos(5 * z)
            + 1332380 * e**4 * np.cos(4 * z)
            + 754318040 * e**4
            - 2318400 * e**3 * np.cos(3 * z)
            + 10 * e**2 * (4444009 * e**4 + 1749888 * e**2 + 393680) * np.cos(2 * z)
            + 242813200 * e**2
            - 1120 * e * (55017 * e**4 + 24470 * e**2 + 7240) * np.cos(z)
            + 39648000
        )
        * np.sin(z)
    )

    return r_diss


def pphi_diss_omega_avg_e_z(omega_avg, e, z, nu, flags_ecc: dict | None = None):
    """
    Dissipative contribution to the angular momentum 'pphi' as a function of
    the orbit-averaged orbital angular frequency 'omega_avg', eccentricity 'e',
    and relativistic anomaly 'z'.

    The name of the corresponding expression in the supplementary material
    "docs/mathematica/dynamical_variables.m" is "pphidissomegaAvgez".
    """

    if flags_ecc is None:
        flags_ecc = {}

    flagPN52 = flags_ecc.get("flagPN52", 1)
    flagPA = flags_ecc.get("flagPA", 1)

    pphi_diss = (
        -1
        / 2100
        * e
        * flagPN52
        * flagPA
        * nu
        * omega_avg ** (4 / 3)
        * (
            7956 * e**6 * np.cos(4 * z)
            + 480 * e**6 * np.cos(6 * z)
            + 259968 * e**6
            - 15575 * e**5 * np.cos(3 * z)
            - 980 * e**5 * np.cos(5 * z)
            + 2016 * e**4 * np.cos(4 * z)
            + 182448 * e**4
            - 4200 * e**3 * np.cos(3 * z)
            + 16 * e**2 * (2501 * e**4 + 1036 * e**2 + 105) * np.cos(2 * z)
            + 109200 * e**2
            - 35 * e * (5269 * e**4 + 3056 * e**2 + 1168) * np.cos(z)
            + 42560
        )
        * np.sin(z)
    )

    return pphi_diss


def prstar_omega_avg_e_z(
    omega_avg,
    e,
    z,
    nu,
    delta,
    chi_A,
    chi_S,
    flags_ecc: dict | None = None,
):
    """
    Radial momentum 'prstar' conjugate to the tortoise 'r' as a function of the
    orbit-averaged orbital angular frequency 'omega_avg', eccentricity 'e',
    and relativistic anomaly 'z', with PA contributions.

    The name of the corresponding expression in the supplementary material
    "docs/mathematica/dynamical_variables.m" is "prstaromegaAvgez".
    """

    if flags_ecc is None:
        flags_ecc = {}

    flagPN1 = flags_ecc.get("flagPN1", 1)
    flagPN32 = flags_ecc.get("flagPN32", 1)
    flagPN2 = flags_ecc.get("flagPN2", 1)
    flagPN52 = flags_ecc.get("flagPN52", 1)
    flagPN3 = flags_ecc.get("flagPN3", 1)
    flagPA = flags_ecc.get("flagPA", 1)

    prstar = (
        -1
        / 6
        * e
        * flagPN1
        * omega_avg
        * (e**2 * (3 - nu) + 6 * e * np.cos(z) + nu + 9)
        * np.sin(z)
        / (1 - e**2) ** (3 / 2)
        - 1
        / 72
        * e
        * flagPN2
        * omega_avg ** (5 / 3)
        * (
            chi_A**2 * (180 - 720 * nu)
            + 360 * chi_A * chi_S * delta
            + 180 * chi_S**2
            + e**4 * (nu**2 + 33 * nu + 9)
            + 36 * e**3 * (nu - 5) * np.cos(z)
            + e**2
            * (
                72 * chi_A**2 * (1 - 4 * nu) * np.cos(z) ** 2
                + 144 * chi_A * chi_S * delta * np.cos(z) ** 2
                + 72 * chi_S**2 * np.cos(z) ** 2
                - 2 * nu**2
                - 6 * nu
                + 18 * np.cos(2 * z)
                - 414
            )
            + e
            * (
                216 * chi_A**2 * (1 - 4 * nu) * np.cos(z)
                + 432 * chi_A * chi_S * delta * np.cos(z)
                + 216 * chi_S**2 * np.cos(z)
                - 108 * (nu - 3) * np.cos(z)
            )
            + nu**2
            - 279 * nu
            + 36 * (1 - e**2) ** (3 / 2) * (2 * nu - 5)
            + 585
        )
        * np.sin(z)
        / (1 - e**2) ** (5 / 2)
        + (1 / 10368)
        * e
        * flagPN3
        * omega_avg ** (7 / 3)
        * (
            -288 * chi_A**2 * (180 * nu**2 - 2323 * nu + 547)
            + 576 * chi_A * chi_S * delta * (467 * nu - 547)
            - 288 * chi_S**2 * (256 * nu**2 - 799 * nu + 547)
            - 8 * e**6 * (5 * nu**3 + 72 * nu**2 - 405 * nu - 999)
            - 432 * e**5 * (nu**2 - 65 * nu + 93) * np.cos(z)
            + e**4
            * (
                1728 * chi_A**2 * (20 * nu**2 - 113 * nu + 27) * np.cos(z) ** 2
                - 3456 * chi_A * chi_S * delta * (5 * nu - 27) * np.cos(z) ** 2
                - 1728 * chi_S**2 * (5 * nu - 27) * np.cos(z) ** 2
                + 120 * nu**3
                + 6336 * nu**2
                - 1296 * nu * (3 * nu - 4) * np.cos(4 * z)
                + 42120 * nu
                + 24 * (486 - 90 * nu) * np.cos(2 * z)
                - 93960
            )
            + e**3
            * (
                2592
                * chi_A**2
                * (4 * nu - 1)
                * (13 * nu + (3 * nu - 2) * np.cos(2 * z) - 88)
                * np.cos(z)
                - 5184
                * chi_A
                * chi_S
                * delta
                * (35 * nu + (9 * nu - 2) * np.cos(2 * z) - 88)
                * np.cos(z)
                + 2592
                * chi_S**2
                * (
                    11 * nu**2
                    - 57 * nu
                    + (3 * nu**2 - 15 * nu + 2) * np.cos(2 * z)
                    + 88
                )
                * np.cos(z)
                - 864
                * (
                    -47 * nu**2
                    + 199 * nu
                    + (36 * nu**2 - 54 * nu + 3) * np.cos(2 * z)
                    - 195
                )
                * np.cos(z)
            )
            + e**2
            * (
                864
                * chi_A**2
                * (
                    160 * nu**2
                    - 1306 * nu
                    + (88 * nu**2 - 31 * nu + 3) * np.cos(2 * z)
                    + 318
                )
                - 1728
                * chi_A
                * chi_S
                * delta
                * (130 * nu + (61 * nu - 3) * np.cos(2 * z) - 318)
                + 864
                * chi_S**2
                * (
                    42 * nu**2
                    - 226 * nu
                    + (18 * nu**2 - 103 * nu + 3) * np.cos(2 * z)
                    + 318
                )
                - 120 * nu**3
                + 12384 * nu**2
                - 162 * nu * (-12 + 41 * np.pi**2)
                - 27
                * (576 * nu**2 + nu * (-5008 + 123 * np.pi**2) + 1008)
                * np.cos(2 * z)
                + 241704
            )
            + e
            * (
                13824 * chi_A**2 * (6 * nu**2 + 28 * nu - 7) * np.cos(z)
                - 6912 * chi_A * chi_S * delta * (nu + 28) * np.cos(z)
                - 1728 * chi_S**2 * (11 * nu**2 + 4 * nu + 56) * np.cos(z)
                - 216 * (42 * nu**2 + nu * (-3722 + 123 * np.pi**2) + 666) * np.cos(z)
            )
            + 40 * nu**3
            - 44064 * nu**2
            - 162 * nu * (-6748 + 205 * np.pi**2)
            + (1 - e**2) ** (3 / 2)
            * (
                3456 * chi_A**2 * (6 * nu**2 - 46 * nu + 11)
                - 6912 * chi_A * chi_S * delta * (9 * nu - 11)
                + 3456 * chi_S**2 * (4 * nu**2 - 16 * nu + 11)
                + 864 * e**2 * (4 * nu**2 + 41 * nu - 75)
                + 15552 * e * (2 * nu - 5) * np.cos(z)
                + 22464 * nu**2
                + 18 * nu * (-9008 + 123 * np.pi**2)
                - 64800
            )
            - 162648
        )
        * np.sin(z)
        / (1 - e**2) ** (7 / 2)
        + (1 / 3)
        * e
        * flagPN32
        * omega_avg ** (4 / 3)
        * (2 * chi_A * delta + chi_S * (2 - nu))
        * (3 * e * np.cos(z) + 5)
        * np.sin(z)
        / (1 - e**2) ** 2
        + e * omega_avg ** (1 / 3) * np.sin(z) / np.sqrt(1 - e**2)
        - 1
        / 144000
        * flagPN52
        * omega_avg**2
        * (
            e**4
            * (
                750
                * chi_A
                * delta
                * (81 * nu * np.cos(2 * z) + 115 * nu + 864)
                * np.sin(2 * z)
                - 750
                * chi_S
                * (98 * nu**2 + 9 * nu * (6 * nu - 19) * np.cos(2 * z) + 47 * nu - 864)
                * np.sin(2 * z)
            )
            + e**3
            * (
                3000
                * chi_A
                * delta
                * (194 * nu + 3 * (59 * nu - 16) * np.cos(2 * z) + 768)
                * np.sin(z)
                - 3000
                * chi_S
                * (
                    124 * nu**2
                    - 230 * nu
                    + 3 * (34 * nu**2 - 145 * nu + 16) * np.cos(2 * z)
                    - 768
                )
                * np.sin(z)
            )
            + e**2
            * (
                3000 * chi_A * delta * (335 * nu - 312) * np.sin(2 * z)
                - 3000 * chi_S * (154 * nu**2 - 977 * nu + 312) * np.sin(2 * z)
            )
            + e
            * (1 - e**2) ** (3 / 2)
            * (
                -192000 * chi_A * delta * (nu - 3) * np.sin(z)
                + 96000 * chi_S * (nu**2 - 8 * nu + 6) * np.sin(z)
            )
            + e
            * (
                3000 * chi_A * delta * (565 * nu - 912) * np.sin(z)
                - 3000 * chi_S * (206 * nu**2 - 1855 * nu + 912) * np.sin(z)
            )
            + flagPA
            * nu
            * (1 - e**2) ** 3
            * (
                e**7
                * (
                    1096360525 * np.cos(z)
                    + 71824371 * np.cos(3 * z)
                    - 487973 * np.cos(5 * z)
                    - 16069 * np.cos(7 * z)
                )
                + 4
                * e**6
                * (
                    82306925 * np.cos(2 * z)
                    + 149614 * np.cos(4 * z)
                    + 8149 * np.cos(6 * z)
                    + 97254900
                )
                + 100
                * e**5
                * (3443358 * np.cos(z) + 166565 * np.cos(3 * z) - 665 * np.cos(5 * z))
                + 800 * e**4 * (102149 * np.cos(2 * z) + 17 * np.cos(4 * z) + 142268)
                + 800 * e**3 * (89912 * np.cos(z) + 2177 * np.cos(3 * z))
                + 12800 * e**2 * (776 * np.cos(2 * z) + 1755)
                + 6566400 * e * np.cos(z)
                + 1843200
            )
        )
        / (1 - e**2) ** 3
    )  # + flagPA * nu * omega_avg**(8/3) * (2972/105 + 176/5 * nu) + flagPA * nu * omega_avg**3 *
    # (-256/5 * np.pi + 1744/15 * (delta * chi_A + chi_S) - 1184/15 * nu * chi_S)
    # #+ flagPA * nu * omega_avg**(10/3) * (-68206/2835 - 32362/315 * nu - 608/15 * nu**2)

    return prstar


def dot_prstar_omega_avg_e_z(
    omega_avg,
    e,
    z,
    nu,
    delta,
    chi_A,
    chi_S,
    flags_ecc: dict | None = None,
):
    """
    Time derivative of the tortoise radial momentum 'dot_prstar'
    as a function of the orbit-averaged orbital angular frequency 'omega_avg',
    eccentricity 'e', and relativistic anomaly 'z'.

    The name of the corresponding expression in the supplementary material
    "docs/mathematica/dynamical_variables.m" is "prstardotomegaAvgez".
    """

    if flags_ecc is None:
        flags_ecc = {}

    flagPN1 = flags_ecc.get("flagPN1", 1)
    flagPN32 = flags_ecc.get("flagPN32", 1)
    flagPN2 = flags_ecc.get("flagPN2", 1)
    flagPN52 = flags_ecc.get("flagPN52", 1)
    flagPN3 = flags_ecc.get("flagPN3", 1)

    dot_prstar = (
        (1 / 6)
        * e
        * flagPN1
        * (-3 * e * (5 * np.cos(2 * z) + 3) + (e**2 * (nu - 21) - nu - 27) * np.cos(z))
        * (e * omega_avg * np.cos(z) + omega_avg) ** 2
        / (1 - e**2) ** 3
        - 1
        / 72
        * e
        * flagPN2
        * omega_avg ** (8 / 3)
        * (e * np.cos(z) + 1) ** 2
        * (
            -18
            * chi_A**2
            * (4 * nu - 1)
            * (
                e * (3 * e * np.cos(3 * z) + 14 * np.cos(2 * z) + 2)
                + (5 * e**2 + 16) * np.cos(z)
            )
            + 36
            * chi_A
            * chi_S
            * delta
            * (
                e * (3 * e * np.cos(3 * z) + 14 * np.cos(2 * z) + 2)
                + (5 * e**2 + 16) * np.cos(z)
            )
            + 18
            * chi_S**2
            * (
                e * (3 * e * np.cos(3 * z) + 14 * np.cos(2 * z) + 2)
                + (5 * e**2 + 16) * np.cos(z)
            )
            + 18
            * e
            * (
                3 * e**2 * nu
                - 33 * e**2
                - 3 * e * (nu + 2) * np.cos(3 * z)
                - 17 * nu
                + (5 * e**2 * (nu - 11) - 23 * nu - 1) * np.cos(2 * z)
                - 7
            )
            + 144 * (1 - e**2) ** (3 / 2) * (2 * nu - 5) * np.cos(z)
            + (
                e**4 * (nu**2 + 141 * nu - 531)
                - 2 * e**2 * (nu**2 + 174 * nu + 1215)
                + nu**2
                - 1035 * nu
                + 1125
            )
            * np.cos(z)
        )
        / (1 - e**2) ** 4
        - 1
        / 5184
        * e
        * flagPN3
        * omega_avg ** (10 / 3)
        * (e * np.cos(z) + 1) ** 2
        * (
            -36
            * chi_A**2
            * (
                3
                * e
                * (
                    108 * e**2 * nu**2 * np.cos(4 * z)
                    - 28 * e**2 * nu**2
                    - 339 * e**2 * nu * np.cos(4 * z)
                    - 2417 * e**2 * nu
                    + 78 * e**2 * np.cos(4 * z)
                    + 606 * e**2
                    + 6
                    * e
                    * (
                        5 * e**2 * (4 * nu**2 - 37 * nu + 9)
                        + 52 * nu**2
                        - 309 * nu
                        + 75
                    )
                    * np.cos(3 * z)
                    - 512 * nu**2
                    - 608 * nu
                    + 4
                    * (
                        e**2 * (140 * nu**2 - 2423 * nu + 597)
                        - 32 * nu**2
                        - 592 * nu
                        + 159
                    )
                    * np.cos(2 * z)
                    + 196
                )
                + 2
                * (
                    3 * e**4 * (100 * nu**2 - 1597 * nu + 393)
                    - 3 * e**2 * (268 * nu**2 + 8493 * nu - 2155)
                    - 1584 * nu**2
                    + 4172 * nu
                    - 836
                )
                * np.cos(z)
            )
            + 72
            * chi_A
            * chi_S
            * delta
            * (
                3
                * e
                * (
                    81 * e**2 * nu * np.cos(4 * z)
                    + 83 * e**2 * nu
                    - 78 * e**2 * np.cos(4 * z)
                    - 606 * e**2
                    + 6 * e * (5 * e**2 * (nu - 9) + 37 * nu - 75) * np.cos(3 * z)
                    - 504 * nu
                    + 4 * (e**2 * (143 * nu - 597) - 74 * nu - 159) * np.cos(2 * z)
                    - 196
                )
                + 2
                * (
                    3 * e**4 * (121 * nu - 393)
                    - 15 * e**2 * (7 * nu + 431)
                    - 2524 * nu
                    + 836
                )
                * np.cos(z)
            )
            - 36
            * chi_S**2
            * (
                3
                * e
                * (
                    27 * e**2 * nu**2 * np.cos(4 * z)
                    + 45 * e**2 * nu**2
                    - 135 * e**2 * nu * np.cos(4 * z)
                    - 173 * e**2 * nu
                    + 78 * e**2 * np.cos(4 * z)
                    + 606 * e**2
                    - 30
                    * e
                    * (e**2 * (nu - 9) - 2 * nu**2 + 13 * nu - 15)
                    * np.cos(3 * z)
                    - 212 * nu**2
                    + 832 * nu
                    + 4
                    * (
                        e**2 * (54 * nu**2 - 251 * nu + 597)
                        - 51 * nu**2
                        + 104 * nu
                        + 159
                    )
                    * np.cos(2 * z)
                    + 196
                )
                + 2
                * (
                    3 * e**4 * (48 * nu**2 - 217 * nu + 393)
                    - 3 * e**2 * (14 * nu**2 + 57 * nu - 2155)
                    - 1280 * nu**2
                    + 4220 * nu
                    - 836
                )
                * np.cos(z)
            )
            + 27
            * e
            * (
                12 * e**4 * nu**2
                - 1500 * e**4 * nu
                + 5004 * e**4
                + 24 * e**3 * nu * (3 * nu - 4) * np.cos(5 * z)
                + 64 * e**2 * nu**2
                + 24 * e**2 * nu * (9 * nu - 26) * np.cos(4 * z)
                + 5720 * e**2 * nu
                + 4824 * e**2
                - e
                * (
                    24 * e**2 * (14 * nu**2 - 47 * nu - 90)
                    - 552 * nu**2
                    + nu * (5384 - 123 * np.pi**2)
                    + 1008
                )
                * np.cos(3 * z)
                + 1436 * nu**2
                - 8172 * nu
                + 246 * np.pi**2 * nu
                + 2
                * (
                    10 * e**4 * (nu**2 - 125 * nu + 417)
                    + e**2 * (-524 * nu**2 + 5100 * nu + 2100)
                    + 802 * nu**2
                    + nu * (-10138 + 369 * np.pi**2)
                    - 4134
                )
                * np.cos(2 * z)
                - 6036
            )
            + (1 - e**2) ** (3 / 2)
            * (
                -6912 * chi_A**2 * (6 * nu**2 - 46 * nu + 11) * np.cos(z)
                + 13824 * chi_A * chi_S * delta * (9 * nu - 11) * np.cos(z)
                - 6912 * chi_S**2 * (4 * nu**2 - 16 * nu + 11) * np.cos(z)
                - 7776 * e * (2 * nu - 5) * (5 * np.cos(2 * z) + 3)
                - 36
                * (
                    24 * e**2 * (14 * nu**2 + 157 * nu - 375)
                    + 1104 * nu**2
                    - 7352 * nu
                    + 123 * np.pi**2 * nu
                    - 6840
                )
                * np.cos(z)
            )
            + (
                4 * e**6 * (5 * nu**3 + 234 * nu**2 - 10935 * nu + 14067)
                - 12 * e**4 * (5 * nu**3 + 1056 * nu**2 - 6417 * nu - 55917)
                + 3
                * e**2
                * (
                    20 * nu**3
                    + 21408 * nu**2
                    + 9 * nu * (-4036 + 615 * np.pi**2)
                    - 124308
                )
                - 20 * nu**3
                + 92664 * nu**2
                + 108 * nu * (-11651 + 369 * np.pi**2)
                - 185004
            )
            * np.cos(z)
        )
        / (1 - e**2) ** 5
        + (1 / 6)
        * e
        * flagPN32
        * omega_avg ** (7 / 3)
        * (2 * chi_A * delta + chi_S * (2 - nu))
        * (3 * e * (3 * np.cos(2 * z) + 1) + (6 * e**2 + 22) * np.cos(z))
        * (e * np.cos(z) + 1) ** 2
        / (1 - e**2) ** (7 / 2)
        - 1
        / 384
        * e
        * flagPN52
        * omega_avg**3
        * (e * np.cos(z) + 1) ** 2
        * (
            chi_A
            * delta
            * (
                e
                * (
                    405 * e**2 * nu * np.cos(4 * z)
                    + 311 * e**2 * nu
                    - 96 * e**2 * np.cos(4 * z)
                    + 4512 * e**2
                    + 48 * e * (59 * nu + 12) * np.cos(3 * z)
                    + 2680 * nu
                    + 12 * (e**2 * (71 * nu + 912) + 670 * nu + 16) * np.cos(2 * z)
                    + 1344
                )
                + 8
                * (
                    e**4 * (17 * nu + 624)
                    + 8 * e**2 * (80 * nu + 423)
                    + 1197 * nu
                    - 1200
                )
                * np.cos(z)
            )
            + chi_S
            * (
                e
                * (
                    -270 * e**2 * nu**2 * np.cos(4 * z)
                    - 250 * e**2 * nu**2
                    + 903 * e**2 * nu * np.cos(4 * z)
                    - 1315 * e**2 * nu
                    - 96 * e**2 * np.cos(4 * z)
                    + 4512 * e**2
                    - 48 * e * (34 * nu**2 - 131 * nu - 12) * np.cos(3 * z)
                    - 1232 * nu**2
                    + 5896 * nu
                    - 12
                    * (
                        e**2 * (58 * nu**2 + 235 * nu - 912)
                        + 308 * nu**2
                        - 1634 * nu
                        - 16
                    )
                    * np.cos(2 * z)
                    + 1344
                )
                - 8
                * (
                    e**4 * (22 * nu**2 + 205 * nu - 624)
                    + 4 * e**2 * (80 * nu**2 - 49 * nu - 846)
                    + 414 * nu**2
                    - 3639 * nu
                    + 1200
                )
                * np.cos(z)
            )
            + (1 - e**2) ** (3 / 2)
            * (
                -2048 * chi_A * delta * (nu - 3) * np.cos(z)
                + 1024 * chi_S * (nu**2 - 8 * nu + 6) * np.cos(z)
            )
        )
        / (1 - e**2) ** (9 / 2)
        + e
        * omega_avg ** (4 / 3)
        * (e * np.cos(z) + 1) ** 2
        * np.cos(z)
        / (1 - e**2) ** 2
    )

    return dot_prstar


def dot_r_pphi_e_z(
    pphi,
    e,
    z,
    nu,
    delta,
    chi_A,
    chi_S,
    flags_ecc: dict | None = None,
):
    """
    Time derivative of the separation 'r' as a function of the angular
    momentum 'pphi', eccentricity 'e', and relativistic anomaly 'z',
    with post-adiabatic (PA) contributions. The PA contributions are set to
    zero by default.
    The name of the corresponding expression in the supplementary material
    "docs/mathematica/dynamical_variables.m" is "rdotpphiez".
    """
    if flags_ecc is None:
        flags_ecc = {}

    flagPN1 = flags_ecc.get("flagPN1", 1)
    flagPN32 = flags_ecc.get("flagPN32", 1)
    flagPN2 = flags_ecc.get("flagPN2", 1)
    flagPN52 = flags_ecc.get("flagPN52", 1)
    flagPN3 = flags_ecc.get("flagPN3", 1)
    flagPA = flags_ecc.get("flagPA", 0)

    dot_r = (
        e**2
        * flagPN32
        * (-e + np.cos(z))
        * (2 * chi_A * delta + chi_S * (2 - nu))
        * np.sin(z)
        / pphi**4
        + (1 / 2)
        * e
        * flagPN1
        * (e**2 * (1 - nu) - 6 * e * np.cos(z) + nu - 3)
        * np.sin(z)
        / pphi**3
        + (1 / 8)
        * e
        * flagPN2
        * (
            chi_A**2 * (16 * nu - 4)
            - 8 * chi_A * chi_S * delta
            - 4 * chi_S**2
            + e**4 * (3 * nu**2 - 9 * nu + 7)
            + 12 * e**3 * (nu - 3) * np.cos(z)
            + e**2
            * (
                chi_A**2 * (12 - 48 * nu)
                + 24 * chi_A * chi_S * delta
                + 12 * chi_S**2
                - 6 * nu**2
                + 10 * nu
                + 6 * (2 * nu + 1) * np.cos(2 * z)
                + 28
            )
            + e
            * (
                8 * chi_A**2 * (4 * nu - 1) * np.cos(z)
                - 16 * chi_A * chi_S * delta * np.cos(z)
                - 8 * chi_S**2 * np.cos(z)
                + 4 * (11 * nu - 29) * np.cos(z)
            )
            + 3 * nu**2
            + 27 * nu
            - 81
        )
        * np.sin(z)
        / pphi**5
        - 1
        / 384
        * e
        * flagPN3
        * (
            96 * chi_A**2 * (28 * nu**2 - 503 * nu + 123)
            + 192 * chi_A * chi_S * delta * (123 - 67 * nu)
            + 96 * chi_S**2 * (32 * nu**2 - 123 * nu + 123)
            + 24 * e**6 * (5 * nu**3 - 24 * nu**2 + 47 * nu - 33)
            + 144 * e**5 * (3 * nu**2 - 13 * nu + 27) * np.cos(z)
            + e**4
            * (
                -288 * chi_A**2 * (12 * nu**2 - 79 * nu + 19)
                + 192 * chi_A * chi_S * delta * (25 * nu - 57)
                - 96 * chi_S**2 * (8 * nu**2 - 41 * nu + 57)
                - 360 * nu**3
                + 864 * nu**2
                - 144 * nu * (3 * nu - 4) * np.cos(4 * z)
                + 1992 * nu
                + 144 * (2 * nu**2 - 9 * nu - 5) * np.cos(2 * z)
                - 5880
            )
            + e**3
            * (
                96
                * chi_A**2
                * (4 * nu - 1)
                * (3 * nu * np.cos(2 * z) + 5 * nu - 88)
                * np.cos(z)
                - 192
                * chi_A
                * chi_S
                * delta
                * (9 * nu * np.cos(2 * z) + 27 * nu - 88)
                * np.cos(z)
                + 96
                * chi_S**2
                * (11 * nu**2 + 3 * nu * (nu - 5) * np.cos(2 * z) - 49 * nu + 88)
                * np.cos(z)
                - 96
                * (
                    -47 * nu**2
                    + 205 * nu
                    + (30 * nu**2 - 8 * nu + 1) * np.cos(2 * z)
                    - 276
                )
                * np.cos(z)
            )
            + e**2
            * (
                192
                * chi_A**2
                * (
                    34 * nu**2
                    + 210 * nu
                    + (18 * nu**2 - 9 * nu + 1) * np.cos(2 * z)
                    - 54
                )
                - 384
                * chi_A
                * chi_S
                * delta
                * (-13 * nu + (16 * nu - 1) * np.cos(2 * z) + 54)
                + 192
                * chi_S**2
                * (
                    -12 * nu**2
                    + 32 * nu
                    + (6 * nu**2 - 27 * nu + 1) * np.cos(2 * z)
                    - 54
                )
                + 360 * nu**3
                + 4224 * nu**2
                - nu * (5432 + 615 * np.pi**2)
                - (288 * nu**2 + nu * (16880 - 123 * np.pi**2) + 3216) * np.cos(2 * z)
                - 3480
            )
            + e
            * (
                768 * chi_A**2 * (8 * nu**2 - 107 * nu + 26) * np.cos(z)
                - 768 * chi_A * chi_S * delta * (37 * nu - 52) * np.cos(z)
                + 192 * chi_S**2 * (35 * nu**2 - 136 * nu + 104) * np.cos(z)
                + 8 * (318 * nu**2 + nu * (-8954 + 123 * np.pi**2) + 4758) * np.cos(z)
            )
            - 120 * nu**3
            + 528 * nu**2
            + 4 * nu * (-8654 + 123 * np.pi**2)
            + 31752
        )
        * np.sin(z)
        / pphi**7
        + e * np.sin(z) / pphi
        - 1
        / 144000
        * flagPN52
        * (
            e**5
            * (
                -27000 * chi_A * delta * (25 * nu - 32) * np.sin(z)
                + 27000 * chi_S * (14 * nu**2 - 51 * nu + 32) * np.sin(z)
            )
            + e**4
            * (
                2250
                * chi_A
                * delta
                * (113 * nu + (27 * nu - 32) * np.cos(2 * z) - 544)
                * np.sin(2 * z)
                - 2250
                * chi_S
                * (
                    70 * nu**2
                    - 475 * nu
                    + (18 * nu**2 - 73 * nu + 32) * np.cos(2 * z)
                    + 544
                )
                * np.sin(2 * z)
            )
            + e**3
            * (
                9000
                * chi_A
                * delta
                * (-9 * nu + (59 * nu - 32) * np.cos(2 * z) + 224)
                * np.sin(z)
                - 9000
                * chi_S
                * (
                    18 * nu**2
                    + 211 * nu
                    + (34 * nu**2 - 153 * nu + 32) * np.cos(2 * z)
                    - 224
                )
                * np.sin(z)
            )
            + e**2
            * (
                9000 * chi_A * delta * (93 * nu - 400) * np.sin(2 * z)
                - 9000 * chi_S * (42 * nu**2 - 455 * nu + 400) * np.sin(2 * z)
            )
            + e
            * (
                9000 * chi_A * delta * (133 * nu - 480) * np.sin(z)
                - 9000 * chi_S * (62 * nu**2 - 535 * nu + 480) * np.sin(z)
            )
            + flagPA
            * nu
            * (
                e**7
                * (
                    272575525 * np.cos(z)
                    + 27079671 * np.cos(3 * z)
                    - 288473 * np.cos(5 * z)
                    - 16069 * np.cos(7 * z)
                )
                + 4
                * e**6
                * (
                    28467125 * np.cos(2 * z)
                    + 139414 * np.cos(4 * z)
                    + 8149 * np.cos(6 * z)
                    + 28281300
                )
                + 100
                * e**5
                * (1482462 * np.cos(z) + 114317 * np.cos(3 * z) - 665 * np.cos(5 * z))
                + 800 * e**4 * (64901 * np.cos(2 * z) + 17 * np.cos(4 * z) + 64940)
                + 800 * e**3 * (65288 * np.cos(z) + 2177 * np.cos(3 * z))
                + 12800 * e**2 * (776 * np.cos(2 * z) + 1323)
                + 6566400 * e * np.cos(z)
                + 1843200
            )
        )
        / pphi**6
    )

    return dot_r


def dot_r_diss_ecc_pphi_e_z(pphi, e, z, nu, flags_ecc: dict | None = None):
    """
    Dissipative eccentric correction to the time derivative of the separation
    'r' as a function of the angular momentum 'pphi', eccentricity 'e', and
    relativistic anomaly 'z'.
    The name of the corresponding expression in the supplementary material
    "docs/mathematica/dynamical_variables.m" is "rdotdisseccpphiez".
    """

    if flags_ecc is None:
        flags_ecc = {}

    flagPN52 = flags_ecc.get("flagPN52", 1)
    flagPA = flags_ecc.get("flagPA", 1)

    dot_r_diss_ecc = (
        -1
        / 144000
        * e
        * flagPN52
        * flagPA
        * nu
        * (
            e
            * (
                -288473 * e**5 * np.cos(5 * z)
                - 16069 * e**5 * np.cos(7 * z)
                + 557656 * e**4 * np.cos(4 * z)
                + 32596 * e**4 * np.cos(6 * z)
                + 113125200 * e**4
                - 66500 * e**3 * np.cos(5 * z)
                + 1885600 * e**2 * np.cos(4 * z)
                + 52307200 * e**2
                + e * (27079671 * e**4 + 11431700 * e**2 + 6637600) * np.cos(3 * z)
                + 100 * (1138685 * e**4 + 512296 * e**2 + 126208) * np.cos(2 * z)
                + 13324800
            )
            + 25
            * (10903021 * e**6 + 5929848 * e**4 + 2003968 * e**2 + 102912)
            * np.cos(z)
        )
        / pphi**6
    )

    return dot_r_diss_ecc
