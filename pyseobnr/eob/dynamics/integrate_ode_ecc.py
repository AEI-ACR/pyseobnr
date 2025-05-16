"""
Contains functions associated with evolving the equations of motion
"""

from __future__ import annotations

import enum
import logging
from typing import Literal

import lal
import numpy as np
import pygsl_lite.errno as errno
import pygsl_lite.odeiv2 as odeiv2

from ..utils.containers import EOBParams
from ..utils.utils_eccentric import (
    compute_starting_values,
    interpolate_dynamics_ecc,
    r_omega_avg_e_z,
)
from ..waveform.waveform_ecc import SEOBNRv5RRForceEcc
from .initial_conditions_aligned_ecc_opt import compute_IC_ecc_opt
from .integrate_ode import control_y_new
from .postadiabatic_C_fast import compute_combined_dynamics
from .rhs_aligned import compute_H_and_omega
from .rhs_aligned_ecc import compute_x, get_rhs_ecc, get_rhs_ecc_secular

logger = logging.getLogger(__name__)

#: Type of supported ODE integrator, following the GSL conventions
IntegratorType = Literal["rk8pd", "rkf45"]


@enum.unique
class ColsEccDyn(enum.IntEnum):
    """
    Column indices for the dynamics of an eccentric system.

    .. note::

        The first columns up until ``pphi`` (included) point to
        the same dynamical variables as in the QC models SEOBNRv5HM
        and SEOBNRv5PHM.
    """

    #: Time
    t = 0

    #: Relative separation
    r = 1

    #: Azimuthal angle
    phi = 2

    #: (Tortoise) radial momentum (conjugate to the tortoise radius)
    pr = 3

    #: Orbital angular momentum
    pphi = 4

    #: Eccentricity
    e = 5

    #: Relativistic anomaly
    z = 6

    #: PN approximation for the dimensionless orbit-averaged orbital frequency
    x = 7

    #: Hamiltonian
    H = 8

    #: Instantaneous orbital frequency
    omega = 9


def ODE_system_RHS_ecc_opt(t: float, z: np.ndarray, args: tuple) -> np.ndarray:
    """
    GSL interface around the dynamics equations for eccentric
    aligned-spin systems.

    :param float t: The current time
    :param np.array z: The dynamics variables, stored as ``(q, p, Kep)``.
        ``q`` and ``p`` have the same meaning as for
        :py:func:`~.integrate_ode.ODE_system_RHS_opt` and
        :py:func:`~.rhs_aligned.get_rhs`.
        ``Kep`` contains the `Keplerian parameters`, respectively the
        eccentricity ``e`` as ``Kep[0]`` and ``z`` as ``Kep[1]``.
    :param tuple args: additional arguments to the right hand side function.

    The content of ``args`` should be a tuple whose content is as follow:

    #. The Hamiltonian to use, of type
       :py:class:`~pyseobnr.eob.hamiltonian.Hamiltonian_C.Hamiltonian_C`,
    #. The radiation reaction force class of type
       :py:class:`~pyseobnr.eob.waveform.waveform_ecc.RadiationReactionForceEcc`,
    #. ``chi_1`` (float): z-component of the primary spin,
    #. ``chi_2`` (float): z-component of the secondary spin,
    #. ``m_1`` (float): Mass of the primary,
    #. ``m_2`` (float): Mass of the secondary,
    #. :py:class:`~pyseobnr.eob.utils.containers.EOBParams`: The EOB parameters.

    Returns:
        np.array: The dynamics equations, as in
        :py:func:`~pyseobnr.eob.dynamics.integrate_ode.ODE_system_RHS_opt`,
        but with two additional values, respectively for
        :math:`\\dot{e}` and :math:`\\dot{z}`.
        See :py:func:`~.rhs_aligned_ecc.get_rhs_ecc` for details

    .. seealso::
        :py:func:`~.rhs_aligned_ecc.get_rhs_ecc`
    """

    return get_rhs_ecc(t, z, *args)


def ODE_system_RHS_ecc_secular_opt(
    t: float, Kep: np.ndarray, args: tuple
) -> np.ndarray:
    """
    GSL interface around the secular dynamics equations for eccentric
    aligned-spin systems.

    :param float t: The current time
    :param np.array Kep: The dynamics variables, stored as ``(e, z, x)``,
        where ``e`` is the eccentricity, ``z`` is the relativistic anomaly,
        and :math:`x = (M \\omega)^{2/3}` is the dimensionless
        orbit-averaged orbital frequency.
    :param tuple args: Additional arguments to the right hand side function.

    The content of ``args`` should be a tuple whose content is as follow:

    #. Instance of the radiation reaction forces helper class for the
        eccentric model :py:class:`SEOBNRv5RRForceEcc`

    Returns:
        np.array: The dynamics equations for ``e``, ``z``, and ``x``.
        See :py:func:`~.rhs_aligned_ecc.get_rhs_ecc_secular` for details
    """

    return get_rhs_ecc_secular(t, Kep, *args)


def _raise_ODE_errors(case, params=None):
    """
    Contains a list of possible error messages related to the
    solution of the ODEs.

    Args:
    case (str): Type of error
    params (EOBParams): EOB parameters of the system
    """

    if case == "start":
        error_message = (
            "Internal function call failed: Input domain error. "
            "The evaluation of the ODEs failed at the start of the "
            "evolution. Please, review the physical sense of "
            "the input parameters."
        )

    elif case == "evaluation":
        error_message = (
            "Internal function call failed: Input domain error. "
            "The evaluation of the ODEs failed. Please, review "
            "the physical sense of the input parameters."
        )

    elif case == "unbound_orbit":
        error_message = (
            "Internal function call failed: Input domain error. "
            "The system transitioned to an unbound configuration. "
            "This is probably related to the application of the model "
            "in a region outside its domain of validity. "
            "Please, review the physical sense of the input parameters."
        )

    elif case == "long_wf":
        error_message = (
            "Very long waveform for the input parameters: "
            f"q = {params.p_params.m_1/params.p_params.m_2}, "
            f"chi_1 = {params.p_params.chi_1}, "
            f"chi_2 = {params.p_params.chi_2}, "
            f"omega_avg = {params.ecc_params.omega_avg}, "
            f"eccentricity = {params.ecc_params.eccentricity}, "
            f"rel_anomaly = {params.ecc_params.rel_anomaly}. "
            "Please, review the physical sense of the input parameters."
        )

    elif case == "short_evolution":
        error_message = (
            "Internal function call failed: Input domain error. "
            "The time length of the dynamics is below 200 M. Aborting "
            "waveform generation since this could cause non-physical "
            "waveforms or problems in subsequent steps of the model. "
            "Please, review the physical sense of the input parameters."
        )

    elif case == "final_separation":
        error_message = (
            "Internal function call failed: Input domain error. "
            "The final separation is larger than 10 M. Please, review the "
            "physical sense of the input parameters."
        )

    elif case == "final_eccentricity":
        error_message = (
            "Internal function call failed: Input domain error. "
            "The final value of eccentricity "
            f"(e_final = {params.ecc_params.eccentricity}) "
            f"is larger than the threshold value of 0.25. "
            "Aborting the waveform generation because the system has not "
            "circularized enough and hence we cannot assume that the "
            "merger-ringdown signal will be similar to the one of a "
            "quasi-circular system. "
            "Please review the physical sense of the input parameters."
        )

    else:
        error_message = "Unknown ODE error."

    logger.error(error_message)
    raise ValueError(error_message)


def _validate_frequency(value, expected_value):
    """
    Validates the system's starting dimensionless orbit-averaged orbital
    frequency x = (M omega_avg)^{2/3}.

    Args:
    value (float): Computed value of the frequency
    expected_value (float): Expected value of the frequency
    """

    if abs(1 - value / expected_value) > 0.1:
        error_message = (
            "Internal function call failed: Input domain error. "
            "The computed value of the dimensionless starting orbit-"
            f"averaged frequency (x_avg_computed = {value}) differs "
            "from the expected value "
            f"(x_avg_expected = {expected_value}) by "
            f"{round(abs(1 - value/expected_value)*100, 3)}% > 10%. "
            "Aborting the waveform generation since this is related "
            "to the application of the model in a region outside its "
            "domain of validity. "
            "Please, review the physical sense of the input parameters."
        )
        logger.error(error_message)
        raise ValueError(error_message)


def compute_dynamics_ecc_opt(
    *,
    m_1: float,
    m_2: float,
    chi_1: float,
    chi_2: float,
    eccentricity: float,
    rel_anomaly: float,
    H,
    RR: SEOBNRv5RRForceEcc,
    params: EOBParams,
    integrator: IntegratorType = "rk8pd",
    atol: float = 1e-12,
    rtol: float = 1e-11,
    h_0: None | float = 1.0,
    y_init: None | np.ndarray = None,
    r_stop: None | float = None,
    step_back: float = 100,
) -> tuple[np.array, np.array, np.array, np.array]:
    """
    Main function to integrate the eccentric dynamics.

    The RHS of the employed ODE system is given by Eqs. (11), (17), (18a),
    (18c) and (19) in [Gamboa2024a]_ .
    See :py:mod:`.rhs_aligned_ecc` for more details.

    :param float m_1: Mass of the primary
    :param float m_2: Mass of the secondary
    :param float chi_1: z-component of the primary spin
    :param float chi_2: z-component of the secondary spin
    :param float eccentricity: Eccentricity of the system
    :param float rel_anomaly: Relativistic anomaly
    :param H: Instance of the Hamiltonian of the system
    :param SEOBNRv5RRForceEcc RR: Instance of the radiation-reaction force
        of the system
    :param EOBParams params: EOB parameters of the system
    :param str integrator: Either ``rkf45`` for the
        `Runge-Kutta-Fehlberg (4,5)` integrator, or ``rk8pd`` for the
        `Runge-Kutta Prince-Dormand (8, 9)` (default) integrator.
    :param float atol: Absolute tolerance of the ODE solver
    :param float rtol: Relative tolerance of the ODE solver
    :param float h_0: Initial time step of the ODE integrator
    :param y_init: Initial conditions. If ``None``, the initial conditions are
        computed from
        :py:func:`~.initial_conditions_aligned_ecc_opt.compute_IC_ecc_opt`
        with the provided values of eccentricity and relativistic anomaly.
        In other case, it should be in the form
        ``np.array([r0, 0.0, pr0, pphi0, eccentricity, rel_anomaly])``
    :param float r_stop: Minimum final separation for the dynamics. If ``None``
        a default of 2.5 will be used.
    :param float step_back: Step back for the start of the fine dynamics

    :returns: coarse, fine, coarse + fine, and raw dynamics arrays.
        Each row of these arrays contains
        ``(t, r, phi, pr, pphi, e, z, x, H_val, omega)``

    .. seealso::
        * :py:func:`interpolate_dynamics_ecc` for the interpolation of the
            eccentric dynamics
        * :py:func:`~.initial_conditions_aligned_ecc_opt.compute_IC_ecc_opt`
            for the initial conditions computation
        * :py:func:`~.rhs_aligned.compute_H_and_omega`
            for the computation of the Hamiltonian and the orbital frequency
    """

    # Step 1: Specify the ODE integrator settings

    num_eqs = 6  # Number of evolution equations in the ODE solver
    sys = odeiv2.system(
        ODE_system_RHS_ecc_opt, None, num_eqs, [H, RR, chi_1, chi_2, m_1, m_2, params]
    )
    if integrator == "rkf45":
        integrator_gsl_type = odeiv2.step_rkf45
    elif integrator == "rk8pd":
        integrator_gsl_type = odeiv2.step_rk8pd
    else:
        raise ValueError("Incorrect value for the numerical integrator type.")
    s = odeiv2.pygsl_lite_odeiv2_step(integrator_gsl_type, num_eqs)
    c = control_y_new(atol, rtol)
    e = odeiv2.pygsl_lite_odeiv2_evolve(num_eqs)

    # Time step
    if h_0:
        h = h_0
    else:
        h = 2 * np.pi / params.ecc_params.omega_avg / 5

    # Step 2: Compute the initial conditions of the ODE system

    if y_init is None:
        r0, pphi0, pr0 = compute_IC_ecc_opt(
            m_1=m_1,
            m_2=m_2,
            chi_1=chi_1,
            chi_2=chi_2,
            eccentricity=eccentricity,
            rel_anomaly=rel_anomaly,
            H=H,
            RR=RR,
            params=params,
        )
        phi0 = 0.0
        e0 = eccentricity
        z0 = rel_anomaly
        y0 = np.array([r0, phi0, pr0, pphi0, e0, z0])

    else:
        y0 = y_init.copy()
        r0 = y0[0]
        phi0 = y0[1]
        pr0 = y0[2]
        pphi0 = y0[3]
        e0 = y0[4]
        z0 = y0[5]

    grad = H.dynamics(
        np.array([r0, phi0]), np.array([pr0, pphi0]), chi_1, chi_2, m_1, m_2
    )
    params.p_params.omega = grad[3]
    RR.evolution_equations.compute(e=e0, omega=grad[3], z=z0)
    x0 = RR.evolution_equations.get("xavg_omegainst")
    params.ecc_params.x_avg = x0
    _validate_frequency(x0, params.ecc_params.omega_avg ** (2 / 3))

    t = 0.0
    y = y0
    res_gsl = [y]
    ts = [t]
    x_values = [x0]

    # Put a bound in the maximum possible separation as 5 times the
    # Newtonian value of the apastron. The purpose of this bound is to
    # avoid the cases in which the stopping conditions fail and the
    # black holes start to go away from each other indefinitely
    r_max = 5 * ((1.0 - e0**2) / x0 / (1.0 - e0))

    # Define the stopping conditions
    t_max = params.ecc_params.t_max
    if r_stop is None:
        r_stop_ecc = 2.5
    else:
        r_stop_ecc = r_stop

    # Step 3: Integrate the ODEs

    while t < t_max:
        # Take a step
        try:
            status, t, h, y = e.apply(c, s, sys, t, t_max, h, y)
        except Exception:
            if t == 0.0:
                _raise_ODE_errors("start")
            else:
                try:
                    h = 0.1
                    status, t, h, y = e.apply(c, s, sys, t, t_max, h, y)
                except Exception:
                    if y[0] < 3.5:
                        ts = ts[:-1]
                        res_gsl = res_gsl[:-1]
                        x_values = x_values[:-1]
                        params.ecc_params.stopping_condition = "ODE_stopped"
                        break
                    else:
                        _raise_ODE_errors("evaluation")

        if status != errno.GSL_SUCCESS:
            ts = ts[:-1]
            res_gsl = res_gsl[:-1]
            x_values = x_values[:-1]
            params.ecc_params.stopping_condition = "ODE_status"
            break

        # Compute the error for the step controller and append last step
        e.get_yerr()
        res_gsl.append(y)
        ts.append(t)
        x_values.append(params.ecc_params.x_avg)
        r = y[0]

        # Checks
        if r > r_max:
            _raise_ODE_errors("unbound_orbit")

        if r <= r_stop_ecc:
            params.ecc_params.stopping_condition = "r <= r_stop"
            break

    # Assemble arrays
    ts = np.array(ts)
    dyn: np.array = np.c_[np.array(res_gsl), np.array(x_values)]
    params.ecc_params.r_final = dyn[-1, 0]

    # Step 4: Perform sanity-checks of the ODE solution

    if t >= t_max:
        _raise_ODE_errors("long_wf", params)
    if ts[-1] < 200:
        _raise_ODE_errors("short_evolution")
    if r > 10:
        _raise_ODE_errors("final_separation")
    if dyn[-1, 4] > 0.25:
        params.ecc_params.eccentricity = dyn[-1, 4]
        _raise_ODE_errors("final_eccentricity", params)

    dynamics_raw = np.c_[ts, dyn]

    # Step 5: Computation of fine dynamics and assembling of the full dynamics

    t_desired = ts[-1] - step_back

    # Determine the index for the start of fine integration
    idx_close = np.argmin(np.abs(ts - t_desired))
    if ts[idx_close] > t_desired:
        idx_close -= 1

    # Guard against the case where when the dynamics is short
    # there is less than step_back time between the start
    # of the ODE integration and the end of the dynamics
    # In that case make the fine dynamics be _all_ dynamics
    # except the 1st element
    step_back_total = False
    if t_desired < ts[1]:
        idx_close = 1
        step_back = ts[-1] - ts[idx_close]
        step_back_total = True

    # Define the coarse and fine dynamics arrays
    dynamics_low = np.c_[ts[:idx_close], dyn[:idx_close]]
    dynamics_fine = np.c_[ts[idx_close:], dyn[idx_close:]]

    # Augment the dynamics array with the values of H_val and omega
    # x is already computed here
    assert dynamics_low.shape[1] >= ColsEccDyn.x
    dynamics_low_H_omega = compute_H_and_omega(dynamics_low, chi_1, chi_2, m_1, m_2, H)
    dynamics_low = np.c_[dynamics_low, dynamics_low_H_omega]

    # Interpolate (r, phi, pr, pphi, e, z)
    # In particular, we discard x because it will be recomputed afterwards
    dynamics_fine = interpolate_dynamics_ecc(
        # 1 (time) + 4 (EOB vars) + 2 (e & z): Should be 7
        dynamics_fine[:, : ColsEccDyn.x],
        step_back=step_back,
        step_back_total=step_back_total,
    )
    dynamics_fine_H_omega = compute_H_and_omega(
        dynamics_fine, chi_1, chi_2, m_1, m_2, H
    )
    dynamics_fine_x = compute_x(
        e=dynamics_fine[:, ColsEccDyn.e],
        z=dynamics_fine[:, ColsEccDyn.z],
        omega=dynamics_fine_H_omega[:, 1],
        RR=RR,
    )
    dynamics_fine = np.c_[dynamics_fine, dynamics_fine_x, dynamics_fine_H_omega]

    # Full dynamics array
    dynamics = np.vstack((dynamics_low, dynamics_fine))

    return dynamics_low, dynamics_fine, dynamics, dynamics_raw


def compute_background_qc_dynamics(
    *,
    duration_ecc: float,
    m_1: float,
    m_2: float,
    chi_1: float,
    chi_2: float,
    H,
    RR,
    params: EOBParams,
    r_stop: None | float = None,
) -> tuple[np.array, np.array]:
    """
    Computation of the background QC dynamics given a certain time length.

    Args:
        duration_ecc (float): Duration of the eccentric dynamics
        m_1 (float): Mass of the primary
        m_2 (float): Mass of the secondary
        chi_1 (float): z-component of the dimensionless spin vector of the primary
        chi_2 (float): z-component of the dimensionless spin vector of the secondary
        H (Hamiltonian): The Hamiltonian to use (an instance of Hamiltonian class)
        RR: QC RR force
        params (EOBParams): EOB parameters of the system
        r_stop (float): Minimum final separation for the QC dynamics

    Returns:
        tuple: (dyn_qc, dyn_fine_qc). The dynamics and fine dynamics arrays
        for the QC evolution.
    """

    nu = params.p_params.nu

    # Maximum frequency of the QC model, corresponding to an initial
    # separation of r ~ 10 M
    omega_start_qc_max = 1 / 10**1.5

    # Minimum frequency for 10 Hz and 5 total solar masses
    omega_start_qc_min = 10 * np.pi * 5 * lal.MTSUN_SI

    # Minimum duration of the dynamics for applying the following test
    min_duration = 250

    if duration_ecc <= min_duration:
        buffer = 1.0
        omega_start_qc = omega_start_qc_max

    else:
        buffer = 0.9
        # Estimate the starting frequency of the QC dynamics given the
        # the time length of the eccentric dynamics, using the first terms
        # from Eq. (6) of https://arxiv.org/pdf/2304.11185.pdf
        tau = nu / 5.0 * duration_ecc
        PN_terms = (
            1
            + (743 / 4032 + nu * 11 / 48) * tau ** (-1.0 / 4)
            - np.pi / 5 * tau ** (-3.0 / 8)
            + (19583 / 254016 + nu * 24401 / 193536 + nu**2 * 31 / 288)
            * tau ** (-1.0 / 2)
            + (-11891 / 53760 + nu * 109 / 1920) * np.pi * tau ** (-5.0 / 8)
        )
        if PN_terms <= 0:
            # Sometimes there is a negative root. In this case, just use
            # the first terms
            PN_terms = 1 + (743 / 4032 + nu * 11 / 48) * tau ** (-1.0 / 4)

        omega_start_qc = buffer * (tau ** (-1.0 / 4) / 4.0 * PN_terms) ** (3.0 / 2.0)

        if omega_start_qc > omega_start_qc_max:
            buffer = 1.0
            omega_start_qc = omega_start_qc_max

        elif omega_start_qc < omega_start_qc_min:
            buffer = 1.0
            omega_start_qc = omega_start_qc_min

    params.ecc_params.omega_start_qc = omega_start_qc

    step_back = 250.0
    PA_order = 8
    dyn_low_qc, dyn_fine_qc = compute_combined_dynamics(
        omega0=omega_start_qc,
        m_1=m_1,
        m_2=m_2,
        chi_1=chi_1,
        chi_2=chi_2,
        H=H,
        RR=RR,
        params=params,
        tol=1e-11,
        backend="ode",
        step_back=step_back,
        PA_order=PA_order,
        # in v5EHM we use r_stop = None when we cannot determine that value from the ISCO,
        # while in v5HM we use r_stop = -1.
        r_stop=-1 if r_stop is None else r_stop,
    )

    dyn_qc = np.vstack((dyn_low_qc, dyn_fine_qc))
    t_qc = dyn_qc[:, 0]

    # If the QC duration is smaller than the eccentric duration, then
    # we iterate until the QC evolution is larger
    while t_qc[-1] < duration_ecc:
        buffer = buffer - 0.1
        if buffer >= 0.1:
            omega_start_qc = buffer * omega_start_qc
            params.ecc_params.omega_start_qc = omega_start_qc
            dyn_low_qc, dyn_fine_qc = compute_combined_dynamics(
                omega0=omega_start_qc,
                m_1=m_1,
                m_2=m_2,
                chi_1=chi_1,
                chi_2=chi_2,
                H=H,
                RR=RR,
                params=params,
                tol=1e-11,
                backend="ode",
                step_back=step_back,
                PA_order=PA_order,
                # in v5EHM we use r_stop = None when we cannot determine that value from the ISCO,
                # while in v5HM we use r_stop = -1.
                r_stop=-1 if r_stop is None else r_stop,
            )
            dyn_qc = np.vstack((dyn_low_qc, dyn_fine_qc))
            t_qc = dyn_qc[:, 0]

        else:
            logger.warning("QC evolution shorter than the eccentric evolution.")
            break

    return dyn_qc, dyn_fine_qc


def compute_dynamics_ecc_backwards_opt(
    *,
    m_1: float,
    m_2: float,
    chi_1: float,
    chi_2: float,
    eccentricity: float,
    rel_anomaly: float,
    H,
    RR: SEOBNRv5RRForceEcc,
    params: EOBParams,
    integrator: IntegratorType = "rk8pd",
    atol: float = 1e-12,
    rtol: float = 1e-11,
    y_init: None | np.ndarray = None,
    e_stop: float = 0.9,
    r_stop: None | float = None,
    t_stop: float = None,
) -> tuple[np.array, np.array]:
    """
    Backwards integration of the full equations of motion of the
    eccentric dynamics.

    The RHS of the employed ODE system is given by Eqs. (11), (17), (18a),
    (18c) and (19) in [Gamboa2024a]_ .
    See :py:mod:`.rhs_aligned_ecc` for more details.

    :param float m_1: Mass of the primary
    :param float m_2: Mass of the secondary
    :param float chi_1: z-component of the primary spin
    :param float chi_2: z-component of the secondary spin
    :param float eccentricity: Eccentricity of the system
    :param float rel_anomaly: Relativistic anomaly
    :param H: Instance of the Hamiltonian of the system
    :param SEOBNRv5RRForceEcc RR: Instance of the radiation-reaction force
        of the system
    :param params: EOB parameters of the system
    :param str integrator: Either ``rkf45`` for the
        `Runge-Kutta-Fehlberg (4,5)` integrator, or ``rk8pd`` for the
        `Runge-Kutta Prince-Dormand (8, 9)` (default) integrator.
    :param float atol: Absolute tolerance of the ODE solver
    :param float rtol: Relative tolerance of the ODE solver
    :param y_init: Initial conditions for the backwards integration.
        The array should be in the form
        ``np.array([r0, phi0, pr0, pphi0, e0, z0])``
    :param float e_stop: Maximum allowed value of eccentricity for the
        backwards integration
    :param float r_stop: Value of separation (r) at which we stop the
        backwards integration
    :param float t_stop: Value of time (t) in geometric units at which we
        stop the backwards integration. It can be positive or negative
        (in any case, this function only integrates backwards in time)

    :returns: Array with initial conditions and dynamics array whose rows
        contain the variables: ``(t, r, phi, pr, pphi, e, z, x, H, omega)``

    .. seealso::
        * :py:func:`~.initial_conditions_aligned_ecc_opt.compute_IC_ecc_opt`
            for the initial conditions computation
        * :py:func:`~.rhs_aligned.compute_H_and_omega`
            for the computation of the Hamiltonian and the orbital frequency
    """

    # Step 1: Specify the ODE integrator settings

    num_eqs = 6  # Number of evolution equations
    sys = odeiv2.system(
        ODE_system_RHS_ecc_opt, None, num_eqs, [H, RR, chi_1, chi_2, m_1, m_2, params]
    )
    if integrator == "rkf45":
        integrator_gsl_type = odeiv2.step_rkf45
    elif integrator == "rk8pd":
        integrator_gsl_type = odeiv2.step_rk8pd
    else:
        raise ValueError("Incorrect value for the numerical integrator type.")
    s = odeiv2.pygsl_lite_odeiv2_step(integrator_gsl_type, num_eqs)
    c = control_y_new(atol, rtol)
    e = odeiv2.pygsl_lite_odeiv2_evolve(num_eqs)

    h = -1.0  # Backwards time step

    # Step 2: Determine the initial conditions for the ODE system

    if y_init is None:
        r0, pphi0, pr0 = compute_IC_ecc_opt(
            m_1=m_1,
            m_2=m_2,
            chi_1=chi_1,
            chi_2=chi_2,
            eccentricity=eccentricity,
            rel_anomaly=rel_anomaly,
            H=H,
            RR=RR,
            params=params,
        )
        phi0 = 0.0
        e0 = eccentricity
        z0 = rel_anomaly
        y0 = np.array([r0, phi0, pr0, pphi0, e0, z0])

    else:
        y0 = y_init.copy()
        r0 = y0[0]
        phi0 = y0[1]
        pr0 = y0[2]
        pphi0 = y0[3]
        e0 = y0[4]
        z0 = y0[5]

    grad = H.dynamics(
        np.array([r0, phi0]), np.array([pr0, pphi0]), chi_1, chi_2, m_1, m_2
    )
    params.p_params.omega = grad[3]
    RR.evolution_equations.compute(e=e0, omega=grad[3], z=z0)
    x0 = RR.evolution_equations.get("xavg_omegainst")
    params.ecc_params.x_avg = x0

    t = 0.0
    y = y0
    res_gsl = []
    ts = []
    x_values = []

    t_max = -params.ecc_params.t_max

    # Step 3: Integrate the ODEs

    while t > t_max:
        # Take a step
        try:
            status, t, h, y = e.apply(c, s, sys, t, t_max, h, y)
        except Exception:
            if t == 0.0:
                _raise_ODE_errors("start")
            else:
                try:
                    h = -0.1
                    status, t, h, y = e.apply(c, s, sys, t, t_max, h, y)
                except Exception:
                    _raise_ODE_errors("evaluation")

        if status != errno.GSL_SUCCESS:
            ts = ts[:-1]
            res_gsl = res_gsl[:-1]
            x_values = x_values[:-1]
            break

        # Compute the error for the step controller and append last step
        e.get_yerr()
        res_gsl.append(y)
        ts.append(t)
        x = params.ecc_params.x_avg
        x_values.append(x)
        ecc = y[4]

        # Handle termination conditions
        if t_stop:
            if t < -abs(t_stop):
                break
            elif ecc > e_stop:
                logger.warning(
                    "The backwards integration reached the maximum allowed "
                    f"value for eccentricity (e_stop = {e_stop}). "
                    "Stopping the integration."
                )
                break
        elif r_stop:
            r = y[0]
            if r > r_stop:
                break

    # Step 4: Assemble the dynamics array

    dynamics = np.c_[np.array(ts), np.array(res_gsl), np.array(x_values)]

    dynamics_H_omega = compute_H_and_omega(dynamics, chi_1, chi_2, m_1, m_2, H)
    dynamics = np.c_[dynamics, dynamics_H_omega]

    dynamics = np.flip(dynamics, axis=0)

    return y0, dynamics


def compute_dynamics_ecc_secular_opt(
    *,
    eccentricity: float,
    rel_anomaly: float,
    omega_avg: float,
    RR: SEOBNRv5RRForceEcc,
    params: EOBParams,
    integrator: IntegratorType = "rk8pd",
    atol: float = 1e-12,
    rtol: float = 1e-11,
    h_0: None | float = -1.0,
    r_stop: None | float = None,
) -> tuple[float, float, float, float]:
    """
    Main function to integrate the secular eccentric dynamics backwards
    in time.

    The RHS of the employed ODE system is given by Eqs. (18a)-(18c)
    in [Gamboa2024a]_ .
    See :py:mod:`.rhs_aligned_ecc` for more details.

    :param float eccentricity: Eccentricity of the system
    :param float rel_anomaly: Relativistic anomaly
    :param SEOBNRv5RRForceEcc RR: Instance of the radiation reaction
        force containing an initialized instance of the secular
        evolution equations
    :param EOBParams params: EOB parameters of the system
    :param str integrator: Either ``rkf45`` for the
        `Runge-Kutta-Fehlberg (4,5)` integrator, or ``rk8pd`` for the
        `Runge-Kutta Prince-Dormand (8, 9)` (default) integrator.
    :param float atol: Absolute tolerance of the ODE solver
    :param float rtol: Relative tolerance of the ODE solver
    :param float h_0: Initial time step of the ODE integrator
    :param float r_stop: Separation at which the ODE solver is stopped

    :returns: (t_start, eccentricity, rel_anomaly, omega_avg), the values
        of starting time, eccentricity, relativistic anomaly, and
        orbit-averaged orbital frequency that correspond to ``r = r_stop``
    """

    # Step 1: Specify the ODE integrator settings

    num_eqs = 3  # Number of evolution equations in the ODE solver
    sys = odeiv2.system(
        ODE_system_RHS_ecc_secular_opt,
        None,
        num_eqs,
        [RR],
    )
    if integrator == "rkf45":
        integrator_gsl_type = odeiv2.step_rkf45
    elif integrator == "rk8pd":
        integrator_gsl_type = odeiv2.step_rk8pd
    else:
        raise ValueError("Incorrect value for the numerical integrator type.")
    s = odeiv2.pygsl_lite_odeiv2_step(integrator_gsl_type, num_eqs)
    c = control_y_new(atol, rtol)
    e = odeiv2.pygsl_lite_odeiv2_evolve(num_eqs)

    # Backwards time step
    h = -abs(h_0)

    # Step 2: Specify the initial conditions of the ODE system

    t = 0.0
    y = np.array([eccentricity, rel_anomaly, omega_avg ** (2.0 / 3.0)])
    res_gsl = [y]
    ts = [t]

    flags_ecc_diss = params.ecc_params.flags_ecc.copy()
    flags_ecc_diss["flagPA"] = 1

    # Define the stopping conditions
    t_max = -params.ecc_params.t_max

    # Step 3: Integrate the ODEs

    while t > t_max:
        # Take a step
        try:
            status, t, h, y = e.apply(c, s, sys, t, t_max, h, y)
        except Exception:
            if t == 0.0:
                _raise_ODE_errors("start")
            else:
                try:
                    h = -0.1
                    status, t, h, y = e.apply(c, s, sys, t, t_max, h, y)
                except Exception:
                    _raise_ODE_errors("evaluation")

        if status != errno.GSL_SUCCESS:
            ts = ts[:-1]
            res_gsl = res_gsl[:-1]
            params.ecc_params.stopping_condition = "ODE_status"
            break

        # Compute the error for the step controller and append last step
        e.get_yerr()
        res_gsl.append(y)
        ts.append(t)

        r = r_omega_avg_e_z(
            y[2] ** 1.5,
            y[0],
            y[1],
            params.p_params.nu,
            params.p_params.delta,
            params.p_params.chi_A,
            params.p_params.chi_S,
            flags_ecc_diss,
        )

        # Reduce time step if r is close to r_stop. Useful when r_stop is
        # close to an apastron
        if abs(r - r_stop) < 0.2:
            h = -1.0

        # Stopping condition
        if r >= r_stop:
            threshold = False
            break
        elif abs(r - r_stop) <= 0.0001:
            threshold = True
            break

    # Assemble arrays
    dynamics = np.c_[np.array(ts), np.array(res_gsl)]
    dynamics = np.flip(dynamics, axis=0)

    # Step 4: Compute the exact values of starting time, eccentricity,
    # rel_anomaly, and omega_avg that correspond to r = r_stop

    t_start, eccentricity, rel_anomaly, omega_avg = compute_starting_values(
        r_stop,
        dynamics,
        threshold,
        params,
    )

    return t_start, eccentricity, rel_anomaly, omega_avg
