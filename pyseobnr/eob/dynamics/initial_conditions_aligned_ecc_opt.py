"""
Computes the aligned-spin eccentric initial conditions in polar coordinates.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import root, root_scalar

from ..hamiltonian import Hamiltonian
from ..utils.containers import EOBParams
from ..utils.utils_eccentric import (
    dot_prstar_omega_avg_e_z,
    dot_r_diss_ecc_pphi_e_z,
    dot_r_pphi_e_z,
    pphi_diss_omega_avg_e_z,
    pphi_omega_avg_e_z,
    prstar_omega_avg_e_z,
    r_diss_omega_avg_e_z,
    r_omega_avg_e_z,
)
from ..waveform.waveform_ecc import RadiationReactionForceEcc

logger = logging.getLogger(__name__)


def IC_cons_ecc(
    u: tuple[float, float] | list[float] | np.array,
    omega_inst: float,
    pr_star: float,
    dot_prstar: float,
    m_1: float,
    m_2: float,
    chi_1: float,
    chi_2: float,
    H: Hamiltonian,
) -> np.array:
    """
    Equations defining the "conservative" part of the initial conditions.
    See Eqs. (110a, 110b) in [Gamboa2024a]_ .

    Args:
        u (tuple): 2-uple containing the unknowns r and pphi.
        omega_inst (float): Desired starting instantaneous orbital frequency,
            in geometric units
        pr_star (float): Radial momentum in tortoise coordinates
        dot_prstar (float): Time derivative of the radial momentum in tortoise coordinates
        m_1 (float): Mass of the primary
        m_2 (float): Mass of the secondary
        chi_1 (float): z-component of the dimensionless spin vector of the primary
        chi_2 (float): z-component of the dimensionless spin vector of the secondary
        H (Hamiltonian): The Hamiltonian to use (an instance of Hamiltonian class)

    Returns:
        [np.array]: The desired equations evaluated at u
    """

    r, pphi = u
    q = np.array([r, 0.0])
    p = np.array([pr_star, pphi])

    grad = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)
    dHdr = grad[0]
    dHdpphi = grad[3]
    xi = grad[5]

    return np.array([dHdpphi - omega_inst, dot_prstar + xi * dHdr])


def IC_diss_ecc(
    u: float | np.ndarray,
    r: float,
    pphi: float,
    dot_r_cons: float,
    dot_r_diss_ecc: float,
    m_1: float,
    m_2: float,
    chi_1: float,
    chi_2: float,
    eccentricity: float,
    rel_anomaly: float,
    H: Hamiltonian,
    RR: RadiationReactionForceEcc,
    params: EOBParams,
):
    """
    Equations defining the "dissipative" part of the initial conditions.
    See Eq. (116) in [Gamboa2024a]_ .

    Args:
        u (float, np.ndarray): Guess for pr
        r (float): Starting separation
        pphi (float): Starting angular momentum
        dot_r_cons (float): Time derivative of r computed with the conservative dynamics
        dot_r_diss_ecc (float): Time derivative of r computed with post-adiabatic corrections
        m_1 (float): Mass of the primary
        m_2 (float): Mass of the secondary
        chi_1 (float): z-component of the dimensionless spin vector of the primary
        chi_2 (float): z-component of the dimensionless spin vector of the secondary
        eccentricity (float): Desired initial eccentricity
        rel_anomaly (float): Desired relativistic anomaly
        H (Hamiltonian): The Hamiltonian to use (an instance of Hamiltonian class)
        RR (RadiationReactionForceEcc): Function that returns the RR force with eccentric corrections
        params (EOBParams): EOB parameters of the system

    Returns:
        [float]: The desired equations evaluated at u
    """

    # cast an array of 1 element to a float
    pr = np.array(u).item()

    q = np.array([r, 0.0])
    p = np.array([pr, pphi])
    dynamics = H.dynamics(q, p, chi_1, chi_2, m_1, m_2)
    auxderivs = H.auxderivs(q, p, chi_1, chi_2, m_1, m_2)
    hess = H.hessian(q, p, chi_1, chi_2, m_1, m_2)
    dHdr = dynamics[0]
    dHdpr = dynamics[2]
    omega = dynamics[3]  # dH/dL
    H_val = dynamics[4]
    xi = dynamics[5]
    dxidr = auxderivs[3]
    d2Hdr2 = hess[0, 0]
    d2HdrdL = hess[3, 0]

    x = params.ecc_params.omega_avg ** (2.0 / 3.0)
    Kep = np.array([eccentricity, rel_anomaly, x])

    assert isinstance(RR, RadiationReactionForceEcc)
    RR_f = RR.RR(q, p, Kep, omega, omega, H_val, params)

    dotH = dHdpr * RR_f[0] + omega * RR_f[1]

    rdot = (
        dot_r_cons
        + dot_r_diss_ecc
        - dotH * d2HdrdL / (omega * d2Hdr2 + dHdr * (dxidr * omega / xi - d2HdrdL))
    )

    return rdot - xi * dHdpr


def compute_IC_ecc_opt(
    *,
    m_1: float,
    m_2: float,
    chi_1: float,
    chi_2: float,
    eccentricity: float,
    rel_anomaly: float,
    H: Hamiltonian,
    RR: RadiationReactionForceEcc,
    params: EOBParams,
) -> tuple[float, float, float]:
    """
    Compute the initial conditions for an aligned-spin eccentric BBH.

    Args:
        m_1 (float): Mass of the primary
        m_2 (float): Mass of the secondary
        chi_1 (float): z-component of the dimensionless spin vector of the primary
        chi_2 (float): z-component of the dimensionless spin vector of the secondary
        eccentricity (float): Desired initial eccentricity
        rel_anomaly (float): Desired relativistic anomaly
        H (Hamiltonian): The Hamiltonian object to use
        RR (RadiationReactionForceEcc): The RR force to use
        params (EOBParams): EOB parameters of the system

    Returns:
        tuple: Initial values of (r, pphi, pr)
    """

    nu = params.p_params.nu
    delta = params.p_params.delta
    chi_A = params.p_params.chi_A
    chi_S = params.p_params.chi_S
    flags_ecc = params.ecc_params.flags_ecc.copy()
    flags_ecc["flagPA"] = 0  # For the conservative part we put flagPA = 0

    # Step 1: Determine initial guesses for the root-finding procedure

    omega_avg = params.ecc_params.omega_avg
    omega_inst = params.ecc_params.omega_inst

    r0_guess = r_omega_avg_e_z(
        omega_avg, eccentricity, rel_anomaly, nu, delta, chi_A, chi_S, flags_ecc
    )
    pphi0_guess = pphi_omega_avg_e_z(
        omega_avg, eccentricity, rel_anomaly, nu, delta, chi_A, chi_S, flags_ecc
    )
    params.ecc_params.r_start_guess = r0_guess.copy()

    _validate_initial_separation(
        "predicted",
        params=params,
    )

    # Step 2: Compute the initial values of the separation and angular
    # momentum, by solving conservative equations and then adding
    # post-adiabatic contributions

    pr_star = prstar_omega_avg_e_z(
        omega_avg, eccentricity, rel_anomaly, nu, delta, chi_A, chi_S, flags_ecc
    )
    dot_prstar = dot_prstar_omega_avg_e_z(
        omega_avg, eccentricity, rel_anomaly, nu, delta, chi_A, chi_S, flags_ecc
    )

    r0_cons, pphi0_cons = compute_roots_cons(
        r0_guess,
        pphi0_guess,
        pr_star,
        dot_prstar,
        m_1,
        m_2,
        chi_1,
        chi_2,
        omega_inst,
        eccentricity,
        rel_anomaly,
        H,
        params,
    )

    # Validate the initial values from the conservative part of the ICs
    params.ecc_params.r_start_ICs = r0_cons
    _validate_initial_angular_momentum(pphi0_cons, pphi0_guess)
    _validate_initial_separation(
        "computed",
        params=params,
    )

    # Add the dissipative contributions
    flags_ecc["flagPA"] = 1
    r0_diss = r_diss_omega_avg_e_z(omega_avg, eccentricity, rel_anomaly, nu, flags_ecc)
    pphi0_diss = pphi_diss_omega_avg_e_z(
        omega_avg, eccentricity, rel_anomaly, nu, flags_ecc
    )
    r0 = r0_cons + r0_diss
    pphi0 = pphi0_cons + pphi0_diss
    params.ecc_params.r_start_ICs = r0

    # Validate the initial value of 'r' after adding the PA contributions
    params.ecc_params.validate_separation = True
    _validate_initial_separation(
        "computed",
        params=params,
        compare_r=False,
    )

    # Step 3: Compute the initial value of the radial momentum

    flags_ecc["flagPA"] = 0
    dot_r_cons = dot_r_pphi_e_z(
        pphi0_cons, eccentricity, rel_anomaly, nu, delta, chi_A, chi_S, flags_ecc
    )

    flags_ecc["flagPA"] = 1
    dot_r_diss_ecc = dot_r_diss_ecc_pphi_e_z(
        pphi0_cons, eccentricity, rel_anomaly, nu, flags_ecc
    )

    pr0 = compute_root_diss(
        r0=r0,
        pphi0=pphi0,
        dot_r_cons=dot_r_cons,
        dot_r_diss_ecc=dot_r_diss_ecc,
        m_1=m_1,
        m_2=m_2,
        chi_1=chi_1,
        chi_2=chi_2,
        omega_avg=omega_avg,
        eccentricity=eccentricity,
        rel_anomaly=rel_anomaly,
        H=H,
        RR=RR,
        params=params,
    )

    return r0, pphi0, pr0


def compute_roots_cons(
    r0_guess: float,
    pphi0_guess: float,
    pr_star: float,
    dot_prstar: float,
    m_1: float,
    m_2: float,
    chi_1: float,
    chi_2: float,
    omega_inst: float,
    eccentricity: float,
    rel_anomaly: float,
    H: Hamiltonian,
    params: EOBParams,
) -> tuple[float, float]:
    """
    Wrapper for the computation of the roots for the conservative
    equations.

    Args:
        r0_guess (float): Value of the starting separation based on PN
        pphi0_guess (float): Value of the starting angular momentum based on PN
        pr_star (float): Radial momentum in tortoise coordinates
        dot_prstar (float): Time derivative of the radial momentum in tortoise coordinates
        m_1 (float): Mass of the primary
        m_2 (float): Mass of the secondary
        chi_1 (float): z-component of the dimensionless spin vector of the primary
        chi_2 (float): z-component of the dimensionless spin vector of the secondary
        omega_inst (float): Initial instantaneous orbital frequency in geometric units
        eccentricity (float): Desired initial eccentricity
        rel_anomaly (float): Desired relativistic anomaly
        H (Hamiltonian): The Hamiltonian object to use
        params (EOBParams): EOB parameters of the system

    Returns:
        tuple: The initial conditions: (r, pphi)
    """

    r0, pphi0, successfull_root, count = find_root_cons(
        [r0_guess, pphi0_guess],
        pr_star,
        dot_prstar,
        m_1,
        m_2,
        chi_1,
        chi_2,
        omega_inst,
        H,
    )

    if params.ecc_params.IC_messages and count > 1:
        logger.warning(
            f"The root-finding computation was done {count} times "
            "for the parameters: "
            f"m_1 = {m_1}, m_2 = {m_2}, q = {m_1/m_2}, "
            f"chi_1z = {chi_1}, chi_2z = {chi_2}, "
            f"omega_inst = {omega_inst}, "
            f"omega_avg = {params.ecc_params.omega_avg}, "
            f"eccentricity = {eccentricity}, and "
            f"rel_anomaly = {rel_anomaly}. "
        )

    if not successfull_root:
        error_message = (
            "Internal function call failed: Input domain error. "
            "The solution for the conservative part of the initial "
            "conditions failed for all the root-finding methods employed. "
            "If the eccentricity is high, then it is likely that we are "
            "outside the model's regime of validity. Please, review the "
            "physical sense of the input parameters."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    return r0, pphi0


def find_root_cons(
    z: tuple[float, float] | list[float] | np.array,
    pr_star: float,
    dot_prstar: float,
    m_1: float,
    m_2: float,
    chi_1: float,
    chi_2: float,
    omega_inst: float,
    H: Hamiltonian,
    count: int = 0,
) -> tuple[float, float, bool, int]:
    """
    Finds the roots for the conservative equations.
    First, the function tries with the root-finding method 'hybr' with a
    'factor' of 0.01 to avoid giving large steps in the root-finding procedure.
    If this is not successful, then tries the other available methods.
    This function is useful when the root-solver struggles to find a solution.

    Args:
        z (tuple): 2-uple containing the unknowns r and pphi
        pr_star (float): Radial momentum in tortoise coordinates
        dot_prstar (float): Time derivative of the radial momentum in tortoise coordinates
        m_1 (float): Mass of the primary
        m_2 (float): Mass of the secondary
        chi_1 (float): z-component of the dimensionless spin vector of the primary
        chi_2 (float): z-component of the dimensionless spin vector of the secondary
        omega_inst (float): Desired starting instantaneous orbital frequency, in geometric units
        H (Hamiltonian): The Hamiltonian to use (an instance of Hamiltonian class)
        count (int): Counter of attempts to find solutions for the initial conditions

    Returns:
        tuple: (r, pphi, successful_root, count). The initial conditions: (r, pphi),
                whether there was a successful root (successful_root)
                and the number of attempts to find the solution (count)
    """

    successful_root = False

    try:
        count += 1
        res_cons = root(
            IC_cons_ecc,
            z,
            args=(omega_inst, pr_star, dot_prstar, m_1, m_2, chi_1, chi_2, H),
            tol=6e-12,
            method="hybr",
            options={"factor": 0.1, "eps": 1e-3},
        )
        if not res_cons.success:
            # Try the new roots as initial guesses
            r0_guess, pphi0_guess = res_cons.x
            count += 1
            res_cons = root(
                IC_cons_ecc,
                np.array([r0_guess, pphi0_guess]),
                args=(
                    omega_inst,
                    pr_star,
                    dot_prstar,
                    m_1,
                    m_2,
                    chi_1,
                    chi_2,
                    H,
                ),
                tol=6e-12,
                method="hybr",
                options={"factor": 0.1, "eps": 1e-3},
            )
            if not res_cons.success:
                r0, pphi0 = res_cons.x
                raise ValueError

        successful_root = True
        r0, pphi0 = res_cons.x

    except Exception:
        root_methods = [
            "broyden1",
            "diagbroyden",
            "lm",
            "broyden2",
            "anderson",
            "linearmixing",
            "excitingmixing",
            "krylov",
            "df-sane",
        ]
        tol_root = 6e-12
        possible_sols_vals = {}
        possible_sols_roots = {}

        for method in root_methods:
            try:
                count += 1
                res_cons = root(
                    IC_cons_ecc,
                    z,
                    args=(
                        omega_inst,
                        pr_star,
                        dot_prstar,
                        m_1,
                        m_2,
                        chi_1,
                        chi_2,
                        H,
                    ),
                    tol=tol_root,
                    method=method,
                )
                if res_cons.success:
                    successful_root = True
                    break

                # Test if the new roots give a successful guess. If not,
                # then compute the value of the roots and add them in a
                # dictionary
                count += 1
                res_cons_new = root(
                    IC_cons_ecc,
                    res_cons.x,
                    args=(
                        omega_inst,
                        pr_star,
                        dot_prstar,
                        m_1,
                        m_2,
                        chi_1,
                        chi_2,
                        H,
                    ),
                    tol=tol_root,
                )
                if res_cons_new.success:
                    res_cons = res_cons_new
                    successful_root = True
                    break

                root_val = IC_cons_ecc(
                    res_cons.x,
                    omega_inst,
                    pr_star,
                    dot_prstar,
                    m_1,
                    m_2,
                    chi_1,
                    chi_2,
                    H,
                )
                if np.abs(root_val[0]) <= 1e-4 and np.abs(root_val[1]) <= 1e-4:
                    possible_sols_vals[method] = res_cons.x
                    possible_sols_roots[method] = root_val
                else:
                    pass
            except Exception:
                pass

        if successful_root:
            r0, pphi0 = res_cons.x

        elif possible_sols_roots:
            # We take the key from the minimum possible root.
            # We are trusting that, if one of the roots is very small,
            # then the other root will also be small. (At least, is
            # greater than 1e-4, since that is the condition for
            # existence of possible_sols_vals.)
            root_0 = {}
            root_1 = {}
            for key in possible_sols_roots:
                root_0[key] = np.abs(possible_sols_roots[key][0])
                root_1[key] = np.abs(possible_sols_roots[key][1])

            min_root_0_key = min(root_0, key=root_0.get)
            min_root_0 = root_0[min_root_0_key]
            min_root_1_key = min(root_1, key=root_1.get)
            min_root_1 = root_1[min_root_1_key]

            if min_root_0_key == min_root_1_key:
                key_min = min_root_0_key
            elif min_root_0 < min_root_1:
                key_min = min_root_0_key
            elif min_root_1 < min_root_0:
                key_min = min_root_1_key

            r0, pphi0 = possible_sols_vals[key_min]

        # Test if r0 and pphi0 were obtained from the previous functions
        # If not, then set them equal to the initial guesses
        try:
            r0, pphi0
        except Exception:
            r0, pphi0 = z

    return r0, pphi0, successful_root, count


def compute_root_diss(
    r0: float,
    pphi0: float,
    dot_r_cons: float,
    dot_r_diss_ecc: float,
    m_1: float,
    m_2: float,
    chi_1: float,
    chi_2: float,
    omega_avg: float,
    eccentricity: float,
    rel_anomaly: float,
    H: Hamiltonian,
    RR: RadiationReactionForceEcc,
    params: EOBParams,
) -> float:
    """
    Compute the initial value of the radial momentum, including dissipative
    contributions. One can use a PN formula or a resummed formula (default).

    Args:
        r0 (float): Starting separation
        pphi0 (float): Starting angular momentum
        dot_r_cons (float): Time derivative of r computed using the
            conservative dynamics
        dot_r_diss_ecc (float): Time derivative of r computed using
            post-adiabatic corrections
        m_1 (float): Mass of the primary
        m_2 (float): Mass of the secondary
        chi_1 (float): z-component of the dimensionless spin vector of
            the primary
        chi_2 (float): z-component of the dimensionless spin vector of
            the secondary
        omega_avg (float): Orbit-averaged orbital frequency
        eccentricity (float): Desired initial eccentricity
        rel_anomaly (float): Desired relativistic anomaly
        H (Hamiltonian): The Hamiltonian to use (an instance of
            Hamiltonian class)
        RR (RadiationReactionForceEcc): Function that returns the RR force
            with eccentric corrections
        params (EOBParams): EOB parameters of the system

    Returns:
        [float]: The initial radial momentum in tortoise coordinates

    """

    # First, compute a PN estimate
    nu = params.p_params.nu
    delta = params.p_params.delta
    chi_A = params.p_params.chi_A
    chi_S = params.p_params.chi_S
    flags_ecc = params.ecc_params.flags_ecc.copy()
    flags_ecc["flagPA"] = 1  # For the dissipative part, flagPA = 1 always
    pr0_guess_PN = prstar_omega_avg_e_z(
        omega_avg,
        eccentricity,
        rel_anomaly,
        nu,
        delta,
        chi_A,
        chi_S,
        flags_ecc=flags_ecc,
    )

    # Next, select the method to determine the dissipative contribution to pr

    if params.ecc_params.dissipative_ICs == "PN":
        pr0 = pr0_guess_PN

    elif params.ecc_params.dissipative_ICs == "root":
        try:
            res_diss = root_scalar(
                IC_diss_ecc,
                bracket=[-1, 1],
                args=(
                    r0,
                    pphi0,
                    dot_r_cons,
                    dot_r_diss_ecc,
                    m_1,
                    m_2,
                    chi_1,
                    chi_2,
                    eccentricity,
                    rel_anomaly,
                    H,
                    RR,
                    params,
                ),
                xtol=1e-12,
                rtol=1e-10,
            )

            if not res_diss.converged:
                raise ValueError

            pr0 = res_diss.root

        except Exception:
            try:
                # Try another root finder
                res_diss = root(
                    IC_diss_ecc,
                    pr0_guess_PN,
                    args=(
                        r0,
                        pphi0,
                        dot_r_cons,
                        dot_r_diss_ecc,
                        m_1,
                        m_2,
                        chi_1,
                        chi_2,
                        eccentricity,
                        rel_anomaly,
                        H,
                        RR,
                        params,
                    ),
                    tol=6e-12,
                    method="hybr",
                    options={"factor": 0.1, "eps": 1e-3},
                )

                if not res_diss.success:
                    raise ValueError

                pr0 = res_diss.x[0]

            except Exception:
                error_message = (
                    "Internal function call failed: Input domain error. "
                    "The solution for the dissipative part of the initial "
                    "conditions failed. If the eccentricity is high, "
                    "then it is likely that we are outside the model's "
                    "regime of validity. Please, review the physical "
                    "sense of the input parameters."
                )
                logger.error(error_message)
                raise ValueError(error_message)

    else:
        raise ValueError("Choose a valid value for 'dissipative_ICs'.")

    return pr0


def _validate_initial_separation(
    stage: str,
    params: EOBParams,
    compare_r: bool = True,
):
    """
    Tests on the initial value of the relative separation.

    Args:
        stage (str): String indicating whether the agreement between the
            ICs solution and the PN guess
        params (EOBParams): EOB parameters of the system
        compare_r (bool): If False, then avoid doing the test for
            the comparison between computed and predicted 'r'
    """

    # A starting separation below r_min will raise an error
    r_min = params.ecc_params.r_min

    if stage == "predicted":
        r0_guess = params.ecc_params.r_start_guess
        if r0_guess < 6.0:
            # This case is very likely to raise an error
            error_message = (
                "Internal function call failed: Input domain error. "
                "The predicted post-Newtonian conservative initial separation "
                f"(r0 = {r0_guess} M) is smaller than 6 M. "
                "Aborting the waveform generation since the given input "
                "values go outside the validity regime of the model. "
                "Please, review the physical sense of the input parameters."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        elif r0_guess < r_min and params.ecc_params.validate_separation:
            error_message = (
                "Internal function call failed: Input domain error. "
                "The predicted post-Newtonian conservative initial separation "
                f"(r0 = {r0_guess} M) is smaller than the minimum "
                f"separation allowed by the model (r_min = {r_min} M). "
                "Aborting the waveform generation since the given input "
                "values go outside the validity regime of the model. "
                "Please, review the physical sense of the input parameters."
            )
            logger.error(error_message)
            raise ValueError(error_message)

    elif stage == "computed":
        r0 = params.ecc_params.r_start_ICs
        r0_guess = params.ecc_params.r_start_guess

        if r0 < r_min and params.ecc_params.validate_separation:
            error_message = (
                "Internal function call failed: Input domain error. "
                f"The computed initial separation (r0 = {r0} M) is smaller "
                "than the minimum separation allowed by the model "
                f"(r_min = {r_min} M). "
                "Aborting the waveform generation since the given input "
                "values go outside the validity regime of the model. "
                "Please, review the physical sense of the input parameters."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        if abs(1 - r0 / r0_guess) > 0.01 and compare_r:
            error_message = (
                "Internal function call failed: Input domain error. "
                f"The computed conservative initial separation (r0 = {r0} M) "
                "differs from the predicted post-Newtonian value "
                f"(r0_guess = {r0_guess} M) by "
                f"{round(abs(1 - r0 / r0_guess) * 100, 3)}% > 1%. "
                "Aborting the waveform generation since the given input "
                "values go outside the validity regime of the model. "
                "Please, review the physical sense of the input parameters."
            )
            logger.error(error_message)
            raise ValueError(error_message)

    return


def _validate_initial_angular_momentum(
    pphi0: float,
    pphi0_guess_PN: float,
):
    """
    Test on the initial value of the angular momentum 'pphi0'.

    Args:
        pphi0 (float): Initial angular momentum
        pphi0_guess_PN (float): Initial angular momentum computed from PN
    """

    if abs(1 - pphi0 / pphi0_guess_PN) > 0.015:
        error_message = (
            "Internal function call failed: Input domain error. "
            "The computed conservative initial angular momentum "
            f"(pphi0 = {pphi0}) differs from the predicted post-Newtonian "
            f"value (pphi0_guess_PN = {pphi0_guess_PN}) by "
            f"{round(abs(1 - pphi0 / pphi0_guess_PN) * 100, 3)}% > 1.5%. "
            "Aborting the waveform generation since the given input "
            "values go outside the validity regime of the model. "
            "Please, review the physical sense of the input parameters."
        )
        logger.error(error_message)
        raise ValueError(error_message)

    return
