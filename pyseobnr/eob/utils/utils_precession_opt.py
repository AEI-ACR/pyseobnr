"""
Additional utility functions to manipulate various aspects of precession dynamics.
"""

from math import cos, sqrt
from typing import Any, Dict

import lal
import numpy as np
import quaternion
from numba import *
from numba import jit, types
from scipy.interpolate import CubicSpline
from scri import WaveformModes

from ..fits.EOB_fits import compute_QNM
from ..fits.fits_Hamiltonian import dSO as dSO_poly_fit
from ..hamiltonian import Hamiltonian
from .math_ops_opt import my_cross, my_dot, my_norm

#################################################################################################################

###########              FUNCTIONS TO  APPLY THE MERGER RINGDOWN APPROXIMATION FOR THE ANGLES          ##########

#################################################################################################################


# This function does exactly the same as the LAL counterpart
def SEOBBuildJframeVectors(Jhat_final: np.ndarray):
    """
    This function computes the Jframe unit vectors, with e3J along Jhat.
    Convention: if (ex, ey, ez) is the initial I-frame, e1J chosen such that ex
    is in the plane (e1J, e3J) and ex.e1J>0.
    In the case where e3J and x happen to be close to aligned, we continuously
    switch to another prescription with y playing the role of x.

    Same operation as in SEOBBuildJframeVectors in
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRSpinPrecEOBv4P.c#L2833
    LALSimIMRSpinPrecEOBv4P.c.
    See Appendix A of https://arxiv.org/abs/1607.05661v1
    and Sec. III B of https://arxiv.org/pdf/2004.09442v1.pdf for details.

    Args:
        Jhat_final(np.ndarray): Final total angular momentum vector

    Returns:
        (tuple): Triad of unit vectors defining the J-frame
    """

    e3J = Jhat_final

    exvec = np.array([1.0, 0.0, 0.0])
    eyvec = np.array([0.0, 1.0, 0.0])

    exdote3J = my_dot(exvec, e3J)
    eydote3J = my_dot(eyvec, e3J)

    lambda_fac = 1.0 - abs(exdote3J)

    if lambda_fac < 0.0 or lambda_fac > 1.0:
        print("Problem: lambda=1-|e3J.ex|=%f, should be in [0,1]" % lambda_fac)

    elif lambda_fac > 1e-4:
        normfacx = 1.0 / sqrt(1.0 - exdote3J * exdote3J)

        e1J = (exvec - exdote3J * e3J) / normfacx
    elif lambda_fac < 1e-5:
        normfacy = 1.0 / sqrt(1.0 - eydote3J * eydote3J)
        e1J = (eyvec - eydote3J * e3J) / normfacy
    else:
        weightx = (lambda_fac - 1e-5) / (1e-4 - 1e-5)
        weighty = 1.0 - weightx
        normfacx = 1.0 / sqrt(1.0 - exdote3J * exdote3J)
        normfacy = 1.0 / sqrt(1.0 - eydote3J * eydote3J)
        e1J = (
            weightx * (exvec - exdote3J * e3J) / normfacx
            + weighty * (eyvec - eydote3J * e3J) / normfacy
        )

        e1Jblendednorm = my_norm(e1J)
        e1J /= e1Jblendednorm

    # Get e2J = e3J x e1J
    e2J = my_cross(e3J, e1J)

    e1Jnorm = my_norm(e1J)
    e2Jnorm = my_norm(e2J)
    e3Jnorm = my_norm(e3J)

    e1J /= e1Jnorm
    e2J /= e2Jnorm
    e3J /= e3Jnorm
    # print(f"e1J:{e1J},e2J:{e2J},e3J:{e3J}")
    return e1J, e2J, e3J


def SEOBEulerJ2PFromDynamics(
    t: np.ndarray,
    omega: np.ndarray,
    LNhat: np.ndarray,
):
    """
    This function computes the Euler angles from J-frame to P-frame given the
    Newtonian orbital angular momentum timeseries LNhat(t) in the J frame.
    Similar operation as SEOBEulerJ2PFromDynamics in LALSimIMRSpinPrecEOBv4P.c.
    See Appendix A of https://arxiv.org/abs/1607.05661v1
    and Sec. III B of https://arxiv.org/pdf/2004.09442v1.pdf for details.

    Args:
        t (np.ndarray): Time array
        omega (np.ndarray): EOB orbital frequency
        LNhat (np.ndarray): LNhat vector array array (in J frame!)

    Returns:
        (tuple): Time-dependent quaterions describing the rotation from the co-precessing frame to the J-frame

    """

    LN_in_J_quat = quaternion.as_quat_array(np.c_[np.zeros(len(LNhat)), LNhat])
    # Compute dot product between LNhat and Jf (remember we are in J frame!)
    dtprd = np.einsum("ij,j->i", LNhat, np.array([0, 0, 1.0]))
    # Check if at any point these vectors are aligned or anti-aligned
    # If this is the case we will need to be careful when constructing the
    # rotation from J frame to P frame to avoid numerical issues
    # If the vectors are almost colinear we split the rotation into 2
    # steps: a rotation of Jf into an auxiliary vector aux and a
    # rotation of aux into LN. This ensures that the rotation is smooth
    # without artefacts to due to roundoff
    if np.amin(np.abs(np.abs(dtprd) - 1)) < 1e-3:
        # Pick the x axis of the J frame as the auxiliary vector
        # This should be a good choice since it's perpendicular to
        # final spin
        aux = quaternion.quaternion(0.0, 1.0, 0.0, 0.0)
        # Still, check if LN is degenerate with x axis of the J frame
        dtprd2 = np.einsum("ij,j->i", LNhat, np.array([1, 0, 0.0]))
        if np.amin(np.abs(np.abs(dtprd2) - 1)) < 1e-3:
            # If not x, then y
            aux = quaternion.quaternion(0.0, 0.0, 1.0, 0.0)
    else:
        aux = None

    # Note that because we are working in the J frame the final spin
    # direction is just along z axis, by definition.
    quatJ2P = minimal_quat(quaternion.z, LN_in_J_quat, aux, t, omega)

    return quatJ2P


def SEOBRotatehIlmFromhJlm_opt_v1(
    w_hJlm: WaveformModes,
    modes_lmax: int,
    quatI2J: quaternion,
) -> WaveformModes:
    """
     This function computes the hIlm Re/Im timeseries (fixed sampling) from hJlm
     Re/Im timeseries (same sampling). This is a simple rotation,
     sample-by-sample, with constant Wigner coefficients.

     Args:
        w_hJlm (WaveformModes): WaveformModes object from the scri python package. It contains waveform modes in the final J-frame
        modes_lmax (int): Maximum value of l in modes (l,m)
        quatI2J (float): Quaternion between the I-frame and the J-frame

    Returns:
        (WaveformModes): WaveformModes scri object in the inertial frame

    """
    res = w_hJlm.rotate_decomposition_basis(~quatI2J)

    return res


def seobnrv4P_quaternionJ2P_postmerger_extension(
    t_full: np.ndarray,
    precRate: float,
    euler_angles_attach: np.ndarray,
    euler_angles_derivative_attach: np.ndarray,
    t_attach: float,
    idx: int,
    rd_approx: bool,
    rd_smoothing: bool,
    beta_approx: int = 0,
):
    """
     This function computes the ringdown approximation of the Euler angles in
     the J-frame, assuming simple precession around the final J-vector with a
     precession frequency approximately equal to the differences between the
     lowest overtone of the (2,2) and (2,1) QNM frequencies.

     Args:
        t_full (np.ndarray): Time array of the waveform modes.
        precRate (float): Precessing rate,  differences between the lowest overtone of the
                          (2,2) and (2,1) QNM frequencies. See Eq. (3.4) of 2004.09442.
        euler_angles_attach (np.ndarray): Euler angles (alpha,beta,gamma) at the attachment time.
        euler_angles_derivative_attach (np.ndarray): Time derivative of the Euler angles (alpha,beta,gamma) at the
            attachment time.
        t_attach (float): Attachment time of the dynamics and the ringdown
        idx (int): Index at which the attachment of the ringdown is performed
        rd_approx (bool): If True apply the approximation of the Euler angles, if False use constant angles
        rd_smoothing (bool): If True apply smoothing of the Euler angles, if False do not apply the smoothing
        beta_approx (int): If 0 use constant beta angle, otherwise use small opening angle approximation

    Returns:
        (quaternion.quaternion): Quaternion describing the ringdown approximation of the Euler angles in the J-frame
    """
    alphaAttach, betaAttach, gammaAttach = euler_angles_attach
    t_RD = t_full[idx[-1] + 1 :]

    # Approximate the Euler angles assuming simple precession
    if rd_approx:
        if rd_smoothing:
            # Smoothing: dalpha/dt and dgamma/dt are windowed with a Tanh function. The expressions for alpha and gamma
            # are the integral of the windowed derivative
            idx_smoothing = (
                60  # The window is expensive so only applied on a subset of points
            )

            t_smooth = t_full[idx[-1] + 1 : idx[-1] + 1 + idx_smoothing] - t_attach

            # Parameters needed for the window
            dalphaAttach, dbetaAttach, dgammaAttach = euler_angles_derivative_attach
            dalphaRD = precRate
            dgammaRD = -cos(betaAttach) * precRate
            sigma = 2.5
            ti = 2.0

            # Offsets to ensure continuity
            t_offset = 0.0  # t_smooth[0]
            alpha_smooth_offset = (
                0.5 * dalphaAttach * t_offset
                + 0.5 * dalphaRD * t_offset
                + (-0.125 * dalphaAttach * sigma + 0.125 * dalphaRD * sigma)
                * np.log(np.cosh((4.0 * (t_offset - 1.0 * ti)) / sigma))
            )
            gamma_smooth_offset = (
                0.5 * dgammaAttach * t_offset
                + 0.5 * dgammaRD * t_offset
                + (-0.125 * dgammaAttach * sigma + 0.125 * dgammaRD * sigma)
                * np.log(np.cosh((4.0 * (t_offset - 1.0 * ti)) / sigma))
            )

            # alpha and gamma smoothened
            alphaJ2P = (
                alphaAttach
                - alpha_smooth_offset
                + 0.5 * dalphaAttach * t_smooth
                + 0.5 * dalphaRD * t_smooth
                + (-0.125 * dalphaAttach * sigma + 0.125 * dalphaRD * sigma)
                * np.log(np.cosh((4.0 * (t_smooth - 1.0 * ti)) / sigma))
            )
            gammaJ2P = (
                gammaAttach
                - gamma_smooth_offset
                + 0.5 * dgammaAttach * t_smooth
                + 0.5 * dgammaRD * t_smooth
                + (-0.125 * dgammaAttach * sigma + 0.125 * dgammaRD * sigma)
                * np.log(np.cosh((4.0 * (t_smooth - 1.0 * ti)) / sigma))
            )

            # Time array for evaluating the unwindowed RD approximation
            t_RD = t_full[idx[-1] + idx_smoothing + 1 :]

            # Ensure continuity between windowed and unwindowed regions
            t_attach_2 = t_RD[0] - t_attach
            alphaAttach2 = (
                alphaAttach
                - alpha_smooth_offset
                + 0.5 * dalphaAttach * t_attach_2
                + 0.5 * dalphaRD * t_attach_2
                + (-0.125 * dalphaAttach * sigma + 0.125 * dalphaRD * sigma)
                * np.log(np.cosh((4.0 * (t_attach_2 - 1.0 * ti)) / sigma))
            )
            gammaAttach2 = (
                gammaAttach
                - gamma_smooth_offset
                + 0.5 * dgammaAttach * t_attach_2
                + 0.5 * dgammaRD * t_attach_2
                + (-0.125 * dgammaAttach * sigma + 0.125 * dgammaRD * sigma)
                * np.log(np.cosh((4.0 * (t_attach_2 - 1.0 * ti)) / sigma))
            )

            # Evaluate unwindowed ringdown approximation
            t_attach_RD = t_RD[0]
            t_full += t_full[0]

            cosbetaAttach = cos(betaAttach)
            tmp = (t_RD - t_attach_RD) * precRate
            alphaJ2P = np.append(alphaJ2P, alphaAttach2 + tmp)

            betaJ2P = np.ones(len(t_smooth) + len(t_RD)) * betaAttach
            # betaJ2P = betaAttach + (t_full[idx[-1] + 1 :] - t_attach) * dbetaAttach

            gammaJ2P = np.append(
                gammaJ2P, gammaAttach2 - cosbetaAttach * (t_RD - t_attach_RD) * precRate
            )

        else:
            t_RD = t_full[idx[-1] + 1 :]
            t_full += t_full[0]

            cosbetaAttach = cos(betaAttach)
            tmp = (t_RD - t_attach) * precRate
            alphaJ2P = alphaAttach + tmp

            if beta_approx:
                y = np.exp(-tmp)
                betaJ2P = -2.0 * np.arctan2(2 * y, 1.0) + betaAttach
            else:
                betaJ2P = np.ones(len(t_RD)) * betaAttach

            gammaJ2P = gammaAttach - cosbetaAttach * (t_RD - t_attach) * precRate

        alphaJ2P = np.unwrap(alphaJ2P)
        betaJ2P = np.unwrap(betaJ2P)
        gammaJ2P = np.unwrap(gammaJ2P)

    else:
        t_RD = t_full[idx[-1] + 1 :]
        alphaJ2P = np.ones(len(t_RD)) * alphaAttach
        betaJ2P = np.ones(len(t_RD)) * betaAttach
        gammaJ2P = np.ones(len(t_RD)) * gammaAttach

    euler_angles_v1 = np.transpose([alphaJ2P, betaJ2P, gammaJ2P])
    quat_postMerger = quaternion.from_euler_angles(
        euler_angles_v1, beta=None, gamma=None
    )

    return quat_postMerger


def minimal_rotation_mine(
    R: quaternion.quaternion,
    t: quaternion.quaternion,
    v: quaternion.quaternion,
    omega: np.ndarray,
    domega: np.ndarray,
    iterations: int = 2,
):
    """
    Adjust frame so that there is no rotation about z' axis.
    The output of this function is a frame that rotates the z axis onto the same z' axis as the
    input frame, but with minimal rotation about that axis.  This is done by pre-composing the input
    rotation with a rotation about the z axis through an angle gamma, where
        dgamma/dt = 2*(dR/dt * z * R.conjugate()).w
    This ensures that the angular velocity has no component along the z' axis.
    Note that this condition becomes easier to impose the closer the input rotation is to a
    minimally rotating frame, which means that repeated application of this function improves its
    accuracy.  By default, this function is iterated twice, though a few more iterations may be
    called for.

    Args:
        R (quaternion.quaternion) : Quaternion describing rotation.
        t (quaternion.quaternion) : Quaternion times at which R is measured.
        v (quaternion.quaternion) : Corresponding times at which R is measured.
        omega (np.ndarray): Orbital frequency array from EOB dynamics.
        domega (np.ndarray): Time derivative of the orbital frequency array from EOB dynamics.
        iterations (int):  Repeat the minimization to refine the result. Defaults to 2.

    Returns:
        (quaternion.quaternion): Quaternion with the minimum rotation condition applied on it.

    """
    from scipy.interpolate import CubicSpline

    if iterations == 0:
        return R
    R = quaternion.as_float_array(R)

    # To compute dR/dt we first compute dR/domega
    # and then dR/dt = dR/domega*domega/dt
    # This is done to reduce the noise in the spline
    # derivative
    dRdomega = CubicSpline(omega, R).derivative()(omega)

    Rdot = dRdomega * domega[:, None]
    R = quaternion.from_float_array(R)
    Rdot = quaternion.from_float_array(Rdot)
    halfgammadot = quaternion.as_float_array(Rdot * v * np.conjugate(R))[:, 0]
    halfgamma = CubicSpline(t, halfgammadot).antiderivative()(t)
    Rgamma = np.exp(v * halfgamma)
    return minimal_rotation_mine(
        R * Rgamma, t, v, omega, domega, iterations=iterations - 1
    )


def minimal_quat(
    v_final: quaternion.quaternion,
    V: quaternion.quaternion,
    aux: quaternion.quaternion,
    t_dyn: np.ndarray,
    omega: np.ndarray,
):
    """
    Compute the rotation that aligns V with v_final
    and obeys the minimal rotation condition.

    Args:
        v_final (quaternion.quaternion): z-axis quaternion of the final spin frame.
        V (quaternion.quaternion): Quaternion of LNhat in the final spin frame.
        aux (quaternion.quaternion): An optional auxiliary quaternion to help with numerical stability.
        t_dyn (np.ndarray): Time array from EOB dynamics.
        omega (np.ndarray): Orbital frequency array from EOB dynamics.

    Returns:
        (np.ndarray): Array of quaternions
    """
    omega_intrp = CubicSpline(t_dyn, omega)
    domega = omega_intrp.derivative()(t_dyn)
    if aux is not None:
        step1 = np.sqrt(-aux * v_final)
        step2 = np.sqrt(-V * aux)
        ttl = step2 * step1
    else:
        ttl = np.sqrt(-V * v_final)

    ttl_min = minimal_rotation_mine(ttl, t_dyn, v_final, omega, domega)
    return ttl_min


def inspiral_merger_quaternion_angles(
    t_dynamics: np.ndarray,
    omega_dynamics: np.ndarray,
    t_attach: float,
    Lvec_hat_attach: np.ndarray,
    Jfhat_attach: np.ndarray,
    splines: Dict[Any, Any],
    t_ref: float = None,
):
    """Wrapper function to compute the angles/quaternions necessary to perform the rotations
    from the co-precessing frame (P-frame) to the observer inertial frame (I-frame) passing
    through the final angular momentum frame (J-frame) for the inspiral part of the waveform

    Args:
        t_dynamics (np.ndarray): Time array for dynamics
        omega_dynamics (np.ndarray): EOB orbital frequency
        t_attach (float): Attachment time
        Lvec_hat_attach (np.ndarray): LN at attachment, I frame
        Jfhat_attach (np.ndarray): final spin direction, I frame
        splines (Dict[Any,Any]): Dictionary of splines
        t_ref (float, optional): Reference time if f_ref!=f_min. Defaults to None.

    Returns:
        (tuple): Quantities required to perform the rotations from the co-precessing to the inertial frame (time of
                of the dynamics, time dependent quaternion from the P-frame to the J-frame, Euler angles from the
                J-frame to the I-frame, and the quaternions during the inspiral in the J-frame)

    """

    # Apply ringdown approximation at the attachment time
    Lf_hat_v5 = Lvec_hat_attach
    Jf_hat_v5 = Jfhat_attach

    # Compute e1J, e2J, e3J triad
    e1J, e2J, e3J = SEOBBuildJframeVectors(Jf_hat_v5)
    # LN on the final dynamics time grid
    tmp_LN = splines["everything"](omega_dynamics)[:, 10:13]

    # Normalize LN_hat to ensure working with unit quaternions
    tmp_LN_norm = np.sqrt(np.einsum("ij,ij->i", tmp_LN, tmp_LN))
    tmp_LN = (tmp_LN.T / tmp_LN_norm).T

    # LN as expressed in J frame
    Ze1J = np.dot(tmp_LN, e1J)
    Ze2J = np.dot(tmp_LN, e2J)
    Ze3J = np.dot(tmp_LN, e3J)
    LN_in_J = np.c_[Ze1J, Ze2J, Ze3J]

    # Compute the Euler angles from the inertial frame to the final J-frame  (I2J angles)
    rotJ_matrix = np.vstack([e1J, e2J, e3J]).T
    quatI2J = quaternion.from_rotation_matrix(rotJ_matrix)

    # The inverse transformation, J->I
    quatJ2I = quatI2J.conjugate()
    alpha_J2I, beta_J2I, gamma_J2I = quaternion.as_euler_angles(quatJ2I)

    quatJ2P_dyn = SEOBEulerJ2PFromDynamics(t_dynamics, omega_dynamics, LN_in_J)
    # Compute Euler angles from the quaternions
    # eAngles_min = quaternion.as_euler_angles(quatJ2P_dyn).T

    # gammaJ2P_dyn = np.unwrap(eAngles_min[2])

    # We still have the residual freedom from minimal rotation condition to perform a
    # global rotation around the J_f . We fix this freedom by demanding that
    # at the reference time, the *I2P* map is the identity, i.e. the
    # inertial frame and co-precessing frame are the same. To do so,
    # we only need to ensure that at t_ref, gammaJ2P is the same as
    # gamma_J2I.
    # NB: due to quaternions being a double cover of SO(3), q and -q
    # are the same rotation.

    if t_ref is None:
        # gamma_ref = gammaJ2P_dyn[0]
        q2JP_ref = quatJ2P_dyn[0]
    else:
        # igammaJ2P_dyn = CubicSpline(t_dynamics, gammaJ2P_dyn)
        # gamma_ref = igammaJ2P_dyn(t_ref)
        q2JP_ref = quaternion.squad(quatJ2P_dyn, t_dynamics, t_ref)

    # We need to shift things so that gamma_J2P = gamma_J2I at reference time
    # The following is a rotation around final J
    # quatJ2P_dyn2 = quatJ2P_dyn*np.exp((-gamma_ref / 2 + gamma_J2I / 2) * quaternion.z)

    shift = q2JP_ref.conjugate() * quatJ2I
    quatJ2P_dyn *= shift

    eAngles_min = quaternion.as_euler_angles(quatJ2P_dyn).T
    alphaJ2P_dyn = np.unwrap(eAngles_min[0])
    betaJ2P_dyn = np.unwrap(eAngles_min[1])
    gammaJ2P_dyn = np.unwrap(eAngles_min[2])

    ialphaJ2P_dyn = CubicSpline(t_dynamics, alphaJ2P_dyn)
    ibetaJ2P_dyn = CubicSpline(t_dynamics, betaJ2P_dyn)
    igammaJ2P_dyn = CubicSpline(t_dynamics, gammaJ2P_dyn)

    alpha_attach = ialphaJ2P_dyn(t_attach)
    beta_attach = ibetaJ2P_dyn(t_attach)
    gamma_attach = igammaJ2P_dyn(t_attach)

    euler_angles_attach = [alpha_attach, beta_attach, gamma_attach]
    euler_angles_derivative_attach = [
        ialphaJ2P_dyn.derivative()(t_attach),
        ibetaJ2P_dyn.derivative()(t_attach),
        igammaJ2P_dyn.derivative()(t_attach),
    ]

    # Compute sign from the angle between the final total and orbital angular momenta to check if we
    # are in the prograde or retrograde case
    cos_angle = my_dot(Jf_hat_v5, Lf_hat_v5)
    flip = 1
    if cos_angle < 0:
        flip = -1

    return (
        t_dynamics,
        quatJ2P_dyn,
        quatI2J,
        euler_angles_attach,
        euler_angles_derivative_attach,
        flip,
    )


def compute_omegalm_P_frame(omegalm: complex, m: int, rotation_term: float):
    """
    Function to compute the co-precessing frame QNM frequencies
    from the J-frame QNMs as described in arXiv:2301.06558.

    Args:
        omegalm (complex): QNM frequency in the J-frame
        m (int): m index of the (l,m) waveform multipole
        rotation_term (float): (1-abs(cos[beta]))*precRate, where the precRate is the difference between the lowest overtone
                                of the 22 and 21 QNM frequencies

    Returns:
        (complex): QNM frequency in the P-frame
    """

    omega_complex_P_frame = omegalm - m * rotation_term

    return omega_complex_P_frame


@jit(nopython=True, cache=True)
def custom_swsh(beta: np.ndarray, gamma: np.ndarray, lmax: int):
    """
    Function to compute the spin-weighted spherical harmonics necessary to
    compute the polarizations in the inertial frame from the co-precessing
    frame modes [(2,|2|),(2,|1|),(3,|3|),(3,|2|),(4,|4|),(4,|3|),(5,|5|)].

    Args:
        beta (np.ndarray): Euler angle beta between the co-precessing frame and the inertial frame
        gamma (np.ndarray): Euler angle gamma between the co-precessing frame and the inertial frame
        lmax (int): Maximum ell to use

    Returns:
        (dict): Dictionary containing specific values of the spin-weighted spherical harmonics
    """

    cBH = np.cos(beta / 2.0)
    sBH = np.sin(beta / 2.0)

    cBH2 = cBH * cBH
    cBH3 = cBH2 * cBH
    cBH4 = cBH3 * cBH

    sBH2 = sBH * sBH
    sBH3 = sBH2 * sBH
    sBH4 = sBH3 * sBH

    expGamma = np.exp(1j * gamma)
    expGamma2 = expGamma * expGamma

    swsh = {}

    swsh[2, 2] = 0.5 * np.sqrt(5.0 / np.pi) * cBH4 * expGamma2
    swsh[2, 1] = np.sqrt(5.0 / np.pi) * cBH3 * sBH * expGamma
    swsh[2, -2] = 0.5 * np.sqrt(5.0 / np.pi) * sBH4 / expGamma2
    swsh[2, -1] = np.sqrt(5.0 / np.pi) * sBH3 * cBH / expGamma

    if lmax >= 3:
        cBH5 = cBH4 * cBH
        sBH5 = sBH4 * sBH
        expGamma3 = expGamma2 * expGamma
        swsh[3, 3] = -np.sqrt(10.5 / np.pi) * cBH5 * sBH * expGamma3
        swsh[3, 2] = (
            0.25 * np.sqrt(7.0 / np.pi) * cBH4 * (6.0 * (cBH2 - sBH2) - 4.0) * expGamma2
        )
        swsh[3, -3] = np.sqrt(10.5 / np.pi) * sBH5 * cBH / expGamma3
        swsh[3, -2] = (
            0.25 * np.sqrt(7.0 / np.pi) * sBH4 * (6.0 * (cBH2 - sBH2) + 4.0) / expGamma2
        )

    if lmax >= 4:
        cBH6 = cBH5 * cBH
        sBH6 = sBH5 * sBH
        expGamma4 = expGamma3 * expGamma
        swsh[4, 4] = 3.0 * np.sqrt(7.0 / np.pi) * cBH6 * sBH2 * expGamma4
        swsh[4, 3] = (
            -0.75
            * np.sqrt(3.5 / np.pi)
            * cBH5
            * sBH
            * (8.0 * (cBH2 - sBH2) - 4.0)
            * expGamma3
        )
        swsh[4, -4] = 3.0 * np.sqrt(7.0 / np.pi) * sBH6 * cBH2 / expGamma4
        swsh[4, -3] = (
            0.75
            * np.sqrt(3.5 / np.pi)
            * sBH5
            * cBH
            * (8.0 * (cBH2 - sBH2) + 4.0)
            / expGamma3
        )

    if lmax == 5:
        cBH7 = cBH6 * cBH
        sBH7 = sBH6 * sBH
        expGamma5 = expGamma4 * expGamma
        swsh[5, 5] = -np.sqrt(330.0 / np.pi) * cBH7 * sBH3 * expGamma5
        swsh[5, -5] = np.sqrt(330.0 / np.pi) * sBH7 * cBH3 / expGamma5

    return swsh


def interpolate_quats(
    quat: quaternion.quaternion, t_intrp: np.ndarray, t_full: np.ndarray
):
    """
    Function to interpolate the quaternions from the co-precessing frame to the final
    J-frame into the equal-spacing and finer time grid of the waveform modes

    Args:
        quat (quaternion): Quaternion from the co-precessing frame to the final J-frame
        t_intrp (np.ndarray): Time array of the EOB dynamics
        t_full (np.ndarray): Time array of waveform modes


    Returns:
        (quaternion): Quaternion interpolated to the finer time grid of the waveform modes
    """
    angles = quaternion.as_euler_angles(quat)
    sp = CubicSpline(t_intrp, np.unwrap(angles.T).T)
    intrp_angles = sp(t_full)
    return quaternion.from_euler_angles(intrp_angles)
