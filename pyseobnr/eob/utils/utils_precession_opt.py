import numpy as np
from numba import *
from numba import jit
from numba import types

from ..fits.EOB_fits import compute_QNM
from .math_ops_opt import my_cross,my_dot,my_norm
from ..fits.fits_Hamiltonian import dSO as dSO_poly_fit
from ..hamiltonian import Hamiltonian
from scri import WaveformModes

import lal
import quaternion
from math import cos,sqrt

from scipy.interpolate import CubicSpline

def compute_spins_EOBdyn_opt(omega: np.ndarray, splines:dict):
    """Function to evaluate the splines of the projections
    of the spins onto the unit Newtonian orbital angular
    momentum (LN_hat) and unit the orbital angular momentum (L_hat),
    as well the components of the spin vectors and the LN_hat
    vector on a grid of the orbital frequency of the non-
    precessing EOB evolution

    Args:
        omega (np.ndarray): Orbital frequency from the non-precessing EOB evolution
        splines (dict): Dictionary containing the splines in orbital frequency
                        of the vector components of the spins, LN_hat and L_hat as well as
                        the spin projections onto LN_hat and L_hat

    Returns:
        (tuple): Dimensionless spin vectors and LN_hat, as well as the projections of the
                dimensionless spins onto LN_hat and L_hat
    """


    tmp = splines["everything"](omega)

    chi1LN_EOB = tmp[:,0]
    chi2LN_EOB = tmp[:,1]

    chi1L_EOB = tmp[:,2]
    chi2L_EOB = tmp[:,3]

    chi1v_EOB = tmp[:,4:7]
    chi2v_EOB = tmp[:,7:10]
    LN_EOB = tmp[:,10:13]

    return chi1L_EOB, chi2L_EOB, chi1LN_EOB, chi2LN_EOB, chi1v_EOB, chi2v_EOB,LN_EOB

def augment_dynamics_precessing_opt(dynamics: np.ndarray,
                                    chi1L_EOB: np.ndarray, chi2L_EOB: np.ndarray,
                                    chi1LN_EOB: np.ndarray, chi2LN_EOB: np.ndarray,
                                    chi1v_EOB: np.ndarray, chi2v_EOB: np.ndarray,
                                    m_1:float, m_2:float,
                                    H: Hamiltonian):
    """Compute dynamical quantities we need for the waveform (low sampling rate dynamics)

    Args:
        dynamics (np.ndarray): The dynamics array: t,r,phi,pr,pphi
        chi1L_EOB (np.ndarray): Projection of the primary spin vector onto the L vector
        chi2L_EOB (np.ndarray): Projection of the seconday spin vector onto the L vector
        chi1LN_EOB (np.ndarray): Projection of the primary spin vector onto the LN vector
        chi2LN_EOB (np.ndarray): Projection of the seconday spin vector onto the LN vector
        chi1v_EOB (np.ndarray): Spin vector of the primary
        chi2v_EOB (np.ndarray): Spin vector of the secondary
        m_1 (float): Mass component of the primary
        m_2 (float): Mass component of the secondary
        H (Hamiltonian): Hamiltonian class


    Returns:
        (np.ndarray): Dynamical variables (r,phi,pr,pphi), as well as the Hamiltonian, orbital frequency,
                      circular orbital frequency, and the projections of the dimensionless spin vectors LN_hat
    """

    ms = m_1 + m_2
    nu = m_1*m_2/(m_1+m_2)**2
    X1 = m_1/ms
    X2 = m_2/ms

    # Compute dSO
    ap  = X1*chi1LN_EOB + X2*chi2LN_EOB
    am  = X1*chi1LN_EOB - X2*chi2LN_EOB
    dSO_new = dSO_poly_fit(nu, ap, am)

    result = []
    p_c = np.zeros(2)
    for i, row in enumerate(dynamics):
        q = row[1:3]
        p = row[3:5]
        chi1_L = chi1L_EOB[i]
        chi2_L = chi2L_EOB[i]
        chi1_LN = chi1LN_EOB[i]
        chi2_LN = chi2LN_EOB[i]
        chi1_v = chi1v_EOB[i]
        chi2_v = chi2v_EOB[i]

        # Evaluate a few things: H, omega,omega_circ
        H.calibration_coeffs['dSO'] = dSO_new[i]

        dyn = H.dynamics(q, p, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L)
        omega = dyn[3]
        H_val = dyn[4]
        p_c[1] = p[1]
        omega_c = H.omega(q, p_c, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L)

        result.append([H_val, omega, omega_c, chi1_LN, chi2_LN])
    result = np.array(result)


    return np.c_[dynamics,result]


def augment_dynamics_precessing_fine_opt(
    dynamics: np.ndarray,
    chi1L_EOB: np.ndarray,
    chi2L_EOB: np.ndarray,
    chi1LN_EOB: np.ndarray,
    chi2LN_EOB: np.ndarray,
    chi1v_EOB: np.ndarray,
    chi2v_EOB: np.ndarray,
    m_1: float,
    m_2: float,
    H:Hamiltonian,
):
    """Compute dynamical quantities we need for the waveform (fine sampling rate dynamics)

    Args:
        dynamics (np.ndarray): The dynamics array: t,r,phi,pr,pphi
        chi1L_EOB (np.ndarray): Projection of the primary spin vector onto the L vector
        chi2L_EOB (np.ndarray): Projection of the seconday spin vector onto the L vector
        chi1LN_EOB (np.ndarray): Projection of the primary spin vector onto the LN vector
        chi2LN_EOB (np.ndarray): Projection of the seconday spin vector onto the LN vector
        chi1v_EOB (np.ndarray): Spin vector of the primary
        chi2v_EOB (np.ndarray): Spin vector of the secondary
        m_1 (float): Mass component of the primary
        m_2 (float): Mass component of the secondary
        H (class): Hamiltonian class


    Returns:
        (np.ndarray): Dynamical variables (r,phi,pr,pphi), as well as the Hamiltonian, orbital frequency,
                      circular orbital frequency, and the projections of the dimensionless spin vectors LN_hat
    """
    t = dynamics[:, 0]
    N = len(t)
    ms = m_1 + m_2
    nu = m_1 * m_2 / (m_1 + m_2) ** 2
    X1 = m_1 / ms
    X2 = m_2 / ms
    result = []
    idx_nan = N
    count = 0
    p_c = np.zeros(2)
    ap = X1 * chi1LN_EOB + X2 * chi2LN_EOB
    am = X1 * chi1LN_EOB - X2 * chi2LN_EOB

    dSO_new = dSO_poly_fit(nu, ap, am)
    rs = [dynamics[0,1]]
    for i, row in enumerate(dynamics):
        q = row[1:3]
        p = row[3:5]
        chi1_L = chi1L_EOB[i]
        chi2_L = chi2L_EOB[i]

        chi1_LN = chi1LN_EOB[i]
        chi2_LN = chi2LN_EOB[i]

        chi1_v = chi1v_EOB[i]
        chi2_v = chi2v_EOB[i]

        r = q[0]
        H.calibration_coeffs["dSO"] = dSO_new[i]
        p_c[1] = p[1]
        # Evaluate a few things: H, omega,omega_circ
        dyn = H.dynamics(
            q, p, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L
        )
        omega = dyn[3]
        H_val = dyn[4]

        # Use some if statements to stop evaluating omega when the dynamics becomes unphysical
        if np.isnan(omega) or omega>1:
            count += 1
            idx_max = i
            break

        if rs[-1]>=r:
            omega_c = H.omega(
                q, p_c, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L
            )
            if np.isnan(omega_c) or omega_c>1:
                count += 1
                idx_max = i

                break


            result.append([H_val, omega, omega_c, chi1_LN, chi2_LN])
            idx_max = i
        else:
            count += 1
            idx_max = i
            break
        rs.append(r)

    result = np.array(result)

    # Remove part of the dynamics which becomes unphysical (typically at small separations < 2M)
    if count >0:
        dynamics1 = dynamics[:idx_max]
    else:
        dynamics1 = dynamics[:len(result)]

    return np.c_[dynamics1, result]


def project_spins_augment_dynamics_opt(
    m_1: float,
    m_2: float,
    H:Hamiltonian,
    omega_low: np.ndarray,
    omega_fine: np.ndarray,
    dynamics_low: np.ndarray,
    dynamics_fine: np.ndarray,
    splines: dict,
):
    """Wrapper to compute dynamical quantities needed for the waveform (for both low and high sampling rate dynamics)

    Args:
        m_1 (float): Mass component of the primary
        m_2 (float): Mass component of the secondary
        H (Hamiltonian): Hamiltonian class
        omega_low (np.ndarray): Orbital frequency from the EOB non-precessing evolution (low sampling rate)
        omega_fine (np.ndarray): Orbital frequency from the EOB non-precessing evolution (high sampling rate)
        dynamics_low (np.ndarray): The dynamics array from the EOB non-precessing evolution: t,r,phi,pr,pphi (low sampling rate)
        dynamics_fine (np.ndarray): The dynamics array from the EOB non-precessing evolution: t,r,phi,pr,pphi (high sampling rate)
        splines (dict): Dictionary containing the splines in orbital frequency of the vector components of the spins, LN and L as
                        well as the spin projections onto LN and L

        Returns:
            (np.ndarray): "Augmented" dynamical variables (r,phi,pr,pphi,H,omega,omega_c,chi1LN,chi2LN) for the low and high
                           sampling rate dynamics, as well as LN_hat for low and high sampling rates
        """

    (
        chi1L_EOB_low,
        chi2L_EOB_low,
        chi1LN_EOB_low,
        chi2LN_EOB_low,
        chi1v_EOB_low,
        chi2v_EOB_low,
        tmp_LN_low,
    ) = compute_spins_EOBdyn_opt(omega_low, splines)

    (
        chi1L_EOB_fine,
        chi2L_EOB_fine,
        chi1LN_EOB_fine,
        chi2LN_EOB_fine,
        chi1v_EOB_fine,
        chi2v_EOB_fine,
        tmp_LN_fine,
    ) = compute_spins_EOBdyn_opt(omega_fine, splines)

    dynamics_low = augment_dynamics_precessing_opt(
        dynamics_low,
        chi1L_EOB_low,
        chi2L_EOB_low,
        chi1LN_EOB_low,
        chi2LN_EOB_low,
        chi1v_EOB_low,
        chi2v_EOB_low,
        m_1,
        m_2,
        H,
    )

    dynamics_fine = augment_dynamics_precessing_fine_opt(
        dynamics_fine,
        chi1L_EOB_fine,
        chi2L_EOB_fine,
        chi1LN_EOB_fine,
        chi2LN_EOB_fine,
        chi1v_EOB_fine,
        chi2v_EOB_fine,
        m_1,
        m_2,
        H,
    )

    # Take the same amount of steps as in the fine dynamics
    tmp_LN_fine = tmp_LN_fine[:len(dynamics_fine)]

    #print(f"len(tmp_LN_fine) = {len(tmp_LN_fine)}, len(dynamics_fine) = {len(dynamics_fine)}")
    return dynamics_fine, dynamics_low, tmp_LN_low, tmp_LN_fine

#################################################################################################################

###########              FUNCTIONS TO  APPLY THE MERGER RINGDOWN APPROXIMATION FOR THE ANGLES          ##########

#################################################################################################################


# This function does exactly the same as the LAL counterpart
def SEOBBuildJframeVectors(Jhat_final:np.ndarray):
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

    return e1J, e2J, e3J

def compute_quatEuler_I2Jframe(e1J: np.ndarray, e2J: np.ndarray, e3J: np.ndarray):
    """
    This function computes Euler angles between the inertial (I-)frame and the
    (J-)frame aligned with the final angular momentum of the system (I2J angles)
    given the unit vectors of the J-frame.
    Same operation as in SEOBEulerI2JFrameVectors in
    https://git.ligo.org/lscsoft/lalsuite/-/blob/master/lalsimulation/lib/LALSimIMRSpinPrecEOBv4P.c#L2925
    LALSimIMRSpinPrecEOBv4P.c.
    See Appendix A of https://arxiv.org/abs/1607.05661v1
    and Sec. III B of https://arxiv.org/pdf/2004.09442v1.pdf for details.

    Args:
        e1J (np.ndarray): Unit J-frame vector e1J
        e2J (np.ndarray): Unit J-frame vector e2J
        e3J (np.ndarray): Unit J-frame vector e3J

    Returns:
        (tuple): Time-independent Euler angles describing the rotation from the J-frame to the I-frame
    """
    alphaI2J = np.arctan2(e3J[1], e3J[0])
    betaI2J = np.arccos(e3J[2])
    gammaI2J = np.arctan2(e2J[2], -e1J[2])

    return alphaI2J, betaI2J, gammaI2J

def SEOBEulerJ2PFromDynamics(t: np.ndarray, LNhat: np.ndarray, e1J: np.ndarray, e2J: np.ndarray, e3J: np.ndarray):
    """
    This function computes the Euler angles from J-frame to P-frame given the
    Newtonian orbital angular momentum timeseries LNhat(t) and the basis vectors
    of the J-frame. The minimal rotation condition is computed using the quaternion
    package and the initial condition for gamma = np.pi-alpha ensures that the
    AS limit is satisfied. Note that all quantities in the dynamics and the
    basis vectors eJ are expressed in the initial I-frame.
    Similar operation as SEOBEulerJ2PFromDynamics in LALSimIMRSpinPrecEOBv4P.c.
    See Appendix A of https://arxiv.org/abs/1607.05661v1
    and Sec. III B of https://arxiv.org/pdf/2004.09442v1.pdf for details.

    Args:
        t (np.ndarray): Time array
        LNhat (np.ndarray): LNhat vector array array
        e1J (np.ndarray): Unit J-frame vector e1J
        e2J (np.ndarray): Unit J-frame vector e2J
        e3J (np.ndarray): Unit J-frame vector e3J

    Returns:
        (tuple): Time-dependent quaterions and Euler angles describing the rotation from the co-precessing frame to the J-frame

    """

    Zframe = LNhat
    Ze1J = np.dot(Zframe, e1J)
    Ze2J = np.dot(Zframe, e2J)
    Ze3J = np.dot(Zframe, e3J)

    # Compute Euler angles from the co-precessing to the J-frame
    alphaJ2P = np.arctan2(Ze2J, Ze1J)
    betaJ2P = np.arccos(Ze3J)

    alphaJ2P = np.unwrap(alphaJ2P)
    betaJ2P = np.unwrap(betaJ2P)

    initialGamma = np.pi  -alphaJ2P[0]

    gamma0J2P = np.full(len(alphaJ2P), initialGamma)

    euler_anglesJ2P = np.array([alphaJ2P, betaJ2P, gamma0J2P]).T


    # Apply minimial rotation condition using the quaternion package
    quatJ2P = quaternion.from_euler_angles(euler_anglesJ2P)
    quatJ2P_f = quaternion.minimal_rotation(quatJ2P, t, iterations=3)
    quatJ2P_f = quaternion.unflip_rotors(quatJ2P_f, axis=-1, inplace=False)

    # Compute Euler angles from the quaternions
    eAngles_min = quaternion.as_euler_angles(quatJ2P_f).T
    alphaJ2P = np.unwrap(eAngles_min[0])
    betaJ2P = np.unwrap(eAngles_min[1])
    gammaJ2P = np.unwrap(eAngles_min[2])

    return quatJ2P_f, alphaJ2P, betaJ2P, gammaJ2P


def SEOBRotatehIlmFromhJlm_opt(
    w_hJlm: WaveformModes,
    modes_lmax: int,
    alphaI2J: float,
    betaI2J: float,
    gammaI2J: float,
)-> WaveformModes:

    """
     This function computes the hIlm Re/Im timeseries (fixed sampling) from hJlm
     Re/Im timeseries (same sampling). This is a simple rotation,
     sample-by-sample, with constant Wigner coefficients.

     Args:
        w_hJlm (WaveformModes): WaveformModes object from the scri python package. It contains waveform modes in the final J-frame
        modes_lmax (int): Maximum value of l in modes (l,m)
        alphaI2J (float): Alpha Euler angle between the I-frame and the J-frame
        betaI2J (float): Beta Euler angle between the I-frame and the J-frame
        gammaI2J (float): Gamma Euler angle between the I-frame and the J-frame

    Returns:
        (WaveformModes): WaveformModes scri object in the inertial frame

    """
    quat = quaternion.from_euler_angles(alphaI2J, betaI2J, gammaI2J)
    res = w_hJlm.rotate_decomposition_basis(~quat)

    return res

def seobnrv4P_quaternionJ2P_postmerger_extension(
    t_full: np.ndarray,
    final_spin: float,
    final_mass: float,
    euler_angles_attach: np.ndarray,
    t_attach: float,
    idx: int,
    flip: int,
    rd_approx: bool,
    beta_approx: int = 0,
):
    """
     This function computes the ringdown approximation of the Euler angles in
     the J-frame, assuming simple precession around the final J-vector with a
     precession frequency approximately equal to the differences between the
     lowest overtone of the (2,2) and (2,1) QNM frequencies.

     Args:
        t_full (np.ndarray): Time array of the waveform modes
        final_spin (float): Final spin of the BH
        final_mass (float): Final mass of the BH
        euler_angles_attach (np.ndarray): Euler angles (alpha,beta,gamma) at the attachment time
        t_attach (float): Attachment time of the dynamics and the ringdown
        idx (int): Index at which the attachment of the ringdown is performed
        flip (int): Sign of the direction of the final spin with respect to the final orbital angular momentum
                    It distinguishes between the prograde and the retrograde cases
                    See Sec. III B of https://arxiv.org/pdf/2004.09442v1.pdf for details
        rd_approx (bool): If True apply the approximation of the Euler angles, if False use constant angles
        beta_approx (int): If 0 use constant beta angle, otherwise use small opening angle approximation

    Returns:
        (quaternion): Quaternion describing the ringdown approximation of the Euler angles in the J-frame
    """
    alphaAttach, betaAttach, gammaAttach = euler_angles_attach
    t_RD = t_full[idx[-1] :]

    # Approximate the Euler angles assuming simple precession
    if rd_approx:

        t_full += t_full[0]
        sigmaQNM220 = compute_QNM(2, 2, 0, final_spin, final_mass).conjugate()
        sigmaQNM210 = compute_QNM(2, 1, 0, final_spin, final_mass).conjugate()

        omegaQNM220 = sigmaQNM220.real
        omegaQNM210 = sigmaQNM210.real
        precRate = omegaQNM220 - omegaQNM210

        cosbetaAttach = cos(betaAttach)

        precRate *= flip

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
        alphaJ2P = np.ones(len(t_RD)) * alphaAttach
        betaJ2P = np.ones(len(t_RD)) * betaAttach
        gammaJ2P = np.ones(len(t_RD)) * gammaAttach

    euler_angles_v1 = np.transpose([alphaJ2P, betaJ2P, gammaJ2P])
    quat_postMerger = quaternion.from_euler_angles(
        euler_angles_v1, beta=None, gamma=None
    )

    return quat_postMerger


def quat_ringdown_approx_opt(
    nu: float,
    m_1: float,
    m_2: float,
    idx: int,
    t_full: np.ndarray,
    t_low: np.ndarray,
    t_fine: np.ndarray,
    tmp_LN_low: np.ndarray,
    tmp_LN_fine: np.ndarray,
    final_spin: float,
    final_mass: float,
    t_attach: float,
    Lvec_hat_attach: np.ndarray,
    Jfhat_attach: np.ndarray,
    splines: dict,
    rd_approx: bool,
    beta_approx: int = 0,
):
    """
        Wrapper function to compute the angles/quaternions necessary to perform the rotations
        from the co-precessing frame (P-frame) to the observer inertial frame (I-frame) passing
        through the final angular momentum frame (J-frame), where the ringdown approximation of the
        Euler angles is applied

        Args:
            nu (float): Symmetric mass ratio
            m_1 (float): Mass component of the primary
            m_2 (float): Mass component of the secondary
            idx (int): Index at which the attachment of the ringdown is performed
            t_full (np.ndarray): Time array of the waveform modes
            t_low (np.ndarray): Time array of the EOB non-precessing evolution (low sampling rate)
            t_fine (np.ndarray): Time array of the EOB non-precessing evolution (high sampling rate)
            tmp_LN_low (np.ndarray): Newtonian angular momentum vector at the low sampling rate of the EOB non-precessing evolution
            tmp_LN_fine (np.ndarray): Newtonian angular momentum vector at the high sampling rate of the EOB non-precessing evolution
            tmp_LN_fine (np.ndarray): Time array of the EOB non-precessing evolution (high sampling rate)
            final_spin (float): Final spin of the BH
            final_mass (float): Final mass of the BH
            euler_angles_attach (np.ndarray): Euler angles (alpha,beta,gamma) at the attachment time
            t_attach (float): Attachment time of the dynamics and the ringdown
            Lvec_hat_attach (np.ndarray): Orbital angular momentum unit vector at the attachment time
            Jfhat_hat_attach (np.ndarray): Total angular momentum unit vector at the attachment time
            splines (dict): Dictionary containing the splines in orbital frequency of the vector components of the spins, LN and L as
                            well as the spin projections onto LN and L
            rd_approx (bool): If True apply the approximation of the Euler angles, if False use constant angles
            beta_approx (int): If 0 use constant beta angle, otherwise use small opening angle approximation


            Returns:
                (tuple): Quantities required to perform the rotations from the co-precessing to the inertial frame (time of
                         of the dynamics, time dependent quaternion from the P-frame to the J-frame, Euler angles from the
                         J-frame to the I-frame, and the quaternions at ringdown in the J-frame)
    """

    # Correct merger-RD attachment
    tt0 = t_attach

    idx_restart = np.argmin(np.abs(t_low - t_fine[0]))

    t_dyn = np.concatenate((t_low[:idx_restart], t_fine))
    tmp_LN = np.vstack((tmp_LN_low[:idx_restart], tmp_LN_fine))

    # Apply ringdown approximation at the attachment time
    Lf_hat_v5 = Lvec_hat_attach
    Jf_hat_v5 = Jfhat_attach

    # Compute e1J, e2J, e3J triad
    e1J, e2J, e3J = SEOBBuildJframeVectors(Jf_hat_v5)

    # Compute the Euler angles from the inertial frame to the final J-frame  (I2J angles)
    alphaI2J, betaI2J, gammaI2J = compute_quatEuler_I2Jframe(e1J, e2J, e3J)


    # Compute Euler angles from the final-J and the co-precessing frame (J2P angles)
    quatJ2P_dyn, alphaJ2P_dyn, betaJ2P_dyn, gammaJ2P_dyn = SEOBEulerJ2PFromDynamics(
        t_dyn, tmp_LN, e1J, e2J, e3J
    )

    ialphaJ2P_dyn = CubicSpline(t_dyn, alphaJ2P_dyn)
    ibetaJ2P_dyn = CubicSpline(t_dyn, betaJ2P_dyn)
    igammaJ2P_dyn = CubicSpline(t_dyn, gammaJ2P_dyn)

    alpha_attach = ialphaJ2P_dyn(tt0)
    beta_attach = ibetaJ2P_dyn(tt0)
    gamma_attach = igammaJ2P_dyn(tt0)
    euler_angles_attach = [alpha_attach, beta_attach, gamma_attach]

    # Compute sign from the angle between the final total and orbital angular momenta to check if we
    # are in the prograde or retrograde case
    cos_angle = my_dot(Jf_hat_v5, Lf_hat_v5)
    flip = 1
    if cos_angle < 0:
        final_spin *= -1
        flip = -1

    # Compute ringdown approximation of the Euler angles in the J-frame
    quat_postMerger = seobnrv4P_quaternionJ2P_postmerger_extension(
        t_full,
        final_spin,
        final_mass,
        euler_angles_attach,
        tt0,
        idx,
        flip,
        rd_approx,
        beta_approx,
    )

    return t_dyn, quatJ2P_dyn, quat_postMerger, alphaI2J, betaI2J, gammaI2J

@jit(nopython=True,cache=True)
def custom_swsh(beta: np.ndarray, gamma:np.ndarray,lmax:int):
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

    cBH = np.cos(beta/2.)
    sBH = np.sin(beta/2.)

    cBH2 = cBH*cBH
    cBH3 = cBH2*cBH
    cBH4 = cBH3*cBH

    sBH2 = sBH*sBH
    sBH3 = sBH2*sBH
    sBH4 = sBH3*sBH

    expGamma = np.exp(1j*gamma)
    expGamma2 =expGamma*expGamma

    swsh = {}

    swsh[2,2] = 0.5*np.sqrt(5./np.pi)*cBH4*expGamma2
    swsh[2,1] = np.sqrt(5./np.pi)*cBH3*sBH*expGamma
    swsh[2,-2] = 0.5*np.sqrt(5./np.pi)*sBH4/expGamma2
    swsh[2,-1] = np.sqrt(5./np.pi)*sBH3*cBH/expGamma

    if lmax >= 3:
        cBH5 = cBH4*cBH
        sBH5 = sBH4*sBH
        expGamma3 = expGamma2*expGamma
        swsh[3,3] = -np.sqrt(10.5/np.pi)*cBH5*sBH*expGamma3
        swsh[3,2] = 0.25*np.sqrt(7./np.pi)*cBH4*(6.*(cBH2 - sBH2)-4.)*expGamma2
        swsh[3,-3] = np.sqrt(10.5/np.pi)*sBH5*cBH/expGamma3
        swsh[3,-2] = 0.25*np.sqrt(7./np.pi)*sBH4*(6.*(cBH2 - sBH2)+4.)/expGamma2

    if lmax >= 4:
        cBH6 = cBH5*cBH
        sBH6 = sBH5*sBH
        expGamma4 = expGamma3*expGamma
        swsh[4,4] = 3.*np.sqrt(7./np.pi)*cBH6*sBH2*expGamma4
        swsh[4,3] = -0.75*np.sqrt(3.5/np.pi)*cBH5*sBH*(8.*(cBH2 - sBH2)-4.)*expGamma3
        swsh[4,-4] = 3.*np.sqrt(7./np.pi)*sBH6*cBH2/expGamma4
        swsh[4,-3] = 0.75*np.sqrt(3.5/np.pi)*sBH5*cBH*(8.*(cBH2 - sBH2)+4.)/expGamma3

    if lmax ==5:
        cBH7 = cBH6*cBH
        sBH7 = sBH6*sBH
        expGamma5 = expGamma4*expGamma
        swsh[5,5] = -np.sqrt(330./np.pi)*cBH7*sBH3*expGamma5
        swsh[5,-5] = np.sqrt(330./np.pi)*sBH7*cBH3/expGamma5

    return swsh


def interpolate_quats(quat: quaternion.quaternion, t_intrp: np.ndarray, t_full: np.ndarray):
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
    sp = CubicSpline(t_intrp,np.unwrap(angles.T).T)
    intrp_angles = sp(t_full)
    return quaternion.from_euler_angles(intrp_angles)
