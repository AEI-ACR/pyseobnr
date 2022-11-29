import numpy as np
from numba import *
from numba import jit
from numba import types


from ..fits.EOB_fits import compute_QNM
from .math_ops_opt import my_cross,my_dot,my_norm
from ..fits.fits_Hamiltonian import dSO as dSO_poly_fit

import lal
import quaternion

from scipy.interpolate import CubicSpline



def compute_spins_EOBdyn_opt(dynamics_low: np.ndarray, splines):

    tEOB_low = dynamics_low[:, 0]
    chi1L_spline = splines["chi1_L"]
    chi2L_spline = splines["chi2_L"]
    chi1LN_spline = splines["chi1_LN"]
    chi2LN_spline = splines["chi2_LN"]

    iphaseEOB_low = CubicSpline(tEOB_low, dynamics_low[:, 2])
    omegaEOB_low = iphaseEOB_low.derivative()(tEOB_low)

    chi1L_EOB = chi1L_spline(omegaEOB_low)
    chi2L_EOB = chi2L_spline(omegaEOB_low)

    chi1LN_EOB = chi1LN_spline(omegaEOB_low)
    chi2LN_EOB = chi2LN_spline(omegaEOB_low)

    chi1v_EOB = splines["chi1"](omegaEOB_low)
    chi2v_EOB = splines["chi2"](omegaEOB_low)

    return chi1L_EOB, chi2L_EOB, chi1LN_EOB, chi2LN_EOB, chi1v_EOB, chi2v_EOB

def augment_dynamics_precessing_opt(dynamics,  chi1L_EOB, chi2L_EOB,  chi1LN_EOB, chi2LN_EOB, chi1v_EOB, chi2v_EOB, m_1, m_2, omegaPN_f, H):
    """Compute dynamical quantities we need for the waveform

    Args:
        dynamics (np,ndarray): The dynamics array: t,r,phi,pr,pphi
    """

    ms = m_1 + m_2
    nu = m_1*m_2/(m_1+m_2)**2
    X1 = m_1/ms
    X2 = m_2/ms

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

        #if q[0]<2:
        #    print(f"q = {q[0]}")


        # Evaluate a few things: H, omega,omega_circ
        ap  = X1*chi1_LN + X2*chi2_LN
        am  = X1*chi1_LN - X2*chi2_LN
        p_c[1] = p[1]
        dSO_new = dSO_poly_fit(nu, ap, am)
        H.calibration_coeffs['dSO'] = dSO_new

        dyn = H.dynamics(q, p, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L)
        omega = dyn[3]
        H_val = dyn[4]
        omega_c = H.omega(q, p_c, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L)

        result.append([H_val, omega, omega_c, chi1_LN, chi2_LN, chi1_L, chi2_L])
    result = np.array(result)


    return np.c_[dynamics,result]


def project_spins_augment_dynamics_opt(
    m_1, m_2, H, dynamics_low, dynamics_fine, splines, omegaPN_f
):

    (
        chi1L_EOB_low,
        chi2L_EOB_low,
        chi1LN_EOB_low,
        chi2LN_EOB_low,
        chi1v_EOB_low,
        chi2v_EOB_low,
    ) = compute_spins_EOBdyn_opt(dynamics_low, splines)

    (
        chi1L_EOB_fine,
        chi2L_EOB_fine,
        chi1LN_EOB_fine,
        chi2LN_EOB_fine,
        chi1v_EOB_fine,
        chi2v_EOB_fine,
    ) = compute_spins_EOBdyn_opt(dynamics_fine, splines)

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
        omegaPN_f,
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
        omegaPN_f,
        H,
    )

    return dynamics_fine, dynamics_low


def augment_dynamics_precessing_fine_opt(
    dynamics,
    chi1L_EOB,
    chi2L_EOB,
    chi1LN_EOB,
    chi2LN_EOB,
    chi1v_EOB,
    chi2v_EOB,
    m_1,
    m_2,
    omegaPN_f,
    H,
):
    """Compute dynamical quantities we need for the waveform and also the csi_fine coefficient

    Args:
        dynamics (np,ndarray): The dynamics array: t,r,phi,pr,pphi
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

        #if omega < 1 and abs(chi1_LN)<= 1 and abs(chi2_LN)<= 1 and rs[-1]>r or not np.isnan(omega):
        if np.isnan(omega):
            count += 1
            idx_max = i
            break

        if rs[-1]>=r:
            #if omega<=omegaPN_f:
            #print(f" omega = {omega}, rs[-1] = {rs[-1]}, r = {r}")
            omega_c = H.omega(
                q, p_c, chi1_v, chi2_v, m_1, m_2, chi1_LN, chi2_LN, chi1_L, chi2_L
            )
            if np.isnan(omega_c):
                count += 1
                idx_max = i
                break


            result.append([H_val, omega, omega_c, chi1_LN, chi2_LN, chi1_L, chi2_L])
            idx_max = i
        else:
            #print(f"i = {i}, omega = {omega}, rs[-1] = {rs[-1]}, r = {r}, a1LN = {abs(chi1_LN)}, a2LN = {abs(chi2_LN)}")
            count += 1
            idx_max = i
            break
        rs.append(r)

        #if q[0]<1.5:
        #    print(f"r = {np.round(q[0],5)}, chi1_LN  = {np.round(chi1_LN ,5)}, chi2_LN  = {np.round(chi2_LN ,5)}, omega = {np.round(omega,5)}, omega_c = {np.round(omega_c,5)}")
        #print(f"r = {np.round(q[0],5)}, H = {np.round(H_val,5)}, dSO = {np.round(dSO_new[i],5)}, omega = {np.round(omega,5)}, omega_c = {np.round(omega_c,5)}")

    result = np.array(result)

    if count >1:
        #print(f"count = {count}, idx_max = {idx_max}, len(result) = {len(result)}")
        dynamics1 = dynamics[:idx_max]
    else:
        #print(f"count = {count}, idx_max = {idx_max}, len(result) = {len(result)}")
        dynamics1 = dynamics[:len(result)]

    #print(f"len(dynamics)= {len(dynamics)}, len(dynamics1)= {len(dynamics1)}, len(result) = {len(result)}")
    return np.c_[dynamics1, result]


#################################################################################################################

###########              FUNCTIONS TO  APPLY THE MERGER RINGDOWN APPROXIMATION FOR THE ANGLES          ##########

#################################################################################################################


# This function does exactly the same as the LAL counterpart
def SEOBBuildJframeVectors(Jhat_final):
    """
     This function computes the Jframe unit vectors, with e3J along Jhat.
     Convention: if (ex, ey, ez) is the initial I-frame, e1J chosen such that ex
     is in the plane (e1J, e3J) and ex.e1J>0.
     In the case where e3J and x happen to be close to aligned, we continuously
     switch to another prescription with y playing the role of x.
    """

    e3J = Jhat_final

    exvec = np.array([1.0, 0.0, 0.0])
    eyvec = np.array([0.0, 1.0, 0.0])

    exdote3J = np.dot(exvec, e3J)
    eydote3J = np.dot(eyvec, e3J)

    lambda_fac = 1.0 - abs(exdote3J)

    if lambda_fac < 0.0 or lambda_fac > 1.0:
        print("Problem: lambda=1-|e3J.ex|=%f, should be in [0,1]" % lambda_fac)

    elif lambda_fac > 1e-4:
        normfacx = 1.0 / np.sqrt(1.0 - exdote3J * exdote3J)

        e1J = (exvec - exdote3J * e3J) / normfacx
    elif lambda_fac < 1e-5:

        normfacy = 1.0 / np.sqrt(1.0 - eydote3J * eydote3J)
        e1J = (eyvec - eydote3J * e3J) / normfacy
    else:
        weightx = (lambda_fac - 1e-5) / (1e-4 - 1e-5)
        weighty = 1.0 - weightx
        normfacx = 1.0 / np.sqrt(1.0 - exdote3J * exdote3J)
        normfacy = 1.0 / np.sqrt(1.0 - eydote3J * eydote3J)
        e1J = (
            weightx * (exvec - exdote3J * e3J) / normfacx
            + weighty * (eyvec - eydote3J * e3J) / normfacy
        )

        e1Jblendednorm = my_norm(e1J)  # np.sqrt(np.dot(e1J, e1J))
        e1J /= e1Jblendednorm

    # /* Get e2J = e3J * e1J */
    e2J = my_cross(e3J, e1J)  # cross_product(e3J, e1J)

    e1Jnorm = my_norm(e1J)
    e2Jnorm = my_norm(e2J)
    e3Jnorm = my_norm(e3J)

    e1J /= e1Jnorm
    e2J /= e2Jnorm
    e3J /= e3Jnorm

    return e1J, e2J, e3J

# This function computes Euler angles I2J given the unit vectors of the Jframe.
# Same operation as in SEOBEulerI2JFrameVectors in LALSimIMRSpinPrecEOBv4P.c
def compute_quatEuler_I2Jframe(e1J, e2J, e3J):

    alphaI2J = np.arctan2(e3J[1], e3J[0])
    betaI2J = np.arccos(e3J[2])
    gammaI2J = np.arctan2(e2J[2], -e1J[2])

    return alphaI2J, betaI2J, gammaI2J

def SEOBEulerJ2PFromDynamics(t, Lhat, e1J, e2J, e3J):

    # Compute Lhat
    Zframe = Lhat
    Ze1J = np.dot(Zframe, e1J)
    Ze2J = np.dot(Zframe, e2J)
    Ze3J = np.dot(Zframe, e3J)

    alphaJ2P = np.arctan2(Ze2J, Ze1J)
    betaJ2P = np.arccos(Ze3J)

    alphaJ2P = np.unwrap(alphaJ2P)
    betaJ2P = np.unwrap(betaJ2P)

    #e3PiniIbasis = np.array([0, 0, 1.0])#Zframe[0]
    #e1PiniIbasis = np.array([1.0, 0, 0])
    # /* e2P is obtained by completing the triad */
    #e2PiniIbasis = np.cross(e3PiniIbasis, e1PiniIbasis)

    #e1PiniJbasis = np.array([0.0, 0.0, 0.0])
    #e2PiniJbasis = np.array([0.0, 0.0, 0.0])

    #e1PiniJbasis[2] = np.dot(e1PiniIbasis, e3J)
    #e2PiniJbasis[2] = np.dot(e2PiniIbasis, e3J)
    #initialGamma = np.arctan2(e2PiniJbasis[2], -e1PiniJbasis[2])
    initialGamma = -alphaJ2P[0]

    gamma0J2P = np.full(len(alphaJ2P), initialGamma)

    """
    if (
        e1J[0] == 1
        and e1J[1] == 0
        and e1J[2] == 0
        and e2J[0] == 0
        and e2J[1] == 1
        and e2J[2] == 0
    ):

        alphaJ2P *= 0
        betaJ2P *= 0
        gamma0J2P *= 0
    """
    euler_anglesJ2P = np.array([alphaJ2P, betaJ2P, gamma0J2P]).T

    # This works
    """
    ialpha = CubicSpline(t,np.unwrap(alphaJ2P))
    alpha_dot = ialpha.derivative()(t)
    cos_beta = np.cos(betaJ2P)
    integ = CubicSpline(t,-1.*alpha_dot*cos_beta)

    gamma2 = np.array([integ.integrate(t[0],tt) for tt in t])

    gammaJ2P = np.unwrap(gamma2)+gamma0J2P[0]

    euler_anglesJ2P = np.array([alphaJ2P,betaJ2P,gammaJ2P]).T
    quatJ2P_f = quaternion.from_euler_angles(euler_anglesJ2P)
    #quatJ2P_f = quaternion.minimal_rotation(quatJ2P_f, t,iterations=3)
    #quatJ2P_f=  quaternion.unflip_rotors(quatJ2P_f, axis=-1, inplace=False)

    """
    # This also works
    quatJ2P = quaternion.from_euler_angles(euler_anglesJ2P)
    quatJ2P_f = quaternion.minimal_rotation(quatJ2P, t, iterations=3)
    quatJ2P_f = quaternion.unflip_rotors(quatJ2P_f, axis=-1, inplace=False)

    eAngles_min = quaternion.as_euler_angles(quatJ2P_f).T
    alphaJ2P = np.unwrap(eAngles_min[0])
    betaJ2P = np.unwrap(eAngles_min[1])
    gammaJ2P = np.unwrap(eAngles_min[2])

    return quatJ2P_f, alphaJ2P, betaJ2P, gammaJ2P


#  This is the same wrapper function for the PN orbital angular momentum in LALSimIMRPhenomX_precession.c
def simIMRPhenomXLPNAnsatz(
    v,  # Input velocity
    LNorm,  # Orbital angular momentum normalization
    L0,  # Newtonian orbital angular momentum (i.e. LN = 1.0*LNorm)
    L1,  # 0.5PN Orbital angular momentum
    L2,  # 1.0PN Orbital angular momentum
    L3,  # 1.5PN Orbital angular momentum
    L4,  # 2.0PN Orbital angular momentum
    L5,  # 2.5PN Orbital angular momentum
    L6,  # 3.0PN Orbital angular momentum
    L7,  # 3.5PN Orbital angular momentum
    L8,  # 4.0PN Orbital angular momentum
    L8L,  # 4.0PN logarithmic orbital angular momentum term
):
    x = v * v
    x2 = x * x
    x3 = x * x2
    x4 = x * x3
    sqx = np.sqrt(x)

    """
      Here LN is the Newtonian pre-factor: LN = \eta / \sqrt{x} :

      L = L_N \sum_a L_a x^{a/2}
        = L_N [ L0 + L1 x^{1/2} + L2 x^{2/2} + L3 x^{3/2} + ... ]

    """

    return LNorm * (
        L0
        + L1 * sqx
        + L2 * x
        + L3 * (x * sqx)
        + L4 * x2
        + L5 * (x2 * sqx)
        + L6 * x3
        + L7 * (x3 * sqx)
        + L8 * x4
        + L8L * x4 * np.log(x)
    )


def orbital_angular_momentum_coeffs(eta, chi1L, chi2L):
    """
        4PN orbital angular momentum + leading order in spin at all PN orders terms.
              - Marsat, CQG, 32, 085008, (2015), arXiv:1411.4118
              - Siemonsen et al, PRD, 97, 064010, (2018), arXiv:1606.08832
    """
    pi2 = np.pi * np.pi
    chi1L2 = chi1L * chi1L
    chi2L2 = chi2L * chi2L

    delta = np.sqrt(abs(1.0 - 4.0 * eta))
    # Using same PN orders as PhenomXPHM and PhenomTPHM

    L0 = 1.0
    L1 = 0.0
    L2 = 3.0 / 2.0 + eta / 6.0
    L3 = (5 * (chi1L * (-2 - 2 * delta + eta) + chi2L * (-2 + 2 * delta + eta))) / 6.0
    L4 = (81 + (-57 + eta) * eta) / 24.0
    L5 = (
        -7
        * (
            chi1L * (72 + delta * (72 - 31 * eta) + eta * (-121 + 2 * eta))
            + chi2L * (72 + eta * (-121 + 2 * eta) + delta * (-72 + 31 * eta))
        )
    ) / 144.0
    L6 = (10935 + eta * (-62001 + eta * (1674 + 7 * eta) + 2214 * pi2)) / 1296.0
    L7 = (
        chi2L
        * (
            -324
            + eta * (1119 - 2 * eta * (172 + eta))
            + delta * (324 + eta * (-633 + 14 * eta))
        )
        - chi1L
        * (
            324
            + eta * (-1119 + 2 * eta * (172 + eta))
            + delta * (324 + eta * (-633 + 14 * eta))
        )
    ) / 32.0
    L8 = (
        2835 / 128.0
        - (
            eta
            * (
                -10677852
                + 100 * eta * (-640863 + eta * (774 + 11 * eta))
                + 26542080 * np.euler_gamma
                + 675 * (3873 + 3608 * eta) * pi2
            )
        )
        / 622080.0
        - (64 * eta * np.log(16)) / 3.0
    )

    # This is the log(x) term at 4PN, x^4/2 * log(x)
    L8L = -(64.0 / 3.0) * eta

    # Leading order in spin at all PN orders, note that the 1.5PN terms are already included. Here we have additional 2PN and 3.5PN corrections.
    L4 += (
        0
        * (
            chi1L2 * (1 + delta - 2 * eta)
            + 4 * chi1L * chi2L * eta
            - chi2L2 * (-1 + delta + 2 * eta)
        )
        / 2.0
    )
    L7 += (
        0
        * (
            3
            * (chi1L + chi2L)
            * eta
            * (
                chi1L2 * (1 + delta - 2 * eta)
                + 4 * chi1L * chi2L * eta
                - chi2L2 * (-1 + delta + 2 * eta)
            )
        )
        / 4.0
    )

    return L0, L1, L2, L3, L4, L5, L6, 0 * L7, 0 * L8, 0 * L8L


def compute_finalJ(eta, m_1, m_2, omega_peak, splines):

    chi1_peak = splines["chi1"](omega_peak)
    chi2_peak = splines["chi2"](omega_peak)
    LN_peak = splines["L_N"](omega_peak)

    LN_peak /= my_norm(LN_peak)
    chi1L_peak = np.dot(chi1_peak, LN_peak)
    chi2L_peak = np.dot(chi2_peak, LN_peak)

    L0, L1, L2, L3, L4, L5, L6, L7, L8, L8L = orbital_angular_momentum_coeffs(
        eta, chi1L_peak, chi2L_peak
    )

    v_peak = omega_peak ** (1.0 / 3.0)
    LN_norm_v5 = eta / v_peak

    LnormPN = simIMRPhenomXLPNAnsatz(
        v_peak,
        LN_norm_v5,  # Orbital angular momentum normalization
        L0,  # Newtonian orbital angular momentum (i.e. LN = 1.0*LNorm)
        L1,  # 0.5PN Orbital angular momentum
        L2,  # 1.0PN Orbital angular momentum
        L3,  # 1.5PN Orbital angular momentum
        L4,  # 2.0PN Orbital angular momentum
        L5,  # 2.5PN Orbital angular momentum
        L6,  # 3.0PN Orbital angular momentum
        L7,  # 3.5PN Orbital angular momentum
        L8,  # 4.0PN Orbital angular momentum
        L8L,  # 4.0PN logarithmic orbital angular momentum term
    )

    spin1_v5P_final = m_1 * m_1 * chi1_peak
    spin2_v5P_final = m_2 * m_2 * chi2_peak
    LpeakPN = LN_peak * LnormPN
    # print(spin1_v5P_final, spin2_v5P_final,LpeakPN)
    Jpeak = spin1_v5P_final + spin2_v5P_final + LpeakPN
    #print(f"LpeakPN = {LpeakPN}, Jpeak = {Jpeak}")
    #print(f"tpeakPN: Lvec = {LpeakPN}, s1 = {spin1_v5P_final}, s2 = {spin2_v5P_final}, Jf = {Jpeak}")

    Jhat_peak = Jpeak / my_norm(Jpeak)
    Lhat_final = LpeakPN / my_norm(LpeakPN)

    return Jpeak, Jhat_peak, Lhat_final


# Code up same waveform rotation as in LAL: SEOBRotatehIlmFromhJlm
def SEOBWignerDAmp(l, m, mp, beta):
    return lal.WignerdMatrix(l, m, mp, beta)


def SEOBWignerDPhase(m, mp, alpha, gamma):
    return m * alpha + mp * gamma


def SEOBRotatehIlmFromhJlm(
    hJlm,  # hJlm time series, complex values on fixed sampling
    modes_lmax,  # Input: maximum value of l in modes (l,m)
    alphaI2J,  # Input: Euler angle alpha I->J
    betaI2J,  # Input: Euler angle beta I->J
    gammaI2J,  # Input: Euler angle gamma I->J
):

    """
     This function computes the hIlm Re/Im timeseries (fixed sampling) from hJlm
     Re/Im timeseries (same sampling). This is a simple rotation,
     sample-by-sample, with constant Wigner coefficients.
     See the comment before SEOBWignerDAmp for explanation of conventions,
     and Appendix A of Babak et al, Phys. Rev. D 95, 024010, 2017 [arXiv:1607.05661] for a general
     discussion.
    """

    amp_wigner = 0.0
    phase_wigner = 0.0
    D_wigner = 0.0

    # /* Loop on l */
    hIlm = {}
    for l in range(2, modes_lmax + 1):
        # print(f"l = {l}")
        # /* Loop on m */
        for m in range(-l, l + 1):
            # print(f"l = {l}, m = {m}")
            mode_lm = str(l) + "," + str(m)
            hIlm[mode_lm] = 0.0
            # hIlm[l,m] = 0.
            # /* Loop on mp - exclude value 0, since hPl0=0 in our approximation */
            for mp in range(-l, l + 1):
                # /* Get hJlm mode */
                # hJlmpmode = XLALSphHarmTimeSeriesGetMode(hJlm, l, mp);
                # hJlmpmode_data = hJlmpmode->data->data;
                # hJlmpmode_data = hJlm[l, mp]
                mode_lmp = str(l) + "," + str(mp)

                if mode_lmp in hJlm.keys():
                    hJlmpmode_data = hJlm[mode_lmp]

                else:
                    hJlmpmode_data = hJlm[l, mp]

                # /* Compute constant Wigner coefficient */
                amp_wigner = SEOBWignerDAmp(l, m, mp, betaI2J)
                phase_wigner = SEOBWignerDPhase(m, mp, alphaI2J, gammaI2J)
                D_wigner = amp_wigner * np.exp(
                    -1.0j * phase_wigner
                )  # /* mind the conjugation Dlmmpstar */
                # /* Evaluate mode contribution */
                # for i in range(retLen):
                # hIlmmode_data[i] += D_wigner * hJlmpmode_data[i];
                # hIlm[l,m] += hJlmpmode_data
                hIlm[mode_lm] += D_wigner * hJlmpmode_data

    return hIlm


def SEOBRotatehIlmFromhJlm_opt(
    w_hJlm,  # hJlm time series, complex values on fixed sampling
    modes_lmax,  # Input: maximum value of l in modes (l,m)
    alphaI2J,  # Input: Euler angle alpha I->J
    betaI2J,  # Input: Euler angle beta I->J
    gammaI2J,  # Input: Euler angle gamma I->J
):

    """
     This function computes the hIlm Re/Im timeseries (fixed sampling) from hJlm
     Re/Im timeseries (same sampling). This is a simple rotation,
     sample-by-sample, with constant Wigner coefficients.

    """
    quat = quaternion.from_euler_angles(alphaI2J, betaI2J, gammaI2J)
    res = w_hJlm.rotate_decomposition_basis(~quat)

    return res


# Ringdown approximation for the Euler angles
def seobnrv4P_quaternionJ2P_postmerger_extension(
    t_full,
    final_spin,
    final_mass,
    euler_angles_attach,
    t_attach,
    idx,
    flip,
    rd_approx,
    beta_approx: int = 0,
):

    alphaAttach, betaAttach, gammaAttach = euler_angles_attach
    t_RD = t_full[idx[-1] :]

    # Approximate the Euler angles assuming simple precession
    if rd_approx:

        t_full += t_full[0]
        sigmaQNM220 = compute_QNM(2, 2, 0, final_spin, final_mass).conjugate()
        sigmaQNM210 = compute_QNM(2, 1, 0, final_spin, final_mass).conjugate()

        omegaQNM220 = np.real(sigmaQNM220)
        omegaQNM210 = np.real(sigmaQNM210)
        precRate = omegaQNM220 - omegaQNM210

        cosbetaAttach = np.cos(betaAttach)

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
    nu,
    m_1,
    m_2,
    idx,
    t_full,
    t_low,
    t_fine,
    tmp_LN_low,
    tmp_LN_fine,
    final_spin,
    final_mass,
    t_attach,
    omega_peak,
    Lvec_hat_attach,
    Jfhat_attach,
    splines,
    rd_approx,
    beta_approx: int = 0,
):

    # Correct merger-RD attachment
    tt0 = t_attach

    idx_restart = np.argmin(np.abs(t_low - t_fine[0]))

    t_dyn = np.concatenate((t_low[:idx_restart], t_fine))
    tmp_LN = np.vstack((tmp_LN_low[:idx_restart], tmp_LN_fine))

    tmp_LN_norm = np.sqrt(np.einsum("ij,ij->i", tmp_LN, tmp_LN))
    tmp_LN = (tmp_LN.T / tmp_LN_norm).T

    # Apply ringdown approximation at the attachment time
    #Jf_v5, Jf_hat_v5, Lf_hat_v5 = compute_finalJ( nu, m_1, m_2, omega_peak, splines)
    #print(f"PN : Lf_hat_v5 = {Lf_hat_v5}, Jf_hat_v5 = {Jf_hat_v5}")
    Lf_hat_v5 = Lvec_hat_attach
    Jf_hat_v5 = Jfhat_attach
    #print(f"EOB: Lf_hat_v5 = {Lf_hat_v5}, Jf_hat_v5 = {Jf_hat_v5}")

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

    # print(f"alpha_attach = {alpha_attach}, beta_attach = {beta_attach}, gamma_attach = {gamma_attach}")
    # print(f"alphaI2J = {alphaI2J}, betaI2J = {betaI2J}, gammaI2J = {gammaI2J}")

    cos_angle = my_dot(Jf_hat_v5, Lf_hat_v5)
    flip = 1
    if cos_angle < 0:
        final_spin *= -1
        flip = -1

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

    # quat_postMerger =  quaternion.unflip_rotors(quat_postMerger, axis=-1, inplace=False)

    return t_dyn, quatJ2P_dyn, quat_postMerger, alphaI2J, betaI2J, gammaI2J

@jit(nopython=True,cache=True)
def custom_swsh(beta, gamma):

    cBH = np.cos(beta/2.)
    sBH = np.sin(beta/2.)

    cBH2 = cBH*cBH
    cBH3 = cBH2*cBH
    cBH4 = cBH3*cBH
    cBH5 = cBH4*cBH
    cBH6 = cBH5*cBH
    cBH7 = cBH6*cBH

    sBH2 = sBH*sBH
    sBH3 = sBH2*sBH
    sBH4 = sBH3*sBH
    sBH5 = sBH4*sBH
    sBH6 = sBH5*sBH
    sBH7 = sBH6*sBH

    expGamma = np.exp(1j*gamma)
    expGamma2 =expGamma*expGamma
    expGamma3 = expGamma2*expGamma
    expGamma4 = expGamma3*expGamma
    expGamma5 = expGamma4*expGamma

    swsh = {}

    swsh[2,2] = 0.5*np.sqrt(5./np.pi)*cBH4*expGamma2
    swsh[2,1] = np.sqrt(5./np.pi)*cBH3*sBH*expGamma
    swsh[3,3] = -np.sqrt(10.5/np.pi)*cBH5*sBH*expGamma3
    swsh[3,2] = 0.25*np.sqrt(7./np.pi)*cBH4*(6.*(cBH2 - sBH2)-4.)*expGamma2
    swsh[4,4] = 3.*np.sqrt(7./np.pi)*cBH6*sBH2*expGamma4
    swsh[4,3] = -0.75*np.sqrt(3.5/np.pi)*cBH5*sBH*(8.*(cBH2 - sBH2)-4.)*expGamma3
    swsh[5,5] = -np.sqrt(330./np.pi)*cBH7*sBH3*expGamma5

    swsh[2,-2] = 0.5*np.sqrt(5./np.pi)*sBH4/expGamma2
    swsh[2,-1] = np.sqrt(5./np.pi)*sBH3*cBH/expGamma
    swsh[3,-3] = np.sqrt(10.5/np.pi)*sBH5*cBH/expGamma3
    swsh[3,-2] = 0.25*np.sqrt(7./np.pi)*sBH4*(6.*(cBH2 - sBH2)+4.)/expGamma2
    swsh[4,-4] = 3.*np.sqrt(7./np.pi)*sBH6*cBH2/expGamma4
    swsh[4,-3] = 0.75*np.sqrt(3.5/np.pi)*sBH5*cBH*(8.*(cBH2 - sBH2)+4.)/expGamma3
    swsh[5,-5] = np.sqrt(330./np.pi)*sBH7*cBH3/expGamma5
    return swsh


def interpolate_quats(quat,t_intrp,t_full):
    angles = quaternion.as_euler_angles(quat)
    sp = CubicSpline(t_intrp,np.unwrap(angles.T).T)
    intrp_angles = sp(t_full)
    return quaternion.from_euler_angles(intrp_angles)