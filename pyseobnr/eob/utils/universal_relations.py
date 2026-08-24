import math

import numpy as np
from scipy.optimize import fsolve

#
# UNIVERSAL RELATIONS AS GIVEN IN
# https://lscsoft.docs.ligo.org/lalsuite/lalsimulation/_l_a_l_sim_universal_relations_8c_source.html
#


def SmoothTransitionFunction(x, a, b, flip=False):
    """A function to transition from 0 for x<=a to 1 for x>=b in a smooth way.

    Args:
        x (flaot): The input value
        a (float, optional): The lower end of the transition window.
        b (flaot, optional): The upper end of the transition window.
        flip (bool, optional): Whether to transition from 1 to 0 instead. Defaults to False.

    Returns:
        float: The trnansitioned value if x in [a,b], else 0 or 1.
    """
    if a > b:
        return SmoothTransitionFunction(x, b, a, flip)
    if flip:
        return 1.0 - SmoothTransitionFunction(x, a, b, False)
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    scaled = (x - a) / (b - a)
    x1 = np.exp(-1 / scaled)
    x2 = np.exp(-1 / (1 - scaled))
    return x1 / (x1 + x2)


def polyfitUniversalRelation(x, coeffs):
    return coeffs[0] + x * (
        coeffs[1] + x * (coeffs[2] + x * (coeffs[3] + x * coeffs[4]))
    )


def UniversalRelationlambda3TidalVSlambda2Tidal(lambda2bar):
    """
    Eq. (60) with coeffs from 1st row of Table I of  https://arxiv.org/pdf/1311.0872.pdf
    Gives the dimensionless l=3 tidal deformability: lambda3bar = 2/15 k3 C^7
    where k3 is the l=3 Love number and C is the compactness. It is a
    function the dimensionless l=2 tidal deformability: lambda2bar = 2/3 k2 C^5.
    Compared to NR for 1 <= lambda2bar <= 3000
    """

    coeffs = [-1.15, 1.18, 2.51e-2, -1.31e-3, 2.52e-5]
    if lambda2bar < 0.0:
        raise ValueError("lambda2bar must be non-negative")
    elif 0.0 <= lambda2bar < 0.01:
        return (
            0.4406491912035266 * lambda2bar
            - 34.63232296075433 * lambda2bar**2
            + 1762.112913125107 * lambda2bar**3
        )
    else:
        lnx = math.log(lambda2bar)
    lny = polyfitUniversalRelation(lnx, coeffs)
    return math.exp(lny)


def UniversalRelationomega02TidalVSlambda2Tidalv2(lambda2bar):
    """
    Eq. (7) of https://arxiv.org/pdf/2109.08145
    Gives the l=2 f-mode frequency M_{NS}omega_{02}
    as a function the dimensionless l=2 tidal deformability: lambda2bar = 2/3 k2 C^5
    where k2 is the l=2 Love number and C is the compactness.
    Note that we rescale to this quantity to one solar mass
    """
    if lambda2bar < 1.0:
        # We have extended the quasi-universal relation by solving the linear system
        # that you get through an ansatz f(x) = a*x**3 + b*x**2 + c*x + d
        # and demanding f'(0) = 0 and continuity of the first two derivatives at
        # lambda2bar = 1
        return (
            0.003534695863104191 * lambda2bar**3
            - 0.009806179431989909 * lambda2bar**2
            + 0.19080067100008258
        )
    constant = 2 * np.pi * 1.4 * 0.004925490947641268  # 2 pi * M_Sun_in_s * 1000 (kHz)
    coeffs = np.array([4.2590, -0.47874, -0.45353, 0.14439, -0.016194, 0.00064163])
    return np.poly1d(coeffs[::-1])(np.log10(lambda2bar)) * constant


def UniversalRelationomega02TidalVSlambda2Tidal(lambda2bar):
    """
    Eq. (3.5) with coeffs from 1st column of Table I of https://arxiv.org/pdf/1408.3789.pdf
    Gives the l=2 f-mode frequency M_{NS}omega_{02}
    as a function the dimensionless l=2 tidal deformability: lambda2bar = 2/3 k2 C^5
    where k2 is the l=2 Love number and C is the compactness.
    Compared to NR for 0 <= log(lambda2bar) <= 9, that is
    1 <= lambda2bar <= 8100
    """

    coeffs = [1.82e-1, -6.836e-3, -4.196e-3, 5.215e-4, -1.857e-5]
    if lambda2bar < 0.0:
        raise ValueError("lambda2bar must be non-negative")
    elif 0.0 <= lambda2bar < 1.0:
        lnx = 0.0
    elif 1.0 <= lambda2bar < math.exp(9.0):
        lnx = math.log(lambda2bar)
    else:
        lnx = 9.0
    return polyfitUniversalRelation(lnx, coeffs)


def UniversalRelationomega03TidalVSlambda3Tidal(lambda3bar):
    """
    Eq. (3.5) with coeffs from 2nd column of Table I of https://arxiv.org/pdf/1408.3789.pdf
    Gives the l=3 f-mode frequency M_{NS}omega_{03}
    as a function the dimensionless l=3 tidal deformability: lambda3bar = 2/15 k3 C^5
    where k3 is the l=3 Love number and C is the compactness.
    Compared to NR for -1 <= log(lambda3bar) <= 10, that is
    0.37 <= lambda3bar <= 20000
    """

    coeffs = [2.245e-1, -1.5e-2, -1.412e-3, 1.832e-4, -5.561e-6]
    if lambda3bar < 0.0:
        raise ValueError("lambda3bar must be non-negative")
    elif 0.0 <= lambda3bar < math.exp(-1.0):
        lnx = -1.0
    elif math.exp(-1.0) <= lambda3bar < math.exp(10.0):
        lnx = math.log(lambda3bar)
    else:
        lnx = 10.0
    return polyfitUniversalRelation(lnx, coeffs)


def UniversalRelationQuadMonVSlambda2Tidal(lambda2bar):
    """
    Eq. (15) with coeffs from third row of Table I of https://arxiv.org/pdf/1608.02582.pdf (Yagi-Yunes)

    Gives the spin-induced quadrupole coefficient as a function the dimensionless l=2
    tidal deformability: lambda2bar = 2/3 k2 C^5.
    This coefficient quadparam relates the spin-induced quadrupole to the square of the spin,
    according to :math:`Q = -quadparam*\\chi^2*m^3`.

    It takes the value 1 for BH, and can reach ~10 for NS.
    The notation is Qbar in Yagi-Yunes. In the PN literature, the notation is often kappa (e.g. in
    https://arxiv.org/pdf/1501.01529.pdf). In https://arxiv.org/pdf/gr-qc/9709032.pdf the notation is a.
    The Yagi-Yunes fit does not cover the BH limit, where lambda2bar->0 and kappa->1.
    We extend it with a polynomial below lambda2bar=1. so that the function and its two
    first derivatives are smooth at the junction, while enforcing the BH limit.
    """
    coeffs = [0.1940, 0.09163, 0.04812, -4.283e-3, 1.245e-4]
    if lambda2bar < 0.0:
        raise ValueError("Invalid argument. Expected lambda2bar to be non-negative.")
    elif 0.0 <= lambda2bar < 1.0:
        return 1.0 + lambda2bar * (
            0.427688866723244
            + lambda2bar * (-0.324336526985068 + lambda2bar * 0.1107439432180572)
        )
    else:
        lnx = math.log(lambda2bar)
        lny = polyfitUniversalRelation(lnx, coeffs)
        return math.exp(lny)


def UniversalRelationSpinInducedOctupoleVSSpinInducedQuadrupole(qm_def):
    """
    Quasi universal relation between spin-induced quadrupole and
    spin-induced octupole moment based on Yagi & Yunes arxiv:1608.02582;
    Table 2 of the review also given explicitly in NRTidalv2 paper https://arxiv.org/abs/1905.06011
    """
    coeffs = [0.003131, 2.071, -0.7152, 0.2458, -0.03309]
    lnx = math.log(qm_def)
    lny = polyfitUniversalRelation(lnx, coeffs)
    return math.exp(lny)


def UniversalRelationSpinInducedHexadecupoleVSSpinInducedQuadrupole(qm_def):
    """
    Quasi universal relation between spin-induced hexadecupole and
    spin-induced octupole moment based on Yagi & Yunes arxiv:1608.02582;
    Table 2 of the review also given explicitly in NRTidalv2 paper https://arxiv.org/abs/1905.06011
    """
    coeffs = [-0.02287, 3.849, -1.540, 0.5863, -8.337e-2]
    lnx = math.log(qm_def)
    lny = polyfitUniversalRelation(lnx, coeffs)
    return math.exp(lny)


def UniversalRelationMomentOfInertiaVSlambda2Tidal(lambda2):
    """
    Eq. (15) with coeffs from 1st row of Table I of https://arxiv.org/pdf/1608.02582
    Gives the moment of inertia I of a neutron star. It is a
    function the dimensionless l=2 tidal deformability: lambda2bar = 2/3 k2 C^5.
    Compared to NR for 1 <= lambda2bar <= 10^4
    """
    coeffs = [1.496, 0.05951, 0.02238, -6.953e-4, 8.345e-6]
    lnx = math.log(lambda2)
    lny = polyfitUniversalRelation(lnx, coeffs)
    return math.exp(lny)


def SimpleTransitionFunction(x, cutoff=0.9):
    """A simple function that returns x in the interval [-cutoff,cutoff]

    Args:
        x (float): The value

    Returns:
        float: The cut transitioned value
    """
    return cutoff * (1 - 2 / (1 + np.exp(2 * x / cutoff)))


def UniversalRelationSpinShiftomega02VSlambda2Tidal(lambda2, spin, omega02):
    """
    Eq. (5.7) of http://arxiv.org/abs/2103.06100
    Gives the spin shift due to the Corriolis force of a neutron star. It is a
    function the dimensionless l=2 tidal deformability: lambda2bar = 2/3 k2 C^5.
    We set its absolute value to be <= the f-mode frequency, as the model
    developes unphysical poles otherwise due to this simplified universal relation
    """
    # I = UniversalRelationMomentOfInertiaVSlambda2Tidal(lambda2)
    # spinshift = 3/2 * spin/I # 2*pi * (0.22 + 0.192)/2
    # return omega02 * SimpleTransitionFunction(spinshift/omega02)

    a1 = [-0.193, 0.220]
    a2 = [-0.0294, -0.0170]
    Omega = OmegaVSChi(lambda2, spin)
    # print(Omega)

    # I = UniversalRelationMomentOfInertiaVSlambda2Tidal(lambda2)
    # Omega = spin/I # * 2*np.pi# Note that we need to compute the ratio
    # omega_02 / Omega
    i = int((1 + np.sign(spin)) / 2)
    ratio = np.abs((Omega / omega02 * 2 * np.pi))
    return omega02 * SimpleTransitionFunction(ratio * (a1[i] + a2[i] * ratio))


def UniversalRelationSpinShiftomega02VSlambda2Tidalv2(lambda2, spin, omega02):
    """
    Eq. (5.7) of http://arxiv.org/abs/2103.06100
    Gives the spin shift due to the Corriolis force of a neutron star. It is a
    function the dimensionless l=2 tidal deformability: lambda2bar = 2/3 k2 C^5.
    We set its absolute value to be <= the f-mode frequency, as the model
    developes unphysical poles otherwise due to this simplified universal relation
    """
    # I = UniversalRelationMomentOfInertiaVSlambda2Tidal(lambda2)
    # spinshift = 3/2 * spin/I # 2*pi * (0.22 + 0.192)/2
    # return omega02 * SimpleTransitionFunction(spinshift/omega02)

    a1 = [0.517, -0.235]
    a2 = [-0.542, -0.491]
    _val_I = UniversalRelationMomentOfInertiaVSlambda2Tidal(lambda2)
    Omega = spin / _val_I * 2 * np.pi
    i = int((1 + np.sign(Omega)) / 2)
    ratio = np.abs((Omega / omega02))
    return omega02 * SimpleTransitionFunction(ratio * (a1[i] + a2[i] * ratio))


def UniversalRelationSpinShiftomega03VSlambda2Tidal(lambda2, spin, omega03):
    """
    Eq. (5.8) of http://arxiv.org/abs/2103.06100
    Gives the spin shift due to the Corriolis force of a neutron star. It is a
    function the dimensionless l=2 tidal deformability: lambda2bar = 2/3 k2 C^5.
    We set its absolute value to be <= the f-mode frequency, as the model
    developes unphysical poles otherwise due to this simplified universal relation
    """
    _val_I = UniversalRelationMomentOfInertiaVSlambda2Tidal(lambda2)
    spinshift = 5 / 2 * spin / _val_I
    # Omega = OmegaVSChi(lambda2, spin)
    # spinshift = 5/2 * Omega
    # print(spinshift/omega03)
    return omega03 * SimpleTransitionFunction(spinshift / omega03)


def OmegaKeplerian(C):
    """
    Table 1, Eq. (1) of https://iopscience.iop.org/article/10.3847/1538-4357/ac7b86/pdf
    Gives Omega_K
    """
    # MSun [m] = 1476.6250380501249
    prefac = 10**3 / 1476.6250380501249  # convert C from dimensionless to M_Sun / km
    Om_star = C ** (3 / 2)  # Sqrt(M/R^3) in units of individual star mass
    return Om_star * np.poly1d([0.552, 3.304, -35.211, 180.61, -326.48][::-1])(
        prefac * C
    )


def Compactness(lambda2bar):
    """
    Eq. (D4) of https://arxiv.org/pdf/2009.08467
    """
    # return np.poly1d([0.3616998, -0.0354818, 0.0006193849][::-1])(np.log(lambda2bar)) # Gamba Eq. (D4)
    if lambda2bar < 1:
        return 0.5 - lambda2bar * (0.5 - 0.360)
    return np.poly1d([0.360, -0.0355, 0.000705][::-1])(
        np.log(lambda2bar)
    )  # YY Eq. (78)


def Radius(C):
    return 1.0 / C


def KeplerianMomentOfInertia(I_star, C):
    """
    Eq. (4) of https://arxiv.org/pdf/2309.05643
    """
    return np.exp(
        np.poly1d([-2.661, 0.0, -0.6221, 0.03786, 0.01445][::-1])(np.log(I_star * C**3))
    ) / C ** (6)


def FractionalSpinningMomentOfInertia(Om_n):
    """
    Eq. (1) of https://arxiv.org/pdf/2309.05643
    """
    return np.poly1d([0.0, 0.4864, 0.4542, -0.4218, 0.4797][::-1])(Om_n**2)


def SpinningMomentOfInertia(I_star, I_K, Om_n):
    """
    Eq. (2) of https://arxiv.org/pdf/2309.05643
    """
    return I_star + (I_K - I_star) * FractionalSpinningMomentOfInertia(Om_n)


def chiVSLambda2(lambda2bar, Omega):
    I_star = UniversalRelationMomentOfInertiaVSlambda2Tidal(lambda2bar)
    C = Compactness(lambda2bar)
    I_K = KeplerianMomentOfInertia(I_star, C)
    Om_K = OmegaKeplerian(C)

    Om_n = Omega / Om_K
    # print(Om_n)
    if Om_n >= 1.0:
        print(
            f"Neutron star is spinning faster than Keplerian limit: {Omega} > {Om_K} M. \n"
            "f-mode resonance might be wrong."
        )
        Om_n = 1.0

    return SpinningMomentOfInertia(I_star, I_K, Om_n) * Omega


def OmegaVSChi(lambda2bar, chi):
    if lambda2bar < 1:
        lambda2bar = 1.0
    I_star = UniversalRelationMomentOfInertiaVSlambda2Tidal(lambda2bar)
    C = Compactness(lambda2bar)
    I_K = KeplerianMomentOfInertia(I_star, C)
    Om_K = OmegaKeplerian(C)
    chi_abs = np.abs(chi)

    def FindCorrectOmega(Om):
        return chi_abs - SpinningMomentOfInertia(I_star, I_K, Om / Om_K) * Om

    return fsolve(FindCorrectOmega, chi / I_star)[0]


def MomentOfInertia(lambda2bar, chi):
    if lambda2bar < 1:
        lambda2bar = 1.0
    I_star = UniversalRelationMomentOfInertiaVSlambda2Tidal(lambda2bar)
    C = Compactness(lambda2bar)
    I_K = KeplerianMomentOfInertia(I_star, C)
    Om_K = OmegaKeplerian(C)
    chi_abs = np.abs(chi)

    def FindCorrectOmega(Om):
        return chi_abs - SpinningMomentOfInertia(I_star, I_K, Om / Om_K) * Om

    Om = fsolve(FindCorrectOmega, chi / I_star)[0]
    return SpinningMomentOfInertia(I_star, I_K, Om / Om_K)


#
# FUNCTIONS TO INCLUDE STOPPING CONDITION FOR THE DYNAMICS
#


def NSStoppingConditions(EOBpars):
    """Returns the stopping conditions for BNS, NSBH binaries due to tidal effects,
    in particular [omega_NR,omega_Resonance] the instantaneous orbital frequency where NR
    universal relations tell us to disrupt, and the instantaneous orbital frequency where
    the neutron star exhibits 22 f-mode resonance.

    Assumes the tidal parameters have already been rescaled by the respective X_i's

    Args:
        EOBpars (EOBpars): The container of the EOB Parameters
    """
    X_1 = EOBpars.p_params.X_1
    X_2 = EOBpars.p_params.X_2
    lambda2Tidal1 = EOBpars.tidal_params.lambda2Tidal1
    omega02Tidal1 = EOBpars.tidal_params.omega02Tidal1
    spinshiftomega02Tidal1 = EOBpars.tidal_params.spinshiftomega02Tidal1
    lambda2Tidal2 = EOBpars.tidal_params.lambda2Tidal2
    omega02Tidal2 = EOBpars.tidal_params.omega02Tidal2
    spinshiftomega02Tidal2 = EOBpars.tidal_params.spinshiftomega02Tidal2
    omega03Tidal1 = EOBpars.tidal_params.omega03Tidal1
    omega03Tidal2 = EOBpars.tidal_params.omega03Tidal2
    spinshiftomega03Tidal1 = EOBpars.tidal_params.spinshiftomega03Tidal1
    spinshiftomega03Tidal2 = EOBpars.tidal_params.spinshiftomega03Tidal2

    k2T = 3 * (lambda2Tidal1 * X_2 / X_1 + lambda2Tidal2 * X_1 / X_2)
    if k2T > 500:
        omega22_NR = 0.0
    elif k2T <= 0.0:
        omega22_NR = 1.0
    else:
        omega22_NR = (
            0.3596 * (1 + 2.4384e-2 * k2T - 1.7167e-5 * k2T**2) / (1 + 6.8865e-2 * k2T)
        )

    # Note that we assume the convention: BH => omega02Tidal = 0.0
    omega_stop_resonance = min(
        [
            (omega02Tidal1 + spinshiftomega02Tidal1) / 2,
            (omega02Tidal2 + spinshiftomega02Tidal2) / 2,
            # (omega03Tidal1 + spinshiftomega03Tidal1)/3,
            # (omega03Tidal2 + spinshiftomega03Tidal2)/3,
        ]
    )
    if omega_stop_resonance == 0.0:
        omega_stop_resonance = min(
            [
                max(
                    [
                        omega02Tidal1 + spinshiftomega02Tidal1,
                        omega02Tidal2 + spinshiftomega02Tidal2,
                    ]
                )
                / 2,
                max(
                    [
                        omega03Tidal1 + spinshiftomega03Tidal1,
                        omega03Tidal2 + spinshiftomega03Tidal2,
                    ]
                )
                / 3,
            ]
        )

    if omega_stop_resonance == 0:
        omega_stop_resonance = 1.0

    if omega22_NR == 0:
        omega22_NR = 1.0

    return [omega22_NR / 2, omega_stop_resonance]
