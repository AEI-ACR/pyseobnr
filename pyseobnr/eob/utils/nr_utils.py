"""BBH final spins and masses calculation utilities

The functions (and corresponding dependencies) come from :py:mod:`lalinference.imrtgr.nrutils`
and are copied verbatim (LAL version 7.24, Feb. 2025):

* :py:func:`.bbh_final_spin_non_precessing_HBR2016`
* :py:func:`.bbh_final_mass_non_precessing_UIB2016`
* :py:func:`.bbh_final_spin_precessing_HBR2016`

"""

import numpy as np


def calc_isco_radius(a):
    """
    Calculate the ISCO radius of a Kerr BH as a function of the Kerr parameter using
    eqns. 2.5 and 2.8 from Ori and Thorne, Phys Rev D 62, 24022 (2000)

    Parameters
    ----------
    a : Kerr parameter

    Returns
    -------
    ISCO radius
    """

    a = np.minimum(np.array(a), 1.0)  # Only consider a <=1, to avoid numerical problems

    # Ref. Eq. (2.5) of Ori, Thorne Phys Rev D 62 124022 (2000)
    z1 = 1.0 + (1.0 - a**2.0) ** (1.0 / 3) * (
        (1.0 + a) ** (1.0 / 3) + (1.0 - a) ** (1.0 / 3)
    )
    z2 = np.sqrt(3.0 * a**2 + z1**2)
    a_sign = np.sign(a)
    return 3 + z2 - np.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)) * a_sign


def _bbh_HBR2016_setup(m1, m2, chi1, chi2):
    """
    Setup function for the Hofmann, Barausse, and Rezzolla final spin fits to vectorize the masses
    and spins and calculate the mass ratio.
    """

    # Vectorize if arrays are provided as input
    m1 = np.vectorize(float)(np.array(m1))
    m2 = np.vectorize(float)(np.array(m2))
    chi1 = np.vectorize(float)(np.array(chi1))
    chi2 = np.vectorize(float)(np.array(chi2))

    return m1, m2, chi1, chi2, m2 / m1


def _bbh_HBR2016_ell(m1, m2, chi1z, chi2z, version):
    """
    Compute the orbital angular momentum ell used by the Hofmann, Barausse, and Rezzolla final spin fit
    [from ApJL 825, L19 (2016)], henceforth HBR.
    The three versions available correspond to the three choices of fit coefficients given in Table 1
    of that paper, with 6, 16, and 20 coefficients, respectively.

    version can thus be "M1J2", "M3J3", or "M3J4"
    """

    # Set upper bounds for sums and coefficients from Table 1 in HBR; k00 is calculated from
    # Eq. (11) in the paper

    if version == "M1J2":
        # nM = 1
        nJ = 2

        k00 = -3.82116
        k01 = -1.2019
        k02 = -1.20764
        k10 = 3.79245
        k11 = 1.18385
        k12 = 4.90494
        xi = 0.41616

    elif version == "M3J3":
        # nM = 3
        nJ = 3

        k00 = -5.83164
        k01 = 2.87025
        k02 = -1.53315
        k03 = -3.78893
        k10 = 32.9127
        k11 = -62.9901
        k12 = 10.0068
        k13 = 56.1926
        k20 = -136.832
        k21 = 329.32
        k22 = -13.2034
        k23 = -252.27
        k30 = 210.075
        k31 = -545.35
        k32 = -3.97509
        k33 = 368.405
        xi = 0.463926

    elif version == "M3J4":
        # nM = 3
        nJ = 4

        k00 = -5.97723
        k01 = 3.39221
        k02 = 4.48865
        k03 = -5.77101
        k04 = -13.0459
        k10 = 35.1278
        k11 = -72.9336
        k12 = -86.0036
        k13 = 93.7371
        k14 = 200.975
        k20 = -146.822
        k21 = 387.184
        k22 = 447.009
        k23 = -467.383
        k24 = -884.339
        k30 = 223.911
        k31 = -648.502
        k32 = -697.177
        k33 = 753.738
        k34 = 1166.89
        xi = 0.474046

    else:
        raise ValueError('Unknown version--should be either "M1J2", "M3J3", or "M3J4".')

    # Calculate eta, atot, and aeff; note that HBR call the symmetric mass ratio nu instead of eta

    m = m1 + m2
    q = m2 / m1
    eta = m1 * m2 / (m * m)

    atot = (chi1z + q * q * chi2z) / ((1.0 + q) * (1.0 + q))  # Eq. (12) in HBR
    aeff = atot + xi * eta * (
        chi1z + chi2z
    )  # Inline equation below Eq. (7); see also Eq. (15) for the precessing analogue

    # Calculate ISCO energy and angular momentum
    r_isco = calc_isco_radius(aeff)
    e_isco = (1.0 - 2.0 / 3.0 / r_isco) ** 0.5  # Eq. (2) in HBR
    l_isco = (
        2.0 / (3.0 * 3.0**0.5) * (1.0 + 2.0 * (3.0 * r_isco - 2.0) ** 0.5)
    )  # Eq. (3) in HBR

    # The following expressions give the double sum in Eq. (13) in HBR (without the overall factor of eta that
    # is put in below) specialized to the nJ values for the three versions, i.e., nJ = 2 => M1J2,
    # nJ = 3 => M3J3, nJ = 4 => M3J4
    if nJ >= 2:
        aeff2 = aeff * aeff
        ksum = k00 + k01 * aeff + k02 * aeff2 + eta * (k10 + k11 * aeff + k12 * aeff2)
    if nJ >= 3:
        eta2 = eta * eta
        eta3 = eta2 * eta
        aeff3 = aeff2 * aeff
        ksum = (
            ksum
            + (k03 + eta * k13) * aeff3
            + eta2 * (k20 + k21 * aeff + k22 * aeff2 + k23 * aeff3)
            + eta3 * (k30 + k31 * aeff + k32 * aeff2 + k33 * aeff3)
        )
    if nJ >= 4:
        ksum = ksum + (k04 + eta * k14 + eta2 * k24 + eta3 * k34) * aeff3 * aeff

    # Calculate the absolute value of ell
    ell = abs((l_isco - 2.0 * atot * (e_isco - 1.0)) + eta * ksum)  # Eq. (13) in HBR

    return ell


def bbh_final_spin_non_precessing_HBR2016(m1, m2, chi1z, chi2z, version="M3J3"):
    """
    Calculate the (signed) dimensionless spin of the final BH resulting from the
    merger of two black holes with aligned spins using the fit from Hofmann, Barausse, and
    Rezzolla ApJL 825, L19 (2016), henceforth HBR.

    The three versions available correspond to the three choices of fit coefficients given in Table 1 of that
    paper, with 6, 16, and 20 coefficients, respectively.

    version can thus be "M1J2", "M3J3", or "M3J4"

    m1, m2: component masses
    chi1z, chi2z: components of the dimensionless spins of the two BHs along the orbital angular momentum
    """

    # Calculate q and vectorize the masses and spins if arrays are provided as input

    m1, m2, chi1z, chi2z, q = _bbh_HBR2016_setup(m1, m2, chi1z, chi2z)

    # Calculate the final spin

    atot = (chi1z + chi2z * q * q) / ((1.0 + q) * (1.0 + q))  # Eq. (12) in HBR

    ell = _bbh_HBR2016_ell(m1, m2, chi1z, chi2z, version)

    return atot + ell / (
        1.0 / q + 2.0 + q
    )  # Eq. (12) in HBR, writing the symmetric mass ratio in terms of q


def bbh_final_spin_precessing_HBR2016(
    m1, m2, chi1, chi2, tilt1, tilt2, phi12, version="M3J3"
):
    """
    Calculate the dimensionless spin of the final BH resulting from the
    merger of two black holes with precessing spins using the fit from
    Hofmann, Barausse, and Rezzolla ApJL 825, L19 (2016), henceforth HBR.

    The three versions available correspond to the three choices of fit coefficients given in Table 1 of that paper,
    with 6, 16, and 20 coefficients, respectively.

    version can thus be "M1J2", "M3J3", or "M3J4"

    m1, m2: component masses
    chi1, chi2: dimensionless spins of two BHs
    tilt1, tilt2: tilt angles of the spins from the orbital angular momentum
    phi12: angle between in-plane spin components
    """

    # Vectorize the function if arrays are provided as input
    if (
        np.size(m1)
        * np.size(m2)
        * np.size(chi1)
        * np.size(chi2)
        * np.size(tilt1)
        * np.size(tilt2)
        * np.size(phi12)
        > 1
    ):
        return np.vectorize(bbh_final_spin_precessing_HBR2016)(
            m1, m2, chi1, chi2, tilt1, tilt2, phi12, version
        )

    # Calculate q and vectorize the masses and spins if arrays are provided as input

    m1, m2, chi1, chi2, q = _bbh_HBR2016_setup(m1, m2, chi1, chi2)

    # Vectorize the spin angles if arrays are provided as input
    tilt1 = np.vectorize(float)(np.array(tilt1))
    tilt2 = np.vectorize(float)(np.array(tilt2))
    phi12 = np.vectorize(float)(np.array(phi12))

    # Set eps (\epsilon_\beta or \epsilon_\gamma) to the value given below Eq. (18) in HBR

    eps = 0.024

    # Computing angles defined in Eq. (17) of HBR. The betas and gammas expressions are for the starred
    # quantities computed using the second (approximate) equality in Eq. (18) in HBR
    cos_beta = np.cos(tilt1)
    cos_betas = np.cos(tilt1 + eps * np.sin(tilt1))
    cos_gamma = np.cos(tilt2)
    cos_gammas = np.cos(tilt2 + eps * np.sin(tilt2))
    cos_alpha = (
        (1 - cos_beta * cos_beta) * (1 - cos_gamma * cos_gamma)
    ) ** 0.5 * np.cos(
        phi12
    ) + cos_beta * cos_gamma  # This rewrites
    # the inner product definition of cos_alpha in terms of cos_beta, cos_gamma, and phi12

    # Define a shorthand and compute the final spin

    q2 = q * q

    ell = _bbh_HBR2016_ell(m1, m2, chi1 * cos_betas, chi2 * cos_gammas, version)

    # Compute the final spin value [Eq. (16) in HBR], truncating the argument of the square root at zero
    # if it becomes negative
    sqrt_arg = (
        chi1 * chi1
        + chi2 * chi2 * q2 * q2
        + 2.0 * chi1 * chi2 * q2 * cos_alpha
        + 2.0 * (chi1 * cos_betas + chi2 * q2 * cos_gammas) * ell * q
        + ell * ell * q2
    )
    if sqrt_arg < 0.0:
        print(
            "bbh_final_spin_precessing_HBR2016(): The argument of the square root is %f; truncating it to zero."
            % sqrt_arg
        )
        sqrt_arg = 0.0

    # Return the final spin value [Eq. (16) in HBR]
    return sqrt_arg**0.5 / ((1.0 + q) * (1.0 + q))


def bbh_UIBfits_setup(m1, m2, chi1, chi2):
    """
    common setup function for UIB final-state and luminosity fit functions
    """

    # Vectorize the function if arrays are provided as input
    m1 = np.vectorize(float)(np.array(m1))
    m2 = np.vectorize(float)(np.array(m2))
    chi1 = np.vectorize(float)(np.array(chi1))
    chi2 = np.vectorize(float)(np.array(chi2))

    if np.any(m1 < 0):
        raise ValueError("m1 must not be negative")
    if np.any(m2 < 0):
        raise ValueError("m2 must not be negative")

    if np.any(abs(chi1) > 1):
        raise ValueError("chi1 has to be in [-1, 1]")
    if np.any(abs(chi2) > 1):
        raise ValueError("chi2 has to be in [-1, 1]")

    # binary masses
    m = m1 + m2
    if np.any(m <= 0):
        raise ValueError("m1+m2 must be positive")
    msq = m * m
    m1sq = m1 * m1
    m2sq = m2 * m2

    # symmetric mass ratio
    eta = m1 * m2 / msq
    if np.any(eta > 0.25):
        print(
            "Truncating eta from above to 0.25. This should only be necessary in some rounding corner cases, "
            "but better check your m1 and m2 inputs..."
        )
        eta = np.minimum(eta, 0.25)
    if np.any(eta < 0.0):
        print(
            "Truncating negative eta to 0.0. This should only be necessary in some rounding corner cases, "
            "but better check your m1 and m2 inputs..."
        )
        eta = np.maximum(eta, 0.0)
    eta2 = eta * eta
    eta3 = eta2 * eta
    eta4 = eta2 * eta2

    # spin variables (in m = 1 units)
    S1 = chi1 * m1sq / msq  # spin angular momentum 1
    S2 = chi2 * m2sq / msq  # spin angular momentum 2
    Stot = S1 + S2  # total spin
    Shat = (chi1 * m1sq + chi2 * m2sq) / (
        m1sq + m2sq
    )  # effective spin, = msq*Stot/(m1sq+m2sq)
    Shat2 = Shat * Shat
    Shat3 = Shat2 * Shat
    Shat4 = Shat2 * Shat2

    # asymmetric spin combination (spin difference), where the paper assumes m1>m2
    # to make our implementation symmetric under simultaneous exchange of m1<->m2 and chi1<->chi2,
    # we flip the sign here when m2>m1
    chidiff = chi1 - chi2
    if np.any(m2 > m1):
        chidiff = np.sign(m1 - m2) * chidiff
    chidiff2 = chidiff * chidiff

    # typical squareroots and functions of eta
    sqrt2 = 2.0**0.5
    sqrt3 = 3.0**0.5
    sqrt1m4eta = (1.0 - 4.0 * eta) ** 0.5

    return (
        m,
        eta,
        eta2,
        eta3,
        eta4,
        Stot,
        Shat,
        Shat2,
        Shat3,
        Shat4,
        chidiff,
        chidiff2,
        sqrt2,
        sqrt3,
        sqrt1m4eta,
    )


def bbh_final_mass_non_precessing_UIB2016(m1, m2, chi1, chi2, version="v2"):
    """
    Calculate the final mass with the aligned-spin NR fit
    by Xisco Jimenez Forteza, David Keitel, Sascha Husa et al.
    [LIGO-P1600270] [https://arxiv.org/abs/1611.00332]
    versions v1 and v2 use the same ansatz,
    with v2 calibrated to additional SXS and RIT data

    m1, m2: component masses
    chi1, chi2: dimensionless spins of two BHs
    Results are symmetric under simultaneous exchange of m1<->m2 and chi1<->chi2.
    """

    (
        m,
        eta,
        eta2,
        eta3,
        eta4,
        Stot,
        Shat,
        Shat2,
        Shat3,
        Shat4,
        chidiff,
        chidiff2,
        sqrt2,
        sqrt3,
        sqrt1m4eta,
    ) = bbh_UIBfits_setup(m1, m2, chi1, chi2)

    if version == "v1":
        # rational-function Pade coefficients (exact) from Eq. (22) of 1611.00332v1
        b10 = 0.487
        b20 = 0.295
        b30 = 0.17
        b50 = -0.0717
        # fit coefficients from Tables VII-X of 1611.00332v1
        # values at increased numerical precision copied from
        # https://gravity.astro.cf.ac.uk/cgit/cardiff_uib_share/tree/Luminosity_and_Radiated_Energy/UIBfits/LALInference/EradUIB2016_pyform_coeffs.txt
        # git commit 636e5a71462ecc448060926890aa7811948d5a53
        a2 = 0.5635376058169299
        a3 = -0.8661680065959881
        a4 = 3.181941595301782
        b1 = -0.15800074104558132
        b2 = -0.15815904609933157
        b3 = -0.14299315232521553
        b5 = 8.908772171776285
        f20 = 3.8071100104582234
        f30 = 25.99956516423936
        f50 = 1.552929335555098
        f10 = 1.7004558922558886
        f21 = 0.0
        d10 = -0.12282040108157262
        d11 = -3.499874245551208
        d20 = 0.014200035799803777
        d30 = -0.01873720734635449
        d31 = -5.1830734185518725
        f11 = 14.39323998088354
        f31 = -232.25752840151296
        f51 = -0.8427987782523847

    elif version == "v2":
        # rational-function Pade coefficients (exact) from Eq. (22) of LIGO-P1600270-v4
        b10 = 0.346
        b20 = 0.211
        b30 = 0.128
        b50 = -0.212
        # fit coefficients from Tables VII-X of LIGO-P1600270-v4
        # values at increased numerical precision copied from
        # https://dcc.ligo.org/DocDB/0128/P1600270/004/FinalStateUIB2016_suppl_Erad_coeffs.txt
        a2 = 0.5609904135313374
        a3 = -0.84667563764404
        a4 = 3.145145224278187
        b1 = -0.2091189048177395
        b2 = -0.19709136361080587
        b3 = -0.1588185739358418
        b5 = 2.9852925538232014
        f20 = 4.271313308472851
        f30 = 31.08987570280556
        f50 = 1.5673498395263061
        f10 = 1.8083565298668276
        f21 = 0.0
        d10 = -0.09803730445895877
        d11 = -3.2283713377939134
        d20 = 0.01118530335431078
        d30 = -0.01978238971523653
        d31 = -4.91667749015812
        f11 = 15.738082204419655
        f31 = -243.6299258830685
        f51 = -0.5808669012986468

    else:
        raise ValueError('Unknown version -- should be either "v1" or "v2".')

    # Calculate the radiated-energy fit from Eq. (28) of LIGO-P1600270-v4
    Erad = (
        (
            ((1.0 + -2.0 / 3.0 * sqrt2) * eta + a2 * eta2 + a3 * eta3 + a4 * eta4)
            * (
                1.0
                + b10
                * b1
                * Shat
                * (f10 + f11 * eta + (16.0 - 16.0 * f10 - 4.0 * f11) * eta2)
                + b20
                * b2
                * Shat2
                * (f20 + f21 * eta + (16.0 - 16.0 * f20 - 4.0 * f21) * eta2)
                + b30
                * b3
                * Shat3
                * (f30 + f31 * eta + (16.0 - 16.0 * f30 - 4.0 * f31) * eta2)
            )
        )
        / (
            1.0
            + b50
            * b5
            * Shat
            * (f50 + f51 * eta + (16.0 - 16.0 * f50 - 4.0 * f51) * eta2)
        )
        + d10 * sqrt1m4eta * eta2 * (1.0 + d11 * eta) * chidiff
        + d30 * Shat * sqrt1m4eta * eta * (1.0 + d31 * eta) * chidiff
        + d20 * eta3 * chidiff2
    )

    # Convert to actual final mass
    Mf = m * (1.0 - Erad)

    return Mf
