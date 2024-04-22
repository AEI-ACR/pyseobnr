"""
Mode mixing fitting function and coefficients from [Berti2014]_.

Used to model mode mixing in the (3,2) and (4,3) modes.
"""

import numpy as np
from numba import jit


@jit(nopython=True)
def mu_fit(j: float, p1: float, p2: float, p3: float, p4: float) -> float:
    """
    Equation 11 from [Berti2014]_ to get the spherical-spheroidal mode mixing coefficients

    Args:
        j (float): dimensionless spin parameter
        p1 (float): first fitting parameter
        p2 (float): second fitting parameter
        p3 (float): third fitting parameter
        p4 (float): fourth fitting parameter

    Returns:
        float: the spherical-spheroidal mode mixing coefficient
    """
    return p1 * abs(j) ** (p2) + p3 * abs(j) ** (p4)


@jit(nopython=True)
def mu(m: int, l: int, lp: int, j: float) -> float:  # noqa: E741
    """
    Equation 11 from [Berti2014]_ applied to the (2,2), (3,2), (3,3) and (4,3) modes using the
    appropriate fitting parameters

    Data files containing the fits
    https://git.ligo.org/waveforms/reviews/seobnrv5/-/blob/main/aligned/docs/swsh_fits.dat
    (copied from https://pages.jh.edu/eberti2/ringdown/ on 02/12/2022)

    Args:
        m (int): m index of the relevant mode
        l (int): :math:`\\ell` index of the relevant mode
        lp (int): index of modes with the same m and lp<=l which give rise to mode mixing
        j (float): dimensionless spin parameter

    Returns:
        float: the spherical-spheroidal mode mixing coefficient for the specified mode
    """

    # (3,2) mode

    if m == 2 and l == 2 and lp == 2:
        return (
            1
            + mu_fit(j, -7.39789e-03, 2.88933e00, -6.61369e-03, 1.71287e01)
            + 1j * mu_fit(j, 1.53046e-02, 1.21928e00, -9.34293e-03, 2.49915e01)
        )

    elif m == 2 and l == 3 and lp == 2:
        return (
            mu_fit(j, -1.03512e-01, 1.22285e00, -5.74989e-02, 8.70536e00)
            + 1j * mu_fit(j, -1.60040e-02, 9.53385e-01, 1.00344e-02, 1.47550e01)
        ) * -1
        # see footnote 4 of 1902.02731
    elif m == 2 and l == 3 and lp == 3:
        return (
            1
            + mu_fit(j, -2.51626e-02, 2.43323e00, -1.30029e-02, 1.09648e01)
            + 1j * mu_fit(j, -3.23812e-02, 9.21248e-01, 1.88165e-02, 1.06988e01)
        )

    # (4,3) mode

    elif m == 3 and l == 3 and lp == 3:
        return (
            1
            + mu_fit(j, -1.20858e-02, 2.94266e00, -1.12402e-02, 1.72866e01)
            + 1j * mu_fit(j, -1.72988e-02, 8.51080e-01, 1.43925e-02, 5.98677e00)
        )

    elif m == 3 and l == 4 and lp == 3:
        return (
            mu_fit(j, -1.32724e-01, 1.24145e00, -7.34932e-02, 8.51670e00)
            + 1j * mu_fit(j, -1.15656e-02, 9.01101e-01, 7.32291e-03, 1.17007e01)
        ) * -1
        # see footnote 4 of 1902.02731
    elif m == 3 and l == 4 and lp == 4:
        return (
            1
            + mu_fit(j, -3.42425e-02, 2.59986e00, -2.12509e-02, 1.25368e01)
            + 1j * mu_fit(j, -4.54823e-02, 9.25823e-01, 2.84355e-02, 1.14204e01)
        )

    else:
        raise NotImplementedError("Coefficient not implemented.")


# Auxiliary functions to obtain the spheroidal mode input values
# See Eq. 73 to Eq. 80 of [SEOBNRv5HM-notes]_


def rho(ell, m, j, h_ellm, h_mm):
    """
    Auxiliary function to obtain the spheroidal mode input values, see Eq.(73) of [SEOBNRv5HM-notes]_

    Args:
        ell (int): :math:`\\ell` index of the relevant mode
        m (int): m index of the relevant mode
        j (float): dimensionless spin parameter
        h_ellm (float): amplitude of the (:math:`\\ell,m`) mode at attachment time
        h_mm (float): amplitude of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time

    Returns:
        float: the value of rho
    """
    return abs(mu(m, ell, m, j)) * (h_mm) / (h_ellm * abs(mu(m, m, m, j)))


def dphi(ell, m, j, phi_ellm, phi_mm):
    """
    Auxiliary function to obtain the spheroidal mode input values, see Eq.(74) of [SEOBNRv5HM-notes]_

    Args:
        ell (int): :math:`\\ell` index of the relevant mode
        m (int): m index of the relevant mode
        j (float): dimensionless spin parameter
        phi_ellm (float): phase of the (:math:`\\ell,m`) mode at attachment time
        phi_mm (float): phase of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time

    Returns:
        float: the value of dphi
    """
    return phi_mm - phi_ellm - np.angle(mu(m, ell, m, j)) + np.angle(mu(m, m, m, j))


def F(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm):
    """
    Auxiliary function to obtain the spheroidal mode input values, see Eq.(75) of [SEOBNRv5HM-notes]_

    Args:
        ell (int): :math:`\\ell` index of the relevant mode
        m (int): m index of the relevant mode
        j (float): dimensionless spin parameter
        h_ellm (float): amplitude of the (:math:`\\ell,m`) mode at attachment time
        h_mm (float): amplitude of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time
        phi_ellm (float): phase of the (:math:`\\ell,m`) mode at attachment time
        phi_mm (float): phase of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time

    Returns:
        float: the value of F
    """
    return np.sqrt(
        (1 - rho(ell, m, j, h_ellm, h_mm) * np.cos(dphi(ell, m, j, phi_ellm, phi_mm)))
        ** 2
        + (rho(ell, m, j, h_ellm, h_mm) ** 2)
        * (np.sin(dphi(ell, m, j, phi_ellm, phi_mm))) ** 2
    )


def alpha(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm):
    """
    Auxiliary function to obtain the spheroidal mode input values, see Eq.(76) of [SEOBNRv5HM-notes]_

    Args:
        ell (int): :math:`\\ell` index of the relevant mode
        m (int): m index of the relevant mode
        j (float): dimensionless spin parameter
        h_ellm (float): amplitude of the (:math:`\\ell,m`) mode at attachment time
        h_mm (float): amplitude of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time
        phi_ellm (float): phase of the (:math:`\\ell,m`) mode at attachment time
        phi_mm (float): phase of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time

    Returns:
        float: the value of alpha
    """
    return np.arctan2(
        -rho(ell, m, j, h_ellm, h_mm) * np.sin(dphi(ell, m, j, phi_ellm, phi_mm)),
        (1 - rho(ell, m, j, h_ellm, h_mm) * np.cos(dphi(ell, m, j, phi_ellm, phi_mm))),
    )


@jit(nopython=True)
def rho_dot(ell, m, j, h_ellm, h_mm, hdot_ellm, hdot_mm):
    """
    Auxiliary function to obtain the spheroidal mode input values, see Eq.(77) of [SEOBNRv5HM-notes]_

    Args:
        ell (int): :math:`\\ell` index of the relevant mode
        m (int): m index of the relevant mode
        j (float): dimensionless spin parameter
        h_ellm (float): amplitude of the (:math:`\\ell,m`) mode at attachment time
        h_mm (float): amplitude of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time
        hdot_ellm (float): amplitude's first derivative of the (:math:`\\ell,m`) mode at attachment time
        hdot_mm (float): amplitude's first derivative of the (:math:`\\ell^{\\prime} = m,m`)
            mode at attachment time

    Returns:
        float: the value of rho_dot
    """
    return abs(mu(m, ell, m, j)) * (hdot_mm / h_ellm - h_mm * hdot_ellm / (h_ellm**2))


@jit(nopython=True)
def dphi_dot(omega_ellm, omega_mm):
    """
    Auxiliary function to obtain the spheroidal mode input values, see Eq.(78) of [SEOBNRv5HM-notes]_

    Args:
        omega_ellm (float): frequency of the (:math:`\\ell,m`) mode at attachment time
        omega_mm (float): frequency of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time

    Returns:
        float: the value of dphi_dot
    """
    return omega_mm - omega_ellm


def F_dot(
    ell, m, j, h_ellm, h_mm, hdot_ellm, hdot_mm, omega_ellm, omega_mm, phi_ellm, phi_mm
):
    """
    Auxiliary function to obtain the spheroidal mode input values, see Eq.(79) of [SEOBNRv5HM-notes]_

    Args:
        ell (int): :math:`\\ell` index of the relevant mode
        m (int): m index of the relevant mode
        j (float): dimensionless spin parameter
        h_ellm (float): amplitude of the (:math:`\\ell,m`) mode at attachment time
        h_mm (float): amplitude of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time
        hdot_ellm (float): amplitude's first derivative of the (:math:`\\ell,m`) mode at attachment time
        hdot_mm (float): amplitude's first derivative of the (:math:`\\ell^{\\prime} = m,m`)
            mode at attachment time
        omega_ellm (float): frequency of the (:math:`\\ell,m`) mode at attachment time
        omega_mm (float): frequency of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time
        phi_ellm (float): phase of the (:math:`\\ell,m`) mode at attachment time
        phi_mm (float): phase of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time

    Returns:
        float: the value of F_dot
    """
    return (
        rho(ell, m, j, h_ellm, h_mm)
        * rho_dot(ell, m, j, h_ellm, h_mm, hdot_ellm, hdot_mm)
        + rho(ell, m, j, h_ellm, h_mm)
        * np.sin(dphi(ell, m, j, phi_ellm, phi_mm))
        * dphi_dot(omega_ellm, omega_mm)
        - rho_dot(ell, m, j, h_ellm, h_mm, hdot_ellm, hdot_mm)
        * np.cos(dphi(ell, m, j, phi_ellm, phi_mm))
    ) / F(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm)


def alpha_dot(
    ell, m, j, h_ellm, h_mm, hdot_ellm, hdot_mm, omega_ellm, omega_mm, phi_ellm, phi_mm
):
    """
    Auxiliary function to obtain the spheroidal mode input values, see Eq.(80) of [SEOBNRv5HM-notes]_

    Args:
        ell (int): :math:`\\ell` index of the relevant mode
        m (int): m index of the relevant mode
        j (float): dimensionless spin parameter
        h_ellm (float): amplitude of the (:math:`\\ell,m`) mode at attachment time
        h_mm (float): amplitude of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time
        hdot_ellm (float): amplitude's first derivative of the (:math:`\\ell,m`) mode at attachment time
        hdot_mm (float): amplitude's first derivative of the (:math:`\\ell^{\\prime} = m,m`)
            mode at attachment time
        omega_ellm (float): frequency of the (:math:`\\ell,m`) mode at attachment time
        omega_mm (float): frequency of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time
        phi_ellm (float): phase of the (:math:`\\ell,m`) mode at attachment time
        phi_mm (float): phase of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time

    Returns:
        float: the value of alpha_dot
    """
    return (
        rho(ell, m, j, h_ellm, h_mm) ** 2 * dphi_dot(omega_ellm, omega_mm)
        - rho(ell, m, j, h_ellm, h_mm)
        * np.cos(dphi(ell, m, j, phi_ellm, phi_mm))
        * dphi_dot(omega_ellm, omega_mm)
        - rho_dot(ell, m, j, h_ellm, h_mm, hdot_ellm, hdot_mm)
        * np.sin(dphi(ell, m, j, phi_ellm, phi_mm))
    ) / (F(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm) ** 2)


# Spheroidal mode input values
# See Eq. 81 to Eq. 84 of [SEOBNRv5HM-notes]_


def h_ellm0_nu(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm):
    """
    Computes the amplitude of the spheroidal :math:`(\\ell,m,0)` mode at at attachment time
    from the :math:`(\\ell,m)` and (:math:`\\ell^{\\prime} = m,m`) spherical modes

    Args:
        ell (int): :math:`\\ell` index of the relevant mode
        m (int): m index of the relevant mode
        j (float): dimensionless spin parameter
        h_ellm (float): amplitude of the (:math:`\\ell,m`) mode at attachment time
        h_mm (float): amplitude of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time
        phi_ellm (float): phase of the (:math:`\\ell,m`) mode at attachment time
        phi_mm (float): phase of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time

    Returns:
        float: the amplitude of the spheroidal :math:`(\\ell,m,0)` mode at at attachment time
    """
    return (
        h_ellm
        * F(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm)
        / np.abs(mu(m, ell, ell, j))
    )


def phi_ellm0(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm):
    """
    Computes the phase of the spheroidal :math:`(\\ell,m,0)` mode at at attachment time
    from the (:math:`\\ell,m`) and (:math:`\\ell^{\\prime} = m,m`) spherical modes

    Args:
        ell (int): :math:`\\ell` index of the relevant mode
        m (int): m index of the relevant mode
        j (float): dimensionless spin parameter
        h_ellm (float): amplitude of the (:math:`\\ell,m`) mode at attachment time
        h_mm (float): amplitude of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time
        phi_ellm (float): phase of the (:math:`\\ell,m`) mode at attachment time
        phi_mm (float): phase of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time

    Returns:
        float: the phase of the spheroidal :math:`(\\ell,m,0)` mode at at attachment time
    """
    return (
        phi_ellm
        + np.angle(mu(m, ell, ell, j))
        + alpha(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm)
    )


def hdot_ellm0_nu(
    ell, m, j, h_ellm, h_mm, hdot_ellm, hdot_mm, omega_ellm, omega_mm, phi_ellm, phi_mm
):
    """
    Computes the amplitude's first derivative of the spheroidal :math:`(\\ell,m,0)` mode at at attachment time
    from the (:math:`\\ell,m`) and (:math:`\\ell^{\\prime} = m,m`) spherical modes

    Args:
        ell (int): :math:`\\ell` index of the relevant mode
        m (int): m index of the relevant mode
        j (float): dimensionless spin parameter
        h_ellm (float): amplitude of the (:math:`\\ell,m`) mode at attachment time
        h_mm (float): amplitude of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time
        hdot_ellm (float): amplitude's first derivative of the (:math:`\\ell,m`) mode at attachment time
        hdot_mm (float): amplitude's first derivative of the (:math:`\\ell^{\\prime} = m,m`)
            mode at attachment time
        omega_ellm (float): frequency of the (:math:`\\ell,m`) mode at attachment time
        omega_mm (float): frequency of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time
        phi_ellm (float): phase of the (:math:`\\ell,m`) mode at attachment time
        phi_mm (float): phase of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time

    Returns:
        float: the amplitude's first derivative of the spheroidal :math:`(\\ell,m,0)` mode at at attachment time
    """
    return (
        hdot_ellm * F(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm)
        + h_ellm
        * F_dot(
            ell,
            m,
            j,
            h_ellm,
            h_mm,
            hdot_ellm,
            hdot_mm,
            omega_ellm,
            omega_mm,
            phi_ellm,
            phi_mm,
        )
    ) / np.abs(mu(m, ell, ell, j))


def omega_ellm0(
    ell, m, j, h_ellm, h_mm, hdot_ellm, hdot_mm, omega_ellm, omega_mm, phi_ellm, phi_mm
):
    """
    Computes the frequency of the spheroidal :math:`(\\ell,m,0)` mode at at attachment time
    from the (:math:`\\ell,m`) and (:math:`\\ell^{\\prime} = m,m`) spherical modes

    Args:
        ell (int): :math:`\\ell` index of the relevant mode
        m (int): m index of the relevant mode
        j (float): dimensionless spin parameter
        h_ellm (float): amplitude of the (:math:`\\ell,m`) mode at attachment time
        h_mm (float): amplitude of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time
        hdot_ellm (float): amplitude's first derivative of the (:math:`\\ell,m`) mode at attachment time
        hdot_mm (float): amplitude's first derivative of the (:math:`\\ell^{\\prime} = m,m`)
            mode at attachment time
        omega_ellm (float): frequency of the (:math:`\\ell,m`) mode at attachment time
        omega_mm (float): frequency of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time
        phi_ellm (float): phase of the (:math:`\\ell,m`) mode at attachment time
        phi_mm (float): phase of the (:math:`\\ell^{\\prime} = m,m`) mode at attachment time

    Returns:
        float: the frequency of the spheroidal :math:`(\\ell,m,0)` mode at at attachment time
    """
    return omega_ellm + alpha_dot(
        ell,
        m,
        j,
        h_ellm,
        h_mm,
        hdot_ellm,
        hdot_mm,
        omega_ellm,
        omega_mm,
        phi_ellm,
        phi_mm,
    )
