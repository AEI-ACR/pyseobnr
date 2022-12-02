import numpy as np
from numba import jit
# Mode mixing fitting function and coefficients from Berti, Klein https://arxiv.org/abs/1408.1860. Used to model mode mixing in the (3,2) and (4,3) modes.

@jit(nopython=True)
def mu_fit(j: float, p1: float, p2: float, p3: float, p4: float) -> float:
    """
    Equation 11 from https://arxiv.org/abs/1408.1860 to get the spherical-spheroidal mode mixing coefficients

    Parameters
    ----------
    j:
        Dimensionless spin parameter
    p1,p2,p3,p4:
        Fitting parameters

    Return the spherical-spheroidal mode mixing coefficient
    """
    return p1 * abs(j) ** (p2) + p3 * abs(j) ** (p4)

@jit(nopython=True)
def mu(m: int, l: int, lp: int, j: float) -> float:
    """
    Equation 11 from https://arxiv.org/abs/1408.1860 applied to the (2,2), (3,2), (3,3) and (4,3) modes using the appropriate fitting parameters
    Data files containing the fits https://git.ligo.org/waveforms/reviews/seobnrv5/-/blob/main/aligned/docs/swsh_fits.dat
    (copied from https://pages.jh.edu/eberti2/ringdown/ on 02/12/2022)

    Parameters
    ----------
    m,l:
        Relevant mode
    lp:
        Modes with the same m and lp<=l which give rise to mode mixing
    j:
        Dimensionless spin parameter

    Return the spherical-spheroidal mode mixing coefficient for the specified mode
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


def rho(ell, m, j, h_ellm, h_mm):
    return abs(mu(m, ell, m, j)) * (h_mm) / (h_ellm * abs(mu(m, m, m, j)))


# Auxiliary functions to obtain the spheroidal mode input values 
# See Eq. 73 to Eq. 80 of https://git.ligo.org/waveforms/reviews/seobnrv5/-/blob/main/aligned/docs/SEOBNRv5HM.pdf

def dphi(ell, m, j, phi_ellm, phi_mm):
    return phi_mm - phi_ellm - np.angle(mu(m, ell, m, j)) + np.angle(mu(m, m, m, j))


def F(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm):
    return np.sqrt(
        (1 - rho(ell, m, j, h_ellm, h_mm) * np.cos(dphi(ell, m, j, phi_ellm, phi_mm)))
        ** 2
        + (rho(ell, m, j, h_ellm, h_mm) ** 2)
        * (np.sin(dphi(ell, m, j, phi_ellm, phi_mm))) ** 2
    )

def alpha(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm):
    return np.arctan2(
        -rho(ell, m, j, h_ellm, h_mm) * np.sin(dphi(ell, m, j, phi_ellm, phi_mm)),
        (1 - rho(ell, m, j, h_ellm, h_mm) * np.cos(dphi(ell, m, j, phi_ellm, phi_mm))),
    )

@jit(nopython=True)
def rho_dot(ell, m, j, h_ellm, h_mm, hdot_ellm, hdot_mm):
    return abs(mu(m, ell, m, j)) * (hdot_mm / h_ellm - h_mm * hdot_ellm / (h_ellm ** 2))

@jit(nopython=True)
def dphi_dot(omega_ellm, omega_mm):
    return omega_mm - omega_ellm

def F_dot(
    ell, m, j, h_ellm, h_mm, hdot_ellm, hdot_mm, omega_ellm, omega_mm, phi_ellm, phi_mm
):
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
    return (
        rho(ell, m, j, h_ellm, h_mm) ** 2 * dphi_dot(omega_ellm, omega_mm)
        - rho(ell, m, j, h_ellm, h_mm)
        * np.cos(dphi(ell, m, j, phi_ellm, phi_mm))
        * dphi_dot(omega_ellm, omega_mm)
        - rho_dot(ell, m, j, h_ellm, h_mm, hdot_ellm, hdot_mm)
        * np.sin(dphi(ell, m, j, phi_ellm, phi_mm))
    ) / (F(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm) ** 2)


# Spheroidal mode input values 
# See Eq. 81 to Eq. 84 of https://git.ligo.org/waveforms/reviews/seobnrv5/-/blob/main/aligned/docs/SEOBNRv5HM.pdf

def h_ellm0_nu(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm):
    return (
        h_ellm
        * F(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm)
        / np.abs(mu(m, ell, ell, j))
    )

def phi_ellm0(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm):
    return (
        phi_ellm
        + np.angle(mu(m, ell, ell, j))
        + alpha(ell, m, j, h_ellm, h_mm, phi_ellm, phi_mm)
    )

def hdot_ellm0_nu(
    ell, m, j, h_ellm, h_mm, hdot_ellm, hdot_mm, omega_ellm, omega_mm, phi_ellm, phi_mm
):
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
