"""
Classes and functions dedicated to the processing of the anti-symmetric modes.
The fits are returned by :py:func:`.fits_iv_mrd_antisymmetric`.
"""

from __future__ import annotations

from collections.abc import Collection, Container
from dataclasses import dataclass
from typing import Final, TypedDict

import numpy as np
import quaternion
from pygsl_lite import spline
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline

from ..fits.antisymmetric_modes import get_predictor_from_fits
from ..fits.MR_fits import MergerRingdownFits
from .compute_MR import MRAnzatze


def sigma(chiA, chiB, mA, mB):
    """Computes spin difference vector :math:`\\Sigma=S_2/m_2 - S_1/m_1`.

    :param chiA: dimensionless spin of the primary
    :param chiB: dimensionless spin of the secondary
    :param mA: mass of the primary
    :param mB: mass of the secondary
    :returns: symmetric spin difference :math:`\\Sigma`

    See notation section §1 of [Estelles2025]_.
    """
    return mB * chiB - mA * chiA


def project(v, axis):
    """Computes projection of vector ``v`` into vector ``axis``"""
    # performances of einsum are better than doing
    # np.sum(axis * v, axis=1)
    return np.einsum("ij,ij->i", v, axis)


def compute_time_for_asym(t_old, phi_orb) -> tuple[np.ndarray, np.ndarray]:
    """Computes new time array for the full resolution of the orbital timescale.

    This is needed to accurately construct the anti-symmetric modes.
    See section §3.2 of [Estelles2025]_.

    Since orbital phase has to be interpolated for this computation, its interpolated
    array on the new time array is also returned.

    :param t_old: previous time array
    :param phi_orb: orbital phase on the previous time array
    :returns: tuple containing the new time array and the interpolated orbital phase
        on this time array.
    """

    n = len(t_old)
    intrp_orb = spline.cspline(n)
    intrp_orb.init(t_old, phi_orb)

    t_new = [0]
    t_aux = 0
    exit_flag = False
    nn = 32
    while t_aux < t_old[-1]:
        w = intrp_orb.eval_deriv_vector([t_aux])
        period = 2 * np.pi / w[0]
        dt_aux = period / nn

        for j in range(2):
            tt = t_aux + (j + 1) * dt_aux
            if tt < t_old[-1]:
                t_new.append(tt)
            else:
                exit_flag = True

        if exit_flag:
            break
        t_aux = t_new[-1]

    t_new = np.array(t_new)

    # Orbital phase in new time-array
    phi_orb_interp = intrp_orb.eval_e_vector(t_new)

    return t_new, phi_orb_interp


def get_all_dynamics(
    dyn: dict[str, np.ndarray],
    t: np.ndarray,
    mA: float,
    mB: float,
) -> dict[str, np.ndarray]:
    """Computes the in-plane basis vectors :math:`\\boldsymbol{n}` and
    :math:`\\boldsymbol{\\lambda}` as well
    as the needed projections of the individual spins onto these vectors (and L_N)
    for the evaluation of the anti-symmetric modes.

    :param dyn: dictionary of dynamic quantities on the angles. In particular, the quantities
        ``orbphase``, ``L_N``, ``q_copr``, ``chiA`` and ``chiB`` should be provided and
        should be arrays of time size as ``t``.
    :param t: time array for the anti-symmetric modes, see :py:func:`.compute_time_for_asym`
    :param mA: mass of the primary
    :param mB: mass of the secondary
    :return: dictionary with the needed quantities to evaluate the PN anti-symmetric modes. Contains
        a copy of the keys passed in ``dyn``.

    .. seealso::

        Section §3.2 (and notation section) of [Estelles2025]_.
    """

    orbital_phase = dyn["orbphase"]
    intrp = CubicSpline(t, orbital_phase)
    omega = intrp.derivative()(t)
    v = omega ** (1.0 / 3)

    L_N = dyn["L_N"]
    q = dyn["q_copr"]
    chiA = dyn["chiA"]
    chiB = dyn["chiB"]

    q2 = quaternion.from_float_array(
        np.column_stack(
            (np.cos(orbital_phase / 2), (L_N.T * np.sin(orbital_phase / 2)).T)
        )
    )

    q2_times_q = q2 * q
    x = quaternion.as_float_array(q2_times_q * quaternion.x * q2_times_q.conjugate())[
        :, 1:
    ]

    lamb = np.cross(L_N, x)
    Sigma = sigma(chiA, chiB, mA, mB)
    S1 = mA**2 * chiA
    S2 = mB**2 * chiB
    S = S1 + S2
    Sigma_l = project(Sigma, L_N)
    Sigma_n = project(Sigma, x)
    Sigma_lambda = project(Sigma, lamb)
    S_n = project(S, x)
    S_lambda = project(S, lamb)
    # delta = (mA - mB) / (mA + mB)
    S1_l = project(S1, L_N)
    S2_l = project(S2, L_N)
    S_l = project(S, L_N)
    S1_n = project(S1, x)
    S2_n = project(S2, x)
    S1_lambda = project(S1, lamb)
    S2_lambda = project(S2, lamb)

    dyn_ret = {
        "t": t,
        "L_N": L_N,
        "n_hat": x,
        "Sigma_l": Sigma_l,
        "Sigma_n": Sigma_n,
        "Sigma_lambda": Sigma_lambda,
        "S_n": S_n,
        "S_lambda": S_lambda,
        "S_l": S_l,
        "S1_n": S1_n,
        "S1_lambda": S1_lambda,
        "S1_l": S1_l,
        "S2_n": S2_n,
        "S2_lambda": S2_lambda,
        "S2_l": S2_l,
        "v": v,
        "orbphase": orbital_phase,
        "exp_orbphase": dyn["exp_orbphase"],
    }

    return dyn_ret


@dataclass
class ParameterForFits:
    """Holds the parameters for the fits corresponding to the anti-symmetric modes.

    The vectors :math:`\\boldsymbol{n}` and :math:`\\boldsymbol{\\lambda}` are defined
    as being, respectively:

        * the unit vector pointing from the lighter to the heavier black hole
        * the unit vector parallel to the rate of change of :math:`\\boldsymbol{n}` :
          :math:`\\frac{d \\boldsymbol{n}}{dt}`

    The vector :math:`\\boldsymbol{\\ell} = \\boldsymbol{n} \\times \\boldsymbol{\\lambda}`
    completes the triad.

    """

    #: Mass of the primary
    mA: float

    #: Mass of the secondary
    mB: float

    #: Projection of the total spin along the :math:`\\boldsymbol{n}`
    S_n: float

    #: Projection of the total spin along the :math:`\\boldsymbol{\\lambda}`
    S_lamb: float

    #: Projection of the spin difference along the :math:`\\boldsymbol{n}`
    Sigma_n: float

    #: Projection of the spin difference along the :math:`\\boldsymbol{\\lambda}`
    Sigma_lamb: float

    #: Projection of dimensionless spin :math:`\\chi_1` along the :math:`\\boldsymbol{n}`
    chi1n: float

    #: Projection of the dimensionless spin :math:`\\chi_1` along the :math:`\\boldsymbol{\\lambda}`
    chi1lamb: float

    #: Projection of dimensionless spin :math:`\\chi_1` along the :math:`\\boldsymbol{\\ell}`
    chi1L: float

    #: Projection of dimensionless spin :math:`\\chi_2` along the :math:`\\boldsymbol{n}`
    chi2n: float

    #: Projection of the dimensionless spin :math:`\\chi_1` along the :math:`\\boldsymbol{\\lambda}`
    chi2lamb: float

    #: Projection of dimensionless spin :math:`\\chi_2` along the :math:`\\boldsymbol{\\ell}`
    chi2L: float

    #: Effective spin along vector :math:`\\boldsymbol{\\ell}`, defined
    #: as :math:`\\frac{q * \\chi_{1, \\ell} + \\chi_{2, \\ell}{1 + q}`.
    chi_eff: float

    #: Spin difference along vector :math:`\\boldsymbol{\\ell}`, defined
    #: as :math:`\\frac{1}{2}(\\chi_{1, \\ell} - \\chi_{2, \\ell})`.
    chi_a: float

    #: Index of the attachment time in the time array
    idx_attach: int


def get_params_for_fit(
    dyn_all: dict[str, np.ndarray],
    t: np.ndarray,
    mA: float,
    mB: float,
    q: float,
    t_attach: float,
    interpolation_step_back: int = 10,
) -> ParameterForFits:
    """Evaluates spin projections and other quantities at the attachment time, needed to evaluate
    the IV and MRD coefficient fits for the anti-symmetric modes.

    :param dyn_all: the dynamics as returned by the function :py:func:`~.get_all_dynamics`
    :param t: the time array for the resolution of the anti-symmetric modes. The array depends
        on the resolution of the inspiral time scale see
        :py:func:`.compute_time_for_asym` for details.
    :param mA: mass of the primary
    :param mB: mass of the secondary
    :param q: mass ratio
    :param t_attach: attachment time
    :param interpolation_step_back: number of time steps to go prior to the index of the
        attachment time for evaluating the interpolants
    :return: the corresponding parameters for the fits

    .. seealso::

        * section 5 of [Estelles2025]_
        * :py:func:`~.get_all_dynamics` for the computation of the dynamics
    """

    # Check that t_attach is not greater than last point in dynamics
    if t_attach >= t[-1]:
        t_attach = t[-1]
        idx_attach = len(t) - 1
    else:
        idx_attach = np.argmin(np.abs(t - t_attach))

    stacked_elements: Final = (
        "S_n",
        "S_lambda",
        "Sigma_n",
        "Sigma_lambda",
        "S1_l",
        "S2_l",
        "S1_n",
        "S2_n",
        "S1_lambda",
        "S2_lambda",
    )
    stacked_ = np.column_stack(tuple(dyn_all[_] for _ in stacked_elements))

    interpolated_ = CubicSpline(
        dyn_all["t"][idx_attach - interpolation_step_back :],
        stacked_[idx_attach - interpolation_step_back :, :],
    )(t_attach)

    S_n, S_lamb, Sigma_n, Sigma_lamb, S1_l, S2_l, S1_n, S2_n, S1_lamb, S2_lamb = tuple(
        float(_) for _ in interpolated_.ravel()
    )

    chi1L = S1_l / mA**2
    chi2L = S2_l / mB**2
    chi1n = S1_n / mA**2
    chi2n = S2_n / mB**2
    chi1lamb = S1_lamb / mA**2
    chi2lamb = S2_lamb / mB**2
    chi_eff = (q * chi1L + chi2L) / (1 + q)
    # chi_hat = (chi_eff - 38 * nu * (chi1L + chi2L) / 113) / (1 - 76 * nu / 113)
    chi_a = 0.5 * (chi1L - chi2L)

    return ParameterForFits(
        S_n=S_n,
        S_lamb=S_lamb,
        Sigma_n=Sigma_n,
        Sigma_lamb=Sigma_lamb,
        chi_a=chi_a,
        mA=mA,
        mB=mB,
        chi1L=chi1L,
        chi2L=chi2L,
        chi1n=chi1n,
        chi2n=chi2n,
        chi1lamb=chi1lamb,
        chi2lamb=chi2lamb,
        chi_eff=chi_eff,
        idx_attach=int(idx_attach),
    )


def compute_asymmetric_PN(
    dyn: dict[str, np.ndarray],
    mA: float,
    mB: float,
    modes_to_compute: Container[tuple[int, int]],
    nlo22: bool,
) -> dict[tuple[int, int], np.ndarray]:
    """Evaluates PN expressions for the anti-symmetric modes in the co-rotation frame, and
    rotates to co-precessing frame.

    :param dyn: the dynamics as computed through :py:func:`.get_all_dynamics` and
        :py:func:`.get_params_for_fit`.
    :param mA: mass of the primary
    :param mB: mass of the secondary
    :param modes_to_compute: indicates the modes to compute
    :param nlo22: computes the next-to-leading order for the ``(2,2)`` mode if set to ``True``.
    :return: dictionary with the PN anti-symmetric modes.

    The expressions are known at next-to-leading order for the dominant ``(2,2)`` mode,
    and at leading order for the ``(3,3)`` and ``(4,4)`` modes.
    The expressions are taken from [Boyle2014]_.

    .. seealso::

        Section §III.B of [Estelles2025]_.
    """

    M = mA + mB
    delta = (mA - mB) / M
    nu = mA * mB / M**2
    Sigma_n = dyn["Sigma_n"]
    Sigma_lambda = dyn["Sigma_lambda"]
    S_lambda = dyn["S_lambda"]
    S_n = dyn["S_n"]
    v = dyn["v"]

    # Notice minus sign from conventions
    prefac = -1 / (1 / (8 * nu * v**2) * np.sqrt(5 / np.pi))

    res = {}

    # this is = np.exp(-1j * orbphase)
    exp_orb_phase = dyn["exp_orbphase"]

    if (2, 2) in modes_to_compute:
        leading_order_22 = -(v**2) * (Sigma_lambda + 1j * Sigma_n) / 2
        if nlo22:
            NLO_22 = (
                v**4
                * (
                    182 * 1j * delta * S_n
                    + 19 * delta * S_lambda
                    + 14j * (7 - 20 * nu) * Sigma_n
                    + (5 - 43 * nu) * Sigma_lambda
                )
                / 84.0
            )
        else:
            NLO_22 = 0.0
        res[(2, 2)] = exp_orb_phase**2 * prefac * (leading_order_22 + NLO_22)

    if (3, 3) in modes_to_compute:
        leading_order_33 = (
            -np.sqrt(10.0 / 21)
            * v**3
            * (S_n - 1j * S_lambda + delta * (Sigma_n - 1j * Sigma_lambda))
        )
        res[(3, 3)] = exp_orb_phase**3 * prefac * leading_order_33

    if (4, 4) in modes_to_compute:
        leading_order_44 = (
            9
            * np.sqrt(5.0 / 7)
            * v**4
            * (
                delta * (S_lambda + 1j * S_n)
                + (1 - 3 * nu) * (Sigma_lambda + 1j * Sigma_n)
            )
            / 8
        )
        total_44 = prefac * leading_order_44
        res[(4, 4)] = exp_orb_phase**4 * total_44

    return res


def apply_nqc_phase_antisymmetric(
    inspiral_modes: dict[tuple[int, int], np.ndarray],
    t_modes: np.ndarray,
    polar_dynamics: np.ndarray | tuple[float, float, float],
    t_attach: float,
    ivs: IVSAntisymmetric,
    nqc_flags: dict,
) -> None:
    """Applies a non-quasicircular correction (NQC) to the phase of the anti-symmetric mode.

    It is based on a simplified version of the full SEOBNRv5HM NQC corrections.

    :param inspiral_modes: antisymmetric modes
    :param t_modes: time array for the anti-symmetric modes
    :param polar_dynamics: (r, pr, omega_orb)
    :param t_attach: attachment time
    :param ivs: parameters of the fits for IV, see :py:func:`.fits_iv_mrd_antisymmetric`.
    :param nqc_flags: indicates the modes that should be corrected, see
        :py:func:`.apply_antisym_factorized_correction` for details

    .. note::

        The modes in ``inspiral_modes`` are modified in-place.

    .. seealso::

        Section 3.3 of [Estelles2025]_.

    """

    # Polar dynamics

    r = polar_dynamics[0]
    pr = polar_dynamics[1]
    omega_orb = polar_dynamics[2]
    fits_dict = ivs

    # Loop over every mode

    for ell_m, mode in inspiral_modes.items():
        if nqc_flags[ell_m]:

            phase = np.unwrap(np.angle(mode))
            ell, m = ell_m

            # Compute coefficient

            # Eq (4) in LIGO DCC document T1100433v2
            rOmega = r * omega_orb
            p1 = pr / rOmega

            nrTimePeak = t_attach
            if nrTimePeak > t_modes[-1]:
                # If the predicted time is after the end of the dynamics
                # use the end of the dynamics
                nrTimePeak = t_modes[-1]

            idx = np.argmin(np.abs(t_modes - nrTimePeak))
            N = 5
            left = np.max((0, idx - N))
            right = np.min((idx + N, len(t_modes)))

            # Now we (should) have calculated the a values.
            # We now compute the frequency NQCs (b1,b2) by solving Eq.(11) of T1100433v2
            # Populate the P matrix in LHS of Eq.(11) of T1100433v2
            intrp_p1 = InterpolatedUnivariateSpline(t_modes[left:right], p1[left:right])

            P = -intrp_p1.derivatives(nrTimePeak)[1]
            # Build the RHS of Eq.(11)
            # Compute frequency and derivative at the right time
            intrp_phase = InterpolatedUnivariateSpline(
                t_modes[left:right], phase[left:right]
            )
            omega = intrp_phase.derivatives(nrTimePeak)[1]

            omega = np.abs(omega)

            # Compute the NR fits
            nromega = np.abs(fits_dict["omega"][(ell, m)])

            # Assemble RHS
            omegas = nromega - omega

            # Solve the equation P*coeffs = omegas
            b1 = omegas / P

            # Phase correction
            phase_corr = b1 * pr / rOmega
            nqc = np.cos(phase_corr) + 1j * np.sin(phase_corr)

            # Apply correction to mode
            inspiral_modes[ell_m] *= nqc


def apply_antisym_factorized_correction(
    antisym_modes: dict[tuple[int, int], np.ndarray],
    v_orb: np.ndarray,
    ivs_asym: IVSAntisymmetric,
    idx_attach: int,
    t: np.ndarray,
    t_attach: float,
    nu: float,
    corr_power: int,
    interpolation_step_back: int = 10,
    modes_to_apply: Collection[tuple[int, int]] | None = None,
) -> dict[tuple[int, int], bool]:
    """Applies amplitude corrections to the PN antisymmetric modes.

    The corrections improve accuracy in late-inspiral and plunge.

    The correction factor ``b_6`` is computed by matching the corrected amplitudes with the calibrated
    amplitudes at the attachment time; see [Estelles2025]_ Eq.21 and Eq.23b for details.

    :param antisym_modes: anti-symmetric modes
    :param v_orb: orbital velocity
    :param ivs_asym: values of the fits, see :py:func:`.fits_iv_mrd_antisymmetric`
    :param t: time array of the modes
    :param idx_attach: index of the attachment time
    :param t_attach: attachment time
    :param nu: symmetric mass ratio
    :param corr_power: exponentiation of the orbital velocity at ``t_attach``, see Eq. 23b in [Estelles2025]_.
    :param interpolation_step_back: number of time steps before ``idx_attach`` to step back
        for the interpolating amplitude and orbital velocity at ``t_attach``.
    :param modes_to_apply: indicates the modes that should be corrected. The returned flags are not
        affected by this content. Defaults to ``[(2,2)]``
    :returns: a dictionary of flags (booleans) indicating if the NQC should be applied to the corresponding
        antisymmetric modes.

    .. note::

        The NQC correction are applied if the average amplitude of the modes for the last 10% of the inspiral
        is below a certain threshold (1e-8), since for negligible amplitudes is recommended to
        deactivate the NQC corrections.

        If selected for correction, the modes in ``antisym_modes`` are modified in-place.

    .. seealso::

        See section §III.B of [Estelles2025]_.
    """

    if modes_to_apply is None:
        modes_to_apply = [(2, 2)]

    assert idx_attach < len(t)
    assert set(modes_to_apply) <= {(2, 2), (3, 3), (4, 4)} and len(
        modes_to_apply
    ) == len(set(modes_to_apply))

    # Check that t_attach is not greater than last point in dynamics
    if t_attach > t[-1]:
        t_attach = t[-1]

    nqc_flags = {}
    for ell_m in antisym_modes.keys():

        # First we check if NQCs will be applied
        # We don't want corrections to small enough modes
        len_mode = len(antisym_modes[ell_m])
        idx_for_mean = int(0.1 * len_mode)

        known_merger = np.mean(
            np.abs(antisym_modes[ell_m][idx_attach - idx_for_mean : idx_attach])
        )
        if known_merger < 1e-8:
            nqc_flags[ell_m] = False
            continue
        else:
            nqc_flags[ell_m] = True

        # Now we apply the amplitude corrections
        if ell_m in modes_to_apply:
            known, v_attach = CubicSpline(
                t[idx_attach - interpolation_step_back :],
                np.column_stack(
                    (
                        np.abs(
                            antisym_modes[ell_m][idx_attach - interpolation_step_back :]
                        ),
                        v_orb[idx_attach - interpolation_step_back :],
                    )
                ),
            )(t_attach)

            b6 = (nu * ivs_asym["amp"][ell_m] - known) / (known * v_attach**corr_power)
            correction = 1 + b6 * v_orb**corr_power
            antisym_modes[ell_m] *= correction

    return nqc_flags


# Apply boundaries based on value range from input data (NR and TPL waveforms)
# @todo move those to a companion file
amp22_bounds: Final = [0.0, 0.6]
omega22_bounds: Final = [0.05, 0.5]
c122_bounds: Final = [0.05, 0.15]
c222_bounds: Final = [-1.4, -0.4]
omega33_bounds: Final = [0.03, 0.8]
omega44_bounds: Final = [0.03, 1.42]


class IVSAntisymmetric(TypedDict):
    """
    Helper type for anti-symmetric fits

    :meta private:
    """

    #: Amplitude of the (l,m) anti-symmetric mode at the attachment time
    amp: dict[tuple[int, int], float | np.ndarray]

    #: Wave-frequency of the (l,m) anti-symmetric mode at the attachment time
    omega: dict[tuple[int, int], float | np.ndarray]


def fits_iv_mrd_antisymmetric(
    params_for_fits: ParameterForFits,
    nu: float,
    modes_to_compute: Collection[tuple[int, int]],
) -> tuple[IVSAntisymmetric, MRAnzatze]:
    """Evaluate the anti-symmetric IVs and MRD fits coefficients.

    See section §III.D.5 of [Estelles2025]_.

    :param params_for_fits: the parameters for computing the fits, as returned by
         :py:func:`.get_params_for_fit`
    :param nu: symmetric mass ratio
    :param modes_to_compute: a sequence indicating the modes on which the anti-symmetric corrections
        would need to be applied.
    :returns:
        a tuple containing the coefficients of the corrections to be applied for the anti-symmetric modes
        (see :py:func:`.apply_nqc_phase_antisymmetric` and :py:func:`.apply_antisym_factorized_correction`)
        and the coefficients of the merger-ringdown ansatze
        (see :py:func:`~pyseobnr.eob.waveform.compute_hlms.compute_IMR_modes`)

    """

    assert set(modes_to_compute) <= {(2, 2), (3, 3), (4, 4)} and len(
        modes_to_compute
    ) == len(set(modes_to_compute))

    # Compute needed quantities

    Sigma_inplane = np.sqrt(params_for_fits.Sigma_n**2 + params_for_fits.Sigma_lamb**2)
    S_inplane = np.sqrt(params_for_fits.S_n**2 + params_for_fits.S_lamb**2)

    phiS = np.arctan2(params_for_fits.S_lamb, params_for_fits.S_n)
    cos_2phiS = np.cos(2 * phiS)
    sin_2phiS = np.sin(2 * phiS)

    phiSigma = np.arctan2(params_for_fits.Sigma_lamb, params_for_fits.Sigma_n)
    cos_2phiSigma = np.cos(2 * phiSigma)
    sin_2phiSigma = np.sin(2 * phiSigma)

    # Aligned-spin MRD coefficients
    MR_fits = MergerRingdownFits(
        params_for_fits.mA,
        params_for_fits.mB,
        [0.0, 0.0, params_for_fits.chi1L],
        [0.0, 0.0, params_for_fits.chi2L],
    )
    c1f_dict = MR_fits.c1f()
    c2f_dict = MR_fits.c2f()
    d1f_dict = MR_fits.d1f()
    d2f_dict = MR_fits.d2f()

    ivs_asym: IVSAntisymmetric = {"amp": {}, "omega": {}}

    # 22 fits

    if (2, 2) in modes_to_compute:

        X = np.array(
            [
                nu,
                S_inplane,
                Sigma_inplane,
                params_for_fits.chi_eff,
                params_for_fits.chi_a,
                cos_2phiSigma,
                sin_2phiSigma,
            ]
        )
        habs_22_nu = (
            np.abs(
                get_predictor_from_fits(
                    nb_dimensions=X.shape[0], ell=2, emm=2, quantity="habs"
                ).predict(X)
            )
            * Sigma_inplane
        )

        X = np.array(
            [
                nu,
                params_for_fits.S_n,
                params_for_fits.S_lamb,
                params_for_fits.chi_eff,
                params_for_fits.Sigma_n,
                params_for_fits.Sigma_lamb,
                params_for_fits.chi_a,
            ]
        )
        omega_peak = np.abs(
            get_predictor_from_fits(
                nb_dimensions=X.shape[0], ell=2, emm=2, quantity="omega_peak"
            ).predict(X)
        )

        X = np.array(
            [
                nu,
                S_inplane,
                Sigma_inplane,
                params_for_fits.chi_eff,
                params_for_fits.chi_a,
                cos_2phiSigma,
                sin_2phiSigma,
            ]
        )
        omega_22 = (
            get_predictor_from_fits(
                nb_dimensions=X.shape[0], ell=2, emm=2, quantity="omegalm"
            ).predict(X)
            + omega_peak
        )

        X = np.array(
            [
                nu,
                params_for_fits.chi_eff,
                params_for_fits.chi_a,
                cos_2phiS,
                sin_2phiS,
            ]
        )
        c1f_22 = get_predictor_from_fits(
            nb_dimensions=X.shape[0], ell=2, emm=2, quantity="c1f"
        ).predict(X)

        X = np.array(
            [
                nu,
                c1f_22,
                params_for_fits.chi_eff,
                params_for_fits.chi_a,
                cos_2phiS,
                sin_2phiS,
            ]
        )
        c2f_22 = get_predictor_from_fits(
            nb_dimensions=X.shape[0], ell=2, emm=2, quantity="c2f"
        ).predict(X)

        # Apply boundaries based on 0.002 and 0.998 quantile
        habs_22_nu = min(habs_22_nu, amp22_bounds[1])
        omega_22 = max(min(omega_22, omega22_bounds[1]), omega22_bounds[0])
        c1f_22 = max(min(c1f_22, c122_bounds[1]), c122_bounds[0])
        c2f_22 = max(min(c2f_22, c222_bounds[1]), c222_bounds[0])

        ivs_asym["amp"][(2, 2)] = habs_22_nu
        ivs_asym["omega"][(2, 2)] = np.abs(omega_22)

        c1f_dict[(2, 2)] = c1f_22
        c2f_dict[(2, 2)] = c2f_22

    if (3, 3) in modes_to_compute or (4, 4) in modes_to_compute:
        X = np.array(
            [
                nu,
                S_inplane,
                Sigma_inplane,
                params_for_fits.chi_eff,
                params_for_fits.chi_a,
                cos_2phiSigma,
                sin_2phiSigma,
            ]
        )

        # 33 fits
        if (3, 3) in modes_to_compute:
            omega_33 = get_predictor_from_fits(
                nb_dimensions=X.shape[0], ell=3, emm=3, quantity="omegalm"
            ).predict(X)

            omega_33 = max(min(omega_33, omega33_bounds[1]), omega33_bounds[0])

            ivs_asym["amp"][(3, 3)] = 0.0
            ivs_asym["omega"][(3, 3)] = np.abs(omega_33)

        # 44 fits
        if (4, 4) in modes_to_compute:
            omega_44 = get_predictor_from_fits(
                nb_dimensions=X.shape[0], ell=4, emm=4, quantity="omegalm"
            ).predict(X)

            omega_44 = max(min(omega_44, omega44_bounds[1]), omega44_bounds[0])

            ivs_asym["amp"][(4, 4)] = 0.0
            ivs_asym["omega"][(4, 4)] = np.abs(omega_44)

    mrd_fits = MRAnzatze(c1f=c1f_dict, c2f=c2f_dict, d1f=d1f_dict, d2f=d2f_dict)

    return ivs_asym, mrd_fits
