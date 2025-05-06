from __future__ import annotations

from unittest.mock import patch

import numpy as np

import pytest

from pyseobnr.eob.fits.EOB_fits import EOBNonQCCorrectionImpl
from pyseobnr.generate_waveform import generate_modes_opt


def EOBNonQCCorrection(
    r: np.ndarray,
    phi: np.ndarray | None,
    pr: np.ndarray,
    pphi: np.ndarray,
    omega: np.ndarray,
    coeffs: dict,
) -> np.ndarray:
    """
    Evaluate the NQC correction, given the coefficients.

    Previous implementation, kept here for correctness checks. See
    :py:class:`~pyseobnr.eob.fits.EOB_fits.EOBNonQCCorrectionImpl` for reference
    documentation

    """
    sqrtR = np.sqrt(r)
    rOmega = r * omega
    rOmegaSq = rOmega * rOmega
    p = pr
    mag = 1.0 + (p * p / rOmegaSq) * (
        coeffs["a1"]
        + coeffs["a2"] / r
        + (coeffs["a3"] + coeffs["a3S"]) / (r * sqrtR)
        + coeffs["a4"] / (r * r)
        + coeffs["a5"] / (r * r * sqrtR)
    )
    phase = coeffs["b1"] * p / rOmega + p * p * p / rOmega * (
        coeffs["b2"] + coeffs["b3"] / sqrtR + coeffs["b4"] / r
    )

    nqc = mag * np.exp(1j * phase)
    return nqc


def test_nqc_coefficients():
    """Checks equivalence of 2 implementations of the apply nqc corrections"""

    q = 5.3
    chi_1 = 0.9
    chi_2 = 0.3
    omega0 = 0.0137

    class MyException(Exception):
        pass

    with patch(
        "pyseobnr.generate_waveform.SEOBNRv5HM.apply_nqc_corrections"
    ) as p_apply_nqc_corrections:

        p_apply_nqc_corrections.side_effect = MyException

        with pytest.raises(MyException):
            _, _, model = generate_modes_opt(q, chi_1, chi_2, omega0, debug=True)

        p_apply_nqc_corrections.assert_called_once()
        args, kwargs = p_apply_nqc_corrections.call_args

        hlms = args[0] if "hlms" not in kwargs else kwargs["hlms"]
        nqc_coeffs = args[1] if "nqc_coeffs" not in kwargs else kwargs["nqc_coeffs"]
        polar_dynamics = (
            args[2] if "polar_dynamics" not in kwargs else kwargs["polar_dynamics"]
        )

    keys = set(nqc_coeffs.keys()) & set(hlms.keys())

    r, pr, omega_orb = polar_dynamics
    nqc_apply = EOBNonQCCorrectionImpl(r=r, phi=None, pr=pr, pphi=None, omega=omega_orb)

    for k in sorted(keys):
        nqc_coeffs_mode = nqc_coeffs[k]
        correction1 = EOBNonQCCorrection(r, None, pr, None, omega_orb, nqc_coeffs_mode)
        correction2 = nqc_apply.get_nqc_multiplier(coeffs=nqc_coeffs_mode)

        np.testing.assert_array_almost_equal(
            np.abs(correction1), np.abs(correction2), decimal=10
        )
        np.testing.assert_array_almost_equal(
            np.angle(correction1), np.angle(correction2), decimal=10
        )
