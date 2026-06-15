"""
Test that the aligned-spin tidal model SEOBNRv5THM has not changed.

Follows the procedures used for SEOBNRv5HM and SEOBNRv5EHM tests in
``test_SEOBNRv5HM.py`` / ``test_SEOBNRv5EHM.py``.

SEOBNRv5THM is the spin-aligned tidal model for binary neutron stars
(see ``SEOBNRv5THM_opt`` in ``pyseobnr/models/SEOBNRv5HM.py``).
"""

from pathlib import Path

import numpy as np
import pandas as pd

import pytest

from pyseobnr.generate_waveform import GenerateWaveform, generate_modes_opt

from .helpers import compare_frames

folder_data = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# A representative BNS configuration used across several tests.
# ---------------------------------------------------------------------------
_M1_BNS = 1.4
_M2_BNS = 1.3
_MTOT_BNS = _M1_BNS + _M2_BNS
_LAMBDA1 = 500.0
_LAMBDA2 = 800.0


def _basic_bns_params_dict(
    *,
    approximant: str = "SEOBNRv5THM",
    postadiabatic: bool = True,
) -> dict:
    """Return a representative BNS parameter dictionary."""
    return {
        "mass1": _M1_BNS,
        "mass2": _M2_BNS,
        "spin1x": 0.0,
        "spin1y": 0.0,
        "spin1z": 0.05,
        "spin2x": 0.0,
        "spin2y": 0.0,
        "spin2z": -0.03,
        "lambda2Tidal1": _LAMBDA1,
        "lambda2Tidal2": _LAMBDA2,
        "deltaT": 1.0 / 4096.0,
        "deltaF": 0.125,
        "f22_start": 30.0,
        "f_ref": 30.0,
        "phi_ref": 0.0,
        "distance": 100.0,
        "inclination": np.pi / 3.0,
        "f_max": 2048.0,
        "approximant": approximant,
        "postadiabatic": postadiabatic,
    }


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestSmokeTHM:
    def test_smoke_no_tides(self):
        """The model runs without errors in the BBH limit (zero tides)."""

        q = 1.2
        chi_1 = 0.05
        chi_2 = -0.05
        omega0 = 0.012

        _, _, model = generate_modes_opt(
            q=q,
            chi1=chi_1,
            chi2=chi_2,
            omega_start=omega0,
            approximant="SEOBNRv5THM",
            settings={"M": _MTOT_BNS},
            debug=True,
        )
        assert model.dynamics is not None
        assert model.dynamics.shape[1] == 8  # t, r, phi, pr, pphi, H, Omega, Omega_circ
        assert model.success is True

    def test_smoke_with_tides(self):
        """The model runs with non-zero tidal parameters."""

        q = _M1_BNS / _M2_BNS
        chi_1 = 0.05
        chi_2 = -0.03
        omega0 = 0.012

        _, _, model = generate_modes_opt(
            q=q,
            chi1=chi_1,
            chi2=chi_2,
            omega_start=omega0,
            approximant="SEOBNRv5THM",
            settings={"M": _MTOT_BNS},
            lambda2Tidal1=_LAMBDA1,
            lambda2Tidal2=_LAMBDA2,
            debug=True,
        )
        assert model.dynamics is not None
        # the EOB ``kappaT`` summary parameter must end up positive when
        # tidal deformabilities are non-zero
        assert model.kappaT > 0.0
        # universal relations must have populated the auxiliary tidal
        # parameters (f-mode frequencies)
        assert model.omega02Tidal1 > 0.0
        assert model.omega02Tidal2 > 0.0
        assert model.lambda3Tidal1 > 0.0
        assert model.lambda3Tidal2 > 0.0

    def test_smoke_no_postadiabatic(self):
        """The ODE-only path (postadiabatic disabled) also runs."""

        q = _M1_BNS / _M2_BNS

        _, _, model = generate_modes_opt(
            q=q,
            chi1=0.0,
            chi2=0.0,
            omega_start=0.012,
            approximant="SEOBNRv5THM",
            settings={"M": _MTOT_BNS, "postadiabatic": False},
            lambda2Tidal1=_LAMBDA1,
            lambda2Tidal2=_LAMBDA2,
            debug=True,
        )
        assert model.dynamics is not None
        assert model.success is True


# ---------------------------------------------------------------------------
# Tidal parameter handling
# ---------------------------------------------------------------------------


def test_tidal_parameter_routing_to_model():
    """Check that tidal kwargs reach the underlying model unchanged when
    they are explicitly provided (i.e. no universal relation override)."""

    q = _M1_BNS / _M2_BNS

    omega02Tidal1 = 0.10
    omega02Tidal2 = 0.12
    lambda3Tidal1 = 1000.0
    lambda3Tidal2 = 1200.0
    CES21 = 5.0
    CES22 = 6.0

    _, _, model = generate_modes_opt(
        q=q,
        chi1=0.0,
        chi2=0.0,
        omega_start=0.012,
        approximant="SEOBNRv5THM",
        # The non-deformability tidal parameters live in ``settings``.
        settings={
            "M": _MTOT_BNS,
            "omega02Tidal1": omega02Tidal1,
            "omega02Tidal2": omega02Tidal2,
            "lambda3Tidal1": lambda3Tidal1,
            "lambda3Tidal2": lambda3Tidal2,
            "CES21": CES21,
            "CES22": CES22,
        },
        lambda2Tidal1=_LAMBDA1,
        lambda2Tidal2=_LAMBDA2,
        debug=True,
    )

    # The model rescales lambda_l by m_i ** (2l + 1) and omega_{0l} by 1 / m_i.
    np.testing.assert_allclose(
        model.lambda2Tidal1,
        _LAMBDA1 * model.m_1**5,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        model.lambda2Tidal2,
        _LAMBDA2 * model.m_2**5,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        model.omega02Tidal1,
        omega02Tidal1 / model.m_1,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        model.omega02Tidal2,
        omega02Tidal2 / model.m_2,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        model.lambda3Tidal1,
        lambda3Tidal1 * model.m_1**7,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        model.lambda3Tidal2,
        lambda3Tidal2 * model.m_2**7,
        rtol=1e-12,
    )
    assert model.CES21 == CES21
    assert model.CES22 == CES22


def test_tidal_parameters_zero_recovers_bbh_limit():
    """With both lambda_2's set to zero, the EOB ``kappaT`` parameter
    vanishes and all derived tidal quantities are zero / unity."""

    q = _M1_BNS / _M2_BNS

    _, _, model = generate_modes_opt(
        q=q,
        chi1=0.0,
        chi2=0.0,
        omega_start=0.012,
        approximant="SEOBNRv5THM",
        settings={"M": _MTOT_BNS},
        lambda2Tidal1=0.0,
        lambda2Tidal2=0.0,
        debug=True,
    )
    assert model.kappaT == 0.0
    assert model.lambda2Tidal1 == 0.0
    assert model.lambda2Tidal2 == 0.0
    assert model.omega02Tidal1 == 0.0
    assert model.omega02Tidal2 == 0.0
    assert model.lambda3Tidal1 == 0.0
    assert model.lambda3Tidal2 == 0.0
    assert model.CES21 == 1.0
    assert model.CES22 == 1.0


def test_unset_tidal_parameters_filled_by_universal_relations():
    """Tidal parameters absent from ``settings`` are populated from the
    deformabilities via the quasi-universal relations, while any that *are*
    provided in ``settings`` are left untouched."""

    q = _M1_BNS / _M2_BNS

    # Provide the deformabilities and override a single coefficient (CES21);
    # everything else is left to the quasi-universal relations.
    explicit_CES21 = 3.0
    _, _, model = generate_modes_opt(
        q=q,
        chi1=0.11,
        chi2=-0.05,
        omega_start=0.012,
        approximant="SEOBNRv5THM",
        settings={"M": _MTOT_BNS, "CES21": explicit_CES21},
        lambda2Tidal1=_LAMBDA1,
        lambda2Tidal2=_LAMBDA2,
        debug=True,
    )

    # The value explicitly set in ``settings`` is used verbatim (CES2 is not
    # rescaled by the model).
    assert model.CES21 == explicit_CES21

    # Everything left unset is filled in by the universal relations: the f-mode
    # resonances and octupolar deformabilities become positive ...
    assert model.omega02Tidal1 > 0.0
    assert model.omega02Tidal2 > 0.0
    assert model.omega03Tidal1 > 0.0
    assert model.omega03Tidal2 > 0.0
    assert model.spinshiftomega02Tidal1 > 0.0
    assert model.spinshiftomega02Tidal2 < 0.0
    assert model.spinshiftomega03Tidal1 > 0.0
    assert model.spinshiftomega03Tidal2 < 0.0
    assert model.lambda3Tidal1 > 0.0
    assert model.lambda3Tidal2 > 0.0
    # ... and the spin-induced coefficients depart from their black-hole value
    # of 1 (cf. test_tidal_parameters_zero_recovers_bbh_limit).
    assert model.CES22 > 1.0
    assert model.CBS31 > 1.0
    assert model.CBS32 > 1.0
    assert model.CES41 > 1.0
    assert model.CES42 > 1.0

    # The overridden CES21 is genuinely distinct from its universal-relation
    # sibling CES22.
    assert model.CES21 != model.CES22


# ---------------------------------------------------------------------------
# Validation through the user-facing GenerateWaveform interface
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_thm_parameters():
    return _basic_bns_params_dict()


def test_parameters_thm_through_waveform_interface(basic_thm_parameters):
    """Sanity-check the GenerateWaveform interface for SEOBNRv5THM."""

    # 1. Construction works with the basic parameter set.
    GenerateWaveform(basic_thm_parameters)

    # 2. In-plane spin components are rejected for the aligned-spin model.
    bad_spins = basic_thm_parameters | {"spin1x": 0.1}
    with pytest.raises(ValueError):
        GenerateWaveform(bad_spins)

    # 3. The minimal parameter dict (default tides = 0) is accepted.
    minimal_params = {
        "mass1": _M1_BNS,
        "mass2": _M2_BNS,
        "approximant": "SEOBNRv5THM",
    }
    GenerateWaveform(minimal_params)


def test_thm_runs_through_waveform_interface_td(basic_thm_parameters):
    """End-to-end smoke through the GenerateWaveform interface (TD)."""
    wfm_gen = GenerateWaveform(basic_thm_parameters)
    hp, hc = wfm_gen.generate_td_polarizations()
    assert hp.data.length > 0
    assert hc.data.length == hp.data.length
    # waveform must be finite everywhere
    assert np.isfinite(hp.data.data).all()
    assert np.isfinite(hc.data.data).all()


def test_thm_runs_through_waveform_interface_fd(basic_thm_parameters):
    """End-to-end smoke through the GenerateWaveform interface (FD)."""
    wfm_gen = GenerateWaveform(basic_thm_parameters)
    hp, hc = wfm_gen.generate_fd_polarizations()
    assert hp.data.length > 0
    assert hc.data.length == hp.data.length
    assert np.isfinite(hp.data.data).all()
    assert np.isfinite(hc.data.data).all()


def test_thm_supports_spa_fft_path(basic_thm_parameters):
    """Test that SEOBNRv5THM supports the SPA + FFT FD generation method."""
    wfm_gen = GenerateWaveform(basic_thm_parameters)
    hp, hc, freqs = wfm_gen.generate_fd_polarizations_stationary_phase_approximation()
    assert len(freqs) == len(hp.data.data)
    assert len(freqs) == len(hc.data.data)
    assert np.isfinite(hp.data.data).all()
    assert np.isfinite(hc.data.data).all()


def test_spa_fft_smoke(basic_thm_parameters):
    """SPA+FFT FD path runs on the default (equidistant) grid and returns a
    sane, fully-populated frequency series spanning ``[0, f_max]``."""
    wfm_gen = GenerateWaveform(basic_thm_parameters)
    hp, hc, freqs = wfm_gen.generate_fd_polarizations_stationary_phase_approximation()

    # Equidistant path: LAL COMPLEX16FrequencySeries on the full [0, f_max] grid.
    assert len(freqs) == hp.data.length == hc.data.length
    assert freqs[0] == 0.0
    assert np.isclose(freqs[-1], basic_thm_parameters["f_max"])

    hp_arr = np.asarray(hp.data.data)
    hc_arr = np.asarray(hc.data.data)
    assert np.isfinite(hp_arr).all()
    assert np.isfinite(hc_arr).all()
    # The signal band must carry non-zero strain.
    assert np.max(np.abs(hp_arr)) > 0.0
    assert np.max(np.abs(hc_arr)) > 0.0


def test_spa_fft_five_frequency_points(basic_thm_parameters):
    """SPA+FFT FD path evaluated on an arbitrary small (5-point) frequency
    array returns the polarizations exactly on those requested frequencies."""
    frequencies = np.array([50.0, 100.0, 300.0, 900.0, 1500.0])
    params = basic_thm_parameters | {"frequency_array": frequencies}

    wfm_gen = GenerateWaveform(params)
    hp, hc, freqs = wfm_gen.generate_fd_polarizations_stationary_phase_approximation()

    print(f"hp(f) = {hp}")
    print(f"hc(f) = {hc}")

    hp_comparison = np.array(
        [
            -5.11314885e-24 - 2.59102419e-24j,
            1.24883831e-24 + 2.19966531e-24j,
            2.85961594e-25 + 6.14888124e-25j,
            2.86214978e-26 - 1.48880862e-25j,
            -3.16211619e-27 + 6.08474814e-26j,
        ]
    )

    hc_comparison = np.array(
        [
            -2.07281935e-24 + 4.09051908e-24j,
            1.75973225e-24 - 9.99070645e-25j,
            4.91910499e-25 - 2.28769275e-25j,
            -1.19104689e-25 - 2.28971982e-26j,
            4.86779851e-26 + 2.52969295e-27j,
        ]
    )

    # With a user-supplied frequency_array the result is plain numpy arrays,
    # evaluated exactly on the requested points (one output per input frequency).
    assert isinstance(hp, np.ndarray)
    assert isinstance(hc, np.ndarray)
    assert len(hp) == len(hc) == len(freqs) == len(frequencies)
    assert np.allclose(freqs, frequencies)

    # Regression test
    np.testing.assert_allclose(hp, hp_comparison, rtol=3e-2, atol=1e-26)
    np.testing.assert_allclose(hc, hc_comparison, rtol=3e-2, atol=1e-26)


# ---------------------------------------------------------------------------
# Regression test on the FD/TD differences, analogous to test_SEOBNRv5HM.py
# ---------------------------------------------------------------------------


def _get_amp_phase(h):
    return np.abs(h), np.unwrap(np.angle(h))


def _sum_sqr_diff(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def _gen_test_data_thm(test_type: str):
    """Replicate the structure of ``gen_test_data`` from
    ``test_SEOBNRv5HM.py`` but for a BNS configuration with SEOBNRv5THM.
    """
    params_dict = _basic_bns_params_dict()
    wfm_gen = GenerateWaveform(params_dict)

    params_dict2 = params_dict.copy()

    if test_type == "FD":
        hp1, hc1 = wfm_gen.generate_fd_polarizations()
        # Vary intrinsic params: change the secondary tidal deformability.
        params_dict2.update({"lambda2Tidal2": 200.0})
        wfm_gen2 = GenerateWaveform(params_dict2)
        hp2, hc2 = wfm_gen2.generate_fd_polarizations()

        hp1_amp, hp1_phase = _get_amp_phase(hp1.data.data)
        hc1_amp, hc1_phase = _get_amp_phase(hc1.data.data)
        hp2_amp, hp2_phase = _get_amp_phase(hp2.data.data)
        hc2_amp, hc2_phase = _get_amp_phase(hc2.data.data)

        return (
            _sum_sqr_diff(hp1_amp, hp2_amp),
            _sum_sqr_diff(hp1_phase, hp2_phase),
            _sum_sqr_diff(hc1_amp, hc2_amp),
            _sum_sqr_diff(hc1_phase, hc2_phase),
        )

    if test_type == "TD":
        hp1, hc1 = wfm_gen.generate_td_polarizations()
        # Vary extrinsic params: inclination and phi_ref.
        params_dict2.update({"inclination": 0.17, "phi_ref": 0.5})
        wfm_gen2 = GenerateWaveform(params_dict2)
        hp2, hc2 = wfm_gen2.generate_td_polarizations()
        return (
            _sum_sqr_diff(hp1.data.data, hp2.data.data),
            _sum_sqr_diff(hc1.data.data, hc2.data.data),
        )

    raise ValueError(f"Unknown test_type: {test_type}")


# NOTE: the reference values below have to be regenerated locally on first
# run and then frozen into this file (similar to ``test_SEOBNRv5HM.py``).


def test_SEOBNRv5THM_diff_TD():
    """Regression check that SEOBNRv5THM TD output has not changed.

    To (re)generate the expected values:

        >>> np.array(_gen_test_data_thm("TD"))
    """
    expected_result = np.array([5.90713091e-21, 5.87432605e-21])
    new_result = np.array(_gen_test_data_thm("TD"))
    np.testing.assert_allclose(
        new_result, expected_result, rtol=1e-4, err_msg="SEOBNRv5THM TD test failed"
    )


def test_SEOBNRv5THM_diff_FD():
    """Regression check that SEOBNRv5THM FD output has not changed.

    To (re)generate the expected values:

        >>> np.array(_gen_test_data_thm("FD"))
    """
    expected_result = np.array(
        [1.39127128e-24, 1.42410130e03, 2.39367774e-24, 6.52096475e02]
    )
    new_result = np.array(_gen_test_data_thm("FD"))
    np.testing.assert_allclose(
        new_result, expected_result, rtol=4.5e-4, err_msg="SEOBNRv5THM FD test failed"
    )


# ---------------------------------------------------------------------------
# Dynamics regression (gated on CI_TEST_DYNAMIC_REGRESSIONS for parity with
# the HM / EHM dynamics regression tests)
# ---------------------------------------------------------------------------


# @pytest.mark.skipif(
#     "CI_TEST_DYNAMIC_REGRESSIONS" not in os.environ,
#     reason="regressions on dynamics are for specific systems only",
# )
def test_regression_dynamics_thm():
    """Compare the integrated dynamics against a frozen reference frame.

    The reference frame ``frame_thm.csv.gz`` must be generated once and
    placed under ``pyseobnr/test/data/``. Generate it via, e.g.::

        q, chi_1, chi_2, omega0 = 1.077, 0.05, -0.03, 0.00084
        _, _, model = generate_modes_opt(
            q, chi_1, chi_2, omega0,
            approximant="SEOBNRv5THM",
            settings={"M": 2.7},
            lambda2Tidal1=500.0, lambda2Tidal2=800.0,
            debug=True,
        )
        pd.DataFrame(
            data=model.dynamics,
            columns="t, r, phi, pr, pphi, H, Omega, Omega_circular".split(", "),
        ).to_csv("frame_thm.csv.gz", index=False)
    """
    q = 1.077
    chi_1 = 0.05
    chi_2 = -0.03

    # 0.00084 corresponds closely to the omega0 of a 20 Hz f22_start
    # Given as 20 * (2.7 * np.pi * lal.MTSUN_SI) = 0.0008355898535273729
    omega0 = 0.00084

    _, _, model = generate_modes_opt(
        q=q,
        chi1=chi_1,
        chi2=chi_2,
        omega_start=omega0,
        approximant="SEOBNRv5THM",
        settings={"M": _MTOT_BNS},
        lambda2Tidal1=_LAMBDA1,
        lambda2Tidal2=_LAMBDA2,
        debug=True,
    )

    frame_thm = pd.DataFrame(
        data=model.dynamics,
        columns="t, r, phi, pr, pphi, H, Omega, Omega_circular".replace(" ", "").split(
            ","
        ),
    )
    frame_thm_reference = pd.read_csv(folder_data / "frame_thm.csv.gz")

    known_differences_percentage = {
        "r": 1,
        "phi": 1,
        "pr": 1,
        "pphi": 1,
        "H": 1,
        "Omega": 1,
        "Omega_circular": 1,
    }

    compare_frames(
        test_frame=frame_thm,
        reference_frame=frame_thm_reference,
        known_differences_percentage=known_differences_percentage,
        time_tolerance_percent=1,
    )


# Run with ``pytest -s`` to see the printed messages
# print(f"expected_result(TD) = {(np.array(_gen_test_data_thm('TD')))}")
# print(f"expected_result(FD) = {(np.array(_gen_test_data_thm('FD')))}")
