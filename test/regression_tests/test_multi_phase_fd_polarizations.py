"""
Tests for ``GenerateWaveform.generate_multi_phase_fd_polarizations``.

The method evaluates FD polarizations at a caller-supplied list of phi_c
offsets (relative to ``phi_ref``). These tests check basic shape, internal
phase-shift symmetries, and consistency with ``generate_fd_polarizations``
for the trivial single-phase case.
"""

from unittest.mock import patch

import lal
import numpy as np

import pytest

from pyseobnr.generate_waveform import GenerateWaveform


@pytest.fixture
def precessing_params():
    m1, m2 = 50.0, 30.0
    Mt = m1 + m2
    return {
        "mass1": m1,
        "mass2": m2,
        "spin1x": 0.5,
        "spin1y": 0.3,
        "spin1z": 0.2,
        "spin2x": 0.3,
        "spin2y": 0.7,
        "spin2z": -0.1,
        "deltaT": 1.0 / 2048.0,
        "deltaF": 0.125,
        "f_ref": 20.0,
        "f22_start": 0.0157 / (Mt * np.pi * lal.MTSUN_SI),
        "phi_ref": 0.0,
        "distance": 1.0,
        "inclination": np.pi / 3.0,
        "f_max": 1024.0,
        "approximant": "SEOBNRv5PHM",
        "postadiabatic": True,
    }


def test_returns_one_per_phi_c(precessing_params):
    """Output list length and array shape track ``phi_c_values``."""
    phi_c_values = [0.0, 0.7, 1.4]
    out = GenerateWaveform(precessing_params).generate_multi_phase_fd_polarizations(
        phi_c_values
    )

    n_domain = int(round(precessing_params["f_max"] / precessing_params["deltaF"])) + 1
    assert len(out) == len(phi_c_values)
    for entry in out:
        assert entry["h_plus"].shape == (n_domain,)
        assert entry["h_cross"].shape == (n_domain,)
        assert entry["h_cross"].shape == entry["h_plus"].shape


def test_phi_c_offset_equals_phi_ref_shift(precessing_params):
    """Shifting ``phi_c_values`` by ``delta`` matches shifting ``phi_ref`` by ``delta``."""
    delta = 0.4

    p_a = precessing_params | {"phi_ref": 0.0}
    out_a = GenerateWaveform(p_a).generate_multi_phase_fd_polarizations([delta])

    p_b = precessing_params | {"phi_ref": delta}
    out_b = GenerateWaveform(p_b).generate_multi_phase_fd_polarizations([0.0])

    np.testing.assert_allclose(
        out_a[0]["h_plus"], out_b[0]["h_plus"], rtol=1e-10, atol=1e-15
    )
    np.testing.assert_allclose(
        out_a[0]["h_cross"], out_b[0]["h_cross"], rtol=1e-10, atol=1e-15
    )


def test_phi_c_zero_matches_generate_fd_polarizations(precessing_params):
    """``phi_c_values=[0]`` matches ``generate_fd_polarizations`` once the
    epoch-as-phase-shift convention difference is undone."""
    multi = GenerateWaveform(precessing_params).generate_multi_phase_fd_polarizations(
        [0.0]
    )[0]

    hp_std, hc_std = GenerateWaveform(precessing_params).generate_fd_polarizations()

    # generate_multi_phase_fd_polarizations includes the epoch as a phase shift
    # on the FFT output; generate_fd_polarizations leaves it as LAL metadata.
    # Apply the same shift here so the two are on the same convention.
    freqs = np.arange(hp_std.data.length) * hp_std.deltaF
    dt = 1.0 / hp_std.deltaF + float(hp_std.epoch)
    shift = np.exp(-1j * 2 * np.pi * dt * freqs)

    hp_ref = hp_std.data.data * shift
    hc_ref = hc_std.data.data * shift

    np.testing.assert_allclose(multi["h_plus"], hp_ref, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(multi["h_cross"], hc_ref, rtol=1e-10, atol=1e-15)


def test_multiphase_fd_calls_always_with_polarization_settings(precessing_params):
    """generate_prec_hpc_opt is the only function that should be called through
    generate_multi_phase_fd_polarizations"""

    class MyException(Exception):
        pass

    inst = GenerateWaveform(precessing_params | {"polarizations_from_coprec": False})
    with patch(
        "pyseobnr.generate_waveform.generate_prec_hpc_opt",
    ) as p_generate_prec_hpc_opt:
        p_generate_prec_hpc_opt.side_effect = MyException
        with pytest.raises(MyException):
            _ = inst.generate_multi_phase_fd_polarizations([0.0, np.pi])
        p_generate_prec_hpc_opt.assert_called_once()
        assert "settings" in p_generate_prec_hpc_opt.call_args.kwargs
        assert (
            "coprecessing_modes_only"
            in p_generate_prec_hpc_opt.call_args.kwargs["settings"]
        ) and p_generate_prec_hpc_opt.call_args.kwargs["settings"][
            "coprecessing_modes_only"
        ]


def test_multiphase_does_not_mutate_parameters(precessing_params):
    """Parameters should not be mutated via a call to generate_multi_phase_fd_polarizations."""

    precessing_params = precessing_params.copy()
    precessing_params |= {"polarizations_from_coprec": False}

    inst = GenerateWaveform(precessing_params)

    with patch.object(
        GenerateWaveform,
        "generate_td_polarizations",
        autospec=True,
        side_effect=GenerateWaveform.generate_td_polarizations,
    ) as p_generate_td_polarizations:

        _ = inst.generate_multi_phase_fd_polarizations([0.0, np.pi])[0]

        p_generate_td_polarizations.assert_called_once()

    assert not inst.parameters["polarizations_from_coprec"]
