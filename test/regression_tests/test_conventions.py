from __future__ import annotations

import re
from itertools import product
from typing import Any, Callable, cast
from unittest.mock import patch

import lal
import numpy as np
from scipy.interpolate import CubicSpline

import pytest

from pyseobnr.generate_waveform import (
    GenerateWaveform,
    generate_modes_opt,
    generate_prec_hpc_opt,
)
from pyseobnr.models.SEOBNRv5HM import SEOBNRv5HM_opt, SEOBNRv5PHM_opt


@pytest.fixture
def basic_settings():
    m1 = 50.0
    m2 = 30.0
    params_dict = {
        "mass1": m1,
        "mass2": m2,
        "spin1x": 0.2,
        "spin1y": 0,
        "spin1z": -0.3,
        "spin2x": 0,
        "spin2y": 0.7,
        "spin2z": 0.3,
        "deltaT": 1 / 2048.0,
        "deltaF": 0.125,
        "f22_start": 0.0157 / ((m1 + m2) * np.pi * lal.MTSUN_SI),
        "phi_ref": 0.0,
        "distance": 1.0,
        "inclination": np.pi / 3.0,
        "f_max": 1024.0,
        "approximant": "SEOBNRv5PHM",
        "enable_antisymmetric_modes": False,
        "postadiabatic": False,
    }
    return params_dict


def test_convention_settings_affect_waveform_generate_modes_opt():
    """Checks that non-default convention settings actually modify the waveforms.

    ... and setting the option to False has no effect
    """

    option_phm = dict(
        chi1=[0.2, 0.0, -0.3],
        chi2=[0.0, 0.7, 0.3],
        approximant="SEOBNRv5PHM",
    )

    option_hm = dict(
        chi1=-0.3,
        chi2=0.3,
        approximant="SEOBNRv5HM",
    )

    # changing omega_ref, as this changes the code path
    for dict_omega_ref in (
        {},
        dict(
            omega_ref=0.12,
        ),
    ):
        t_hm, modes_no_conventions_hm, model_hm = generate_modes_opt(
            q=1.1, omega_start=0.1, debug=True, **(option_hm | dict_omega_ref)
        )

        t_phm, modes_no_conventions_phm, model_phm = generate_modes_opt(
            q=1.1,
            omega_start=0.1,
            debug=True,
            **(option_phm | dict_omega_ref),
        )

        for option, (kwargs, modes_to_compare_to) in product(
            (True, False),
            zip(
                (option_hm, option_phm),
                (modes_no_conventions_hm, modes_no_conventions_phm),
            ),
        ):
            t_conv, modes, model = generate_modes_opt(
                q=1.1,
                omega_start=0.1,
                debug=True,
                settings=dict(
                    convention_coprecessing_phase22_set_to_0_at_reference_frequency=option
                ),
                **(kwargs | dict_omega_ref),
            )

            if option:
                with pytest.raises(AssertionError):
                    np.testing.assert_allclose(
                        modes["2,2"],
                        modes_to_compare_to["2,2"],
                        rtol=1e-6,
                        atol=1e-10,
                    )

            else:
                np.testing.assert_allclose(
                    modes["2,2"],
                    modes_to_compare_to["2,2"],
                    rtol=1e-6,
                    atol=1e-13,
                )


def test_convention_settings_affect_waveform_generate_waveform(basic_settings):
    """Checks that non-default convention settings actually modify the waveforms from
    the GenerateWaveform interface

    ... and setting the option to False has no effect
    """

    option_phm = basic_settings
    option_hm = basic_settings | {
        "approximant": "SEOBNRv5HM",
        "spin1x": 0,
        "spin1y": 0,
        "spin2x": 0,
        "spin2y": 0,
    }

    for dict_omega_ref in (
        {},
        dict(
            omega_ref=0.12,
        ),
    ):
        # mass is the same for the 2 models
        ref_freq_dict: dict = (
            {
                "f_ref": dict_omega_ref["omega_ref"]
                / (np.pi * (option_phm["mass1"] + option_phm["mass2"]) * lal.MTSUN_SI)
            }
            if "omega_ref" in dict_omega_ref
            else {}
        )

        hp_hm, hc_hm = cast(
            tuple[Any, Any],
            GenerateWaveform(option_hm | ref_freq_dict).generate_td_polarizations(),
        )
        hp_phm, hc_phm = cast(
            tuple[Any, Any],
            GenerateWaveform(option_phm | ref_freq_dict).generate_td_polarizations(),
        )

        for option, (kwargs, (hp_to_compare_to, hc_to_compare_to)) in product(
            (True, False),
            zip(
                (option_hm, option_phm),
                ((hp_hm, hc_hm), (hp_phm, hc_phm)),
            ),
        ):
            final_dict = (
                kwargs
                | ref_freq_dict
                | dict(
                    convention_coprecessing_phase22_set_to_0_at_reference_frequency=option
                )
            )
            hp, hc = cast(
                tuple[Any, Any],
                GenerateWaveform(final_dict).generate_td_polarizations(),
            )

            if option:
                min_len_ = min(len(hp.data.data), len(hp_to_compare_to.data.data))
                assert np.any(
                    hp.data.data[:min_len_] != hp_to_compare_to.data.data[:min_len_]
                )
                with pytest.raises(AssertionError):
                    np.testing.assert_allclose(
                        hp.data.data,
                        hp_to_compare_to.data.data,
                    )

                with pytest.raises(AssertionError):
                    np.testing.assert_allclose(
                        hc.data.data,
                        hc_to_compare_to.data.data,
                    )
            else:
                np.testing.assert_allclose(
                    hp.data.data,
                    hp_to_compare_to.data.data,
                )

                np.testing.assert_allclose(
                    hc.data.data,
                    hc_to_compare_to.data.data,
                )


def check_phase_0_for_same_omega_start_and_omega_ref(modes_22):
    # for SEOBNRv5PHM:
    # normally the test should be performed on the modes on the co-precessing frame
    # but all the 3-frames are the same at reference frequency by construction.
    # so this work...
    assert np.abs(np.angle(modes_22[0])) < 1e-10


def max_phase_diff_mod_pi(a, b):
    """Maximum phase difference between two complex arrays, reduced modulo pi.

    Computed from ``angle(a * conj(b))`` rather than
    ``angle(a) - angle(b)`` so it is robust to the branch cut of ``np.angle``
    at +/-pi: samples sitting on the negative real axis (e.g. the first
    polarisation sample, where the cross polarisation is ~0) can be assigned
    +pi by one computation and -pi by another, producing a spurious 2*pi
    difference. The returned value is the distance to the nearest multiple of
    pi, in [0, pi/2].
    """
    d = np.abs(np.angle(a * np.conj(b)))  # in [0, pi], ~0 when a ~ b (mod sign)
    return np.max(np.minimum(d, np.pi - d))


def check_phase_0_for_different_omega_start_and_omega_ref(
    t, modes_22, reference_time=0, tolerance=1e-10
):

    # the input time has been shifted (starts at 0)
    assert t[0] == 0
    t_idx_ref = np.searchsorted(t, reference_time)

    # check that it is found
    assert t_idx_ref < t.shape[0]

    # we are crossing the 0 of the phase shift convention
    assert np.angle(modes_22[t_idx_ref - 1]) * np.angle(modes_22[t_idx_ref]) < 0

    left: float
    right: float
    left, right = max(0, t_idx_ref - 5), t_idx_ref + 5
    interpolated_angle = CubicSpline(
        t[left:right],
        np.unwrap(np.angle(modes_22[left:right])),
    )

    assert np.abs(interpolated_angle(reference_time)) < tolerance


def _get_internal_model_from_generate_wf(
    settings,
    generate_modes_opt_or_generate_prec_hpc_opt="generate_modes_opt",
    func_prepare_instance: Callable[[GenerateWaveform], None] | None = None,
):
    """Captures the model returned by generate_modes_opt/generate_prec_hpc_opt through a
    higer level call (GenerateWaveform).)"""
    model: SEOBNRv5HM_opt | SEOBNRv5PHM_opt | None = None
    function = (
        generate_modes_opt
        if generate_modes_opt_or_generate_prec_hpc_opt == "generate_modes_opt"
        else generate_prec_hpc_opt
    )
    with patch(
        f"pyseobnr.generate_waveform.{generate_modes_opt_or_generate_prec_hpc_opt}"
    ) as p_generate_modes_func:

        def _generate_modes_func(*args, **kwargs):
            nonlocal model
            t, modes, model = function(*args, **(kwargs | dict(debug=True)))
            return (t, modes, model) if kwargs.get("debug", False) else (t, modes)

        p_generate_modes_func.side_effect = _generate_modes_func

        gen = GenerateWaveform(settings)

        if func_prepare_instance is not None:
            func_prepare_instance(gen)

        t, modes_dict = gen.generate_td_modes()

        p_generate_modes_func.assert_called_once()

        t_model = None
        if hasattr(model, "t"):
            t_model = model.t.copy()
            assert modes_dict[2, 2].shape == t_model.shape

    return t_model, model, modes_dict


def _get_internal_model_from_generate_wf_polarization(settings):
    model = None
    hpc = None

    with patch(
        "pyseobnr.generate_waveform.generate_prec_hpc_opt"
    ) as p_generate_pol_func:

        def _generate_prec_hpc_opt(*args, **kwargs):
            nonlocal model, hpc
            t, hpc, model = generate_prec_hpc_opt(*args, **(kwargs | dict(debug=True)))
            return (t, hpc, model) if kwargs.get("debug", False) else (t, hpc)

        p_generate_pol_func.side_effect = _generate_prec_hpc_opt

        gen = GenerateWaveform(settings)
        hp, hc = gen.generate_td_polarizations()

        p_generate_pol_func.assert_called_once()

        t_model = model.t.copy()
        t_model /= (
            settings["mass1"] + settings["mass2"]
        ) * lal.MTSUN_SI  # because this is modified in place
        assert hp.data.data.shape == t_model.shape

    return t_model, model, -(hp.data.data - 1j * hc.data.data)


def test_convention_coprecessing_phase22_at_0_hm(basic_settings):
    """Checks that settings associated to conventions are properly passed to
    generate_modes_opt and GenerateWaveform interfaces"""

    settings = {
        "convention_coprecessing_phase22_set_to_0_at_reference_frequency": True,
    }

    generate_wf_settings = {
        "spin1x": 0,
        "spin1y": 0,
        "spin2x": 0,
        "spin2y": 0,
        "approximant": "SEOBNRv5HM",
    }

    #
    # same omega start and reference
    #

    t, modes, model = generate_modes_opt(
        q=1.1,
        chi1=-0.3,
        chi2=0.3,
        omega_start=0.01,  # should not use a different f_ref and f0 in this case
        debug=True,
        approximant="SEOBNRv5HM",
        settings=settings,
    )
    assert model.t_ref is None
    assert abs(model.f_ref - model.f0) < 1e-10
    check_phase_0_for_same_omega_start_and_omega_ref(modes["2,2"] * np.exp(-1j * np.pi))

    gen = GenerateWaveform(basic_settings | settings | generate_wf_settings)
    t, modes_dict = gen.generate_td_modes()
    check_phase_0_for_same_omega_start_and_omega_ref(modes_dict[2, 2])

    gen = GenerateWaveform(basic_settings | generate_wf_settings)
    t, modes_dict = gen.generate_td_modes()
    with pytest.raises(AssertionError):
        check_phase_0_for_same_omega_start_and_omega_ref(modes_dict[2, 2])

    #
    # different omega start and reference
    #

    t, modes, model = generate_modes_opt(
        q=1.1,
        chi1=-0.3,
        chi2=0.3,
        omega_start=0.1,  # should have a different f_ref and f0
        debug=True,
        approximant="SEOBNRv5HM",
        settings=settings,
    )
    assert model.t_ref is not None
    assert abs(model.f_ref - model.f0) > 1e-10
    check_phase_0_for_different_omega_start_and_omega_ref(
        t - t[0], modes["2,2"] * np.exp(-1j * np.pi), reference_time=model.t_ref
    )

    m_total = 50
    corresponding_setting_generate_wf = {
        "mass1": 1.1 * m_total / (1 + 1.1),
        "mass2": m_total / (1 + 1.1),
        "approximant": "SEOBNRv5HM",
        "f22_start": 0.1 / (np.pi * m_total * lal.MTSUN_SI),
        "deltaT": m_total * lal.MTSUN_SI / 10,
        "spin1x": 0,
        "spin1y": 0,
        "spin1z": -0.3,
        "spin2x": 0,
        "spin2y": 0,
        "spin2z": 0.3,
    }
    time_model, model, modes_dict = _get_internal_model_from_generate_wf(
        corresponding_setting_generate_wf | settings
    )
    # same sanity checks on the system
    assert model.t_ref is not None
    check_phase_0_for_different_omega_start_and_omega_ref(
        time_model - time_model[0], modes_dict[2, 2], reference_time=model.t_ref
    )

    # now checking without the setting
    time_model, model, modes_dict = _get_internal_model_from_generate_wf(
        corresponding_setting_generate_wf
    )
    with pytest.raises(AssertionError):
        # this should be the same call as above
        check_phase_0_for_different_omega_start_and_omega_ref(
            time_model - time_model[0], modes_dict[2, 2], reference_time=model.t_ref
        )


@pytest.fixture
def phm_testing_parameters() -> tuple[dict, dict]:
    m_total = 50

    return {
        "mass1": 1.1 * m_total / (1 + 1.1),
        "mass2": m_total / (1 + 1.1),
        "approximant": "SEOBNRv5PHM",
        "f22_start": 0.025 / (np.pi * m_total * lal.MTSUN_SI),
        "deltaT": m_total * lal.MTSUN_SI / 10,
        "spin1x": 0.2,
        "spin1y": 0,
        "spin1z": -0.3,
        "spin2x": 0,
        "spin2y": 0.7,
        "spin2z": 0.3,
        "phi_ref": 0.0,
    }, dict(
        q=1.1,
        chi1=np.array([0.2, 0.0, -0.3]),
        chi2=np.array([0.0, 0.7, 0.3]),
        omega_start=0.025,
        debug=True,
        approximant="SEOBNRv5PHM",
    )


def test_convention_coprecessing_phase22_at_0_phm(phm_testing_parameters):
    """Checks that convention associated to phase22 are properly passed
    to generate_modes_opt and GenerateWaveform does the work"""

    settings = {
        "enable_antisymmetric_modes": False,
        "convention_coprecessing_phase22_set_to_0_at_reference_frequency": True,
        "convention_t0_set_to_0_at_coprecessing_amplitude22_peak": True,
        "return_coprec": True,
    }
    corresponding_setting_generate_wf, generate_mode_opt_kwargs = phm_testing_parameters
    corresponding_setting_generate_wf |= dict(polarizations_from_coprec=False)
    M_total = (
        corresponding_setting_generate_wf["mass1"]
        + corresponding_setting_generate_wf["mass2"]
    )

    # generate modes opt
    _, modes, model = generate_modes_opt(**generate_mode_opt_kwargs, settings=settings)
    assert model.t_ref is None
    check_phase_0_for_same_omega_start_and_omega_ref(
        model.coprecessing_modes["2,2"] * np.exp(-1j * np.pi)
    )

    _, modes, model = generate_modes_opt(
        **generate_mode_opt_kwargs, settings={"return_coprec": True}
    )
    with pytest.raises(AssertionError):
        check_phase_0_for_same_omega_start_and_omega_ref(
            model.coprecessing_modes["2,2"] * np.exp(-1j * np.pi)
        )

    # GenerateWaveform
    gen = GenerateWaveform(corresponding_setting_generate_wf | settings)
    t, modes_dict = gen.generate_td_modes()
    check_phase_0_for_same_omega_start_and_omega_ref(modes_dict[2, 2])

    gen = GenerateWaveform(corresponding_setting_generate_wf)
    t, modes_dict = gen.generate_td_modes()
    with pytest.raises(AssertionError):
        check_phase_0_for_same_omega_start_and_omega_ref(modes_dict[2, 2])

    #
    # now with omega_ref != omega_start
    #

    t, modes, model = generate_modes_opt(
        **generate_mode_opt_kwargs, omega_ref=0.12, settings=settings
    )
    assert model.t_ref is not None
    assert abs(model.f_ref - model.f_start) > 1e-10
    check_phase_0_for_different_omega_start_and_omega_ref(
        t - t[0],
        model.coprecessing_modes["2,2"] * np.exp(-1j * np.pi),
        reference_time=model.t_ref,
    )

    # checks the GWSignal interface
    time_model, model, modes_dict = _get_internal_model_from_generate_wf(
        corresponding_setting_generate_wf
        | {"f_ref": 0.12 / (np.pi * M_total * lal.MTSUN_SI)}
        | settings,
        # generate_modes_opt_or_generate_prec_hpc_opt=internal_function_being_called,
    )

    # same sanity checks on the system
    assert model.t_ref is not None
    check_phase_0_for_different_omega_start_and_omega_ref(
        time_model - time_model[0], modes_dict[2, 2], reference_time=model.t_ref
    )

    # check without the setting -> should fail
    time_model, model, modes_dict = _get_internal_model_from_generate_wf(
        corresponding_setting_generate_wf
        | {"f_ref": 0.12 / (np.pi * M_total * lal.MTSUN_SI)},
        # generate_modes_opt_or_generate_prec_hpc_opt=internal_function_being_called,
    )
    with pytest.raises(AssertionError):
        check_phase_0_for_different_omega_start_and_omega_ref(
            time_model - time_model[0], modes_dict[2, 2], reference_time=model.t_ref
        )


def _check_phase_shift_models_with_and_without_conventions(
    _model_w_convention: SEOBNRv5HM_opt | SEOBNRv5PHM_opt,
    _model_wo_phase_convention: SEOBNRv5HM_opt | SEOBNRv5PHM_opt,
):
    _sym_modes_with_phase_convention = _model_w_convention.symmetric_modes_full[2, 2]
    _asym_modes_with_phase_convention = _model_w_convention.antisymmetric_modes_full[
        2, 2
    ]

    _sym_modes_without_phase_convention = (
        _model_wo_phase_convention.symmetric_modes_full[2, 2]
    )
    _asym_modes_without_phase_convention = (
        _model_wo_phase_convention.antisymmetric_modes_full[2, 2]
    )

    t_w_convention = _model_w_convention.t
    t_wo_phase_convention = _model_wo_phase_convention.t

    if _model_w_convention.t_ref is not None:
        # check the convention is honored
        check_phase_0_for_different_omega_start_and_omega_ref(
            t_w_convention - t_w_convention[0],
            _sym_modes_with_phase_convention * np.exp(-1j * np.pi),
            reference_time=_model_w_convention.t_ref,
        )

        # we are comparing the same configurations
        assert _model_wo_phase_convention.t_ref is not None

        with pytest.raises(AssertionError):
            check_phase_0_for_different_omega_start_and_omega_ref(
                t_wo_phase_convention - t_wo_phase_convention[0],
                _sym_modes_without_phase_convention * np.exp(-1j * np.pi),
                reference_time=_model_wo_phase_convention.t_ref,
            )

    for modes_with_convention, modes_wo_convention in (
        (_sym_modes_with_phase_convention, _sym_modes_without_phase_convention),
        (_asym_modes_with_phase_convention, _asym_modes_without_phase_convention),
    ):

        phase_shift_between_conventions = np.angle(
            modes_wo_convention * np.conj(modes_with_convention)
        )
        assert np.abs(phase_shift_between_conventions[0]) > 1e-3
        section_with_amplitude = (
            np.max(
                np.vstack(
                    (
                        np.abs(modes_wo_convention),
                        np.abs(modes_with_convention),
                    )
                ),
                axis=0,
            )
            >= 1e-12
        )
        assert (
            np.max(
                np.abs(
                    phase_shift_between_conventions - phase_shift_between_conventions[0]
                )[section_with_amplitude]
            )
            < 1e-12
        )

    # now we check that the phase difference between symmetric is the same as the phase
    # difference between asym
    # we already check those are constant
    phase_shift_sym = np.angle(
        _sym_modes_without_phase_convention * np.conj(_sym_modes_with_phase_convention)
    )[0]

    phase_shift_asym = np.angle(
        _asym_modes_without_phase_convention
        * np.conj(_asym_modes_with_phase_convention)
    )[0]

    assert phase_shift_sym == pytest.approx(phase_shift_asym, abs=1e-10)


def test_convention_coprecessing_phase22_at_0_phm_with_asym(phm_testing_parameters):
    """Checks that convention associated to phase22 are properly passed
    to generate_modes_opt and GenerateWaveform does the work. The case here is with asymetries
    and the convention acts only on the 22 without asymetries."""

    # we specify convention_t0_set_to_0_at_coprecessing_amplitude22_peak to avoid
    # any displacement due to a different frame invariant amplitude??
    settings = {
        "convention_coprecessing_phase22_set_to_0_at_reference_frequency": True,
        "convention_t0_set_to_0_at_coprecessing_amplitude22_peak": True,
        "enable_antisymmetric_modes": True,
        "return_coprec": True,
    }
    corresponding_setting_generate_wf, generate_mode_opt_kwargs = phm_testing_parameters
    corresponding_setting_generate_wf |= dict(
        polarizations_from_coprec=False,
        convention_t0_set_to_0_at_coprecessing_amplitude22_peak=True,
        enable_antisymmetric_modes=True,
        postadiabatic=False,  # important so that we can compare with more precision
    )

    M_total = (
        corresponding_setting_generate_wf["mass1"]
        + corresponding_setting_generate_wf["mass2"]
    )

    # generate modes opt: we check that the symmetric_modes_full honours the convention
    # but coprecessing does not (contributions from asym on the 22).
    _, modes, model_w_convention = generate_modes_opt(
        **generate_mode_opt_kwargs, settings=settings
    )
    assert model_w_convention.t_ref is None
    check_phase_0_for_same_omega_start_and_omega_ref(
        model_w_convention.symmetric_modes_full[2, 2] * np.exp(-1j * np.pi)
    )
    with pytest.raises(AssertionError):
        check_phase_0_for_same_omega_start_and_omega_ref(
            model_w_convention.coprecessing_modes["2,2"] * np.exp(-1j * np.pi)
        )

    # just checking that disabling the convention does not honour the property we look for
    # the phase shift between the modes with and without the convention enabled should be constant
    _, modes, model_wo_phase_convention = generate_modes_opt(
        **generate_mode_opt_kwargs,
        settings=settings
        | {"convention_coprecessing_phase22_set_to_0_at_reference_frequency": False},
    )
    with pytest.raises(AssertionError):
        check_phase_0_for_same_omega_start_and_omega_ref(
            model_wo_phase_convention.symmetric_modes_full[2, 2] * np.exp(-1j * np.pi)
        )

    with pytest.raises(AssertionError):
        check_phase_0_for_same_omega_start_and_omega_ref(
            model_wo_phase_convention.coprecessing_modes["2,2"] * np.exp(-1j * np.pi)
        )

    _check_phase_shift_models_with_and_without_conventions(
        _model_wo_phase_convention=model_wo_phase_convention,
        _model_w_convention=model_w_convention,
    )

    #
    # GenerateWaveform
    #

    #
    # asym+phase22 convention
    gen = GenerateWaveform(corresponding_setting_generate_wf | settings)
    t, modes_dict = gen.generate_td_modes()
    with pytest.raises(AssertionError):
        check_phase_0_for_same_omega_start_and_omega_ref(modes_dict[2, 2])
    # because of LAL conventions, we undo the *-1
    check_phase_0_for_same_omega_start_and_omega_ref(
        modes_dict[2, 2] * np.conj(-1 * model_w_convention.waveform_modes["2,2"])
    )

    # asym+w/o. phase22 convention
    gen = GenerateWaveform(corresponding_setting_generate_wf)
    t, modes_dict = gen.generate_td_modes()
    with pytest.raises(AssertionError):
        check_phase_0_for_same_omega_start_and_omega_ref(modes_dict[2, 2])
    # make sure postadiabatic is the same for both...
    check_phase_0_for_same_omega_start_and_omega_ref(
        modes_dict[2, 2]
        * (-1 * np.conj(model_wo_phase_convention.waveform_modes["2,2"]))
    )

    #
    # now with omega_ref != omega_start
    #
    t, modes, model_w_convention = generate_modes_opt(
        **generate_mode_opt_kwargs, omega_ref=0.12, settings=settings
    )
    assert model_w_convention.t_ref is not None
    assert abs(model_w_convention.f_ref - model_w_convention.f_start) > 1e-10
    check_phase_0_for_different_omega_start_and_omega_ref(
        t - t[0],
        model_w_convention.symmetric_modes_full[2, 2] * np.exp(-1j * np.pi),
        reference_time=model_w_convention.t_ref,
    )

    # compare the anti-sym modes with and without the convention
    _, _, model_wo_phase_convention = generate_modes_opt(
        **generate_mode_opt_kwargs,
        omega_ref=0.12,
        settings=settings
        | {"convention_coprecessing_phase22_set_to_0_at_reference_frequency": False},
    )

    # compare phase shifts between convention enabled/disabled
    _check_phase_shift_models_with_and_without_conventions(
        _model_wo_phase_convention=model_wo_phase_convention,
        _model_w_convention=model_w_convention,
    )

    #
    # Through the GenerateWF interface
    #
    time_model, model_w_convention, modes_dict = _get_internal_model_from_generate_wf(
        corresponding_setting_generate_wf
        | {"f_ref": 0.12 / (np.pi * M_total * lal.MTSUN_SI)}
        | settings,
    )
    assert model_w_convention is not None

    # same sanity checks on the system
    assert model_w_convention.t_ref is not None
    assert abs(model_w_convention.f_ref - model_w_convention.f_start) > 1e-10

    # check without the setting -> should fail
    time_model, model_wo_phase_convention, modes_dict = (
        _get_internal_model_from_generate_wf(
            corresponding_setting_generate_wf
            | {"f_ref": 0.12 / (np.pi * M_total * lal.MTSUN_SI)},
        )
    )
    assert model_wo_phase_convention is not None
    assert model_wo_phase_convention.t_ref is not None
    assert (
        abs(model_wo_phase_convention.f_ref - model_wo_phase_convention.f_start) > 1e-10
    )

    _check_phase_shift_models_with_and_without_conventions(
        _model_w_convention=model_w_convention,
        _model_wo_phase_convention=model_wo_phase_convention,
    )


def test_convention_coprecessing_phase22_at_0_phm_polarizations_from_coprec(
    phm_testing_parameters,
):
    """Checks that convention associated to phase22 are properly passed to
    generate_modes_opt and GenerateWaveform does the work"""

    corresponding_setting_generate_wf, generate_mode_opt_kwargs = phm_testing_parameters
    generate_prec_hpc_opt_kwargs = generate_mode_opt_kwargs.copy()
    generate_prec_hpc_opt_kwargs.pop("approximant")
    corresponding_setting_generate_wf |= dict(polarizations_from_coprec=True)

    settings = {
        "convention_coprecessing_phase22_set_to_0_at_reference_frequency": True,
        "polarizations_from_coprec": False,
        "return_modes": [(2, 2), (2, 1)],
    }

    # generate modes opt
    _, modes, model = generate_modes_opt(**generate_mode_opt_kwargs, settings=settings)
    assert model.t_ref is None

    _, hpc_polarizations_from_coprec, model1 = generate_prec_hpc_opt(
        **generate_prec_hpc_opt_kwargs,
        settings=settings
        | {
            "polarizations_from_coprec": True,
            "phiref": np.pi / 2 - corresponding_setting_generate_wf["phi_ref"],
            "inclination": 0,
        },
    )
    assert model1.t_ref is None

    hpc_polarized_manually = np.sum(
        [
            lal.SpinWeightedSphericalHarmonic(
                0,
                np.pi / 2 - corresponding_setting_generate_wf["phi_ref"],
                -2,
                2,
                emm,
            )
            * modes[f"2,{emm}"]
            for emm in [-2, -1, 1, 2]
        ],
        axis=0,  # important
    )

    # we have the same polarization
    assert (
        np.max(np.abs(hpc_polarized_manually - hpc_polarizations_from_coprec)) < 1e-10
    )

    assert (
        max_phase_diff_mod_pi(hpc_polarized_manually, hpc_polarizations_from_coprec)
        < 1e-10
    )

    # GenerateWaveform / polarization
    # comparing with and without coprec
    gen = GenerateWaveform(
        corresponding_setting_generate_wf
        | settings
        | {"polarizations_from_coprec": False, "mode_array": [(2, 2), (2, 1)]}
    )
    hp, hc = gen.generate_td_polarizations()

    gen = GenerateWaveform(
        corresponding_setting_generate_wf
        | settings
        | {"polarizations_from_coprec": True, "mode_array": [(2, 2), (2, 1)]}
    )
    hp_copre, hc_copre = gen.generate_td_polarizations()

    assert (
        np.max(
            np.abs(
                (hp.data.data - 1j * hc.data.data)
                - (hp_copre.data.data - 1j * hc_copre.data.data)
            )
        )
        < 1e-10
    )
    assert (
        max_phase_diff_mod_pi(
            hp.data.data - 1j * hc.data.data,
            hp_copre.data.data - 1j * hc_copre.data.data,
        )
        < 1e-10
    )

    #
    # checking validity of the test by removing the option

    gen = GenerateWaveform(
        corresponding_setting_generate_wf
        | settings
        | {
            "polarizations_from_coprec": True,
            "mode_array": [(2, 2), (2, 1)],
            "convention_coprecessing_phase22_set_to_0_at_reference_frequency": False,
        }
    )
    hp_copre_wo, hc_copre_wo = gen.generate_td_polarizations()

    # curiously the magnitudes are equivalent
    assert (
        np.max(
            np.abs(
                (hp.data.data - 1j * hc.data.data)
                - (hp_copre_wo.data.data - 1j * hc_copre_wo.data.data)
            )
        )
        < 1e-10
    )

    # however the angles are different, the previous conditions with the same
    # settings do not hold anymore
    assert not (
        max_phase_diff_mod_pi(
            hp.data.data - 1j * hc.data.data,
            hp_copre_wo.data.data - 1j * hc_copre_wo.data.data,
        )
        < 1e-10
    )


def test_convention_coprecessing_phase22_at_0_phm_polarizations_from_coprec_different_reference_frequency(
    phm_testing_parameters,
):
    """Checks that convention associated to phase22 are properly passed to
    generate_modes_opt and GenerateWaveform does the work"""

    corresponding_setting_generate_wf, generate_mode_opt_kwargs = phm_testing_parameters

    # specific to differences omega_ref / omega_start
    m_total = (
        corresponding_setting_generate_wf["mass1"]
        + corresponding_setting_generate_wf["mass2"]
    )
    generate_mode_opt_kwargs |= {"omega_ref": 0.12}
    corresponding_setting_generate_wf |= {
        "f_ref": 0.12 / (np.pi * m_total * lal.MTSUN_SI)
    }

    # other settings, same as test above
    generate_prec_hpc_opt_kwargs = generate_mode_opt_kwargs.copy()
    generate_prec_hpc_opt_kwargs.pop("approximant")
    corresponding_setting_generate_wf |= dict(polarizations_from_coprec=True)

    settings = {
        "convention_coprecessing_phase22_set_to_0_at_reference_frequency": True,
        "polarizations_from_coprec": False,
        "return_modes": [(2, 2), (2, 1)],
    }

    # generate modes opt
    _, modes, model = generate_modes_opt(**generate_mode_opt_kwargs, settings=settings)
    assert model.t_ref is not None

    _, hpc_polarizations_from_coprec, model1 = generate_prec_hpc_opt(
        **generate_prec_hpc_opt_kwargs,
        settings=settings
        | {
            "polarizations_from_coprec": True,
            "phiref": np.pi / 2 - corresponding_setting_generate_wf["phi_ref"],
            "inclination": 0,
        },
    )
    assert model1.t_ref is not None

    hpc_polarized_manually = np.sum(
        [
            lal.SpinWeightedSphericalHarmonic(
                0,
                np.pi / 2 - corresponding_setting_generate_wf["phi_ref"],
                -2,
                2,
                emm,
            )
            * modes[f"2,{emm}"]
            for emm in [-2, -1, 1, 2]
        ],
        axis=0,  # important
    )

    # we have the same polarization
    assert (
        np.max(np.abs(hpc_polarized_manually - hpc_polarizations_from_coprec)) < 1e-10
    )

    assert (
        max_phase_diff_mod_pi(hpc_polarized_manually, hpc_polarizations_from_coprec)
        < 1e-10
    )

    # GenerateWaveform / polarization
    # comparing with and without coprec
    gen = GenerateWaveform(
        corresponding_setting_generate_wf
        | settings
        | {"polarizations_from_coprec": False, "mode_array": [(2, 2), (2, 1)]}
    )
    hp, hc = gen.generate_td_polarizations()

    gen = GenerateWaveform(
        corresponding_setting_generate_wf
        | settings
        | {"polarizations_from_coprec": True, "mode_array": [(2, 2), (2, 1)]}
    )
    hp_copre, hc_copre = gen.generate_td_polarizations()

    assert (
        np.max(
            np.abs(
                (hp.data.data - 1j * hc.data.data)
                - (hp_copre.data.data - 1j * hc_copre.data.data)
            )
        )
        < 1e-10
    )
    assert (
        max_phase_diff_mod_pi(
            hp.data.data - 1j * hc.data.data,
            hp_copre.data.data - 1j * hc_copre.data.data,
        )
        < 1e-10
    )

    #
    # checking validity of the test by removing the option

    gen = GenerateWaveform(
        corresponding_setting_generate_wf
        | settings
        | {
            "polarizations_from_coprec": True,
            "mode_array": [(2, 2), (2, 1)],
            "convention_coprecessing_phase22_set_to_0_at_reference_frequency": False,
        }
    )
    hp_copre_wo, hc_copre_wo = gen.generate_td_polarizations()

    # curiously the magnitudes are equivalent
    assert (
        np.max(
            np.abs(
                (hp.data.data - 1j * hc.data.data)
                - (hp_copre_wo.data.data - 1j * hc_copre_wo.data.data)
            )
        )
        < 1e-10
    )

    # however the angles are not
    assert not (
        max_phase_diff_mod_pi(
            hp.data.data - 1j * hc.data.data,
            hp_copre_wo.data.data - 1j * hc_copre_wo.data.data,
        )
        < 1e-10
    )


@pytest.mark.parametrize(
    "approximant",
    [
        "SEOBNRv5HM",
        "SEOBNRv5PHM",
    ],
)
def test_convention_t0_set_to_0_at_coprecessing_amplitude22_peak(approximant):
    """Checks that settings associated to conventions are properly passed to
    generate_modes_opt and GenerateWaveform interfaces"""

    settings = {
        "convention_t0_set_to_0_at_coprecessing_amplitude22_peak": True,
    }

    def _get_modes(modes_, model_):
        if approximant == "SEOBNRv5PHM":
            modes_ = model_.coprecessing_modes

        k = list(modes_.keys())[0]
        if isinstance(k, str):
            modes_ = {
                (ell, emm): v
                for k, v in modes_.items()
                if (ell := int(k.split(",")[0])) and (emm := int(k.split(",")[1]))
            }

        return modes_

    q = 1.1
    omega_ref = 0.12
    # min omega_start above which the short WF condition is not true anymore
    # this is lower than the 10 ** (-3/2) needed for HM and works with PHM
    omega_start = 10.5 ** (-3.0 / 2.0)
    m_total = 50
    m1 = q * m_total / (1 + q)
    m2 = m_total / (1 + q)
    chi1 = np.array([0.2, 0.0, -0.3])
    chi2 = np.array([0.0, 0.7, 0.3])
    settings_generate_wf = {}
    settings_generate_modes_opt = {}

    if approximant == "SEOBNRv5HM":
        chi1[:2] = 0
        chi2[:2] = 0
    else:
        settings_generate_wf |= {
            "polarizations_from_coprec": False,
            "return_coprec": True,
            # enabling this makes it difficult to test for the 22 peak
            "enable_antisymmetric_modes": False,
        }
        settings_generate_modes_opt |= {
            "return_coprec": True,
            "enable_antisymmetric_modes": False,
        }

    dict_generate_modes_opt = dict(
        q=q,
        chi1=chi1,
        chi2=chi2,
        omega_start=omega_start,
        omega_ref=omega_ref,
        approximant=approximant,
    )
    if approximant == "SEOBNRv5HM":
        # because omega_ref is not properly propagated into the underlying
        # model from the generate_modes_opt
        settings_generate_modes_opt |= dict(
            f_ref=omega_ref / (m_total * lal.MTSUN_SI * np.pi)
        )

    dict_gen_wf = {
        "mass1": m1,
        "mass2": m2,
        "approximant": approximant,
        "f22_start": omega_start / (np.pi * m_total * lal.MTSUN_SI),
        "f_ref": omega_ref / (np.pi * m_total * lal.MTSUN_SI),
        "deltaT": m_total * lal.MTSUN_SI / 10,
        "spin1x": chi1[0],
        "spin1y": chi1[1],
        "spin1z": chi1[2],
        "spin2x": chi2[0],
        "spin2y": chi2[1],
        "spin2z": chi2[2],
        "phi_ref": 0.0,
    }

    t, modes, model = generate_modes_opt(
        **dict_generate_modes_opt,
        debug=True,
        settings=settings | settings_generate_modes_opt,
    )
    # asserts we have the case we need: omega_ref != omega_start
    assert model.t_ref is not None
    modes = _get_modes(modes, model)
    assert (2, 2) in modes.keys()

    # we check that there is a flip of sign around the computed peak of the 2,2
    #
    # * ---- * ---- *
    #     ^
    #     t_attach = t_max
    #        ^ idx_max
    # ==> we need to check for the 2 cases, and they cannot be true at the same time
    idx_max = np.argmax(np.abs(modes[2, 2]))
    # below the condition "!=" means "exclusive or"
    assert (t[idx_max - 1] * t[idx_max] < 0) != (t[idx_max] * t[idx_max + 1] < 0)

    # we do the same test for the GenerateWaveform interface
    gen = GenerateWaveform(dict_gen_wf | settings | settings_generate_wf)
    t, dict_modes = gen.generate_td_modes()
    # asserts we have the case we need
    assert gen.model.t_ref is not None
    dict_modes = _get_modes(dict_modes, gen.model)
    assert (2, 2) in dict_modes.keys()
    idx_max = np.argmax(np.abs(dict_modes[2, 2]))
    # below the condition "!=" means "exclusive or"
    assert (t[idx_max - 1] * t[idx_max] < 0) != (t[idx_max] * t[idx_max + 1] < 0)

    #
    # without the omega_start != omega_ref
    #

    dict_generate_modes_opt.pop("omega_ref")
    dict_gen_wf.pop("f_ref")

    if approximant == "SEOBNRv5HM":
        # this is needed because omega_ref is not passed property to HM from generate_modes_opt
        settings_generate_modes_opt.pop("f_ref")

    t, modes, model = generate_modes_opt(
        **dict_generate_modes_opt,
        debug=True,
        settings=settings | settings_generate_modes_opt,
    )
    # asserts we have the case we need: omega_ref == omega_start
    assert model.t_ref is None
    modes = _get_modes(modes, model)
    assert (2, 2) in modes.keys()
    idx_max = np.argmax(np.abs(modes[2, 2]))
    assert (t[idx_max - 1] * t[idx_max] < 0) != (t[idx_max] * t[idx_max + 1] < 0)

    # we do the same test for the GenerateWaveform interface
    gen = GenerateWaveform(dict_gen_wf | settings | settings_generate_wf)
    t, dict_modes = gen.generate_td_modes()
    # asserts we have the case we need
    assert gen.model.t_ref is None
    dict_modes = _get_modes(dict_modes, gen.model)
    assert (2, 2) in dict_modes.keys()
    idx_max = np.argmax(np.abs(dict_modes[2, 2]))
    # below the condition "!=" means "exclusive or"
    assert (t[idx_max - 1] * t[idx_max] < 0) != (t[idx_max] * t[idx_max + 1] < 0)


def test_convention_coprecessing_phase22_at_0_ehm_not_supported(basic_settings):
    """Checks that convention associated to phase22 is not supported for EHM and yields an error message"""

    #
    # setting convention_coprecessing_phase22_set_to_0_at_reference_frequency
    #

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Convention 'convention_coprecessing_phase22_set_to_0_at_reference_frequency' "
            "not supported by the model SEOBNRv5EHM"
        ),
    ):
        _, modes = generate_modes_opt(
            q=1.1,
            chi1=-0.3,
            chi2=0.3,
            omega_start=0.1,
            eccentricity=0.1,
            debug=False,
            approximant="SEOBNRv5EHM",
            settings={
                "convention_coprecessing_phase22_set_to_0_at_reference_frequency": True,
            },
        )

    # Setting to False should work
    _, modes = generate_modes_opt(
        q=1.1,
        chi1=-0.3,
        chi2=0.3,
        omega_start=0.1,
        eccentricity=0.1,
        debug=False,
        approximant="SEOBNRv5EHM",
        settings={
            "convention_coprecessing_phase22_set_to_0_at_reference_frequency": False,
        },
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The approximant 'SEOBNRv5EHM' does not support the choice "
            "for a convention setting the phase of the 2,2 mode."
        ),
    ):
        _ = GenerateWaveform(
            basic_settings
            | {
                "approximant": "SEOBNRv5EHM",
                "spin1x": 0,
                "spin1y": 0,
                "spin2x": 0,
                "spin2y": 0,
                "eccentricity": 0.1,
            }
            | {
                "convention_coprecessing_phase22_set_to_0_at_reference_frequency": True,
            }
        )

    # setting to False should work
    _ = GenerateWaveform(
        basic_settings
        | {
            "approximant": "SEOBNRv5EHM",
            "spin1x": 0,
            "spin1y": 0,
            "spin2x": 0,
            "spin2y": 0,
            "eccentricity": 0.1,
        }
        | {
            "convention_coprecessing_phase22_set_to_0_at_reference_frequency": False,
        }
    )

    #
    # setting convention_t0_set_to_0_at_coprecessing_amplitude22_peak
    #

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Convention 'convention_t0_set_to_0_at_coprecessing_amplitude22_peak' "
            "not supported by the model SEOBNRv5EHM"
        ),
    ):
        _, modes = generate_modes_opt(
            q=1.1,
            chi1=-0.3,
            chi2=0.3,
            omega_start=0.1,
            eccentricity=0.1,
            debug=False,
            approximant="SEOBNRv5EHM",
            settings={
                "convention_t0_set_to_0_at_coprecessing_amplitude22_peak": True,
            },
        )

    # Setting to False should work
    _, modes = generate_modes_opt(
        q=1.1,
        chi1=-0.3,
        chi2=0.3,
        omega_start=0.1,
        eccentricity=0.1,
        debug=False,
        approximant="SEOBNRv5EHM",
        settings={
            "convention_t0_set_to_0_at_coprecessing_amplitude22_peak": False,
        },
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The approximant 'SEOBNRv5EHM' does not support the choice "
            "for a convention setting the time at the peak of the 2,2 mode."
        ),
    ):
        _ = GenerateWaveform(
            basic_settings
            | {
                "approximant": "SEOBNRv5EHM",
                "spin1x": 0,
                "spin1y": 0,
                "spin2x": 0,
                "spin2y": 0,
                "eccentricity": 0.1,
            }
            | {"convention_t0_set_to_0_at_coprecessing_amplitude22_peak": True}
        )

    # Setting to False should work
    GenerateWaveform(
        basic_settings
        | {
            "approximant": "SEOBNRv5EHM",
            "spin1x": 0,
            "spin1y": 0,
            "spin2x": 0,
            "spin2y": 0,
            "eccentricity": 0.1,
        }
        | {"convention_t0_set_to_0_at_coprecessing_amplitude22_peak": False}
    )
