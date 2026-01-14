import re
from itertools import product
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
        hp_hm, hc_hm = GenerateWaveform(option_hm).generate_td_polarizations()
        hp_phm, hc_phm = GenerateWaveform(option_phm).generate_td_polarizations()

        for option, (kwargs, (hp_to_compare_to, hc_to_compare_to)) in product(
            (True, False),
            zip(
                (option_hm, option_phm),
                ((hp_hm, hc_hm), (hp_phm, hc_phm)),
            ),
        ):
            hp, hc = GenerateWaveform(
                kwargs
                | dict_omega_ref
                | dict(
                    convention_coprecessing_phase22_set_to_0_at_reference_frequency=option
                )
            ).generate_td_polarizations()

            if option:
                assert np.any(hp.data.data != hp_to_compare_to.data.data)
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


def check_phase_0_for_different_omega_start_and_omega_ref(
    t, modes_22, reference_time=0
):

    # the time has been shifted
    t_idx_ref = np.searchsorted(t, reference_time)

    # check that it is found
    assert t_idx_ref < t.shape[0]

    # we are crossing the 0 of the phase shift convention
    assert np.angle(modes_22[t_idx_ref - 1]) * np.angle(modes_22[t_idx_ref]) < 0

    interpolated_angle = CubicSpline(
        t[t_idx_ref - 5 : t_idx_ref + 5],
        np.unwrap(np.angle(modes_22[t_idx_ref - 5 : t_idx_ref + 5])),
    )

    assert np.abs(interpolated_angle(reference_time)) < 1e-10


def _get_internal_model_from_generate_wf(
    settings, generate_modes_opt_or_generate_prec_hpc_opt="generate_modes_opt"
):
    model = None
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
        t, modes_dict = gen.generate_td_modes()

        p_generate_modes_func.assert_called_once()

        t_model = model.t.copy()
        t_model /= (
            settings["mass1"] + settings["mass2"]
        ) * lal.MTSUN_SI  # because this is modified in place
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

    gen = GenerateWaveform(basic_settings | settings)
    t, modes_dict = gen.generate_td_modes()
    check_phase_0_for_same_omega_start_and_omega_ref(modes_dict[2, 2])

    gen = GenerateWaveform(basic_settings)
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
def phm_testing_parameters():
    m_total = 50

    return {
        "mass1": 1.1 * m_total / (1 + 1.1),
        "mass2": m_total / (1 + 1.1),
        "approximant": "SEOBNRv5PHM",
        "f22_start": 0.1 / (np.pi * m_total * lal.MTSUN_SI),
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
        omega_start=0.1,
        debug=True,
        approximant="SEOBNRv5PHM",
    )


def test_convention_coprecessing_phase22_at_0_phm(phm_testing_parameters):
    """Checks that convention associated to phase22 are properly passed
    to generate_modes_opt and GenerateWaveform does the work"""

    settings = {
        "convention_coprecessing_phase22_set_to_0_at_reference_frequency": True,
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
        t - t[0], modes["2,2"] * np.exp(-1j * np.pi), reference_time=model.t_ref
    )

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

    # check without the setting
    time_model, model, modes_dict = _get_internal_model_from_generate_wf(
        corresponding_setting_generate_wf
        | {"f_ref": 0.12 / (np.pi * M_total * lal.MTSUN_SI)},
        # generate_modes_opt_or_generate_prec_hpc_opt=internal_function_being_called,
    )
    with pytest.raises(AssertionError):
        check_phase_0_for_different_omega_start_and_omega_ref(
            time_model - time_model[0], modes_dict[2, 2], reference_time=model.t_ref
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
    # M_total = (
    #     corresponding_setting_generate_WF["mass1"]
    #     + corresponding_setting_generate_WF["mass2"]
    # )
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
        np.max(
            np.mod(
                np.abs(
                    np.angle(hpc_polarized_manually)
                    - np.angle(hpc_polarizations_from_coprec)
                ),
                np.pi,
            )
        )
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
        np.max(
            np.mod(
                np.abs(
                    np.angle((hp.data.data - 1j * hc.data.data))
                    - np.angle((hp_copre.data.data - 1j * hc_copre.data.data))
                ),
                np.pi,
            )
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
        np.max(
            np.mod(
                np.abs(
                    np.angle((hp.data.data - 1j * hc.data.data))
                    - np.angle((hp_copre_wo.data.data - 1j * hc_copre_wo.data.data))
                ),
                np.pi,
            )
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
        np.max(
            np.mod(
                np.abs(
                    np.angle(hpc_polarized_manually)
                    - np.angle(hpc_polarizations_from_coprec)
                ),
                np.pi,
            )
        )
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
        np.max(
            np.mod(
                np.abs(
                    np.angle((hp.data.data - 1j * hc.data.data))
                    - np.angle((hp_copre.data.data - 1j * hc_copre.data.data))
                ),
                np.pi,
            )
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

    # curiously the magnitudes are the equivalent
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
        np.max(
            np.mod(
                np.abs(
                    np.angle((hp.data.data - 1j * hc.data.data))
                    - np.angle((hp_copre_wo.data.data - 1j * hc_copre_wo.data.data))
                ),
                np.pi,
            )
        )
        < 1e-10
    )


def test_convention_t0_set_to_0_at_coprecessing_amplitude22_peak_hm(basic_settings):
    """Checks that settings associated to conventions are properly passed to
    generate_modes_opt and GenerateWaveform interfaces"""

    settings = {
        "convention_t0_set_to_0_at_coprecessing_amplitude22_peak": True,
    }

    t, modes, model = generate_modes_opt(
        q=1.1,
        chi1=-0.3,
        chi2=0.3,
        omega_start=0.1,
        omega_ref=0.12,
        debug=True,
        approximant="SEOBNRv5HM",
        settings=settings,
    )
    assert model.t_ref is not None
    assert "2,2" in modes.keys()

    # we check that there is a flip of sign around the computed peak of the 2,2
    #
    # * ---- * ---- *
    #     ^
    #     t_attach = t_max
    #        ^ idx_max
    # ==> we need to check for the 2 cases, and they cannot be true at the same time
    idx_max = np.argmax(np.abs(modes["2,2"]))
    # below the condition "!=" means "exclusive or"
    assert (t[idx_max - 1] * t[idx_max] < 0) != (t[idx_max] * t[idx_max + 1] < 0)

    #
    # without the omega_start != omega_ref
    #

    t, modes, model = generate_modes_opt(
        q=1.1,
        chi1=-0.3,
        chi2=0.3,
        omega_start=0.01,
        debug=True,
        approximant="SEOBNRv5HM",
        settings=settings,
    )
    assert model.t_ref is None
    idx_max = np.argmax(np.abs(modes["2,2"]))
    assert (t[idx_max - 1] * t[idx_max] < 0) != (t[idx_max] * t[idx_max + 1] < 0)


def test_convention_t0_set_to_0_at_coprecessing_amplitude22_peak_phm(basic_settings):
    """Checks that settings associated to conventions are properly passed to
    generate_modes_opt and GenerateWaveform interfaces"""

    settings = {
        "convention_t0_set_to_0_at_coprecessing_amplitude22_peak": True,
    }

    t, modes, model = generate_modes_opt(
        q=1.1,
        chi1=np.array([0.2, 0.0, -0.3]),
        chi2=np.array([0.0, 0.7, 0.3]),
        omega_start=0.1,
        omega_ref=0.12,
        debug=True,
        approximant="SEOBNRv5PHM",
        settings=settings | {"return_coprec": True},
    )
    assert model.t_ref is not None
    assert "2,2" in modes.keys()

    # we check that there is a flip of sign around the computed peak of the 2,2
    #
    # * ---- * ---- *
    #     ^
    #     t_attach = t_max
    #        ^ idx_max
    # ==> we need to check for the 2 cases, and they cannot be true at the same time
    idx_max = np.argmax(np.abs(model.coprecessing_modes["2,2"]))
    # below the condition "!=" means "exclusive or"
    assert (t[idx_max - 1] * t[idx_max] < 0) != (t[idx_max] * t[idx_max + 1] < 0)

    #
    # without omega_ref != omega_start

    t, modes, model = generate_modes_opt(
        q=1.1,
        chi1=np.array([0.2, 0.0, -0.3]),
        chi2=np.array([0.0, 0.7, 0.3]),
        omega_start=0.1,
        debug=True,
        approximant="SEOBNRv5PHM",
        settings=settings | {"return_coprec": True},
    )
    assert model.t_ref is None
    assert "2,2" in modes.keys()

    idx_max = np.argmax(np.abs(model.coprecessing_modes["2,2"]))
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
