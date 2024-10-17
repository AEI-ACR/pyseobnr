import re
from typing import get_args
from unittest import mock

import lal
import numpy as np

import pytest

import pyseobnr
from pyseobnr.generate_waveform import (
    GenerateWaveform,
    SupportedApproximants,
    generate_modes_opt,
)
from pyseobnr.models.common import VALID_MODES


@pytest.fixture
def basic_settings():
    m1 = 50.0
    m2 = 30.0
    params_dict = {
        "mass1": m1,
        "mass2": m2,
        "spin1x": 0,
        "spin1y": 0,
        "spin1z": 0.5,
        "spin2x": 0,
        "spin2y": 0,
        "spin2z": 0.1,
        "deltaT": 1 / 2048.0,
        "deltaF": 0.125,
        "f22_start": 0.0157 / ((m1 + m2) * np.pi * lal.MTSUN_SI),
        "phi_ref": 0.0,
        "distance": 1.0,
        "inclination": np.pi / 3.0,
        "f_max": 1024.0,
        "approximant": "SEOBNRv5HM",
        "postadiabatic": False,
    }
    return params_dict


def test_mode_arrays_settings(basic_settings):
    """Checks the behaviour wrt. mode_array setting"""

    # cannot accept both
    value_error = (
        r"Only one setting can be specified between .?ModeArray.? and .?mode_array.?."
    )

    with pytest.raises(
        ValueError,
        match=value_error,
    ):
        _ = GenerateWaveform(basic_settings | {"mode_array": [], "ModeArray": []})

    with pytest.raises(
        ValueError,
        match=value_error,
    ):
        _ = GenerateWaveform(
            basic_settings
            | {"approximant": "SEOBNRv5PHM", "mode_array": [], "ModeArray": []}
        )

    calls_to_check = (
        "generate_td_modes",
        "generate_td_polarizations",
        "generate_fd_polarizations",
    )

    # works with list and tuples
    for approximant in "SEOBNRv5HM", "SEOBNRv5PHM":
        wfm_gen = GenerateWaveform(
            basic_settings
            | {"approximant": approximant, "mode_array": [(2, 2), (3, 3)]}
        )

        for func in calls_to_check:
            _ = getattr(wfm_gen, func)()

        wfm_gen = GenerateWaveform(
            basic_settings
            | {"approximant": approximant, "mode_array": ((2, 2), (3, 3))}
        )

        for func in calls_to_check:
            _ = getattr(wfm_gen, func)()

        # incorrect mode array yields an error
        wfm_gen = GenerateWaveform(
            basic_settings | {"approximant": approximant, "mode_array": [2, 2, 3, 3]}
        )
        for func in calls_to_check:
            with pytest.raises(TypeError):
                _ = getattr(wfm_gen, func)()


def test_mode_arrays_settings_with_lmax_HM(basic_settings):

    approximant: SupportedApproximants = "SEOBNRv5HM"

    # this should work as lmax overrides the modes
    *_, model = generate_modes_opt(
        q=1.1,
        chi1=0,
        chi2=0,
        omega_start=0.1,
        debug=True,
        approximant=approximant,
        settings={"lmax": 2},
    )
    assert {
        tuple(int(k) for k in _.split(",")) for _ in model.waveform_modes.keys()
    } == {(ell, emm) for ell, emm in VALID_MODES if ell <= 2}

    # this also should work
    *_, model = generate_modes_opt(
        q=1.1,
        chi1=0,
        chi2=0,
        omega_start=0.1,
        debug=True,
        approximant=approximant,
        settings={"lmax": 3},
    )
    assert {
        tuple(int(k) for k in _.split(",")) for _ in model.waveform_modes.keys()
    } == {(ell, emm) for ell, emm in VALID_MODES if ell <= 3}

    # this also should work
    *_, model = generate_modes_opt(
        q=1.1,
        chi1=0,
        chi2=0,
        omega_start=0.1,
        debug=True,
        approximant=approximant,
        settings={"lmax": 5},
    )
    assert {
        tuple(int(k) for k in _.split(",")) for _ in model.waveform_modes.keys()
    } == {(ell, emm) for ell, emm in VALID_MODES if ell <= 5}
    assert "5,5" in model.waveform_modes

    # this is consistent between lmax and return_modes, "lmax" should have no effect on the final
    # list of modes

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Setting lmax=3 together with the selection of modes [(2, 2), (3, 2)] "
            "yields inconsistencies for the following modes: [(2, 1), (3, 3)]"
        ),
    ):
        *_, model = generate_modes_opt(
            q=1.1,
            chi1=0,
            chi2=0,
            omega_start=0.1,
            approximant=approximant,
            settings={"lmax": 3, "return_modes": [(2, 2), (3, 2)]},
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Setting lmax=3 together with the selection of modes [(2, 2), (4, 4)] "
            "yields inconsistencies for the following modes: [(2, 1), (3, 2), (3, 3), (4, 4)]"
        ),
    ):
        *_, model = generate_modes_opt(
            q=1.1,
            chi1=0,
            chi2=0,
            omega_start=0.1,
            approximant=approximant,
            settings={"lmax": 3, "return_modes": [(2, 2), (4, 4)]},
        )

    # impossible values of lmax yields an error
    for lmax_tentative in (18, -3):
        with pytest.raises(
            ValueError,
            match=(
                "Incorrect value for lmax=-?\\d+:\\s*the condition 1\\s*<=\\s*lmax\\s*<=\\s*\\d+ is "
                "not satisfied"
            ),
        ):
            *_, model = generate_modes_opt(
                q=1.1,
                chi1=0,
                chi2=0,
                omega_start=0.1,
                debug=True,
                approximant=approximant,
                settings={"lmax": lmax_tentative},
            )


def test_mode_arrays_settings_with_lmax_PHM(basic_settings):

    approximant: SupportedApproximants = "SEOBNRv5PHM"

    # this should work as lmax overrides the modes
    *_, model = generate_modes_opt(
        q=1.1,
        chi1=0,
        chi2=0,
        omega_start=0.1,
        debug=True,
        approximant=approximant,
        settings={"lmax": 2},
    )
    assert {
        tuple(int(k) for k in _.split(",")) for _ in model.waveform_modes.keys()
    } == (
        {(ell, emm) for ell, emm in VALID_MODES if ell <= 2}
        | {(ell, -emm) for ell, emm in VALID_MODES if ell <= 2}
        | {(ell, 0) for ell, _ in VALID_MODES if ell <= 2}
    )

    # this will narrow the modes array to l <= 3
    *_, model = generate_modes_opt(
        q=1.1,
        chi1=0,
        chi2=0,
        omega_start=0.1,
        debug=True,
        approximant=approximant,
        settings={"lmax": 3},
    )
    assert {
        tuple(int(k) for k in _.split(",")) for _ in model.waveform_modes.keys()
    } == {(ell, emm) for ell in range(2, 4) for emm in range(-ell, ell + 1)}

    # this will extend the default modes array to l = 5 modes
    *_, model = generate_modes_opt(
        q=1.1,
        chi1=0,
        chi2=0,
        omega_start=0.1,
        debug=True,
        approximant=approximant,
        settings={"lmax": 5},
    )
    assert {
        tuple(int(k) for k in _.split(",")) for _ in model.waveform_modes.keys()
    } == {(ell, emm) for ell in range(2, 6) for emm in range(-ell, ell + 1)}
    assert "5,5" in model.waveform_modes

    # this is consistent between lmax and return_modes, "lmax" should have no effect on the final
    # list of modes

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Setting lmax=3 together with the selection of modes [(2, 2), (3, 2)] "
            "yields inconsistencies for the following modes: [(2, 1), (3, 3)]"
        ),
    ):
        *_, model = generate_modes_opt(
            q=1.1,
            chi1=0,
            chi2=0,
            omega_start=0.1,
            approximant=approximant,
            settings={"lmax": 3, "return_modes": [(2, 2), (3, 2)]},
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Setting lmax=3 together with the selection of modes [(2, 2), (4, 4)] "
            "yields inconsistencies for the following modes: [(2, 1), (3, 2), (3, 3), (4, 4)]"
        ),
    ):
        *_, model = generate_modes_opt(
            q=1.1,
            chi1=0,
            chi2=0,
            omega_start=0.1,
            approximant=approximant,
            settings={"lmax": 3, "return_modes": [(2, 2), (4, 4)]},
        )

    # impossible values of lmax yields an error
    for lmax_tentative in (18, -3):
        with pytest.raises(
            ValueError,
            match=(
                "Incorrect value for lmax=-?\\d+:\\s*the condition 1\\s*<=\\s*lmax\\s*<=\\s*\\d+ is "
                "not satisfied"
            ),
        ):
            *_, model = generate_modes_opt(
                q=1.1,
                chi1=0,
                chi2=0,
                omega_start=0.1,
                debug=True,
                approximant=approximant,
                settings={"lmax": lmax_tentative},
            )


def test_lmax_in_generate_waveform(basic_settings):
    """Checks the behaviour of GenerateWaveform wrt. lmax"""

    approximants_and_function_to_patch = ("SEOBNRv5HM", "generate_modes_opt"), (
        "SEOBNRv5PHM",
        "generate_modes_opt",
    )

    class MyException(Exception):
        pass

    # Generate TD modes
    for approximant, function_to_patch in approximants_and_function_to_patch:
        wfm_gen = GenerateWaveform(
            basic_settings | {"approximant": approximant, "lmax": 4}
        )

        assert wfm_gen.parameters["lmax"] == 4

        with mock.patch(
            f"pyseobnr.generate_waveform.{function_to_patch}"
        ) as p_generate_modes_func:
            p_generate_modes_func.side_effect = MyException

            with pytest.raises(MyException):
                wfm_gen.generate_td_modes()

            p_generate_modes_func.assert_called_once()
            assert "settings" in p_generate_modes_func.call_args_list[0].kwargs
            assert "lmax" in p_generate_modes_func.call_args_list[0].kwargs["settings"]
            assert (
                p_generate_modes_func.call_args_list[0].kwargs["settings"]["lmax"] == 4
            )

    # Generate TD polarization
    approximants_and_function_to_patch = ("SEOBNRv5HM", "generate_modes_opt"), (
        "SEOBNRv5PHM",
        "generate_prec_hpc_opt",
    )

    for approximant, function_to_patch in approximants_and_function_to_patch:
        wfm_gen = GenerateWaveform(
            basic_settings | {"approximant": approximant, "lmax": 4}
        )

        assert wfm_gen.parameters["lmax"] == 4

        with mock.patch(
            f"pyseobnr.generate_waveform.{function_to_patch}"
        ) as p_generate_modes_func:
            p_generate_modes_func.side_effect = MyException

            with pytest.raises(MyException):
                wfm_gen.generate_td_polarizations()

            p_generate_modes_func.assert_called_once()
            assert "settings" in p_generate_modes_func.call_args_list[0].kwargs
            assert "lmax" in p_generate_modes_func.call_args_list[0].kwargs["settings"]
            assert (
                p_generate_modes_func.call_args_list[0].kwargs["settings"]["lmax"] == 4
            )

    # Generate FD polarization
    approximants_and_function_to_patch = ("SEOBNRv5HM", "generate_modes_opt"), (
        "SEOBNRv5PHM",
        "generate_prec_hpc_opt",
    )

    for approximant, function_to_patch in approximants_and_function_to_patch:
        wfm_gen = GenerateWaveform(
            basic_settings | {"approximant": approximant, "lmax": 4}
        )

        assert wfm_gen.parameters["lmax"] == 4

        with mock.patch(
            f"pyseobnr.generate_waveform.{function_to_patch}"
        ) as p_generate_modes_func:
            p_generate_modes_func.side_effect = MyException

            with pytest.raises(MyException):
                wfm_gen.generate_fd_polarizations()

            p_generate_modes_func.assert_called_once()
            assert "settings" in p_generate_modes_func.call_args_list[0].kwargs
            assert "lmax" in p_generate_modes_func.call_args_list[0].kwargs["settings"]
            assert (
                p_generate_modes_func.call_args_list[0].kwargs["settings"]["lmax"] == 4
            )


def test_f_ref_behaviour(basic_settings):
    """When not set, f_ref should be set to f22_start"""

    assert "f_ref" not in basic_settings  # for this test to work

    approximants = "SEOBNRv5HM", "SEOBNRv5PHM"

    for approximant in approximants:
        wfm_gen = GenerateWaveform(
            basic_settings | {"approximant": approximant, "f_ref": 21, "f22_start": 20}
        )

        assert wfm_gen.parameters["f_ref"] == 21

    # takes value of f22_start if not specified
    for approximant in approximants:
        wfm_gen = GenerateWaveform(
            basic_settings | {"approximant": approximant, "f22_start": 21}
        )

        assert wfm_gen.parameters["f_ref"] == 21

    # inherit default value
    for approximant in approximants:
        new_settings = basic_settings | {"approximant": approximant}
        new_settings.pop("f22_start")
        wfm_gen = GenerateWaveform(new_settings)

        assert wfm_gen.parameters["f_ref"] == wfm_gen.parameters["f22_start"]

    # sanity check
    value_error = r"f_ref has to be positive!"
    with pytest.raises(
        ValueError,
        match=value_error,
    ):
        _ = GenerateWaveform(basic_settings | {"f_ref": -1})


def test_generate_modes_opt_settings_all_models(basic_settings):
    """Checks error reporting of incorrect parameters to generate_modes_opt"""

    with pytest.raises(
        ValueError,
        match="mass-ratio has to be positive and with convention q>=1",
    ):
        _ = generate_modes_opt(
            q=0.9,
            chi1=0,
            chi2=0,
            omega_start=0,
        )

    with pytest.raises(
        ValueError,
        match="omega_start has to be positive",
    ):
        _ = generate_modes_opt(
            q=1.1,
            chi1=0,
            chi2=0,
            omega_start=-0.00001,
        )


def test_generate_modes_opt_settings_hm(basic_settings):
    """Checks error reporting of incorrect parameters to generate_modes_opt for HM"""

    # pytest-mocker is hard to use with context managers
    with mock.patch.object(
        pyseobnr.generate_waveform.SEOBNRv5HM.SEOBNRv5HM_opt,
        "__call__",
        autospec=True,
    ) as p_model_call:
        with pytest.raises(
            ValueError,
            match="Dimensionless spin magnitudes cannot be greater than 1!",
        ):
            _ = generate_modes_opt(
                q=1.1,
                chi1=1.1,
                chi2=0,
                omega_start=0.00001,
                approximant="SEOBNRv5HM",
            )
            p_model_call.assert_not_called()

    with mock.patch.object(
        pyseobnr.generate_waveform.SEOBNRv5HM.SEOBNRv5HM_opt,
        "__call__",
        autospec=True,
    ) as p_model_call:

        def compute_dynamics(self):
            self.t = "something"
            self.waveform_modes = "something else"

        p_model_call.side_effect = compute_dynamics

        _ = generate_modes_opt(
            q=1.1,
            chi1=1,
            chi2=0,
            omega_start=0.00001,
            approximant="SEOBNRv5HM",
        )
        p_model_call.assert_called_once()

    with mock.patch.object(
        pyseobnr.generate_waveform.SEOBNRv5HM.SEOBNRv5HM_opt,
        "__call__",
        autospec=True,
    ) as p_model_call:

        def compute_dynamics(self):
            self.t = "something"
            self.waveform_modes = "something else"

        p_model_call.side_effect = compute_dynamics

        _ = generate_modes_opt(
            q=1.1,
            chi1=1,
            chi2=0,
            omega_start=0.00001,
            approximant="SEOBNRv5HM",
        )
        p_model_call.assert_called_once()


def test_generate_modes_opt_precessing_chi_array_float_int():
    """models should accept chi arrays with only z component, ints and floats"""
    q = 41.83615272380585
    omega0 = 0.02

    for chi_2 in (0, 0.3):
        chi_1 = np.array([0.0, 0.0, 0.98917404])

        approx: SupportedApproximants
        for approx in get_args(SupportedApproximants):
            _ = generate_modes_opt(
                q,
                chi_1,
                chi_2,
                omega0,
                approximant=approx,
                debug=False,
                settings={
                    "beta_approx": None,
                    "M": 154.2059835575123,
                    "dt": 6.103515625e-05,
                },
            )

    for chi_1 in (0, 0.3):
        chi_2 = np.array([0.0, 0.0, 0.98917404])

        approx: SupportedApproximants
        for approx in get_args(SupportedApproximants):
            _ = generate_modes_opt(
                q,
                chi_1,
                chi_2,
                omega0,
                approximant=approx,
                debug=False,
                settings={
                    "beta_approx": None,
                    "M": 154.2059835575123,
                    "dt": 6.103515625e-05,
                },
            )
