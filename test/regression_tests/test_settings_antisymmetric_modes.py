from __future__ import annotations

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


@pytest.fixture
def basic_settings():
    m1 = 50.0
    m2 = 30.0
    params_dict = {
        "q": m1 / m2,
        "chi1": np.array([0, 0, 0.1]),
        "chi2": np.array([0, 0, 0.8]),
        "deltaT": 1 / 2048.0,
        "deltaF": 0.125,
        "f22_start": 0.0157 / ((m1 + m2) * np.pi * lal.MTSUN_SI),
        "phi_ref": 0.0,
        "distance": 1.0,
        "inclination": np.pi / 3.0,
        "f_max": 1024.0,
        "approximant": "SEOBNRv5PHM",
        "enable_antisymmetric_modes": True,
        "postadiabatic": False,
    }
    return params_dict


def test_generate_modes_opt_antisymmetric_works_only_with_PHM(basic_settings):
    """Checks that generate_modes_opt rejects incorrect antisymmetric setting wrt. approximant"""

    for approximant in get_args(SupportedApproximants):
        if approximant == "SEOBNRv5PHM":
            continue

        current_params = basic_settings | {"approximant": approximant}

        with pytest.raises(
            ValueError,
            match=f"Antisymmetric modes not available for approximant {approximant}.",
        ):
            _ = generate_modes_opt(
                q=current_params["q"],
                chi1=current_params["chi1"],
                chi2=current_params["chi2"],
                omega_start=20,
                approximant=approximant,
                settings=current_params,
            )


def test_generate_waveform_antisymmetric_works_only_with_PHM():
    """Checks that GenerateWaveform rejects incorrect antisymmetric setting wrt. approximant"""

    for approximant in get_args(SupportedApproximants):
        if approximant == "SEOBNRv5PHM":
            continue

        m1, m2 = 20, 50
        params_generate_waveform = {
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
            "approximant": approximant,
            "postadiabatic": False,
            "enable_antisymmetric_modes": True,
        }
        with pytest.raises(
            ValueError,
            match="Antisymmetric modes not available for approximant "
            f"{params_generate_waveform['approximant']}.",
        ):
            _ = GenerateWaveform(params_generate_waveform)


@pytest.fixture
def prepare_mock_class_and_call(basic_settings):
    with mock.patch.object(
        pyseobnr.generate_waveform.SEOBNRv5HM.SEOBNRv5PHM_opt,
        "__call__",
        autospec=True,
    ) as p_model_call:

        class MyException(Exception):
            pass

        instance_model: dict = {}

        def _fake_call(self, *args, **kwargs):
            nonlocal instance_model
            instance_model["model"] = self
            raise MyException

        p_model_call.side_effect = _fake_call

        yield basic_settings, p_model_call, instance_model, MyException


def test_generate_modes_opt_without_explicit_antisymmetric_modes_defaults_to_22(
    prepare_mock_class_and_call,
):
    """If `antisymmetric_modes` is not provided, defaults to the 22 mode"""

    (
        current_params,
        p_model_call,
        instance_model,
        MyException,
    ) = prepare_mock_class_and_call

    with pytest.raises(MyException):
        _ = generate_modes_opt(
            q=current_params["q"],
            chi1=current_params["chi1"],
            chi2=current_params["chi2"],
            omega_start=20,
            approximant=current_params["approximant"],
            settings=current_params,
        )

    p_model_call.assert_called_once()
    assert "model" in instance_model
    assert "antisymmetric_modes" in instance_model["model"].settings
    assert instance_model["model"].settings["antisymmetric_modes"] == [(2, 2)]


def test_generate_modes_opt_explicit_antisymmetric_modes_correctness(
    prepare_mock_class_and_call,
):
    """Checks the parameter `antisymmetric_modes`"""

    (
        current_params,
        p_model_call,
        instance_model,
        MyException,
    ) = prepare_mock_class_and_call

    with pytest.raises(MyException):
        _ = generate_modes_opt(
            q=current_params["q"],
            chi1=current_params["chi1"],
            chi2=current_params["chi2"],
            omega_start=20,
            approximant=current_params["approximant"],
            settings=current_params | {"antisymmetric_modes": [(2, 2), (3, 3)]},
        )

    p_model_call.assert_called_once()
    assert "model" in instance_model
    assert "antisymmetric_modes" in instance_model["model"].settings
    assert instance_model["model"].settings["antisymmetric_modes"] == [(2, 2), (3, 3)]


def test_generate_modes_opt_incorrect_antisymmetric_modes_yields_error(
    prepare_mock_class_and_call,
):
    """If `antisymmetric_modes` has non supported modes, raise an error"""

    (
        current_params,
        p_model_call,
        instance_model,
        MyException,
    ) = prepare_mock_class_and_call

    with pytest.raises(
        RuntimeError, match="Redundant modes in 'antisymmetric_modes' settings"
    ):
        # Redundant modes
        _ = generate_modes_opt(
            q=current_params["q"],
            chi1=current_params["chi1"],
            chi2=current_params["chi2"],
            omega_start=20,
            approximant=current_params["approximant"],
            settings=current_params | {"antisymmetric_modes": [(2, 2), (2, 2)]},
        )

    p_model_call.assert_not_called()
    assert "model" not in instance_model

    instance_model.clear()
    p_model_call.reset_mock()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "Incorrect modes in 'antisymmetric_modes' settings: [(3, 2)] "
            "is not in the set of valid modes [(2, 2), (3, 3), (4, 4)]"
        ),
    ):
        # Redundant modes
        _ = generate_modes_opt(
            q=current_params["q"],
            chi1=current_params["chi1"],
            chi2=current_params["chi2"],
            omega_start=20,
            approximant=current_params["approximant"],
            settings=current_params | {"antisymmetric_modes": [(2, 2), (3, 2)]},
        )

    p_model_call.assert_not_called()
    assert "model" not in instance_model


def test_generate_modes_opt_fits_version(
    prepare_mock_class_and_call,
):
    """Checks parameter `antisymmetric_fits_version` when provided and unsupported"""

    (
        current_params,
        p_model_call,
        instance_model,
        MyException,
    ) = prepare_mock_class_and_call

    # ivs_mrd set but incorrect type
    current_params["ivs_mrd"] = tuple()
    with pytest.raises(ValueError, match="Incorrect 'ivs_mrd' parameter provided"):
        _ = generate_modes_opt(
            q=current_params["q"],
            chi1=current_params["chi1"],
            chi2=current_params["chi2"],
            omega_start=20,
            approximant=current_params["approximant"],
            settings=current_params,
        )
    p_model_call.assert_not_called()

    current_params["ivs_mrd"] = {"a": 0, "b": None}
    with pytest.raises(ValueError, match="Incorrect 'ivs_mrd' parameter provided"):
        _ = generate_modes_opt(
            q=current_params["q"],
            chi1=current_params["chi1"],
            chi2=current_params["chi2"],
            omega_start=20,
            approximant=current_params["approximant"],
            settings=current_params,
        )
    p_model_call.assert_not_called()

    # ivs_mrd set and not None: overrides
    current_params["ivs_mrd"] = {"ivs_asym": None, "mrd_ivs": None}
    with pytest.raises(MyException):
        _ = generate_modes_opt(
            q=current_params["q"],
            chi1=current_params["chi1"],
            chi2=current_params["chi2"],
            omega_start=20,
            approximant=current_params["approximant"],
            settings=current_params,
        )

    p_model_call.assert_called_once()


def test_generate_modes_opt_fits_overrides(
    prepare_mock_class_and_call,
):
    """Checks parameter `ivs_mrd` when provided"""

    (
        current_params,
        p_model_call,
        instance_model,
        MyException,
    ) = prepare_mock_class_and_call

    current_params["antisymmetric_fits_version"] = "test-version"

    # type is checked
    for incorrect_ivs_mrd in 43, [], ():
        current_params["ivs_mrd"] = incorrect_ivs_mrd

        with pytest.raises(ValueError, match="Incorrect 'ivs_mrd' parameter provided"):
            _ = generate_modes_opt(
                q=current_params["q"],
                chi1=current_params["chi1"],
                chi2=current_params["chi2"],
                omega_start=20,
                approximant=current_params["approximant"],
                settings=current_params,
            )

        p_model_call.assert_not_called()

    # dict content is checked

    for incorrect_ivs_mrd_dicts in {"ivs_asym": None, "something_else": None}, {
        "ivs_asym": None,
        "mrd_ivs": None,
        "should_strictly_be_the_previous_two_keys": None,
    }:
        current_params["ivs_mrd"] = incorrect_ivs_mrd_dicts
        with pytest.raises(ValueError, match="Incorrect 'ivs_mrd' parameter provided"):
            _ = generate_modes_opt(
                q=current_params["q"],
                chi1=current_params["chi1"],
                chi2=current_params["chi2"],
                omega_start=20,
                approximant=current_params["approximant"],
                settings=current_params,
            )

    p_model_call.assert_not_called()

    # ivs_mrd set and not None: overrides
    p_model_call.reset_mock()
    current_params["ivs_mrd"] = {"ivs_asym": None, "mrd_ivs": None}
    with pytest.raises(MyException):
        _ = generate_modes_opt(
            q=current_params["q"],
            chi1=current_params["chi1"],
            chi2=current_params["chi2"],
            omega_start=20,
            approximant=current_params["approximant"],
            settings=current_params,
        )

    p_model_call.assert_called_once()


def test_enabling_gwsignal_environment_should_not_raises_any_warnings(recwarn):
    """Checks parameter `gwsignal_environment` does not raise when combined w. antisymmetric"""

    m1, m2 = 20, 50
    params_generate_waveform = {
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
        "approximant": "SEOBNRv5PHM",
        "postadiabatic": False,
        "enable_antisymmetric_modes": True,
        "gwsignal_environment": True,
    }

    _ = GenerateWaveform(params_generate_waveform | {"gwsignal_environment": False})

    assert len(recwarn) == 0

    _ = GenerateWaveform(
        {
            k: v
            for k, v in params_generate_waveform.items()
            if k != "gwsignal_environment"
        }
    )

    assert len(recwarn) == 0

    _ = GenerateWaveform(params_generate_waveform)

    assert len(recwarn) == 0
