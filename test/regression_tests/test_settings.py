import decimal
import fractions
import re
from contextlib import ExitStack
from typing import Final, cast, get_args
from unittest import mock

import lal
import numpy as np

import pytest

import pyseobnr
from pyseobnr.eob.hamiltonian.Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C import (
    Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C as Ham_aligned_opt,
)
from pyseobnr.eob.hamiltonian.Ham_AvgS2precess_simple_cython_PA_AD import (
    Ham_AvgS2precess_simple_cython_PA_AD as Ham_prec_pa_cy,
)
from pyseobnr.eob.waveform.waveform import SEOBNRv5RRForce
from pyseobnr.eob.waveform.waveform_ecc import SEOBNRv5RRForceEcc
from pyseobnr.generate_waveform import (
    GenerateWaveform,
    SupportedApproximants,
    generate_modes_opt,
)
from pyseobnr.models.common import VALID_MODES, VALID_MODES_ECC
from pyseobnr.models.SEOBNRv5EHM import SEOBNRv5EHM_opt
from pyseobnr.models.SEOBNRv5HM import SEOBNRv5HM_opt, SEOBNRv5PHM_opt


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


def test_generate_waveform_mode_arrays_settings(basic_settings):
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

    for enable_anti_symmetric in (None, False, True):
        with pytest.raises(
            ValueError,
            match=value_error,
        ):
            additional = {"approximant": "SEOBNRv5PHM"} | (
                {}
                if enable_anti_symmetric is None
                else {"enable_antisymmetric_modes": enable_anti_symmetric}
            )
            _ = GenerateWaveform(
                basic_settings | additional | {"mode_array": [], "ModeArray": []}
            )

    calls_to_check = (
        "generate_td_modes",
        "generate_td_polarizations",
        "generate_fd_polarizations",
    )

    # works with list and tuples
    for enable_anti_symmetric in (None, False, True):

        for approximant in get_args(SupportedApproximants):

            additional = (
                {}
                if enable_anti_symmetric is None
                else {
                    # only available for "SEOBNRv5PHM"
                    "enable_antisymmetric_modes": enable_anti_symmetric
                    and approximant == "SEOBNRv5PHM"
                }
            )

            wfm_gen = GenerateWaveform(
                basic_settings
                | additional
                | {"approximant": approximant, "mode_array": [(2, 2), (3, 3)]}
            )

            for func in calls_to_check:
                _ = getattr(wfm_gen, func)()

            wfm_gen = GenerateWaveform(
                basic_settings
                | additional
                | {"approximant": approximant, "mode_array": ((2, 2), (3, 3))}
            )

            for func in calls_to_check:
                _ = getattr(wfm_gen, func)()

            # incorrect mode array yields an error
            wfm_gen = GenerateWaveform(
                basic_settings
                | additional
                | {"approximant": approximant, "mode_array": [2, 2, 3, 3]}
            )
            for func in calls_to_check:
                with pytest.raises(TypeError):
                    _ = getattr(wfm_gen, func)()


def test_default_settings_HM():
    RR_f = SEOBNRv5RRForce()
    cls = SEOBNRv5HM_opt(
        q=1,
        chi_1=0,
        chi_2=0,
        omega0=0.15,
        H=Ham_aligned_opt,
        RR=RR_f,
    )

    # old value 2.4627455127717882e-05 computed with old value of MTSUN_SI in LAL
    #   previous: MTSUN_SI = 4.925491025543575903411922162094833998e-6
    #   current : MTSUN_SI = 4.925490947641266978197229498498379006e-6
    assert cls.dt == 2.4627454738206332e-05  # 2.4627455127717882e-05
    assert cls.return_modes == [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]


def test_default_settings_PHM():
    RR_f = SEOBNRv5RRForce()
    cls = SEOBNRv5PHM_opt(
        q=1,
        chi1_x=0,
        chi1_y=0,
        chi1_z=0.1,
        chi2_x=0,
        chi2_y=0,
        chi2_z=0.1,
        omega_start=0.15,
        H=Ham_prec_pa_cy,
        RR=RR_f,
    )

    # old value 2.4627455127717882e-05 computed with old value of MTSUN_SI in LAL
    #   previous: MTSUN_SI = 4.925491025543575903411922162094833998e-6
    #   current : MTSUN_SI = 4.925490947641266978197229498498379006e-6
    assert cls.dt == 2.4627454738206332e-05  # 2.4627455127717882e-05
    assert cls.return_modes == [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]


def test_default_settings_EHM():
    RR_f = SEOBNRv5RRForceEcc("Ecc")
    cls = SEOBNRv5EHM_opt(
        q=1,
        chi_1=0.1,
        chi_2=0.1,
        omega_start=0.15,
        eccentricity=0.1,
        rel_anomaly=0.1,
        H=Ham_aligned_opt,
        RR=RR_f,
    )

    # old value 2.4627455127717882e-05 computed with old value of MTSUN_SI in LAL
    #   previous: MTSUN_SI = 4.925491025543575903411922162094833998e-6
    #   current : MTSUN_SI = 4.925490947641266978197229498498379006e-6
    assert cls.dt == 2.4627454738206332e-05  # 2.4627455127717882e-05
    assert cls.return_modes == [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]


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


def test_mode_arrays_settings_with_lmax_EHM(basic_settings):
    """Checks lmax handling in the EHM model"""

    approximant: SupportedApproximants = "SEOBNRv5EHM"

    # this should work as lmax overrides the modes
    with mock.patch.object(
        pyseobnr.generate_waveform.SEOBNRv5EHM.SEOBNRv5EHM_opt,
        "__call__",
        autospec=True,
    ) as p_model_call:

        class LocalException(Exception):
            pass

        class FakeCall:
            def __init__(self):
                self.model = None

            def __call__(self, other_self, *args, **kwargs):

                self.model = other_self
                raise LocalException

        instance_fake_call = FakeCall()
        p_model_call.side_effect = instance_fake_call.__call__

        with pytest.raises(LocalException):
            generate_modes_opt(
                q=1.1,
                chi1=0,
                chi2=0,
                omega_start=0.1,
                debug=True,
                eccentricity=0.1,
                approximant=approximant,
                settings={"lmax": 2},
            )

        p_model_call.assert_called_once()
        assert instance_fake_call.model is not None
        assert set(instance_fake_call.model.return_modes) == (
            {(ell, emm) for ell, emm in VALID_MODES_ECC if ell <= 2}
        )

        p_model_call.reset_mock()
        instance_fake_call.model = None
        with pytest.raises(LocalException):
            # this will narrow the modes array to l <= 3
            generate_modes_opt(
                q=1.1,
                chi1=0,
                chi2=0,
                omega_start=0.1,
                debug=True,
                approximant=approximant,
                settings={"lmax": 3},
            )

        p_model_call.assert_called_once()
        assert instance_fake_call.model is not None
        assert set(instance_fake_call.model.return_modes) == {
            (ell, emm) for ell, emm in VALID_MODES_ECC if ell <= 3
        }

        p_model_call.reset_mock()
        instance_fake_call.model = None
        with pytest.raises(LocalException):

            # this will extend the default modes array to l = 5 modes
            generate_modes_opt(
                q=1.1,
                chi1=0,
                chi2=0,
                omega_start=0.1,
                debug=True,
                approximant=approximant,
                settings={"lmax": 4},
            )

        p_model_call.assert_called_once()
        assert instance_fake_call.model is not None
        assert set(instance_fake_call.model.return_modes) == {
            (ell, emm) for ell, emm in VALID_MODES_ECC if ell <= 4
        }

        p_model_call.reset_mock()
        instance_fake_call.model = None
        with pytest.raises(ValueError, match="Incorrect value for lmax=.*"):
            # this will extend the default modes array to l = 5 modes
            generate_modes_opt(
                q=1.1,
                chi1=0,
                chi2=0,
                omega_start=0.1,
                debug=True,
                approximant=approximant,
                settings={"lmax": 5},
            )

        p_model_call.assert_not_called()
        assert instance_fake_call.model is None

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
    for lmax_tentative in (18, -3, max(_[0] for _ in VALID_MODES_ECC) + 1):
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

    approximants_and_function_to_patch = (
        ("SEOBNRv5HM", "generate_modes_opt"),
        (
            "SEOBNRv5PHM",
            "generate_modes_opt",
        ),
        (
            "SEOBNRv5EHM",
            "generate_modes_opt",
        ),
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
    approximants_and_function_to_patch = (
        ("SEOBNRv5HM", "generate_modes_opt"),
        (
            "SEOBNRv5PHM",
            "generate_prec_hpc_opt",
        ),
        ("SEOBNRv5EHM", "generate_modes_opt"),
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
    approximants_and_function_to_patch = (
        ("SEOBNRv5HM", "generate_modes_opt"),
        (
            "SEOBNRv5PHM",
            "generate_prec_hpc_opt",
        ),
        ("SEOBNRv5EHM", "generate_modes_opt"),
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


def test_generate_waveform_f_ref_f_min_f_max_behaviour(basic_settings):
    """Checks the logic in setting f_ref (and f_min/f_start)"""

    assert "f_ref" not in basic_settings  # for this test to work

    approximants = "SEOBNRv5HM", "SEOBNRv5PHM"

    # When not set, f_ref should be set to f22_start
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
    for approximant in approximants + ("SEOBNRv5EHM",):
        new_settings = basic_settings | {"approximant": approximant}
        new_settings.pop("f22_start")
        wfm_gen = GenerateWaveform(new_settings)

        assert wfm_gen.parameters["f_ref"] == wfm_gen.parameters["f22_start"]

    # sanity checks validity ranges
    for approximant in approximants + ("SEOBNRv5EHM",):
        with pytest.raises(
            ValueError,
            match=r"f_ref has to be positive!",
        ):
            _ = GenerateWaveform(
                basic_settings | {"f_ref": -1, "approximant": approximant}
            )

        with pytest.raises(
            ValueError,
            match=r"f_ref has to be positive!",
        ):
            _ = GenerateWaveform(
                basic_settings | {"f_ref": 0, "approximant": approximant}
            )

        with pytest.raises(
            ValueError,
            match=r"f22_start has to be positive!",
        ):
            _ = GenerateWaveform(
                basic_settings | {"f22_start": 0, "approximant": approximant}
            )

        with pytest.raises(
            ValueError,
            match=r"f22_start has to be positive!",
        ):
            _ = GenerateWaveform(
                basic_settings | {"f22_start": -1e-12, "approximant": approximant}
            )

        # fmax >= f_min
        with pytest.raises(
            ValueError,
            match="'f_max' cannot be smaller than 'f22_start' or 'f_ref'!",
        ):
            _ = GenerateWaveform(
                basic_settings
                | {"f_max": 19, "f22_start": 20, "approximant": approximant}
            )

        # fmax >= f_ref
        with pytest.raises(
            ValueError,
            match="'f_max' cannot be smaller than 'f22_start' or 'f_ref'!",
        ):
            _ = GenerateWaveform(
                basic_settings | {"f_max": 19, "f_ref": 20, "approximant": approximant}
            )

    # f_ref not supported by EHM
    with pytest.raises(
        ValueError,
        match=r"The approximant 'SEOBNRv5EHM' does not support the choice for a "
        r"reference frequency. Please, set 'f_ref' = 'f22_start'.",
    ):
        _ = GenerateWaveform(
            basic_settings
            | {"f_ref": 19, "f22_start": 20, "approximant": "SEOBNRv5EHM"}
        )


def test_generate_modes_opt_settings_can_be_none():
    """Checks that generate_modes_opt allows for None settings"""

    class MyException(Exception):
        pass

    with mock.patch.object(
        pyseobnr.generate_waveform.SEOBNRv5HM.SEOBNRv5HM_opt,
        "__call__",
        autospec=True,
    ) as p_model_call:

        p_model_call.side_effect = MyException

        with pytest.raises(
            MyException,
        ):
            _ = generate_modes_opt(
                q=1.1, chi1=0, chi2=0, omega_start=20, approximant="SEOBNRv5HM"
            )

    with mock.patch.object(
        pyseobnr.generate_waveform.SEOBNRv5HM.SEOBNRv5PHM_opt,
        "__call__",
        autospec=True,
    ) as p_model_call:

        p_model_call.side_effect = MyException

        with pytest.raises(
            MyException,
        ):
            _ = generate_modes_opt(
                q=1.1, chi1=0, chi2=0, omega_start=20, approximant="SEOBNRv5PHM"
            )


def test_generate_modes_opt_settings_all_models():
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


def test_generate_waveform_spins_cannot_be_booleans():
    """Checks that passing a boolean for a spin value returns an error"""

    params: Final = {
        "mass1": 40.0,
        "mass2": 40 / 1.1,
        "spin1z": True,
        "spin2z": 0.3,
        "f22_start": 0.015 / (np.pi * (40.0 + 40 / 1.1) * lal.MTSUN_SI),
        "approximant": "SEOBNRv5HM",
    }

    with pytest.raises(
        ValueError,
        match="Boolean spin values unsupported",
    ):
        GenerateWaveform(params)

    with pytest.raises(
        ValueError,
        match="Boolean spin values unsupported",
    ):
        _ = generate_modes_opt(
            q=1.1,
            chi1=0.1,
            chi2=[0, False, 0.9],
            omega_start=0.015,
        )

    params2: Final = {
        "mass1": 40.0,
        "mass2": 40 / 1.1,
        "spin1z": 0.1,
        "spin2x": 0.0,
        "spin2y": True,
        "spin2z": 0.9,
        "f22_start": 0.015 / (np.pi * (40.0 + 40 / 1.1) * lal.MTSUN_SI),
        "approximant": "SEOBNRv5HM",
    }

    with pytest.raises(
        ValueError,
        match="Boolean spin values unsupported",
    ):
        GenerateWaveform(params2)


def test_generate_modes_opt_spins_cannot_be_booleans():
    """Checks that passing a boolean for a spin value returns an error"""

    with pytest.raises(
        ValueError,
        match="Boolean spin values unsupported",
    ):
        _ = generate_modes_opt(
            q=1.1,
            chi1=True,
            chi2=0.3,
            omega_start=0.015,
        )


def test_generate_modes_opt_params_cannot_be_booleans():
    """Extends the boolean tests on all parameters"""

    with pytest.raises(
        ValueError,
        match="Boolean spin values unsupported",
    ):
        _ = generate_modes_opt(
            q=1.1,
            chi1=True,
            chi2=0.3,
            omega_start=0.015,
        )


def test_generate_waveform_params_cannot_be_booleans():
    """Extends the boolean tests on all parameters"""

    f_min = 0.015 / (np.pi * (40.0 + 40 / 1.1) * lal.MTSUN_SI)
    params: Final = {
        "mass1": 40,
        "mass2": 40 / 1.1,
        "spin1z": 0.1,
        "spin2z": 0.3,
        "f_min": f_min,
        "deltaT": 1 / (8 * 2**10),
        "f_ref": f_min,
        "f22_start": f_min,
        "f_max": 4 * (2**10),
        "deltaF": 0,
        "phi_ref": 0,
        "distance": 100,
        "eccentricity": 0.1,
        "rel_anomaly": 0,
        "inclination": np.pi / 3,
        "conditioning": 1,
        "postadiabatic": True,
    }

    for param_to_test in ["mass1", "mass2"]:

        for wrong_value in [
            True,
            False,
            (params["mass1"],),
            [params["mass1"]],
            f"{params['mass1']}",
            {"mass1": params["mass1"]},
            decimal.Decimal("3.1415926535"),  # decimal is not a "Real" number
        ]:
            new_params = params | {
                param_to_test: wrong_value,
                "approximant": "SEOBNRv5EHM",
            }
            new_params["postadiabatic"] = False

            error_msg = "Only 'float' and 'int' values of masses are supported\\."

            with pytest.raises(
                ValueError,
                match=error_msg,
            ):
                GenerateWaveform(new_params)

        # we also check the values that should be accepted
        for correct_value in [
            float(10),
            np.float64(10),
            10,
            np.int64(10),
            fractions.Fraction(20, 23),
        ]:
            new_params = params | {
                param_to_test: correct_value,
                "approximant": "SEOBNRv5EHM",
            }
            new_params["postadiabatic"] = False
            GenerateWaveform(new_params)

    for param_to_test in params.keys() - {
        "mass1",
        "mass2",
        "conditioning",
        "postadiabatic",
    }:

        # catches errors
        for wrong_value in [
            True,
            False,
            decimal.Decimal("0.1415926535"),
        ]:
            new_params = params | {
                param_to_test: wrong_value,
                "approximant": "SEOBNRv5EHM",
            }
            new_params["postadiabatic"] = False

            error_msg = f"{param_to_test} has to be a real number.*"
            if "spin" in param_to_test:
                error_msg = "Boolean spin values unsupported"

            with pytest.raises(
                ValueError,
                match=error_msg,
            ):
                GenerateWaveform(new_params)

        # does not yield any error
        for correct_value in [
            np.int32(10),
            np.float32(10),
            10,
            fractions.Fraction(200, 23),
        ]:

            new_params = params | {
                param_to_test: correct_value,
                "approximant": "SEOBNRv5EHM",
            }
            if param_to_test == "f_max":
                new_params[param_to_test] += params["f_min"]
            elif param_to_test in ["f_ref", "f22_start"]:
                new_params["f_ref"] = new_params["f22_start"] = new_params[
                    param_to_test
                ]
            elif "spin" in param_to_test or param_to_test in ["eccentricity"]:
                new_params[param_to_test] /= 11

            new_params["postadiabatic"] = False

            GenerateWaveform(new_params)

    # only integer parameters
    for param_to_test in params.keys() & {"conditioning"}:

        for wrong_value in [
            True,
            False,
            0.3,
            np.float32(10),
            fractions.Fraction(20, 23),
            decimal.Decimal("3.1415926535"),
        ]:
            new_params = params | {
                param_to_test: wrong_value,
                "approximant": "SEOBNRv5EHM",
            }

            error_msg = f"{param_to_test} has to be an integer number.*"

            with pytest.raises(
                ValueError,
                match=error_msg,
            ):
                GenerateWaveform(new_params)

        for correct_value in [np.int32(1), 1]:
            new_params = params | {
                param_to_test: correct_value,
                "approximant": "SEOBNRv5EHM",
            }
            new_params["postadiabatic"] = False

            GenerateWaveform(new_params)

    # only boolean parameters
    for param_to_test in params.keys() & {"postadiabatic"}:

        for wrong_value in [0, 3, 0.3]:
            new_params = params | {
                param_to_test: wrong_value,
                "approximant": "SEOBNRv5EHM",
            }

            error_msg = f"{param_to_test} has to be a boolean.*"

            with pytest.raises(
                ValueError,
                match=error_msg,
            ):
                GenerateWaveform(new_params)

        for correct_value in [True, np.True_]:
            new_params = params | {
                param_to_test: correct_value,
                "approximant": "SEOBNRv5EHM",
            }

            new_params["postadiabatic"] = False
            GenerateWaveform(new_params)


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
            # match="chi1 and chi2 have to respect Kerr limit.*",
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

        p_model_call.reset_mock()
        _ = generate_modes_opt(
            q=1.1,
            chi1=0,
            chi2=1,
            omega_start=0.00001,
            approximant="SEOBNRv5HM",
        )
        p_model_call.assert_called_once()


def test_generate_modes_opt_settings_phm(basic_settings):
    """Checks error reporting of incorrect parameters to generate_modes_opt for PHM"""
    theta = np.pi / 4
    phi = 11 * np.pi / 12

    r = 1 + 1e-3

    incorrect_spin = np.array(
        [
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta),
        ]
    )

    rm1 = 0.9
    correct_spin = np.array(
        [
            rm1 * np.sin(theta) * np.cos(phi),
            rm1 * np.sin(theta) * np.sin(phi),
            rm1 * np.cos(theta),
        ]
    )

    # pytest-mocker is hard to use with context managers
    with mock.patch.object(
        pyseobnr.generate_waveform.SEOBNRv5HM.SEOBNRv5PHM_opt,
        "__call__",
        autospec=True,
    ) as p_model_call:

        class LocalException(Exception):
            pass

        p_model_call.side_effect = LocalException

        with pytest.raises(
            ValueError,
            match="Dimensionless spin magnitudes cannot be greater than 1!",
        ):
            _ = generate_modes_opt(
                q=1.1,
                chi1=incorrect_spin,
                chi2=0,
                omega_start=0.00001,
                approximant="SEOBNRv5PHM",
            )
            p_model_call.assert_not_called()

        with pytest.raises(
            ValueError,
            match="Dimensionless spin magnitudes cannot be greater than 1!",
        ):
            _ = generate_modes_opt(
                q=1.1,
                chi1=0,
                chi2=incorrect_spin,
                omega_start=0.00001,
                approximant="SEOBNRv5PHM",
            )
            p_model_call.assert_not_called()

        with pytest.raises(LocalException):
            _ = generate_modes_opt(
                q=1.1,
                chi1=correct_spin,
                chi2=0,
                omega_start=0.00001,
                approximant="SEOBNRv5PHM",
            )
        p_model_call.assert_called_once()

        p_model_call.reset_mock()
        with pytest.raises(LocalException):
            _ = generate_modes_opt(
                q=1.1,
                chi1=0,
                chi2=correct_spin,
                omega_start=0.00001,
                approximant="SEOBNRv5PHM",
            )
        p_model_call.assert_called_once()

        p_model_call.reset_mock()
        with pytest.raises(LocalException):
            _ = generate_modes_opt(
                q=1.1,
                chi1=1,
                chi2=0,
                omega_start=0.00001,
                approximant="SEOBNRv5PHM",
            )
        p_model_call.assert_called_once()

        p_model_call.reset_mock()
        with pytest.raises(LocalException):
            _ = generate_modes_opt(
                q=1.1,
                chi1=0,
                chi2=1,
                omega_start=0.00001,
                approximant="SEOBNRv5PHM",
            )
        p_model_call.assert_called_once()


def test_generate_modes_opt_precessing_chi_array_float_int():
    """models should accept chi arrays with only z component, ints and floats"""
    q = 41.83615272380585
    omega0 = 0.02

    class MyException(Exception):
        pass

    class_map: Final = {
        "SEOBNRv5HM": pyseobnr.generate_waveform.SEOBNRv5HM.SEOBNRv5HM_opt,
        "SEOBNRv5PHM": pyseobnr.generate_waveform.SEOBNRv5HM.SEOBNRv5PHM_opt,
        "SEOBNRv5EHM": pyseobnr.generate_waveform.SEOBNRv5EHM.SEOBNRv5EHM_opt,
    }

    for chi_2 in (0, 0.3):
        chi_1 = np.array([0.0, 0.0, 0.98917404])

        approx: SupportedApproximants
        for approx in get_args(SupportedApproximants):

            # we prevent the execution of the waveform generation by mocking
            with mock.patch.object(
                class_map[approx],
                "__call__",
                autospec=True,
            ) as p_model_call:
                p_model_call.side_effect = MyException

                with pytest.raises(MyException):
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

                p_model_call.assert_called_once()

    for chi_1 in (0, 0.3):
        chi_2 = np.array([0.0, 0.0, 0.98917404])

        approx: SupportedApproximants
        for approx in get_args(SupportedApproximants):

            # we prevent the execution of the waveform generation by mocking
            with mock.patch.object(
                class_map[approx],
                "__call__",
                autospec=True,
            ) as p_model_call:
                p_model_call.side_effect = MyException

                p_model_call.side_effect = MyException

                with pytest.raises(MyException):
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

                p_model_call.assert_called_once()


def test_generate_modes_opt_settings_ehm(basic_settings):
    """Checks error reporting of incorrect parameters to generate_modes_opt for EHM"""

    with mock.patch.object(
        pyseobnr.generate_waveform.SEOBNRv5EHM.SEOBNRv5EHM_opt,
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
                approximant="SEOBNRv5EHM",
            )
            p_model_call.assert_not_called()

    with mock.patch.object(
        pyseobnr.generate_waveform.SEOBNRv5EHM.SEOBNRv5EHM_opt,
        "__call__",
        autospec=True,
    ) as p_model_call:

        approximant: SupportedApproximants
        for approximant in ["SEOBNRv5HM", "SEOBNRv5PHM"]:
            with pytest.raises(
                ValueError,
                match="The selected approximant does not support input values for the "
                "eccentricity and relativistic anomaly.*",
            ):
                _ = generate_modes_opt(
                    q=1.1,
                    chi1=0,
                    chi2=0,
                    eccentricity=0.3,
                    omega_start=0.00001,
                    approximant=approximant,
                )
                p_model_call.assert_not_called()

            with pytest.raises(
                ValueError,
                match="The selected approximant does not support input values for the "
                "eccentricity and relativistic anomaly.*",
            ):
                _ = generate_modes_opt(
                    q=1.1,
                    chi1=0,
                    chi2=0,
                    eccentricity=0,
                    rel_anomaly=0.1,
                    omega_start=0.00001,
                    approximant=approximant,
                )
                p_model_call.assert_not_called()

        for e in -0.0001, 1, 1.00000000000:
            with pytest.raises(
                ValueError,
                # this is a regex
                match=r"The value of eccentricity must be inside the interval \[0, 1\).",
            ):
                _ = generate_modes_opt(
                    q=1.1,
                    chi1=0,
                    chi2=0,
                    eccentricity=e,
                    rel_anomaly=0.1,
                    omega_start=0.00001,
                    approximant="SEOBNRv5EHM",
                )
                p_model_call.assert_not_called()

    with mock.patch.object(
        pyseobnr.generate_waveform.SEOBNRv5HM.SEOBNRv5HM_opt,
        "__call__",
        autospec=True,
    ) as p_model_call:

        def compute_dynamics_hm(self):
            self.t = "something"
            self.waveform_modes = "something else"

        p_model_call.side_effect = compute_dynamics_hm

        # we should be able to call other models if eccentricity and rel_anomaly are both 0
        _ = generate_modes_opt(
            q=1.1,
            chi1=1,
            chi2=0,
            eccentricity=0,
            rel_anomaly=0,
            omega_start=0.00001,
            approximant="SEOBNRv5HM",
        )
        p_model_call.assert_called_once()

    with mock.patch.object(
        pyseobnr.generate_waveform.SEOBNRv5EHM.SEOBNRv5EHM_opt,
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
            approximant="SEOBNRv5EHM",
        )
        p_model_call.assert_called_once()


def test_generate_waveform_ehm():
    """Checks error reporting of incorrect parameters to GenerateWaveform for EHM"""
    params: Final = {
        "mass1": 40.0,
        "mass2": 30.0,
        "spin1z": 0.0,
        "spin2z": 0.0,
        "f22_start": 20.0,
        "f_max": 2048.0,
        "f22_ref": 20.0,
        "phi_ref": 0.0,
        "distance": 1000.0,
        "inclination": 0.0,
        "deltaF": 1.0 / 8.0,
        "deltaT": 1.0 / 8192,
        "eccentricity": 0,
        "approximant": "SEOBNRv5EHM",
    }

    minimal_params_dict: Final = {
        "mass1": 50,
        "mass2": 30,
        "approximant": "SEOBNRv5EHM",
    }

    with mock.patch(
        "pyseobnr.generate_waveform.generate_modes_opt",
    ) as p_generate_modes_opt:

        class MyException(Exception):
            pass

        p_generate_modes_opt.side_effect = MyException

        gen = GenerateWaveform(params)
        with pytest.raises(MyException):
            gen.generate_td_modes()
        p_generate_modes_opt.assert_called_once()

        p_generate_modes_opt.reset_mock()
        gen = GenerateWaveform(params | {"postadiabatic": False})
        with pytest.raises(MyException):
            gen.generate_td_modes()
        p_generate_modes_opt.assert_called_once()

        p_generate_modes_opt.reset_mock()
        gen = GenerateWaveform(params | {"conditioning": 1})
        with pytest.raises(MyException):
            gen.generate_td_modes()
        p_generate_modes_opt.assert_called_once()

        p_generate_modes_opt.reset_mock()
        gen = GenerateWaveform(minimal_params_dict)
        with pytest.raises(MyException):
            gen.generate_td_modes()
        p_generate_modes_opt.assert_called_once()


@pytest.fixture
def get_exit_stack():
    exit_s = ExitStack()
    yield exit_s
    exit_s.close()


def test_post_adiabatic_settings(get_exit_stack):
    """Checks the settings associated to the post-adiabatic features"""

    class MyException(Exception):
        pass

    exit_s = get_exit_stack

    all_classes = [
        pyseobnr.generate_waveform.SEOBNRv5HM.SEOBNRv5PHM_opt,
        pyseobnr.generate_waveform.SEOBNRv5HM.SEOBNRv5HM_opt,
    ]
    all_mocks = []
    for current_class in all_classes:
        p_current_mock = exit_s.enter_context(
            mock.patch.object(
                current_class,
                "__call__",
                autospec=True,
            )
        )

        p_current_mock.side_effect = MyException
        all_mocks += [p_current_mock]

    approx: SupportedApproximants
    approximants_supporting_adiabatic: list[SupportedApproximants] = cast(
        list[SupportedApproximants],
        list(set(get_args(SupportedApproximants)) - {"SEOBNRv5EHM"}),
    )
    for approx in approximants_supporting_adiabatic:
        print(approx)
        with pytest.raises(MyException):
            _ = generate_modes_opt(
                q=1,
                chi1=0,
                chi2=0,
                omega_start=0.1,
                approximant=approx,
                settings={"postadiabatic_type": "numeric", "postadiabatic": True},
            )

        assert sum(_.call_count for _ in all_mocks) == 1

        for current_mock in all_mocks:
            current_mock.reset_mock()

    for approx in approximants_supporting_adiabatic:
        with pytest.raises(ValueError, match="Incorrect value for postadiabatic_type"):
            _ = generate_modes_opt(
                q=1,
                chi1=0,
                chi2=0,
                omega_start=0.1,
                approximant=approx,
                # root was the type used in a previous version of v5HM
                settings={"postadiabatic_type": "root", "postadiabatic": True},
            )

        assert sum(_.call_count for _ in all_mocks) == 0

        for current_mock in all_mocks:
            current_mock.reset_mock()
