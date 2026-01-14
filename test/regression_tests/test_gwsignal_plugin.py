import re
from unittest.mock import patch

import astropy.units as u
import lal
import lalsimulation.gwsignal.core.waveform as wfm
import numpy as np
from gwpy.timeseries import TimeSeries

import pytest


@pytest.fixture
def basic_settings():
    m1 = 50.0
    m2 = 30.0
    Mt = m1 + m2
    distance = 500.0
    inclination = np.pi / 3.0
    phiRef = 0.0
    s1x = s1y = s2x = s2y = 0.0
    s1z = 0.5
    s2z = 0.1
    f_max = 1024.0
    f_min = 0.0157 / (Mt * np.pi * lal.MTSUN_SI)
    dt = 1 / (2 * f_max)
    deltaF = 0.125
    params_dict = {
        "mass1": m1,
        "mass2": m2,
        "spin1x": s1x,
        "spin1y": s1y,
        "spin1z": s1z,
        "spin2x": s2x,
        "spin2y": s2y,
        "spin2z": s2z,
        "deltaT": dt,
        "deltaF": deltaF,
        "f_ref": 20,
        "f22_start": f_min,
        "phi_ref": phiRef,
        "distance": distance,
        "inclination": inclination,
        "f_max": f_max,
        "postadiabatic": False,
        "eccentricity": 0.1,
        "rel_anomaly": np.pi / 6,
    }

    return params_dict


def test_gwsignal_plugin_td_v5phm(basic_settings):
    from pyseobnr.plugins.gwsignal_plugin import SEOBNRv5HM, SEOBNRv5PHM

    arguments_dict = {
        "mass1": basic_settings["mass1"] * u.solMass,
        "mass2": basic_settings["mass2"] * u.solMass,
        "spin1x": basic_settings["spin1x"] * u.dimensionless_unscaled,
        "spin1y": basic_settings["spin1y"] * u.dimensionless_unscaled,
        "spin1z": basic_settings["spin1z"] * u.dimensionless_unscaled,
        "spin2x": basic_settings["spin2x"] * u.dimensionless_unscaled,
        "spin2y": basic_settings["spin2y"] * u.dimensionless_unscaled,
        "spin2z": basic_settings["spin2z"] * u.dimensionless_unscaled,
        "deltaF": basic_settings["deltaF"] * u.Hz,
        "deltaT": basic_settings["deltaT"] * u.s,
        "f22_start": basic_settings["f22_start"] * u.Hz,
        "f22_ref": basic_settings["f_ref"] * u.Hz,
        "f_max": basic_settings["f_max"] * u.Hz,
        "phi_ref": basic_settings["phi_ref"] * u.rad,
        "distance": basic_settings["distance"] * u.Mpc,
        "inclination": basic_settings["inclination"] * u.rad,
        "eccentricity": basic_settings["eccentricity"] * u.dimensionless_unscaled,
        "meanPerAno": basic_settings["rel_anomaly"] * u.rad,
        "condition": 0,
    }

    arguments_dict_aligned_spin = arguments_dict | {
        f"spin{k}{o}": 0.0 * u.dimensionless_unscaled
        for k in (1, 2)
        for o in ("x", "y")
    }

    generator_hm = SEOBNRv5HM()
    generator_phm = SEOBNRv5PHM()

    # wrong types. Curiously the types not used by the models are cast to the target units
    with pytest.raises(
        ValueError,
        match=re.escape("'Mpc' (length) and '' (dimensionless) are not convertible"),
    ):
        _ = wfm.GenerateTDModes(
            arguments_dict | {"eccentricity": 10 * u.Mpc}, generator_hm
        )
    with pytest.raises(
        ValueError,
        match=re.escape("'Hz' (frequency) and 'rad' (angle) are not convertible"),
    ):
        _ = wfm.GenerateTDModes(
            arguments_dict | {"meanPerAno": 20 * u.Hz}, generator_hm
        )
    with pytest.raises(
        ValueError,
        match=re.escape("'Hz' (frequency) and 'solMass' (mass) are not convertible"),
    ):
        _ = wfm.GenerateTDModes(arguments_dict | {"mass1": 20 * u.Hz}, generator_hm)

    # HM error capture
    with pytest.raises(
        ValueError,
        match=r"In-plane spin components must be zero for calling the non-precessing approximant\.",
    ):
        _ = wfm.GenerateTDModes(
            arguments_dict | {"spin1y": 0.1 * u.dimensionless_unscaled}, generator_hm
        )

    for generator, arg_dict in (generator_hm, arguments_dict_aligned_spin), (
        generator_phm,
        arguments_dict,
    ):
        with pytest.raises(
            ValueError,
            match=r"Use the approximant 'SEOBNRv5EHM' for a system with non-zero "
            r"eccentricity or relativistic anomaly\.",
        ):
            _ = wfm.GenerateTDModes(arg_dict, generator)

    arguments_dict_aligned_spin["eccentricity"] = arguments_dict["eccentricity"] = (
        0 * u.dimensionless_unscaled
    )
    arguments_dict_aligned_spin["meanPerAno"] = arguments_dict["meanPerAno"] = 0 * u.rad

    class MyException(Exception):
        pass

    for generator, arg_dict in (generator_hm, arguments_dict_aligned_spin), (
        generator_phm,
        arguments_dict,
    ):

        with patch(
            "pyseobnr.generate_waveform.GenerateWaveform.generate_td_modes",
            autospec=True,
        ) as p_generate_td_polarizations:
            p_generate_td_polarizations.return_value = np.arange(10000), {
                (2, 2): np.arange(10000) + 10j
            }

            hlm = wfm.GenerateTDModes(
                arg_dict,
                generator,
            )
            p_generate_td_polarizations.assert_called_once()

            assert hlm.keys() == {(2, 2)}
            assert (
                np.max(
                    np.abs(
                        hlm[(2, 2)]
                        - TimeSeries(
                            np.arange(10000) + 10j, times=np.arange(10000), name=(2, 2)
                        )
                    )
                )
                < 1e-10
            )

    # capturing parameters passed to the underlying GenerateWaveform
    for generator, arg_dict in (generator_hm, arguments_dict_aligned_spin), (
        generator_phm,
        arguments_dict,
    ):
        with patch(
            "pyseobnr.generate_waveform.GenerateWaveform.generate_td_polarizations",
            autospec=True,
        ) as p_generate_td_polarizations:

            inst = None

            def _side_effect(self, *args, **kwargs):
                nonlocal inst
                inst = self
                raise MyException()

            p_generate_td_polarizations.side_effect = _side_effect

            with pytest.raises(MyException):
                _ = wfm.GenerateTDWaveform(arg_dict, generator)

            assert inst.parameters["spin1x"] == float(arg_dict["spin1x"])
            assert inst.parameters["rel_anomaly"] == float(
                arg_dict["meanPerAno"] / u.rad
            )

    # capturing specific convention parameters
    for generator, arg_dict in (generator_hm, arguments_dict_aligned_spin), (
        generator_phm,
        arguments_dict,
    ):

        with patch(
            "pyseobnr.generate_waveform.GenerateWaveform.generate_td_polarizations",
            autospec=True,
        ) as p_generate_td_polarizations:

            for dict_convention in {
                "convention_coprecessing_phase22_set_to_0_at_reference_frequency": True,
                "convention_t0_set_to_0_at_coprecessing_amplitude22_peak": True,
            }, {
                "convention_coprecessing_phase22_set_to_0_at_reference_frequency": False,
                "convention_t0_set_to_0_at_coprecessing_amplitude22_peak": False,
            }:
                inst = None

                def _side_effect(self, *args, **kwargs):
                    nonlocal inst
                    inst = self
                    raise MyException()

                p_generate_td_polarizations.side_effect = _side_effect

                with pytest.raises(MyException):
                    _ = wfm.GenerateTDWaveform(arg_dict | dict_convention, generator)

                for k, v in dict_convention.items():
                    assert inst.parameters[k] == v


def test_gwsignal_plugin_td_v5ehm(basic_settings):
    from pyseobnr.plugins.gwsignal_plugin import SEOBNRv5EHM

    arguments_dict = {
        "mass1": basic_settings["mass1"] * u.solMass,
        "mass2": basic_settings["mass2"] * u.solMass,
        "spin1x": basic_settings["spin1x"] * u.dimensionless_unscaled,
        "spin1y": basic_settings["spin1y"] * u.dimensionless_unscaled,
        "spin1z": basic_settings["spin1z"] * u.dimensionless_unscaled,
        "spin2x": basic_settings["spin2x"] * u.dimensionless_unscaled,
        "spin2y": basic_settings["spin2y"] * u.dimensionless_unscaled,
        "spin2z": basic_settings["spin2z"] * u.dimensionless_unscaled,
        "deltaF": basic_settings["deltaF"] * u.Hz,
        "deltaT": basic_settings["deltaT"] * u.s,
        "f22_start": basic_settings["f22_start"] * u.Hz,
        "f22_ref": basic_settings["f_ref"] * u.Hz,
        "f_max": basic_settings["f_max"] * u.Hz,
        "phi_ref": basic_settings["phi_ref"] * u.rad,
        "distance": basic_settings["distance"] * u.Mpc,
        "inclination": basic_settings["inclination"] * u.rad,
        "eccentricity": basic_settings["eccentricity"] * u.dimensionless_unscaled,
        "meanPerAno": basic_settings["rel_anomaly"] * u.rad,
        "condition": 0,
    }

    arguments_dict_aligned_spin = arguments_dict | {
        f"spin{k}{o}": 0.0 * u.dimensionless_unscaled
        for k in (1, 2)
        for o in ("x", "y")
    }

    arguments_dict_aligned_spin_start_freq = arguments_dict_aligned_spin | {
        "f22_start": arguments_dict_aligned_spin["f22_ref"]
    }

    generator = SEOBNRv5EHM()

    # wrong types
    with pytest.raises(
        ValueError,
        match=re.escape("'Mpc' (length) and '' (dimensionless) are not convertible"),
    ):
        _ = wfm.GenerateTDModes(
            arguments_dict | {"eccentricity": 10 * u.Mpc}, generator
        )
    with pytest.raises(
        ValueError,
        match=re.escape("'Hz' (frequency) and 'rad' (angle) are not convertible"),
    ):
        _ = wfm.GenerateTDModes(arguments_dict | {"meanPerAno": 20 * u.Hz}, generator)
    with pytest.raises(
        ValueError,
        match=re.escape("'Hz' (frequency) and 'solMass' (mass) are not convertible"),
    ):
        _ = wfm.GenerateTDModes(arguments_dict | {"mass1": 20 * u.Hz}, generator)

    # EHM error capture
    with pytest.raises(
        ValueError,
        match=r"In-plane spin components must be zero for calling the non-precessing approximant\.",
    ):
        _ = wfm.GenerateTDModes(
            arguments_dict | {"spin1y": 0.1 * u.dimensionless_unscaled}, generator
        )

    with pytest.raises(
        ValueError,
        match=r"The approximant 'SEOBNRv5EHM' does not support the choice for a "
        r"reference frequency\. Please, set 'f_ref' = 'f22_start'\.",
    ):
        _ = wfm.GenerateTDModes(
            arguments_dict_aligned_spin,
            generator,
        )

    class MyException(Exception):
        pass

    with patch(
        "pyseobnr.generate_waveform.GenerateWaveform.generate_td_modes",
        autospec=True,
    ) as p_generate_td_polarizations:
        p_generate_td_polarizations.return_value = np.arange(10000), {
            (2, 2): np.arange(10000) + 10j
        }

        hlm = wfm.GenerateTDModes(
            arguments_dict_aligned_spin_start_freq,
            generator,
        )
        p_generate_td_polarizations.assert_called_once()

        assert hlm.keys() == {(2, 2)}
        assert (
            np.max(
                np.abs(
                    hlm[(2, 2)]
                    - TimeSeries(
                        np.arange(10000) + 10j, times=np.arange(10000), name=(2, 2)
                    )
                )
            )
            < 1e-10
        )

    # capturing parameters passed to the underlying GenerateWaveform
    with patch(
        "pyseobnr.generate_waveform.GenerateWaveform.generate_td_polarizations",
        autospec=True,
    ) as p_generate_td_polarizations:

        inst = None

        def _side_effect(self, *args, **kwargs):
            nonlocal inst
            inst = self
            raise MyException()

        p_generate_td_polarizations.side_effect = _side_effect

        with pytest.raises(MyException):
            _ = wfm.GenerateTDWaveform(
                arguments_dict_aligned_spin_start_freq, generator
            )

        assert inst.parameters["eccentricity"] == basic_settings["eccentricity"]
        assert inst.parameters["rel_anomaly"] == basic_settings["rel_anomaly"]

    # capturing specific convention parameters
    with patch(
        "pyseobnr.generate_waveform.GenerateWaveform.__init__",
        autospec=True,
    ) as p_generate_init:

        for dict_convention in {
            "convention_coprecessing_phase22_set_to_0_at_reference_frequency": True,
            "convention_t0_set_to_0_at_coprecessing_amplitude22_peak": True,
        }, {
            "convention_coprecessing_phase22_set_to_0_at_reference_frequency": False,
            "convention_t0_set_to_0_at_coprecessing_amplitude22_peak": False,
        }:
            p_generate_init.reset_mock()

            p_generate_init.side_effect = MyException

            with pytest.raises(MyException):
                _ = wfm.GenerateTDWaveform(
                    arguments_dict_aligned_spin_start_freq | dict_convention, generator
                )

            p_generate_init.assert_called_once()

            for k, v in dict_convention.items():
                assert p_generate_init.call_args.args[1][k] == v
