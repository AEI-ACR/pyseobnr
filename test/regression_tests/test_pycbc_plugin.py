from unittest.mock import patch

import lal
import numpy as np

import pytest


def test_pycbc_basic_installation():
    import pycbc.waveform

    # just to use the package
    assert dir(pycbc.waveform) is not None


@pytest.fixture
def basic_settings():
    m1 = 50.0
    m2 = 30.0
    Mt = m1 + m2
    dt = 1 / 2048.0
    distance = 1.0
    inclination = np.pi / 3.0
    phiRef = 0.0
    s1x = s1y = s2x = s2y = 0.0
    s1z = 0.5
    s2z = 0.1
    f_max = 1024.0
    f_min = 0.0157 / (Mt * np.pi * lal.MTSUN_SI)
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
    }

    # translates pyseobnr parameters to the expectations of pycbc, the opposite
    # transformation will be done inside the plugin
    params_dict["delta_t"] = params_dict.pop("deltaT")
    params_dict["f_lower"] = params_dict.pop("f22_start")
    params_dict["coa_phase"] = params_dict.pop("phi_ref")
    params_dict["f_final"] = params_dict.pop("f_max")
    return params_dict


def test_pycbc_plugin_td_v5hm(basic_settings):
    import pycbc.waveform

    hp, hc = pycbc.waveform.get_td_waveform(approximant="SEOBNRv5HM", **basic_settings)

    assert hp is not None
    assert hc is not None

    with pytest.raises(
        ValueError,
        match=r"In-plane spin components must be zero for calling non-precessing approximant\.",
    ):
        _ = pycbc.waveform.get_td_waveform(
            approximant="SEOBNRv5HM", **(basic_settings | {"spin1y": 0.1})
        )

    with patch(
        "pyseobnr.generate_waveform.GenerateWaveform.generate_td_polarizations",
        autospec=True,
    ) as p_generate_td_polarizations:

        class MyException(Exception):
            pass

        p_generate_td_polarizations.side_effect = MyException
        with pytest.raises(MyException):
            _ = pycbc.waveform.get_td_waveform(
                approximant="SEOBNRv5HM", **basic_settings
            )

    with patch(
        "pyseobnr.generate_waveform.generate_modes_opt",
        autospec=True,
    ) as p_generate_modes_opt:

        class MyException(Exception):
            pass

        p_generate_modes_opt.side_effect = MyException
        with pytest.raises(MyException):
            _ = pycbc.waveform.get_td_waveform(
                approximant="SEOBNRv5HM", **basic_settings
            )

        p_generate_modes_opt.assert_called_once()
        assert "approximant" in p_generate_modes_opt.call_args.kwargs
        assert p_generate_modes_opt.call_args.kwargs["approximant"] == "SEOBNRv5HM"


def test_pycbc_plugin_fd_v5hm(basic_settings):
    import pycbc.waveform

    basic_settings |= {
        "delta_f": 0,  # required for fd
    }

    hp, hc = pycbc.waveform.get_fd_waveform(approximant="SEOBNRv5HM", **basic_settings)

    assert hp is not None
    assert hc is not None

    with patch(
        "pyseobnr.generate_waveform.GenerateWaveform.generate_fd_polarizations",
        autospec=True,
    ) as p_generate_fd_polarizations:

        class MyException(Exception):
            pass

        p_generate_fd_polarizations.side_effect = MyException
        with pytest.raises(MyException):
            _ = pycbc.waveform.get_fd_waveform(
                approximant="SEOBNRv5HM", **basic_settings
            )

    with patch(
        "pyseobnr.generate_waveform.generate_modes_opt",
        autospec=True,
    ) as p_generate_modes_opt:

        class MyException(Exception):
            pass

        p_generate_modes_opt.side_effect = MyException
        with pytest.raises(MyException):
            _ = pycbc.waveform.get_fd_waveform(
                approximant="SEOBNRv5HM", **basic_settings
            )

        p_generate_modes_opt.assert_called_once()
        assert "approximant" in p_generate_modes_opt.call_args.kwargs
        assert p_generate_modes_opt.call_args.kwargs["approximant"] == "SEOBNRv5HM"


def test_pycbc_plugin_td_v5phm(basic_settings):
    import pycbc.waveform

    basic_settings |= {
        "spin1x": basic_settings["spin1x"] - 0.1,
        "spin2y": basic_settings["spin2y"] + 0.1,
    }

    hp, hc = pycbc.waveform.get_td_waveform(approximant="SEOBNRv5PHM", **basic_settings)

    assert hp is not None
    assert hc is not None

    with patch(
        "pyseobnr.generate_waveform.GenerateWaveform.generate_td_polarizations",
        autospec=True,
    ) as p_generate_td_polarizations:

        class MyException(Exception):
            pass

        p_generate_td_polarizations.side_effect = MyException
        with pytest.raises(MyException):
            _ = pycbc.waveform.get_td_waveform(
                approximant="SEOBNRv5PHM", **basic_settings
            )

    with patch(
        "pyseobnr.generate_waveform.generate_prec_hpc_opt",
        autospec=True,
    ) as p_generate_modes_opt:

        class MyException(Exception):
            pass

        p_generate_modes_opt.side_effect = MyException
        with pytest.raises(MyException):
            _ = pycbc.waveform.get_td_waveform(
                approximant="SEOBNRv5PHM", **basic_settings
            )

        p_generate_modes_opt.assert_called_once()
        # no named argument call at the moment
        # assert "chi1" in p_generate_modes_opt.call_args.kwargs
        assert list(p_generate_modes_opt.call_args.args[1]) == [
            basic_settings[f"spin1{_}"] for _ in ("x", "y", "z")
        ]


def test_pycbc_plugin_fd_v5phm(basic_settings):
    import pycbc.waveform

    basic_settings |= {
        "spin1x": basic_settings["spin1x"] - 0.1,
        "spin2y": basic_settings["spin2y"] + 0.1,
        "delta_f": 0,  # required for fd
    }

    hp, hc = pycbc.waveform.get_fd_waveform(approximant="SEOBNRv5PHM", **basic_settings)

    assert hp is not None
    assert hc is not None

    with patch(
        "pyseobnr.generate_waveform.GenerateWaveform.generate_fd_polarizations",
        autospec=True,
    ) as p_generate_fd_polarizations:

        class MyException(Exception):
            pass

        p_generate_fd_polarizations.side_effect = MyException
        with pytest.raises(MyException):
            _ = pycbc.waveform.get_fd_waveform(
                approximant="SEOBNRv5PHM", **basic_settings
            )

    with patch(
        "pyseobnr.generate_waveform.generate_prec_hpc_opt",
        autospec=True,
    ) as p_generate_modes_opt:

        class MyException(Exception):
            pass

        p_generate_modes_opt.side_effect = MyException
        with pytest.raises(MyException):
            _ = pycbc.waveform.get_fd_waveform(
                approximant="SEOBNRv5PHM", **basic_settings
            )

        p_generate_modes_opt.assert_called_once()
        # no named argument call at the moment
        # assert "chi1" in p_generate_modes_opt.call_args.kwargs
        assert list(p_generate_modes_opt.call_args.args[1]) == [
            basic_settings[f"spin1{_}"] for _ in ("x", "y", "z")
        ]


def test_pycbc_f_ref_eq_0_convention(basic_settings):
    import pycbc.waveform

    # pycbc will set this to 0
    basic_settings.pop("f_ref")
    hp, hc = pycbc.waveform.get_td_waveform(approximant="SEOBNRv5HM", **basic_settings)

    assert hp is not None
    assert hc is not None

    basic_settings |= {
        "delta_f": 0,  # required for fd
    }
    hp, hc = pycbc.waveform.get_fd_waveform(approximant="SEOBNRv5HM", **basic_settings)
    assert hp is not None
    assert hc is not None


def test_pycbc_f_final_eq_0_convention(basic_settings):
    import pycbc.waveform

    # pycbc will set this to 0
    basic_settings.pop("f_final")
    hp, hc = pycbc.waveform.get_td_waveform(approximant="SEOBNRv5HM", **basic_settings)

    assert hp is not None
    assert hc is not None

    basic_settings |= {
        "delta_f": 0,  # required for fd
    }
    hp, hc = pycbc.waveform.get_fd_waveform(approximant="SEOBNRv5HM", **basic_settings)
    assert hp is not None
    assert hc is not None
