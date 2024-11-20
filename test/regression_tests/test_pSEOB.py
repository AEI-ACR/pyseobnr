import random
from typing import get_args
from unittest import mock

import pytest

from pyseobnr.eob.waveform.compute_hlms import NQC_correction, compute_IMR_modes
from pyseobnr.generate_waveform import SupportedApproximants, generate_modes_opt


def test_pSEOB_settings_passed_to_underlying_models():
    """Checks the pSEOB related parameters handling in SEOBNRv5HM/SEOBNRv5PHM"""
    q = 41.83615272380585
    omega0 = 0.02
    chi_1 = 0.98917404
    chi_2 = 0.3

    dict_params = "dA_dict", "dw_dict", "domega_dict", "dtau_dict"

    for current_dict in dict_params:
        for approximant in get_args(SupportedApproximants):

            random_dict = {
                k: random.uniform(-1 if current_dict != "dtau" else -0.9999, 1)
                for k in ("2,2", "2,1", "3,3", "3,2", "4,4", "4,3", "5,5")
            }

            with mock.patch(
                "pyseobnr.models.SEOBNRv5HM.NQC_correction", wraps=NQC_correction
            ) as p_NQC_correction, mock.patch(
                "pyseobnr.models.SEOBNRv5HM.compute_IMR_modes", wraps=compute_IMR_modes
            ) as p_compute_IMR_modes:

                *_, model = generate_modes_opt(
                    q,
                    chi_1,
                    chi_2,
                    omega0,
                    approximant=approximant,
                    debug=True,
                    settings={
                        "M": 154.2059835575123,
                        "dt": 6.103515625e-05,
                        current_dict: random_dict,
                    },
                )

                p_NQC_correction.assert_called_once()
                assert "dA_dict" in p_NQC_correction.call_args.kwargs
                assert "dw_dict" in p_NQC_correction.call_args.kwargs

                p_compute_IMR_modes.assert_called_once()
                assert "dw_dict" in p_compute_IMR_modes.call_args.kwargs
                assert "domega_dict" in p_compute_IMR_modes.call_args.kwargs
                assert "dtau_dict" in p_compute_IMR_modes.call_args.kwargs

            for other_dict in dict_params:
                assert hasattr(model, other_dict)

            assert model.settings[current_dict] == random_dict


def test_pSEOB_check_dtau_above_m1_yields_an_error():
    """Checks constraints on th values of the deviations"""
    q = 41.83615272380585
    omega0 = 0.02
    chi_1 = 0.98917404
    chi_2 = 0.3

    for approximant in get_args(SupportedApproximants):

        dtau_dict = {
            k: random.uniform(-0.9999, 1)
            for k in ("2,2", "2,1", "3,3", "3,2", "4,4", "4,3", "5,5")
        }
        dtau_dict[random.choice(list(dtau_dict.keys()))] = -2

        with mock.patch(
            "pyseobnr.models.SEOBNRv5HM.NQC_correction", wraps=NQC_correction
        ) as p_NQC_correction, mock.patch(
            "pyseobnr.models.SEOBNRv5HM.compute_IMR_modes", wraps=compute_IMR_modes
        ) as p_compute_IMR_modes:

            with pytest.raises(
                ValueError,
                match="dtau must be larger than -1, otherwise the remnant rings up instead of ringing",
            ):
                *_, model = generate_modes_opt(
                    q,
                    chi_1,
                    chi_2,
                    omega0,
                    approximant=approximant,
                    debug=True,
                    settings={
                        "M": 154.2059835575123,
                        "dt": 6.103515625e-05,
                        "dtau_dict": dtau_dict,
                    },
                )

            p_NQC_correction.assert_not_called()
            p_compute_IMR_modes.assert_not_called()


def test_pSEOB_settings_passed_with_missing_modes():
    q = 41.83615272380585
    omega0 = 0.02
    chi_1 = 0.98917404
    chi_2 = 0.3

    dict_params = "dA_dict", "dw_dict", "domega_dict", "dtau_dict"

    # when some modes are missing: filled with 0
    for current_dict in dict_params:

        for approximant in get_args(SupportedApproximants):

            # 3,3 is missing from here
            random_dict = {
                k: random.uniform(-1 if current_dict != "dtau" else -0.9999, 1)
                for k in ("2,2", "2,1", "3,2", "4,4", "4,3", "5,5")
            }

            with mock.patch(
                "pyseobnr.models.SEOBNRv5HM.NQC_correction", wraps=NQC_correction
            ) as p_NQC_correction, mock.patch(
                "pyseobnr.models.SEOBNRv5HM.compute_IMR_modes", wraps=compute_IMR_modes
            ) as p_compute_IMR_modes:

                *_, model = generate_modes_opt(
                    q,
                    chi_1,
                    chi_2,
                    omega0,
                    approximant=approximant,
                    debug=True,
                    settings={
                        "M": 154.2059835575123,
                        "dt": 6.103515625e-05,
                        current_dict: random_dict,
                    },
                )

                p_NQC_correction.assert_called_once()
                assert "dA_dict" in p_NQC_correction.call_args.kwargs
                assert "dw_dict" in p_NQC_correction.call_args.kwargs
                if current_dict in ("dA_dict", "dw_dict"):
                    assert (
                        p_NQC_correction.call_args.kwargs[current_dict]
                        == {"3,3": 0.0} | random_dict
                    )

                p_compute_IMR_modes.assert_called_once()
                assert "dw_dict" in p_compute_IMR_modes.call_args.kwargs
                assert "domega_dict" in p_compute_IMR_modes.call_args.kwargs
                assert "dtau_dict" in p_compute_IMR_modes.call_args.kwargs
                if current_dict in ("dw_dict", "domega_dict", "dtau_dict"):
                    assert (
                        p_compute_IMR_modes.call_args.kwargs[current_dict]
                        == {"3,3": 0.0} | random_dict
                    )

            for other_dict in dict_params:
                assert hasattr(model, other_dict)

            # original dict passed into the settings
            assert model.settings[current_dict] == random_dict
            # dict corrected
            assert getattr(model, current_dict) == {"3,3": 0.0} | random_dict
