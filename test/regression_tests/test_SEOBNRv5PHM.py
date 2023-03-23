"""
Test that the precessing SEOBNRv5PHM model has not changed.
Follows the procedure of IMRPhenomXPHM/IMRPhenomTPHM tests
in LALSuite.

This file was automatically generated on 2023-03-23 20:31:34.581748.
It should be regenrated using regenerate_SEOBNRv5PHM.py whenever
understood changes to the model are made.
"""

import pytest
import numpy as np
from pyseobnr.generate_waveform import generate_modes_opt, GenerateWaveform
import lal


def get_amp_phase(h):
    amp = np.abs(h)
    phase = np.unwrap(np.angle(h))
    return amp, phase


def sum_sqr_diff(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def gen_test_data(test_type):
    """
    compute the difference between two waveforms
    and compare to expected value
    """
    m1 = 50.0
    m2 = 30.0
    Mt = m1 + m2
    dt = 1 / 2048.0
    distance = 1.0
    inclination = np.pi / 3.0
    phiRef = 0.0
    approximant = "SEOBNRv5PHM"
    s1x = 0.5
    s1y = 0.3
    s2x = 0.3
    s2y = 0.7
    s1z = 0.2
    s2z = -0.1
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
        "f22_start": f_min,
        "phi_ref": phiRef,
        "distance": distance,
        "inclination": inclination,
        "f_max": f_max,
        "approximant": approximant,
        "postadiabatic": True
    }
    wfm_gen = GenerateWaveform(params_dict)  # We call the generator with the parameters
    params_dict2 = params_dict.copy()

    if test_type == "FD":
        # Test differences in Fourier domain, when varying intrinsic params
        hp1, hc1 = wfm_gen.generate_fd_polarizations()
        params_dict2.update({"mass2": 20.0})
        wfm_gen2 = GenerateWaveform(params_dict2)
        hp2, hc2 = wfm_gen2.generate_fd_polarizations()

        # Since we are in FD, hp,hc are *complex*, decompose into amp/phase
        hp1_amp, hp1_phase = get_amp_phase(hp1.data.data)
        hc1_amp, hc1_phase = get_amp_phase(hc1.data.data)

        hp2_amp, hp2_phase = get_amp_phase(hp2.data.data)
        hc2_amp, hc2_phase = get_amp_phase(hc2.data.data)

        hp_amp_diff = sum_sqr_diff(hp1_amp, hp2_amp)
        hp_phase_diff = sum_sqr_diff(hp1_phase, hp2_phase)

        hc_amp_diff = sum_sqr_diff(hc1_amp, hc2_amp)
        hc_phase_diff = sum_sqr_diff(hc1_phase, hc2_phase)

        return hp_amp_diff, hp_phase_diff, hc_amp_diff, hc_phase_diff

    elif test_type == "TD":
        # Test differences in time domain, when varying extrinsic params
        hp1, hc1 = wfm_gen.generate_td_polarizations()
        params_dict2.update({"inclination": 0.17})
        params_dict2.update({"phi_ref": 0.5})

        wfm_gen2 = GenerateWaveform(params_dict2)
        hp2, hc2 = wfm_gen2.generate_td_polarizations()
        # We are in TD, so hp,hc are *real*, compute differences directly
        hp_diff = sum_sqr_diff(hp1.data.data, hp2.data.data)
        hc_diff = sum_sqr_diff(hc1.data.data, hc2.data.data)
        return hp_diff, hc_diff

# -- test functions ---------------------


def test_SEOBNRv5PHM_diff_TD():
    """
    This test checks that SEOBNRv5HM hasn't changed in the time domain.
    It does this by generating two SEOBNRv5HM waveforms and computing
    their difference (according to their amplitude and phases)
    and compares them to pre-computed values.

    these pre-computed values were computed using the following line:

    `expected_result  =  np.array(gen_test_data("TD"))`

    """

    expected_result = np.array([9.286455953988354e-18,8.842800671532437e-18])

    new_result = np.array(gen_test_data("TD"))
    np.testing.assert_allclose(
        new_result, expected_result, rtol=1e-4, err_msg="SEOBNRv5HM TD test failed"
    )


def test_SEOBNRv5PHM_diff_FD():
    """
    This test checks that SEOBNRv5HM hasn't changed in frequency domain.
    It does this by generating two SEOBNRv5HM waveforms and computing
    their difference (according to their amplitude and phases)
    and compares them to pre-computed values.

    these pre-computed values were computed using the following line:

    `expected_result  =  np.array(gen_test_data("FD"))`

    """

    expected_result = np.array(
	[9.652274107044372e-20,3.391378195301547e+04,8.605764565737375e-20,
 3.440922572212310e+04]
    )

    new_result = np.array(gen_test_data("FD"))
    np.testing.assert_allclose(
        new_result, expected_result, rtol=2.0e-4, err_msg="SEOBNRv5HM FD test failed"
    )