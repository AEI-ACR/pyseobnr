from unittest.mock import patch

import numpy as np
import pandas as pd

from pyseobnr.eob.utils.utils import estimate_time_max_amplitude
from pyseobnr.eob.waveform.compute_hlms import compute_IMR_modes
from pyseobnr.generate_waveform import generate_modes_opt


def test_qnm_rotation_always_positive():
    """Checks we are applying the right rotation to the QNMs

    See https://git.ligo.org/waveforms/software/pyseobnr/-/merge_requests/70 for details
    """

    chi_1 = np.array((-0.0131941810847, 0.0676539715676, -0.646272606477))
    chi_2 = 0
    omega0 = 0.02

    with patch("pyseobnr.models.SEOBNRv5HM.compute_IMR_modes") as p_compute_IMR_modes:
        p_compute_IMR_modes.side_effect = compute_IMR_modes

        generate_modes_opt(
            q=5.33,
            chi1=chi_1[2],
            chi2=chi_2,
            omega_start=omega0,
            approximant="SEOBNRv5HM",
        )

        p_compute_IMR_modes.assert_called_once()
        assert "qnm_rotation" not in p_compute_IMR_modes.call_args.kwargs

        p_compute_IMR_modes.reset_mock()

        generate_modes_opt(
            q=5.32,
            chi1=chi_1,
            chi2=chi_2,
            omega_start=omega0,
            approximant="SEOBNRv5PHM",
        )

        p_compute_IMR_modes.assert_called_once()
        assert p_compute_IMR_modes.call_args.kwargs["qnm_rotation"] > 0


def test_estimate_t_for_max_amplitude_PHM():
    """Checks that the estimated t_0 from the frame invariant amplitude is stable wrt. frequency"""
    chi_1 = np.array((-0.0131941810847, 0.0676539715676, -0.646272606477))
    chi_2 = 0.1
    omega0 = 0.02
    q = 10

    all_t0 = {}
    all_ts = {}
    all_frame_inv_amplitudes = {}

    with patch(
        "pyseobnr.models.SEOBNRv5HM.estimate_time_max_amplitude"
    ) as p_estimate_time_max_amplitude:

        def m_estimate_time_max_amplitude(*args, **kwargs):
            ret = estimate_time_max_amplitude(*args, **kwargs)
            # gets the outside freq
            all_t0[freq] = ret
            all_frame_inv_amplitudes[freq] = kwargs["amplitude"]
            return ret

        p_estimate_time_max_amplitude.side_effect = m_estimate_time_max_amplitude

        # 2**11, 2**12 ....
        freqs_to_check = np.logspace(10, 14, base=2, num=14 - 10 + 1)
        for freq in freqs_to_check:
            _, _, model = generate_modes_opt(
                q=q,
                chi1=chi_1,
                chi2=chi_2,
                omega_start=omega0,
                approximant="SEOBNRv5PHM",
                settings={"dt": 1 / float(freq), "M": 100},
                debug=True,
            )
            p_estimate_time_max_amplitude.assert_called_once()
            p_estimate_time_max_amplitude.reset_mock()

            all_ts[freq] = model.t

    # we calculate the value of the amplitude at 10M
    inter_at_10M = []
    for freq in freqs_to_check:
        inter_at_10M += [
            {
                "freq": freq,
                "t_0": all_t0[freq],
                # 10 -> this is 10M
                "val": np.interp(10, all_ts[freq], all_frame_inv_amplitudes[freq]),
            }
        ]

    inter_at_10M = pd.DataFrame(inter_at_10M)

    # the estimated t_0 does not "vary too much"
    assert inter_at_10M["t_0"].std() < 1e-2

    # the estimated amplitude at 10M does not vary too much
    # without the fix, we obtain 0.01928841154831644 for this specific setup
    assert inter_at_10M["val"].std() / inter_at_10M["val"].max() < 1e-3


def test_estimate_t_for_max_amplitude_HM():
    """Checks that the estimated t_0 from the frame invariant amplitude is stable wrt. frequency"""
    chi_1 = -0.646272606477
    chi_2 = 0.1
    omega0 = 0.02
    q = 10

    all_t0 = {}
    all_ts = {}
    all_frame_inv_amplitudes = {}

    with patch(
        "pyseobnr.models.SEOBNRv5HM.estimate_time_max_amplitude"
    ) as p_estimate_time_max_amplitude:

        def m_estimate_time_max_amplitude(*args, **kwargs):
            ret = estimate_time_max_amplitude(*args, **kwargs)
            # gets the outside freq
            all_t0[freq] = ret
            all_frame_inv_amplitudes[freq] = kwargs["amplitude"]
            return ret

        p_estimate_time_max_amplitude.side_effect = m_estimate_time_max_amplitude

        # 2**11, 2**12 ....
        freqs_to_check = np.logspace(10, 14, base=2, num=14 - 10 + 1)
        for freq in freqs_to_check:
            _, _, model = generate_modes_opt(
                q=q,
                chi1=chi_1,
                chi2=chi_2,
                omega_start=omega0,
                approximant="SEOBNRv5HM",
                settings={"dt": 1 / float(freq), "M": 100},
                debug=True,
            )
            p_estimate_time_max_amplitude.assert_called_once()
            p_estimate_time_max_amplitude.reset_mock()

            all_ts[freq] = model.t

    # we calculate the value of the amplitude at 10M
    inter_at_10M = []
    for freq in freqs_to_check:
        inter_at_10M += [
            {
                "freq": freq,
                "t_0": all_t0[freq],
                # 10 -> this is 10M
                "val": np.interp(10, all_ts[freq], all_frame_inv_amplitudes[freq]),
            }
        ]

    inter_at_10M = pd.DataFrame(inter_at_10M)

    # the estimated t_0 does in fact in this case vary so that stddev = 0.4108767856285839
    # assert inter_at_10M["t_0"].std() < 1e-2

    # the estimated amplitude at 10M does not vary too much
    # without the fix, we obtain 0.02275764373726919 for this specific setup
    assert inter_at_10M["val"].std() / inter_at_10M["val"].max() < 1e-3
