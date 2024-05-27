from unittest.mock import patch

import numpy as np

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
            chi1=chi_1,
            chi2=chi_2,
            omega_start=omega0,
            approximant="SEOBNRv5PHM",
        )

        p_compute_IMR_modes.assert_called_once()
        assert p_compute_IMR_modes.call_args.kwargs["qnm_rotation"] > 0

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
