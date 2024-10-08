import os
from pathlib import Path

import pandas as pd

import pytest

from pyseobnr.generate_waveform import generate_modes_opt

from .helpers import compare_frames

folder_data = Path(__file__).parent.parent / "data"


@pytest.mark.skipif(
    "CI_TEST_DYNAMIC_REGRESSIONS" not in os.environ,
    reason="regressions on dynamics are for specific systems only",
)
def test_regression_dynamic():
    """
    Checks the dynamic against a full data frame
    """
    q = 5.3
    chi_1 = 0.9
    chi_2 = 0.3
    omega0 = 0.0137  # This is the orbital frequency in geometric units with M=1

    _, _, model = generate_modes_opt(q, chi_1, chi_2, omega0, debug=True)

    frame_hm = pd.DataFrame(
        data=model.dynamics,
        columns="t, r, phi, pr, pphi, H, Omega, Omega_circular".replace(" ", "").split(
            ","
        ),
    )
    frame_hm_reference = pd.read_csv(folder_data / "frame_hm.csv.gz")

    known_differences_percentage = {
        "r": 1,
        "phi": 1,
        "pr": 1,
        "pphi": 1,
        "H": 1,
        "Omega": 1,
        "Omega_circular": 1,
    }

    compare_frames(
        test_frame=frame_hm,
        reference_frame=frame_hm_reference,
        known_differences_percentage=known_differences_percentage,
        time_tolerance_percent=1,
    )
