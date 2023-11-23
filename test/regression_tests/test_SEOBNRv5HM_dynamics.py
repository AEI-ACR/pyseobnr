from pathlib import Path

import pandas as pd

from pyseobnr.generate_waveform import generate_modes_opt
from .helpers import compare_frames

folder_data = Path(__file__).parent.parent / "data"


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

    known_differences = {
        "r": 1.303788310647036e-06,  # 1e-10,
        "phi": 7.878629730839748e-07,  # 1e-10,
        "pr": 3.3902689824949483e-07,  # 1e-10,
        "pphi": 1.6233081812089267e-07,  # 1e-10,
        "H": 1.0472285616458521e-07,  # 1e-10,
        "Omega": 1.1044090023060171e-07,  # 1e-10,
        "Omega_circular": 1.344480126674874e-07,  # 1e-10,
    }

    compare_frames(
        test_frame=frame_hm,
        reference_frame=frame_hm_reference,
        known_differences=known_differences,
        time_tolerance=0.09,  # pretty high, the reference frame was generated on a macOS m2
    )
