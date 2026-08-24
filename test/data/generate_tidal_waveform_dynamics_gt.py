"""Regenerates the various frames used for unit testing the dynamics for the tidal model

Run it as follow after installation of ``pyseobnr``:

.. code:: console

    python generate_tidal_waveform_dynamics_gt.py

The files in this same folder will be rewritten.
"""

import pandas as pd
from pyseobnr.generate_waveform import generate_modes_opt

# Physical parameters of the binary
# 0.00084 corresponds closely to the omega0 of a 20 Hz f22_start
# Given as 20 * (2.7 * np.pi * lal.MTSUN_SI)

q, chi_1, chi_2, omega0 = 1.077, 0.05, -0.03, 0.00084
_, _, model = generate_modes_opt(
    q,
    chi_1,
    chi_2,
    omega0,
    approximant="SEOBNRv5THM",
    settings={"M": 2.7},
    lambda2Tidal1=500.0,
    lambda2Tidal2=800.0,
    debug=True,
)

# Generating the frame
pd.DataFrame(
    data=model.dynamics,
    columns="t, r, phi, pr, pphi, H, Omega, Omega_circular".split(", "),
).to_csv("frame_thm.csv.gz", index=False)
