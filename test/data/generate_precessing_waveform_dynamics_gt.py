"""Regenerates the various frames used for unit testing the dynamics for the Eccentric models

Run it as follow after installation of ``pyseobnr``:

.. code:: console

    python generate_precessing_waveform_dynamics_gt.py

The files in this same folder will be rewritten.
"""

import numpy as np
import pandas as pd

from pyseobnr.generate_waveform import generate_modes_opt

# Physical parameters of the binary

q = 1.5
# The new model currently does not support the inclusion of spin
chi_1 = np.array([0.1, 0.2, 0.3])
chi_2 = np.array([0.3, 0.2, 0.1])
omega0 = 0.01
omega_start = omega0


# This gives the new eccentric model, but with the v5EHM infrastructure
settings = {}
_, _, model_phm = generate_modes_opt(
    q,
    chi_1,
    chi_2,
    omega0,
    omega_start,
    approximant="SEOBNRv5PHM",
    settings=settings,
    debug=True,
)


# storing the dynamics in pandas frames
frame_phm = pd.DataFrame(
    columns="t, r, phi, pr, pphi, "
    "H, omega, omega_circ, "
    "chi1, chi2".replace(" ", "").split(","),
    data=list(model_phm.dynamics),
)

frame_phm.to_csv("frame_phm.csv.gz", index=False)
