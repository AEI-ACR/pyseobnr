"""Regenerates the various frames used for unit testing the dynamics for the Eccentric models

Run it as follow after installation of ``pyseobnr``:

.. code:: console

    python generate_eccentric_waveform_dynamics_gt.py

The files in this same folder will be rewritten.
"""

import pandas as pd
from pyseobnr.generate_waveform import generate_modes_opt


# Physical parameters of the binary

q = 1.5
# The new model currently does not support the inclusion of spin
chi_1 = [0.0, 0.0, 0.0]
chi_2 = [0.0, 0.0, 0.0]
omega0 = 0.01
omega_start = omega0
eccentricity = 0.3
rel_anomaly = 0.0


# This gives the new eccentric model, but with the v5EHM infrastructure
settings = {}
_, _, model_ehm = generate_modes_opt(
    q,
    chi_1[2],
    chi_2[2],
    omega0,
    omega_start,
    eccentricity,
    rel_anomaly,
    approximant="SEOBNRv5EHM",
    settings=settings,
    debug=True,
)


# storing the dynamics in pandas frames
frame_ehm = pd.DataFrame(
    columns="t_e, r_e, phi_e, prstar_e, pphi_e, e_e, z_e, x_e, H_e, Omega_e".replace(
        " ", ""
    ).split(","),
    data=list(model_ehm.dynamics),
)

frame_ehm.to_csv("frame_ehm.csv.gz", index=False)
