import os
from pathlib import Path

import numpy as np
import pandas as pd

import pytest

from pyseobnr.eob.dynamics.integrate_ode import compute_dynamics_opt
from pyseobnr.eob.dynamics.rhs_precessing import get_rhs_prec
from pyseobnr.eob.hamiltonian.Ham_AvgS2precess_simple_cython_PA_AD import (
    Ham_AvgS2precess_simple_cython_PA_AD as Ham_precessing_opt,
)
from pyseobnr.eob.waveform.waveform import SEOBNRv5RRForce
from pyseobnr.generate_waveform import generate_modes_opt

from .helpers import compare_frames, create_eob_params

folder_data = Path(__file__).parent.parent / "data"


@pytest.mark.skipif(
    "CI_TEST_DYNAMIC_REGRESSIONS" not in os.environ,
    reason="regressions on dynamics are for specific systems only",
)
def test_regression_dynamic():
    """
    Checks the dynamic against a full data frame
    """
    q = 1.5
    chi_1 = np.array([0.1, 0.2, 0.3])
    chi_2 = np.array([0.3, 0.2, 0.1])
    omega0 = 0.01
    omega_start = omega0
    settings = {}

    _, _, model = generate_modes_opt(
        q,
        chi_1,
        chi_2,
        omega0,
        omega_start,
        approximant="SEOBNRv5PHM",
        settings=settings,
        debug=True,
    )

    frame_phm = pd.DataFrame(
        data=model.dynamics,
        columns="t, r, phi, pr, pphi, "
        "H, omega, omega_circ, "
        "chi1, chi2".replace(" ", "").split(","),
    )
    frame_phm_reference = pd.read_csv(folder_data / "frame_phm.csv.gz")

    known_differences_percentage = {
        "r": 1,
        "phi": 1,
        "pr": 1,
        "pphi": 1,
        "H": 1,
        "omega": 1,
        "omega_circ": 1,
        "chi1": 1,
        "chi2": 1,
    }

    compare_frames(
        test_frame=frame_phm,
        reference_frame=frame_phm_reference,
        known_differences_percentage=known_differences_percentage,
        time_tolerance_percent=1,
    )


def test_compute_rhs_precessing():
    """Checks the behaviour of the omega computation for the aligned spin case"""
    # in particular, we prevent the computation of augment_dynamics too many times
    # and perform it only once per coarse/fine grid.

    m_1 = 49.63230261003650412022
    m_2 = 7.25039023163753793000
    omega_start = 0.012047558533554784

    eob_params_call1, hamiltonian_aligned = create_eob_params(
        m_1=m_1,
        m_2=m_2,
        chi_1=-0.08112979979029,
        chi_2=0.27848911003169274,
        omega=0.3128483050626029,
        omega_circ=0.012666718090257607,
        omega_avg=0.012666718090257607,
        omega_instant=0.013442463859018835,
        x_avg=0.4608462098169872,
        eccentricity=0.1,
        rel_anomaly=0,
    )

    hamiltonian_prec = Ham_precessing_opt(eob_params_call1)
    hamiltonian_prec.eob_params.c_coeffs = hamiltonian_aligned.calibration_coeffs
    hamiltonian_prec.calibration_coeffs.ddSO = 0

    dyn_coarse, dyn_fine = compute_dynamics_opt(
        chi_1=eob_params_call1.p_params.chi_1,
        chi_2=eob_params_call1.p_params.chi_2,
        m_1=eob_params_call1.p_params.m_1,
        m_2=eob_params_call1.p_params.m_2,
        H=hamiltonian_aligned,
        omega0=omega_start,
        RR=SEOBNRv5RRForce(),
        rtol=1e-11,
        atol=1e-12,
        step_back=250.0,
        y_init=None,
        r_stop=1.4,
        params=eob_params_call1,
    )

    RR = SEOBNRv5RRForce()

    # we are replaying the dynamics p/q from a aligned spin system
    # we are keeping the spins constant as well. This is fine for the purpose
    # of checking get_rhs_prec

    for idx, row in enumerate(dyn_fine):

        q = tuple(row[0:2])
        p = tuple(row[2:4])

        chi1_LN = eob_params_call1.p_params.chi_1
        chi2_LN = eob_params_call1.p_params.chi_2
        chi1_L = eob_params_call1.p_params.chi1_L
        chi2_L = eob_params_call1.p_params.chi2_L

        dynamics = hamiltonian_prec.dynamics(
            q,
            p,
            eob_params_call1.p_params.chi1_v,
            eob_params_call1.p_params.chi2_v,
            m_1,
            m_2,
            chi1_LN,
            chi2_LN,
            chi1_L,
            chi2_L,
        )

        H_val = dynamics[4]
        omega = dynamics[3]
        eob_params_call1.dynamics.p_circ = (
            eob_params_call1.dynamics.p_circ[0],
            p[1],
        )

        omega_circ = hamiltonian_prec.omega(
            q,
            eob_params_call1.dynamics.p_circ,
            eob_params_call1.p_params.chi1_v,
            eob_params_call1.p_params.chi2_v,
            m_1,
            m_2,
            chi1_LN,
            chi2_LN,
            chi1_L,
            chi2_L,
        )

        xi = dynamics[5]

        eob_params_call1.p_params.omega_circ = omega_circ
        eob_params_call1.p_params.omega = omega
        eob_params_call1.p_params.H_val = H_val

        RR_f = RR.RR(q, p, omega, omega_circ, H_val, eob_params_call1)

        val = [
            xi * dynamics[2],
            dynamics[3],
            -dynamics[0] * xi + RR_f[0],
            -dynamics[1] + RR_f[1],
        ]

        np.testing.assert_array_equal(
            val,
            get_rhs_prec(
                0,
                row[:4],
                hamiltonian_prec,
                RR,
                m_1,
                m_2,
                eob_params_call1,
            ),
        )
