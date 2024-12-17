import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import pytest

from pyseobnr.eob.dynamics.integrate_ode import compute_dynamics_opt
from pyseobnr.eob.dynamics.integrate_ode_ecc import ColsEccDyn
from pyseobnr.eob.dynamics.rhs_aligned import augment_dynamics, compute_H_and_omega
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


def test_compute_omega():
    """Checks the behaviour of the omega computation for the aligned spin case"""
    # in particular, we prevent the computation of augment_dynamics too many times
    # and perform it only once per coarse/fine grid.

    m_1 = 49.63230261003650412022
    m_2 = 7.25039023163753793000
    omega_start = 0.012047558533554784

    eob_params_call1, hamiltonian1 = create_eob_params(
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

    dyn_coarse, dyn_fine = compute_dynamics_opt(
        chi_1=eob_params_call1.p_params.chi_1,
        chi_2=eob_params_call1.p_params.chi_2,
        m_1=eob_params_call1.p_params.m_1,
        m_2=eob_params_call1.p_params.m_2,
        H=hamiltonian1,
        omega0=omega_start,
        RR=SEOBNRv5RRForce(),
        rtol=1e-11,
        atol=1e-12,
        step_back=250.0,
        y_init=None,
        r_stop=1.4,
        params=eob_params_call1,
    )

    omega_cython = compute_H_and_omega(
        dyn_fine[:, : (ColsEccDyn.pphi + 1)],
        eob_params_call1.p_params.chi_1,
        eob_params_call1.p_params.chi_2,
        m_1,
        m_2,
        hamiltonian1,
    )
    # last column is omega_circle and second-to-last is omega
    dyn_fine_omega = dyn_fine[:, -2]

    # check what the function compute_dynamics_ecc_qc_opt is doing
    assert (omega_cython[:, 1] == dyn_fine_omega).all()

    # check what the function compute_H_and_omega is doing
    dyn_fine = dyn_fine[: (ColsEccDyn.pphi + 1)]
    for idx, row in enumerate(dyn_fine):
        q = row[1:3]
        p = row[3:5]

        dyn = hamiltonian1.dynamics(
            q,
            p,
            eob_params_call1.p_params.chi_1,
            eob_params_call1.p_params.chi_2,
            m_1,
            m_2,
        )

        # omega = dyn[3]
        # H_val = dyn[4]
        assert dyn[4] == omega_cython[idx, 0], f"False for idx {idx}"
        assert dyn[3] == omega_cython[idx, 1]


def test_call_augment_dynamics_only_once():
    """Checks the behaviour of the omega computation for the aligned spin case"""
    # in particular, we prevent the computation of augment_dynamics too many times
    # and perform it only once per coarse/fine grid.

    m_1 = 49.63230261003650412022
    m_2 = 7.25039023163753793000
    omega_start = 0.012047558533554784

    eob_params_call1, hamiltonian1 = create_eob_params(
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

    with patch(
        "pyseobnr.eob.dynamics.integrate_ode.augment_dynamics"
    ) as p_augment_dynamics, patch(
        "pyseobnr.eob.dynamics.integrate_ode.compute_H_and_omega"
    ) as p_compute_H_and_omega:

        p_augment_dynamics.side_effect = augment_dynamics
        p_compute_H_and_omega.side_effect = compute_H_and_omega

        _ = compute_dynamics_opt(
            chi_1=eob_params_call1.p_params.chi_1,
            chi_2=eob_params_call1.p_params.chi_2,
            m_1=eob_params_call1.p_params.m_1,
            m_2=eob_params_call1.p_params.m_2,
            H=hamiltonian1,
            omega0=omega_start,
            RR=SEOBNRv5RRForce(),
            rtol=1e-11,
            atol=1e-12,
            step_back=250.0,
            y_init=None,
            r_stop=1.4,
            params=eob_params_call1,
        )

        assert p_augment_dynamics.call_count == 2
        p_compute_H_and_omega.assert_called_once()
