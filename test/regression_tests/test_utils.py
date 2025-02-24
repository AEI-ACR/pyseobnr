import datetime
import math
import pickle
import tempfile
from pathlib import Path
from typing import Final
from unittest.mock import patch

import numpy as np
from scipy.interpolate import CubicSpline

import pytest

from pyseobnr.eob.utils.containers import EOBParams
from pyseobnr.eob.utils.utils import interpolate_dynamics
from pyseobnr.eob.utils.utils_eccentric import interpolate_dynamics_ecc
from pyseobnr.generate_waveform import generate_modes_opt

from .helpers import create_eob_params


def interpolate_dynamics_ref(dyn_fine, dt=0.1, peak_omega=None, step_back=250.0):
    """
    Same function as interpolate_dynamics, reference implementation
    """

    res = []

    if peak_omega:
        start = max(peak_omega - step_back, dyn_fine[0, 0])
        t_new = np.arange(start, peak_omega, dt)

    else:
        t_new = np.arange(dyn_fine[0, 0], dyn_fine[-1, 0], dt)

    for i in range(1, dyn_fine.shape[1]):
        intrp = CubicSpline(dyn_fine[:, 0], dyn_fine[:, i])
        res.append(intrp(t_new))

    res = np.array(res)
    res = res.T
    return np.c_[t_new, res]


def interpolate_dynamics_ecc_ref(
    dyn_fine, dt=0.1, peak_omega=None, step_back=250.0, step_back_total=False
):
    """
    Same function as interpolate_dynamics_ecc, reference implementation.
    """

    res = []

    if peak_omega:
        start = max(peak_omega - step_back, dyn_fine[0, 0])
        t_new = np.arange(start, peak_omega, dt)

    else:
        t_new = np.arange(dyn_fine[0, 0], dyn_fine[-1, 0], dt)

    for i in range(1, dyn_fine.shape[1]):
        intrp = CubicSpline(dyn_fine[:, 0], dyn_fine[:, i])
        res.append(intrp(t_new))

    res = np.array(res)
    res = res.T

    if step_back_total:
        return np.c_[t_new, res]

    elif peak_omega:
        last_index_fine = np.where(dyn_fine[:, 0] < start)[0][-1]
        assert (
            dyn_fine[: last_index_fine + 1][-1, 0] < t_new[0]
        ), "Problem with the interpolation."
        return np.concatenate((dyn_fine[: last_index_fine + 1], np.c_[t_new, res]))

    else:
        return np.c_[t_new, res]


def test_interpolate_dynamics():
    q = 6.855193
    c1 = 0.35492
    c2 = 0.962056
    om = 0.012885

    # we run this once to record the call to interpolate_dynamics
    with patch(
        "pyseobnr.eob.dynamics.integrate_ode.interpolate_dynamics"
    ) as p_interpolate_dynamics:

        class LocalException(Exception):
            pass

        p_interpolate_dynamics.side_effect = LocalException

        with pytest.raises(LocalException):
            _, _ = generate_modes_opt(
                q,
                chi1=c1,
                chi2=c2,
                omega_start=om,
                approximant="SEOBNRv5HM",
            )

        p_interpolate_dynamics.assert_called_once()
        dyn_fine = p_interpolate_dynamics.call_args_list[0].args[0]
        peak_omega = p_interpolate_dynamics.call_args_list[0].kwargs["peak_omega"]
        step_back = p_interpolate_dynamics.call_args_list[0].kwargs["step_back"]

    t1 = datetime.datetime.now()
    for _ in range(1000):
        res1 = interpolate_dynamics_ref(
            dyn_fine=dyn_fine,
            peak_omega=peak_omega,
            step_back=step_back,
        )

    t2 = datetime.datetime.now()
    for _ in range(1000):
        res2 = interpolate_dynamics(
            dyn_fine=dyn_fine,
            peak_omega=peak_omega,
            step_back=step_back,
        )
    t3 = datetime.datetime.now()

    # comparing with tolerance when MKL is used
    np.testing.assert_allclose(res1, res2, rtol=1e-12)
    assert (t3 - t2).total_seconds() <= 0.8 * (t2 - t1).total_seconds()


def test_interpolate_dynamics_ecc():
    q = 6.855193
    c1 = 0.35492
    c2 = 0.962056
    om = 0.012885
    e = 0.147622

    # we run this once to record the call to interpolate_dynamics_ecc
    with patch(
        "pyseobnr.eob.dynamics.integrate_ode_ecc.interpolate_dynamics_ecc"
    ) as p_interpolate_dynamics_ecc:

        class LocalException(Exception):
            pass

        p_interpolate_dynamics_ecc.side_effect = LocalException

        with pytest.raises(LocalException):
            _, _ = generate_modes_opt(
                q,
                chi1=c1,
                chi2=c2,
                omega_start=om,
                eccentricity=e,
                approximant="SEOBNRv5EHM",
            )

        p_interpolate_dynamics_ecc.assert_called_once()
        dyn_fine = p_interpolate_dynamics_ecc.call_args_list[0].args[0]
        peak_omega = (
            p_interpolate_dynamics_ecc.call_args_list[0].kwargs["peak_omega"]
            if "peak_omega" in p_interpolate_dynamics_ecc.call_args_list[0].kwargs
            else None
        )
        step_back = p_interpolate_dynamics_ecc.call_args_list[0].kwargs["step_back"]
        step_back_total = p_interpolate_dynamics_ecc.call_args_list[0].kwargs[
            "step_back_total"
        ]

    t1 = datetime.datetime.now()
    for _ in range(1000):
        res1 = interpolate_dynamics_ecc_ref(
            dyn_fine=dyn_fine,
            peak_omega=peak_omega,
            step_back=step_back,
            step_back_total=step_back_total,
        )

    t2 = datetime.datetime.now()
    for _ in range(1000):
        res2 = interpolate_dynamics_ecc(
            dyn_fine=dyn_fine,
            peak_omega=peak_omega,
            step_back=step_back,
            step_back_total=step_back_total,
        )
    t3 = datetime.datetime.now()

    # comparing with tolerance when MKL is used
    np.testing.assert_allclose(res1, res2, rtol=1e-12)
    assert (t3 - t2).total_seconds() <= 0.7 * (t2 - t1).total_seconds()


def test_interpolate_ecc():
    x = np.arange(2, 4 * math.pi, 0.5)
    x += np.random.uniform(-0.01, 0.01, x.shape[0])

    y = np.sin(x)
    z = (1 - np.exp(x)) * np.sin(x)

    y += np.random.uniform(-0.01, 0.01, x.shape[0])
    z += np.random.uniform(-0.01, 0.01, x.shape[0])

    arr_interpolate = interpolate_dynamics_ecc(
        np.column_stack((x, y, z)),
        dt=0.1,
        peak_omega=2 * math.pi,
        step_back=2,
        step_back_total=False,
    )
    assert (arr_interpolate[:, 0] >= x[0]).all()
    assert arr_interpolate[0, 0] == x[0]

    # in case step back is too big, we start at x[0]
    arr_interpolate = interpolate_dynamics_ecc(
        np.column_stack((x, y, z)),
        dt=0.1,
        peak_omega=2 * math.pi,
        step_back=5,
        step_back_total=False,
    )
    assert (arr_interpolate[:, 0] >= x[0]).all()
    assert arr_interpolate[0, 0] == x[0]

    arr_interpolate_step_back_total = interpolate_dynamics_ecc(
        np.column_stack((x, y, z)),
        dt=0.1,
        peak_omega=2 * math.pi,
        step_back=2,
        step_back_total=True,
    )

    assert (arr_interpolate_step_back_total[:, 0] >= 2 * math.pi - 2).all()
    assert abs(arr_interpolate_step_back_total[0, 0] - (2 * math.pi - 2)) < 1e-12


cols_phys: Final = (
    "m_1",
    "m_2",
    "M",
    "nu",
    "X_1",
    "X_2",
    "delta",
    "a1",
    "a2",
    "chi_1",
    "chi_2",
    "chi1_v",
    "chi2_v",
    "lN",
    "omega",
    "omega_circ",
    "H_val",
)

cols_flux: Final = (
    "Tlm",
    "rho_coeffs",
    "rho_coeffs_log",
    "f_coeffs",
    "f_coeffs_vh",
    "delta_coeffs",
    "delta_coeffs_vh",
    "rholm",
    "deltalm",
    "prefixes",
    "prefixes_abs",
    "nqc_coeffs",
    "extra_coeffs",
    "extra_coeffs_log",
    "special_modes",
    "extra_PN_terms",
)


# the following two functions are convenient for debugging the eob_params
def _save_eob_params(filename, eob_params):
    """Utility function for saving eob_params to disk

    The function is useful for debugging functions that depend on an instance of EOBParams.
    Only a subset of the parameters is being saved at the moment. Expand as needed.
    """
    dict_phys = {col: np.array(getattr(eob_params.p_params, col)) for col in cols_phys}
    dict_flux = {
        col: np.array(getattr(eob_params.flux_params, col)) for col in cols_flux
    }

    with open(filename, "wb") as f:
        pickle.dump({"p_params": dict_phys, "flux_params": dict_flux}, f)


def _load_eob_params(filename, eob_params):
    """Utility function for loading eob_params from disk"""
    with open(filename, "rb") as f:
        dict_pic = pickle.load(f)

    for col in cols_phys:
        if len(dict_pic["p_params"][col].shape):
            setattr(eob_params.p_params, col, dict_pic["p_params"][col])
        else:
            setattr(eob_params.p_params, col, dict_pic["p_params"][col].item())

    for col in cols_flux:
        if col == "special_modes":
            continue

        # in order to narrow it down to the field that is not properly initialized,
        # we can discard all fields but one...
        # if col not in ["extra_coeffs"]:
        #     continue

        if len(dict_pic["flux_params"][col].shape):
            setattr(eob_params.flux_params, col, dict_pic["flux_params"][col])
        else:
            setattr(eob_params.flux_params, col, dict_pic["flux_params"][col].item())


def test_eob_params_save_load():
    """Smake test on the read/write of the EOBParam utility functions"""
    eob_params, ham = create_eob_params(
        m_1=50,
        m_2=60,
        chi_1=0.3,
        chi_2=0.4,
        omega=0.015,
        omega_circ=0.015,
        omega_avg=0.015,
        omega_instant=0.015,
        x_avg=0.015,
        eccentricity=0.1,
        rel_anomaly=np.pi,
    )

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        _save_eob_params(d / "test.pkl", eob_params=eob_params)
        recovered_eob_params = EOBParams(
            dict(
                m_1=3,
                m_2=3,
                chi_1=0.1,
                chi_2=0.2,
                chi1_v=np.array([0, 0, 0.1]),
                chi2_v=np.array([0, 0, 0.1]),
                lN=np.array([1.0]),
                omega=0.015,
                H_val=0,
                a1=abs(0.1),
                a2=abs(0.1),  # on purpose
            ),
            {},
            {},
        )
        _load_eob_params(d / "test.pkl", recovered_eob_params)

        assert recovered_eob_params.p_params.m_1 == eob_params.p_params.m_1
        assert recovered_eob_params.p_params.m_2 == eob_params.p_params.m_2
        assert recovered_eob_params.p_params.chi_1 == eob_params.p_params.chi_1
        assert recovered_eob_params.p_params.chi_2 == eob_params.p_params.chi_2

        np.testing.assert_allclose(
            np.array(recovered_eob_params.p_params.chi1_v),
            np.array(eob_params.p_params.chi1_v),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            np.array(recovered_eob_params.p_params.chi2_v),
            np.array(eob_params.p_params.chi2_v),
            rtol=1e-12,
        )
