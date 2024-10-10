import datetime
from unittest.mock import patch

import numpy as np
from scipy.interpolate import CubicSpline

import pytest

from pyseobnr.eob.utils.utils import interpolate_dynamics
from pyseobnr.generate_waveform import generate_modes_opt


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


def test_interpolate_dynamics():
    q = 6.855193
    c1 = 0.35492
    c2 = 0.962056
    om = 0.012885

    # we run this once to record the call to interpolate_dynamics
    with patch(
        "pyseobnr.eob.dynamics.integrate_ode.interpolate_dynamics"
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
                approximant="SEOBNRv5HM",
            )

        p_interpolate_dynamics_ecc.assert_called_once()
        dyn_fine = p_interpolate_dynamics_ecc.call_args_list[0].args[0]
        peak_omega = p_interpolate_dynamics_ecc.call_args_list[0].kwargs["peak_omega"]
        step_back = p_interpolate_dynamics_ecc.call_args_list[0].kwargs["step_back"]

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
    assert (t3 - t2).total_seconds() <= 0.7 * (t2 - t1).total_seconds()
