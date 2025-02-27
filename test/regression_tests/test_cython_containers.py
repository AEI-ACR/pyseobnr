import re

import numpy as np

import pytest

from pyseobnr.eob.utils.containers import CalibCoeffs

from .helpers import create_eob_params


def test_0_spins():
    """In case we have 0 spin, the conditions in cython for nu should be the same as for python"""
    eob_pars, hamiltonian = create_eob_params(
        m_1=0.5000000057439491,
        m_2=0.49999999425605085,
        chi_1=0,
        chi_2=0,
        omega=0.17073493094550068,
        omega_circ=0.012761920876346968,
        omega_avg=0.012761920876346968,
        omega_instant=0.016980735748382013,
        x_avg=0.30767334771596205,
        eccentricity=0.14745898920231795,
        rel_anomaly=0,
    )

    assert np.all(np.array(eob_pars.flux_params.f_coeffs) == 0)

    assert np.abs(eob_pars.p_params.nu - 0.25) < 1e-14
    assert not (np.abs(eob_pars.p_params.m_1 / eob_pars.p_params.m_2 - 1) < 1e-14)
    assert (
        np.abs(eob_pars.p_params.m_1 / eob_pars.p_params.m_2 - 1) < (2 / 5 * 1e-6)
        and np.abs(eob_pars.p_params.chi_A) < 1e-14
    )


def test_calibration_parameters():
    """Checks consistency of the operations wrt. calibration parameters"""
    eob_pars, hamiltonian = create_eob_params(
        m_1=0.5000000057439491,
        m_2=0.49999999425605085,
        chi_1=0,
        chi_2=0,
        omega=0.17073493094550068,
        omega_circ=0.012761920876346968,
        omega_avg=0.012761920876346968,
        omega_instant=0.016980735748382013,
        x_avg=0.30767334771596205,
        eccentricity=0.14745898920231795,
        rel_anomaly=0,
    )

    calibration_parameters = eob_pars.c_coeffs

    # this is not a copy
    assert id(calibration_parameters) == id(eob_pars.c_coeffs)

    # returned values are consistent
    assert calibration_parameters.a6 == eob_pars.c_coeffs.a6
    assert calibration_parameters.dSO == eob_pars.c_coeffs.dSO
    assert calibration_parameters.ddSO == eob_pars.c_coeffs.ddSO

    # this is not a dictionary
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "'pyseobnr.eob.utils.containers.CalibCoeffs' object has no attribute 'something'"
        ),
    ):
        calibration_parameters.something = 10

    with pytest.raises(
        AttributeError,
        match=re.escape(
            "'pyseobnr.eob.utils.containers.CalibCoeffs' object has no attribute 'something_else'"
        ),
    ):
        _ = calibration_parameters.something_else

    # corresponding field is assignable
    new_calibration_coeffs = CalibCoeffs(
        {
            "ddSO": 20,
        }
    )

    eob_pars.c_coeffs = new_calibration_coeffs
    # points to the same object
    assert id(new_calibration_coeffs) == id(eob_pars.c_coeffs)
    assert id(calibration_parameters) != id(eob_pars.c_coeffs)

    # default values are 0
    assert eob_pars.c_coeffs.a6 == 0
    assert eob_pars.c_coeffs.ddSO == 20
    assert eob_pars.c_coeffs.dSO == 0

    # cannot instanciate with incorrect values
    with pytest.raises(TypeError, match="must be real number, not NoneType"):
        CalibCoeffs(
            {
                "ddSO": None,
            }
        )

    with pytest.raises(TypeError, match="must be real number, not list"):
        CalibCoeffs(
            {
                "ddSO": [1, 2, 3],
            }
        )

    with pytest.raises(TypeError, match="must be real number, not tuple"):
        CalibCoeffs(
            {
                "ddSO": (1,),
            }
        )

    with pytest.raises(
        TypeError, match="only length-1 arrays can be converted to Python scalars"
    ):
        CalibCoeffs(
            {
                "ddSO": np.array([10, 20]),
            }
        )

    # can cast array of scalar (0 dimension) to double
    calib_coeff = CalibCoeffs(
        {
            "ddSO": np.array(np.pi),
        }
    )
    np.testing.assert_array_equal(calib_coeff.ddSO, np.array(np.pi))
