import re

import numpy as np

import pytest

from pyseobnr.eob.hamiltonian.Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C import (
    Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C as Ham_aligned_opt,
)
from pyseobnr.eob.hamiltonian.Ham_AvgS2precess_simple_cython_PA_AD import (
    Ham_AvgS2precess_simple_cython_PA_AD as Ham_precessing_opt,
)
from pyseobnr.eob.utils.containers import CalibCoeffs

from .helpers import create_eob_params


def test_ctor_aligned():
    """Checks the construction of the Hamiltonian object"""

    # one mandatory argument
    with pytest.raises(
        TypeError,
        match=re.escape(r"__cinit__() takes exactly 1 positional argument (0 given)"),
    ):
        _ = Ham_aligned_opt()

    # not 2
    with pytest.raises(
        TypeError,
        match=re.escape(r"__cinit__() takes exactly 1 positional argument (2 given)"),
    ):
        _ = Ham_aligned_opt(10, 20)

    # should reject None
    with pytest.raises(
        TypeError,
        match=re.escape(
            r"Argument 'eob_params' has incorrect type (expected "
            r"pyseobnr.eob.utils.containers.EOBParams, got NoneType)"
        ),
    ):
        _ = Ham_aligned_opt(None)


def test_ctor_precessing():
    """Checks the construction of the Hamiltonian object"""

    # one mandatory argument
    with pytest.raises(
        TypeError,
        match=re.escape(r"__cinit__() takes exactly 1 positional argument (0 given)"),
    ):
        _ = Ham_precessing_opt()

    # not 2
    with pytest.raises(
        TypeError,
        match=re.escape(r"__cinit__() takes exactly 1 positional argument (2 given)"),
    ):
        _ = Ham_precessing_opt(10, 20)

    # should reject None
    with pytest.raises(
        TypeError,
        match=re.escape(
            r"Argument 'eob_params' has incorrect type (expected "
            r"pyseobnr.eob.utils.containers.EOBParams, got NoneType)"
        ),
    ):
        _ = Ham_precessing_opt(None)


@pytest.fixture
def eob_params():
    eob_params, _ = create_eob_params(
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
    return eob_params


def test_accessors_aligned(eob_params):
    """Checks consistency of various accessors of the Hamiltonian"""

    hamiltonian = Ham_aligned_opt(eob_params)
    coefficients = CalibCoeffs(
        {
            "a6": 10,
            "dSO": 20,
        }
    )

    # accessing through the eob_params of the Hamiltonian
    hamiltonian.eob_params.c_coeffs = coefficients

    assert id(hamiltonian.calibration_coeffs) == id(coefficients)
    assert id(hamiltonian.eob_params.c_coeffs) == id(coefficients)

    coefficients = CalibCoeffs(
        {
            "a6": 10,
            "dSO": 20,
        }
    )
    assert id(hamiltonian.calibration_coeffs) != id(coefficients)
    assert id(eob_params.c_coeffs) != id(coefficients)

    # this replaces the instance of the calibration coefficients inside the current
    # eob_params inside the current Hamiltonian
    hamiltonian.calibration_coeffs = coefficients
    assert id(hamiltonian.calibration_coeffs) == id(coefficients)
    assert id(hamiltonian.eob_params.c_coeffs) == id(coefficients)

    # eob_params has not been changed inside the Hamiltonian
    assert id(hamiltonian.eob_params) == id(eob_params)
    # and the corresponding field in eob_params has been replaced
    assert id(eob_params.c_coeffs) == id(coefficients)


def test_accessors_precessing(eob_params):
    """Checks consistency of various accessors of the Hamiltonian"""

    hamiltonian = Ham_precessing_opt(eob_params)
    coefficients = CalibCoeffs(
        {
            "a6": 10,
            "dSO": 20,
        }
    )

    # accessing through the eob_params of the Hamiltonian
    hamiltonian.eob_params.c_coeffs = coefficients

    assert id(hamiltonian.calibration_coeffs) == id(coefficients)
    assert id(hamiltonian.eob_params.c_coeffs) == id(coefficients)

    coefficients = CalibCoeffs(
        {
            "a6": 10,
            "dSO": 20,
        }
    )
    assert id(hamiltonian.calibration_coeffs) != id(coefficients)
    assert id(eob_params.c_coeffs) != id(coefficients)

    # this replaces the instance of the calibration coefficients inside the current
    # eob_params inside the current Hamiltonian
    hamiltonian.calibration_coeffs = coefficients
    assert id(hamiltonian.calibration_coeffs) == id(coefficients)
    assert id(hamiltonian.eob_params.c_coeffs) == id(coefficients)

    # eob_params has not been changed inside the Hamiltonian
    assert id(hamiltonian.eob_params) == id(eob_params)
    # and the corresponding field in eob_params has been replaced
    assert id(eob_params.c_coeffs) == id(coefficients)


def test_hamiltonian_calls_aligned(eob_params):
    """Checks calls consistency"""
    hamiltonian = Ham_aligned_opt(eob_params)

    call_args = (
        np.array([2.9493765, 131.52477657]),
        np.array([0.0, 2.66907957]),
        eob_params.p_params.chi_1,
        eob_params.p_params.chi_2,
        eob_params.p_params.m_1,
        eob_params.p_params.m_2,
    )
    assert len(hamiltonian(*call_args, verbose=True)) == 8
    assert len(hamiltonian(*call_args, verbose=False)) == 2

    kwargs = {
        k: v for k, v in zip(("q", "p", "chi_1", "chi_2", "m_1", "m_2"), call_args)
    }
    assert len(hamiltonian(**kwargs, verbose=True)) == 8
    assert hamiltonian(**kwargs, verbose=True) == hamiltonian(*call_args, verbose=True)
    assert len(hamiltonian(**kwargs, verbose=False)) == 2
    assert hamiltonian(**kwargs, verbose=False) == hamiltonian(
        *call_args, verbose=False
    )

    # changing the order should not change anything
    kwargs_shuffled = {k: kwargs[k] for k in ("p", "m_2", "chi_1", "chi_2", "m_1", "q")}
    assert hamiltonian(*call_args, verbose=True) == hamiltonian(
        **kwargs_shuffled, verbose=True
    )

    # omega
    assert hamiltonian.omega(*call_args) == hamiltonian.omega(**kwargs_shuffled)

    # dynamics
    np.testing.assert_array_equal(
        hamiltonian.dynamics(*call_args), hamiltonian.dynamics(**kwargs_shuffled)
    )

    # grad
    np.testing.assert_array_equal(
        hamiltonian.grad(*call_args), hamiltonian.grad(**kwargs_shuffled)
    )

    # hessian
    np.testing.assert_array_equal(
        hamiltonian.hessian(*call_args), hamiltonian.hessian(**kwargs_shuffled)
    )

    # csi
    assert hamiltonian.csi(*call_args) == hamiltonian.csi(**kwargs_shuffled)

    # auxderivs
    np.testing.assert_array_equal(
        hamiltonian.auxderivs(*call_args), hamiltonian.auxderivs(**kwargs_shuffled)
    )


@pytest.fixture
def hamiltonian_aligned(eob_params):
    hamiltonian = Ham_aligned_opt(eob_params)

    hamiltonian.calibration_coeffs.a6 = -12.69080059597502
    hamiltonian.calibration_coeffs.dSO = -28.477359513522586
    hamiltonian.calibration_coeffs.ddSO = 0

    hamiltonian.eob_params.p_params.M = 1
    hamiltonian.eob_params.p_params.nu = 0.23437499999999997
    hamiltonian.eob_params.p_params.X_1 = 0.625
    hamiltonian.eob_params.p_params.X_2 = 0.375
    return hamiltonian


def test_hamiltonian_H_gt(hamiltonian_aligned):
    """Checks numerical values returned by H.omega"""

    call_args = (
        np.array([2.9493765, 131.52477657]),
        np.array([0.0, 2.66907957]),
        0.5,
        0.1,
        0.625,
        0.375,
    )

    H_val = (
        4.1430240742549,
        0.4290490169265656,
        0.39226432621194446,
        -0.5415063754847536,
        -0.18845865877579898,
        0.0,
        0.8402955742557364,
        0.037853336144015814,
    )
    np.testing.assert_array_almost_equal(
        hamiltonian_aligned(*call_args, verbose=True), H_val
    )


def test_hamiltonian_omega_gt(hamiltonian_aligned):
    """Checks numerical values returned by H.omega"""

    call_args = (
        np.array([2.9493765, 131.52477657]),
        np.array([0.0, 2.66907957]),
        0.5,
        0.1,
        0.625,
        0.375,
    )

    omega = 0.15863599493180983
    assert abs(hamiltonian_aligned.omega(*call_args) - omega) < 1e-10, (
        hamiltonian_aligned.omega(*call_args),
        omega,
    )


def test_hamiltonian_gradient_gt(hamiltonian_aligned):
    """Checks numerical values returned by H.gradient"""

    call_args = (
        np.array([2.9493765, 131.52477657]),
        np.array([0.0, 2.66907957]),
        0.5,
        0.1,
        0.625,
        0.375,
    )

    grad_gt = (0.016700932983959967, 0.0, 0.0, 0.15863599493180983)
    np.testing.assert_array_almost_equal(hamiltonian_aligned.grad(*call_args), grad_gt)


def test_hamiltonian_dynamics_gt(hamiltonian_aligned):
    """Checks numerical values returned by H.dynamics"""

    call_args = (
        np.array([2.9493765, 131.52477657]),
        np.array([0.0, 2.66907957]),
        0.5,
        0.1,
        0.625,
        0.375,
    )

    dynamics_gt = (
        0.016700932983959967,
        0.0,
        0.0,
        0.15863599493180983,
        4.1430240742549,
        0.4290490169265656,
    )
    np.testing.assert_array_almost_equal(
        hamiltonian_aligned.dynamics(*call_args), dynamics_gt
    )


def test_hamiltonian_calls_precessing(eob_params):
    """Checks calls consistency"""
    hamiltonian = Ham_precessing_opt(eob_params)

    call_args = (
        np.array([2.9493765, 131.52477657]),
        np.array([0.0, 2.66907957]),
        np.array(eob_params.p_params.chi1_v),
        np.array(eob_params.p_params.chi2_v),
        eob_params.p_params.m_1,
        eob_params.p_params.m_2,
        eob_params.p_params.chi_1,
        eob_params.p_params.chi_2,
        eob_params.p_params.chi1_L,
        eob_params.p_params.chi2_L,
    )

    assert len(hamiltonian(*call_args)) == 9

    kwargs = {
        k: v
        for k, v in zip(
            (
                "q",
                "p",
                "chi1_v",
                "chi2_v",
                "m_1",
                "m_2",
                "chi_1",
                "chi_2",
                "chi_L1",
                "chi_L2",
            ),
            call_args,
        )
    }

    assert len(hamiltonian(**kwargs)) == 9
    assert hamiltonian(**kwargs) == hamiltonian(*call_args)

    # changing the order should not change anything
    kwargs_shuffled = {
        k: kwargs[k]
        for k in (
            "chi1_v",
            "q",
            "p",
            "chi_1",
            "chi_L2",
            "chi2_v",
            "m_1",
            "chi_2",
            "chi_L1",
            "m_2",
        )
    }
    assert hamiltonian(*call_args) == hamiltonian(**kwargs_shuffled)

    # omega
    assert hamiltonian.omega(*call_args) == hamiltonian.omega(**kwargs_shuffled)

    # dynamics
    np.testing.assert_array_equal(
        hamiltonian.dynamics(*call_args), hamiltonian.dynamics(**kwargs_shuffled)
    )

    # grad
    np.testing.assert_array_equal(
        hamiltonian.grad(*call_args), hamiltonian.grad(**kwargs_shuffled)
    )

    # hessian
    np.testing.assert_array_equal(
        hamiltonian.hessian(*call_args), hamiltonian.hessian(**kwargs_shuffled)
    )

    # csi
    assert hamiltonian.csi(*call_args) == hamiltonian.csi(**kwargs_shuffled)

    # auxderivs
    np.testing.assert_array_equal(
        hamiltonian.auxderivs(*call_args), hamiltonian.auxderivs(**kwargs_shuffled)
    )


@pytest.fixture
def hamiltonian_precessing(eob_params):
    hamiltonian = Ham_precessing_opt(eob_params)

    hamiltonian.calibration_coeffs.a6 = -12.69080059597502
    hamiltonian.calibration_coeffs.dSO = -33.369866493663814
    hamiltonian.calibration_coeffs.ddSO = 0

    hamiltonian.eob_params.p_params.M = 1
    hamiltonian.eob_params.p_params.nu = 0.23437499999999997
    hamiltonian.eob_params.p_params.X_1 = 0.625
    hamiltonian.eob_params.p_params.X_2 = 0.375

    hamiltonian.eob_params.p_params.chi1_L = 0.22617092
    hamiltonian.eob_params.p_params.chi_1 = 0.23327279836221107
    hamiltonian.eob_params.p_params.chi1_v = np.array(
        [0.25022559, 0.51326609, 0.23226074]
    )
    hamiltonian.eob_params.p_params.chi2_L = -0.16629132
    hamiltonian.eob_params.p_params.chi_2 = -0.1575542799917961
    hamiltonian.eob_params.p_params.chi2_v = np.array(
        [-0.36040643, 0.67769987, -0.02856655]
    )

    hamiltonian.eob_params.p_params.delta = 0.25000000000000006

    return hamiltonian


def test_hamiltonian_precessing_H_gt(hamiltonian_precessing):
    """Checks numerical values returned by H.__call__"""

    call_args = (
        np.array([2.9493765, 131.52477657]),
        np.array([0.0, 2.66907957]),
        np.array([0.25022559, 0.51326609, 0.23226074]),
        np.array([-0.36040643, 0.67769987, -0.02856655]),
        0.625,
        0.375,
        0.23327279836221107,
        -0.1575542799917961,
        0.22617092,
        -0.16629132,
    )

    H_val = (
        4.121621685302856,
        0.44541165678127637,
        0.4017643674822076,
        -0.5382888894397937,
        -0.18074114147514322,
        0.0,
        0.8490220671781932,
        0.008398347546140267,
        0.970872892854496,
    )
    np.testing.assert_array_almost_equal(
        hamiltonian_precessing(*call_args), np.array(H_val)
    )


def test_hamiltonian_precessing_omega_gt(hamiltonian_precessing):
    """Checks numerical values returned by H.omega"""

    call_args = (
        np.array([2.9493765, 131.52477657]),
        np.array([0.0, 2.66907957]),
        np.array([0.25022559, 0.51326609, 0.23226074]),
        np.array([-0.36040643, 0.67769987, -0.02856655]),
        0.625,
        0.375,
        0.23327279836221107,
        -0.1575542799917961,
        0.22617092,
        -0.16629132,
    )

    omega = 0.147808125558441
    assert abs(hamiltonian_precessing.omega(*call_args) - omega) < 1e-10, (
        hamiltonian_precessing.omega(*call_args),
        omega,
    )


def test_hamiltonian_precessing_csi_gt(hamiltonian_precessing):
    """Checks numerical values returned by H.csi"""

    call_args = (
        np.array([2.9493765, 131.52477657]),
        np.array([0.0, 2.66907957]),
        np.array([0.25022559, 0.51326609, 0.23226074]),
        np.array([-0.36040643, 0.67769987, -0.02856655]),
        0.625,
        0.375,
        0.23327279836221107,
        -0.1575542799917961,
        0.22617092,
        -0.16629132,
    )

    csi = 0.44541165678127637
    assert abs(hamiltonian_precessing.csi(*call_args) - csi) < 1e-10, (
        hamiltonian_precessing.csi(*call_args),
        csi,
    )


def test_hamiltonian_precessing_gradient_gt(hamiltonian_precessing):
    """Checks numerical values returned by H.grad"""

    call_args = (
        np.array([2.9493765, 131.52477657]),
        np.array([0.0, 2.66907957]),
        np.array([0.25022559, 0.51326609, 0.23226074]),
        np.array([-0.36040643, 0.67769987, -0.02856655]),
        0.625,
        0.375,
        0.23327279836221107,
        -0.1575542799917961,
        0.22617092,
        -0.16629132,
    )

    grad_gt = (0.02822711163597779, 0.0, 0.0, 0.14780812555844097)
    np.testing.assert_array_almost_equal(
        hamiltonian_precessing.grad(*call_args), grad_gt
    )


def test_hamiltonian_precessing_dynamics_gt(hamiltonian_precessing):
    """Checks numerical values returned by H.dynamics"""

    call_args = (
        np.array([2.9493765, 131.52477657]),
        np.array([0.0, 2.66907957]),
        np.array([0.25022559, 0.51326609, 0.23226074]),
        np.array([-0.36040643, 0.67769987, -0.02856655]),
        0.625,
        0.375,
        0.23327279836221107,
        -0.1575542799917961,
        0.22617092,
        -0.16629132,
    )

    dyn_gt = (
        0.02822711163597779,
        0.0,
        0.0,
        0.14780812555844097,
        4.121621685302856,
        0.44541165678127637,
    )
    np.testing.assert_array_almost_equal(
        hamiltonian_precessing.dynamics(*call_args), dyn_gt
    )


def test_hamiltonian_precessing_hessian_gt(hamiltonian_precessing):
    """Checks numerical values returned by H.hessian"""

    call_args = (
        np.array([2.9493765, 131.52477657]),
        np.array([0.0, 2.66907957]),
        np.array([0.25022559, 0.51326609, 0.23226074]),
        np.array([-0.36040643, 0.67769987, -0.02856655]),
        0.625,
        0.375,
        0.23327279836221107,
        -0.1575542799917961,
        0.22617092,
        -0.16629132,
    )

    hessian_gt = np.array(
        [
            [-0.0270075, 0.0, 0.0, -0.04951159],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.14004028, -0.0],
            [-0.04951159, 0.0, -0.0, 0.02566709],
        ]
    )
    np.testing.assert_array_almost_equal(
        hamiltonian_precessing.hessian(*call_args).ravel(), hessian_gt.ravel()
    )


def test_hamiltonian_precessing_auxderiv_gt(hamiltonian_precessing):
    """Checks numerical values returned by H.dynamics"""

    call_args = (
        np.array([2.9493765, 131.52477657]),
        np.array([0.0, 2.66907957]),
        np.array([0.25022559, 0.51326609, 0.23226074]),
        np.array([-0.36040643, 0.67769987, -0.02856655]),
        0.625,
        0.375,
        0.23327279836221107,
        -0.1575542799917961,
        0.22617092,
        -0.16629132,
    )

    auxderiv_gt = (
        0.14528764138411066,
        0.22695470398693696,
        0.13803046088145557,
        0.15132956113633944,
        -0.0,
        0.0,
        -0.0034893716270903916,
        0.023213209712333593,
        0.030756904931538005,
    )
    np.testing.assert_array_almost_equal(
        hamiltonian_precessing.auxderivs(*call_args), auxderiv_gt
    )
