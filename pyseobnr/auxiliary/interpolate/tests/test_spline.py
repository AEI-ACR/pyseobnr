import sys
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

sys.path.append("../")
from _cubic import myCubicSpline


@pytest.fixture()
def x_pol():
    return np.linspace(-10, 20, 20)


@pytest.fixture()
def x_new_pol():
    return np.linspace(-10, 20, 100)


@pytest.fixture()
def polynomial_indices(x_pol, x_new_pol):
    coeffs = [1, 2, 3]
    pol = np.poly1d(coeffs)
    y = pol(x_pol)
    intrp_mine = myCubicSpline(x_pol, y)
    my_result, indices = intrp_mine(x_new_pol)
    return indices


@pytest.fixture()
def x_trig():
    return np.linspace(-2 * np.pi, 3 * np.pi, 20)


@pytest.fixture()
def x_new_trig():
    return np.linspace(-2 * np.pi, 3 * np.pi, 100)


@pytest.fixture()
def trig_indices(x_trig, x_new_trig):

    y = np.sin(x_trig) ** 2 + 1.2 * np.cos(x_trig)
    intrp_mine = myCubicSpline(x_trig, y)
    my_result, indices = intrp_mine(x_new_trig)
    return indices


@pytest.mark.parametrize(
    "coeffs", [[6.0, 1.11, 1.42, 6.668], [3.0, 1.43, 8.09], [2.0, 14.97]]
)
def test_polynomial_noindices(x_pol, x_new_pol, coeffs):
    """
    Test that interpolation with no search indices still works correctly

    Args:
        coeffs (arrray): Coefficients of the polynomials we will interpolate
    """
    pol = np.poly1d(coeffs)
    y = pol(x_pol)
    intrp = CubicSpline(x_pol, y)
    intrp_mine = myCubicSpline(x_pol, y)
    scipy_result = intrp(x_new_pol)
    my_result, _ = intrp_mine(x_new_pol)
    assert np.allclose(scipy_result, my_result), "The arrays are different!"


@pytest.mark.parametrize(
    "coeffs", [[6.0, 1.11, 1.42, 6.668], [3.0, 1.43, 8.09], [2.0, 14.97]]
)
def test_polynomial_indices(x_pol, x_new_pol, polynomial_indices, coeffs):
    """
    Test that interpolation *with* search indices still works correctly

    Args:
        coeffs (arrray): Coefficients of the polynomials we will interpolate
    """
    pol = np.poly1d(coeffs)
    y = pol(x_pol)
    intrp = CubicSpline(x_pol, y)
    intrp_mine = myCubicSpline(x_pol, y, indices=polynomial_indices)
    scipy_result = intrp(x_new_pol)
    my_result, _ = intrp_mine(x_new_pol)
    assert np.allclose(scipy_result, my_result), "The arrays are different!"


def test_trig_noindices(x_trig, x_new_trig):
    """
    Test that interpolation with no search indices still works correctly
    """
    y = np.sin(x_trig) ** 2 + 1.2 * np.cos(x_trig)
    intrp = CubicSpline(x_trig, y)
    intrp_mine = myCubicSpline(x_trig, y)
    scipy_result = intrp(x_new_trig)
    my_result, _ = intrp_mine(x_new_trig)
    assert np.allclose(scipy_result, my_result), "The arrays are different!"


def test_trig_indices(x_trig, x_new_trig, trig_indices):
    """
    Test that interpolation *with* search indices still works correctly
    """
    y = np.sin(x_trig) ** 2 + 1.2 * np.cos(x_trig)
    intrp = CubicSpline(x_trig, y)
    intrp_mine = myCubicSpline(x_trig, y, indices=trig_indices)
    scipy_result = intrp(x_new_trig)
    my_result, _ = intrp_mine(x_new_trig)
    assert np.allclose(scipy_result, my_result), "The arrays are different!"
