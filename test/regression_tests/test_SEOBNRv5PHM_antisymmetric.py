from __future__ import annotations

import dataclasses
import os
from copy import deepcopy
from functools import cache
from itertools import combinations_with_replacement
from typing import Final, Literal
from unittest import mock

import numpy as np

import pytest

from pyseobnr.eob.fits.antisymmetric_modes import (
    PredictorSparseFits,
    get_fits_asymmetries,
)
from pyseobnr.eob.waveform.compute_antisymmetric import (
    apply_antisym_factorized_correction,
    compute_asymmetric_PN,
    fits_iv_mrd_antisymmetric,
    get_all_dynamics,
    get_params_for_fit,
    get_predictor_from_fits,
)
from pyseobnr.generate_waveform import generate_modes_opt


def test_load_fits():
    """Basic check for the loading of the fits files from python resources

    This should also be checked after packaging, which is normally the case when using tox
    as test driver.
    """
    assert get_fits_asymmetries() is not None
    assert set(get_fits_asymmetries().keys()) == {(2, 2), (3, 3), (4, 4)}


def test_fits_content():
    """various checks on the content of the fits and their shape"""
    # we may be mutating the returned dictionary (which is currently
    # possible) so we make a deepcopy. Should be replaced by immutable
    # dicts instead
    fits_asym = deepcopy(get_fits_asymmetries())
    modes_fits: Final = (2, 2), (3, 3), (4, 4)

    assert set(fits_asym) == set(modes_fits)

    @cache
    def get_combinations(dimension, max_degree):
        feature_combinations = []
        for i in range(1, max_degree + 1):
            for comb in combinations_with_replacement(range(dimension), i):
                feature_combinations.append(
                    {variable: comb.count(variable) for variable in set(comb)}
                )
        return feature_combinations

    dimensions_parts: Final = {
        "omegalm": 7,
        "omega_peak": 7,
        "habs": 7,
        "c1f": 5,
        "c2f": 6,
    }

    quantities_per_mode: Final = {
        (2, 2): set(dimensions_parts.keys()),
        (3, 3): {"omegalm"},
        (4, 4): {"omegalm"},
    }

    # checks the content is the same for all modes
    first_mode_quantities = None
    for current_mode in modes_fits:

        if first_mode_quantities is None:
            first_mode_quantities = fits_asym[current_mode]
            assert len(first_mode_quantities.keys()) > 0
            assert current_mode == (2, 2)  # should be first, see modes_fits
            assert "omega_peak" in first_mode_quantities

        assert set(fits_asym[current_mode].keys()) == quantities_per_mode[current_mode]

        for parts in fits_asym[current_mode]:
            assert set(fits_asym[current_mode][parts].keys()) == set(
                first_mode_quantities[parts].keys()
            )

            len_combination = len(
                get_combinations(
                    dimension=dimensions_parts[parts],
                    max_degree=int(first_mode_quantities[parts]["degree"]),
                )
            )

            assert len_combination + int(
                first_mode_quantities[parts]["include_bias"]
            ) == len(first_mode_quantities[parts]["coefficients"])


def test_PHM_with_antisymmetries_smoke():
    chi_1 = np.array((-0.0131941810847, 0.0676539715676, -0.646272606477))
    chi_2 = 0
    omega0 = 0.02

    with mock.patch(
        "pyseobnr.models.SEOBNRv5HM.apply_antisym_factorized_correction"
    ) as p_apply_antisym_factorized_correction:
        p_apply_antisym_factorized_correction.side_effect = (
            apply_antisym_factorized_correction
        )
        generate_modes_opt(
            q=5.32,
            chi1=chi_1,
            chi2=chi_2,
            omega_start=omega0,
            approximant="SEOBNRv5PHM",
            settings={"enable_antisymmetric_modes": True},
        )

        # basic check that we are actually running the anti-symmetric calculations
        p_apply_antisym_factorized_correction.assert_called_once()


def test_PHM_with_antisymmetries_smoke_all_modes():
    chi_1 = np.array((-0.0131941810847, 0.0676539715676, -0.646272606477))
    chi_2 = 0
    omega0 = 0.02

    with mock.patch(
        "pyseobnr.models.SEOBNRv5HM.apply_antisym_factorized_correction"
    ) as p_apply_antisym_factorized_correction:
        p_apply_antisym_factorized_correction.side_effect = (
            apply_antisym_factorized_correction
        )
        generate_modes_opt(
            q=5.32,
            chi1=chi_1,
            chi2=chi_2,
            omega_start=omega0,
            approximant="SEOBNRv5PHM",
            settings={
                "enable_antisymmetric_modes": True,
                "antisymmetric_modes": [(2, 2), (3, 3), (4, 4)],
            },
        )

        # basic check that we are actually running the anti-symmetric calculations
        p_apply_antisym_factorized_correction.assert_called_once()


def test_PHM_with_antisymmetries_does_not_mutate_fits():
    """Checks that the values of the fits are not mutated during a call to the antisymmetric modes"""
    chi_1 = np.array((-0.0131941810847, 0.0676539715676, -0.646272606477))
    chi_2 = 0
    omega0 = 0.02

    get_fits_asymmetries.cache_clear()
    fits_asym = deepcopy(get_fits_asymmetries())

    with mock.patch(
        "pyseobnr.models.SEOBNRv5HM.apply_antisym_factorized_correction"
    ) as p_apply_antisym_factorized_correction:
        p_apply_antisym_factorized_correction.side_effect = (
            apply_antisym_factorized_correction
        )
        generate_modes_opt(
            q=5.32,
            chi1=chi_1,
            chi2=chi_2,
            omega_start=omega0,
            approximant="SEOBNRv5PHM",
            settings={
                "enable_antisymmetric_modes": True,
                "antisymmetric_modes": [(2, 2), (3, 3), (4, 4)],
            },
        )
        # basic check that we are actually running the anti-symmetric calculations
        p_apply_antisym_factorized_correction.assert_called_once()

    after_call_fits = get_fits_asymmetries()
    assert set(fits_asym.keys()) == set(after_call_fits.keys())
    for k1 in fits_asym:
        assert set(fits_asym[k1].keys()) == set(after_call_fits[k1].keys())

        for k2 in fits_asym[k1].keys():
            assert set(fits_asym[k1][k2].keys()) == set(after_call_fits[k1][k2].keys())

            for k3 in fits_asym[k1][k2].keys():
                assert type(fits_asym[k1][k2][k3]) is type(after_call_fits[k1][k2][k3])
                assert isinstance(fits_asym[k1][k2][k3], float) or isinstance(
                    fits_asym[k1][k2][k3], np.ndarray
                )
                assert np.allclose(fits_asym[k1][k2][k3], after_call_fits[k1][k2][k3])


def test_predictor_sparse_fit():
    """Checks the basic functionality of the sparse fit on a toy example"""

    # without bias term
    max_degree = 1
    coefficients = [0, 2]
    mean = [0, 0]
    std = [10000, 1.1]

    monomial = PredictorSparseFits(
        monomial_coefficients=np.array(coefficients),
        mean=np.array(mean),
        std=np.array(std),
        intercept=0,
        dimension=2,
        max_degree=max_degree,
        polynomials_contain_bias=False,
    )
    assert monomial.intercept == 0

    # x=1, y=1
    predict1 = monomial.predict(np.array([1, 1]))
    predict2 = 2 * 1 / 1.1
    assert predict1 == predict2

    predict1 = monomial.predict(np.array([1, 3]))
    predict2 = 2 * 3 / 1.1
    assert predict1 == predict2

    # max degree = 2
    max_degree = 2
    coefficients = [0, 2, 0.3, 0.9, 3]
    mean = [0, 0, 0, 0, 0]
    std = [1, 1.1, 1, 0.5, 1]

    monomial = PredictorSparseFits(
        monomial_coefficients=np.array(coefficients),
        mean=np.array(mean),
        std=np.array(std),
        intercept=0,
        dimension=2,
        max_degree=max_degree,
        polynomials_contain_bias=False,
    )

    # x=1, y=1
    predict1 = monomial.predict(np.array([1, 1]))
    predict2 = 0 + 2 * 1 / 1.1 + 0.3 * 1**2 + 0.9 * 1 * 1 / 0.5 + 3 * 1**2
    assert predict1 == predict2

    predict1 = monomial.predict(np.array([2, 3]))
    predict2 = 0 + 2 * 3 / 1.1 + 0.3 * 2**2 + 0.9 * 2 * 3 / 0.5 + 3 * 3**2
    assert predict1 == predict2


def test_predictor_sparse_fit_with_bias_term_in_polynomial_feature():
    """Checks the basic functionality of the sparse fit on a toy example

    This time with bias term in the polynomial feature
    """
    # with bias term
    # bias + a1 x + a2 * y
    max_degree = 1
    coefficients = [1, 0, 2]
    mean = [0, 0, 0]
    std = [1, 10000, 1.1]

    monomial = PredictorSparseFits(
        monomial_coefficients=np.array(coefficients),
        mean=np.array(mean),
        std=np.array(std),
        intercept=0,
        dimension=2,
        max_degree=max_degree,
        polynomials_contain_bias=True,
    )
    # coefficient * (1-mean)/std = 1
    assert monomial.intercept == 1

    # x=1, y=1
    predict1 = monomial.predict(np.array([1, 1]))
    predict2 = 1 + 2 * 1 / 1.1
    assert predict1 == predict2

    predict1 = monomial.predict(np.array([1, 3]))
    predict2 = 1 + 2 * 3 / 1.1
    assert predict1 == predict2

    # max degree = 2
    max_degree = 2
    coefficients = [7, 0, 2, 0.3, 0.9, 3]
    mean = [-9, -10000000, 3, 3, -4, 0]
    std = [10, 1, 1.1, 1, 0.5, 1]

    monomial = PredictorSparseFits(
        monomial_coefficients=np.array(coefficients),
        mean=np.array(mean),
        std=np.array(std),
        intercept=3,
        dimension=2,
        max_degree=max_degree,
        polynomials_contain_bias=True,
    )
    # given intercept + coefficient * (1-mean)/std = 1
    assert monomial.intercept == 3 + 7 * (1 - (-9)) / 10

    # x=1, y=1
    predict1 = monomial.predict(np.array([1, 1]))
    predict2 = (
        3  #
        + 7 * (1 - (-9)) / 10  #
        + 0  #
        + 2 * (1 - 3) / 1.1  #
        + 0.3 * (1**2 - 3) / 1  #
        + 0.9 * (1 * 1 - (-4)) / 0.5  #
        + 3 * (1**2 - 0) / 1  #
    )
    assert predict1 == predict2

    predict1 = monomial.predict(np.array([2, 3]))
    predict2 = (
        3  #
        + 7 * (1 - (-9)) / 10  #
        + 0  #
        + 2 * (3 - 3) / 1.1  #
        + 0.3 * (2**2 - 3) / 1  #
        + 0.9 * (2 * 3 - (-4)) / 0.5  #
        + 3 * (3**2 - 0) / 1  #
    )
    assert predict1 == predict2


def _generate_polynomial_features(features, degree, include_bias):
    """Generate polynomial features manually for a single sample

    This is the original implementation that was superseded by PredictorSparseFits and
    used only in tests as reference.
    """
    n_features = len(features)
    feature_combinations = []
    for i in range(degree + 1):
        for comb in combinations_with_replacement(range(n_features), i):
            feature_combinations.append(comb)

    # Generate polynomial features array
    poly_features = np.ones(
        len(feature_combinations) - 1 + include_bias
    )  # Start with an array of ones for bias term
    for i, comb in enumerate(
        feature_combinations[1:], include_bias
    ):  # Skip the first combination as it's the bias term
        poly_features[i] = np.prod([features[j] for j in comb])

    return poly_features


def _predict(input_features, ell, emm, quantity):
    """Manually predict the output for given input features.

    This is the original implementation that was superseded by PredictorSparseFits
    """
    # Get fit
    fit = get_fits_asymmetries()[(ell, emm)][quantity]

    # Generate polynomial features
    poly_features = _generate_polynomial_features(
        input_features, fit["degree"], int(fit["include_bias"])
    )

    # Normalize the polynomial features
    normalized_features = (poly_features - fit["means"]) / fit["stds"]

    # Calculate the prediction
    prediction = fit["intercept"] + np.dot(fit["coefficients"], normalized_features)
    return prediction


def test_PHM_with_antisymmetries_smoke_fits_function():
    """Checks the predictions made by the antisymmetric fits wrt. reference implementation"""
    chi_1 = np.array((-0.0131941810847, 0.0676539715676, -0.646272606477))
    chi_2 = 0
    omega0 = 0.02

    # the mock is just to get the input parameters of the function
    class MyException(Exception):
        pass

    with mock.patch(
        "pyseobnr.models.SEOBNRv5HM.fits_iv_mrd_antisymmetric"
    ) as p_fits_iv_mrd_antisymmetric:
        p_fits_iv_mrd_antisymmetric.side_effect = MyException

        with pytest.raises(MyException):
            generate_modes_opt(
                q=5.32,
                chi1=chi_1,
                chi2=chi_2,
                omega_start=omega0,
                approximant="SEOBNRv5PHM",
                settings={"enable_antisymmetric_modes": True},
            )

        # basic check that we are actually running the anti-symmetric calculations
        p_fits_iv_mrd_antisymmetric.assert_called_once()
        input_parameters = p_fits_iv_mrd_antisymmetric.call_args.kwargs

    # if not true, the test should be adapted
    assert len(input_parameters) == 3
    kwarg, params_for_fit = list(input_parameters.items())[0]
    kwarg_nu, nu = list(input_parameters.items())[1]
    kwarg2, modes_to_compute = list(input_parameters.items())[2]

    assert modes_to_compute == [(2, 2)]

    Sigma_inplane = np.sqrt(params_for_fit.Sigma_n**2 + params_for_fit.Sigma_lamb**2)
    S_inplane = np.sqrt(params_for_fit.S_n**2 + params_for_fit.S_lamb**2)

    phiS = np.arctan2(params_for_fit.S_lamb, params_for_fit.S_n)
    cos_2phiS = np.cos(2 * phiS)
    sin_2phiS = np.sin(2 * phiS)

    phiSigma = np.arctan2(params_for_fit.Sigma_lamb, params_for_fit.Sigma_n)
    cos_2phiSigma = np.cos(2 * phiSigma)
    sin_2phiSigma = np.sin(2 * phiSigma)

    # Check omegalm, that has the same input features for all the modes
    X = np.array(
        [
            nu,
            S_inplane,
            Sigma_inplane,
            params_for_fit.chi_eff,
            params_for_fit.chi_a,
            cos_2phiSigma,
            sin_2phiSigma,
        ]
    )

    quantity = "omegalm"
    for ell, emm in (2, 2), (3, 3), (4, 4):
        fits = get_fits_asymmetries()[(ell, emm)][quantity]
        monomial = PredictorSparseFits(
            monomial_coefficients=fits["coefficients"],
            mean=fits["means"],
            std=fits["stds"],
            intercept=float(fits["intercept"]),
            dimension=X.shape[0],
            max_degree=fits["degree"],
            polynomials_contain_bias=bool(fits["include_bias"]),
        )
        predict1 = monomial.predict(X)
        predict2 = _predict(input_features=X, ell=ell, emm=emm, quantity=quantity)

        assert abs(predict1 - predict2) < 1e-15

    # special 22 fits

    # habs - same features as omegalm
    ell, emm = 2, 2
    quantity = "habs"
    fits = get_fits_asymmetries()[(ell, emm)][quantity]

    monomial = PredictorSparseFits(
        monomial_coefficients=fits["coefficients"],
        mean=fits["means"],
        std=fits["stds"],
        intercept=float(fits["intercept"]),
        dimension=X.shape[0],
        max_degree=fits["degree"],
        polynomials_contain_bias=bool(fits["include_bias"]),
    )

    predict1 = monomial.predict(X)
    predict2 = _predict(X, 2, 2, "habs")

    assert abs(predict1 - predict2) < 1e-15

    # Omega peak
    ell, emm = 2, 2
    quantity = "omega_peak"
    fits = get_fits_asymmetries()[(ell, emm)][quantity]

    X = np.array(
        [
            nu,
            params_for_fit.S_n,
            params_for_fit.S_lamb,
            params_for_fit.chi_eff,
            params_for_fit.Sigma_n,
            params_for_fit.Sigma_lamb,
            params_for_fit.chi_a,
        ]
    )
    monomial = PredictorSparseFits(
        monomial_coefficients=fits["coefficients"],
        mean=fits["means"],
        std=fits["stds"],
        intercept=float(fits["intercept"]),
        dimension=X.shape[0],
        max_degree=fits["degree"],
        polynomials_contain_bias=bool(fits["include_bias"]),
    )

    predict1 = monomial.predict(X)
    predict2 = _predict(X, 2, 2, "omega_peak")

    assert abs(predict1 - predict2) < 1e-15

    # c1f
    ell, emm = 2, 2
    quantity = "c1f"
    fits = get_fits_asymmetries()[(ell, emm)][quantity]

    X = np.array(
        [
            nu,
            params_for_fit.chi_eff,
            params_for_fit.chi_a,
            cos_2phiS,
            sin_2phiS,
        ]
    )
    monomial = PredictorSparseFits(
        monomial_coefficients=fits["coefficients"],
        mean=fits["means"],
        std=fits["stds"],
        intercept=float(fits["intercept"]),
        dimension=X.shape[0],
        max_degree=fits["degree"],
        polynomials_contain_bias=bool(fits["include_bias"]),
    )

    predict1 = monomial.predict(X)
    predict2 = _predict(X, 2, 2, "c1f")

    assert abs(predict1 - predict2) < 1e-15

    # c2f
    ell, emm = 2, 2
    quantity = "c2f"
    fits = get_fits_asymmetries()[(ell, emm)][quantity]

    X = np.array(
        [
            nu,
            predict1,  # This fit depends on the previous coefficient, c1f
            params_for_fit.chi_eff,
            params_for_fit.chi_a,
            cos_2phiS,
            sin_2phiS,
        ]
    )
    monomial = PredictorSparseFits(
        monomial_coefficients=fits["coefficients"],
        mean=fits["means"],
        std=fits["stds"],
        intercept=float(fits["intercept"]),
        dimension=X.shape[0],
        max_degree=fits["degree"],
        polynomials_contain_bias=bool(fits["include_bias"]),
    )

    predict1 = monomial.predict(X)
    predict2 = _predict(X, 2, 2, "c2f")

    assert abs(predict1 - predict2) < 1e-15


def _generate_dynamics_array(a1, a2):
    q = 1.0

    # a1 = 0.99
    ph1 = 0.0

    # a2 = 0.99
    ph2 = 0.0

    mtot = 150.0

    t1 = np.pi / 5
    t2 = t1
    omega0 = 0.018
    delta_t = 1.0 / 16384.0

    chiA = a1 * np.array(
        [np.sin(t1) * np.cos(ph1), np.sin(t1) * np.sin(ph1), np.cos(t1)]
    )
    chiB = a2 * np.array(
        [np.sin(t2) * np.cos(ph2), np.sin(t2) * np.sin(ph2), np.cos(t2)]
    )

    settings = {
        "ell_max": 4,
        "M": mtot,
        "dt": delta_t,
        "postadiabatic": True,
        "return_coprec": True,
        "postadiabatic_type": "analytic",
        "enable_antisymmetric_modes": True,
        "ivs_mrd": None,
        "antisymmetric_modes": [(2, 2), (3, 3), (4, 4)],
    }

    with mock.patch(
        "pyseobnr.models.SEOBNRv5HM.get_all_dynamics"
    ) as p_get_all_dynamics, mock.patch(
        "pyseobnr.models.SEOBNRv5HM.get_params_for_fit"
    ) as p_get_params_for_fit:

        class RunGetAllDynamics(Exception):
            pass

        # original function, we need some of the input values
        p_get_all_dynamics.side_effect = get_all_dynamics
        # we stop here
        p_get_params_for_fit.side_effect = RunGetAllDynamics

        with pytest.raises(RunGetAllDynamics):
            _ = generate_modes_opt(
                q,
                chiA,
                chiB,
                omega0,
                approximant="SEOBNRv5PHM",
                settings=settings,
                debug=True,
            )

        p_get_all_dynamics.assert_called_once()
        p_get_params_for_fit.assert_called_once()
        dyn_EOB, t_array, m1, m2 = tuple(
            p_get_all_dynamics.call_args.kwargs[_] for _ in ("dyn", "t", "mA", "mB")
        )
        t_attach = p_get_params_for_fit.call_args_list[0][1]["t_attach"]

        assert "Sigma_n" not in dyn_EOB

    return dict(
        q=q,
        chiA=chiA,
        chiB=chiB,
        omega0=omega0,
        dyn_EOB=dyn_EOB,
        t_array=t_array,
        m1=m1,
        m2=m2,
        t_attach=t_attach,
        settings=settings,
    )


def test_nqc_flag_computation_for_antisymmetric_modes():
    """Checks for a regression in the calculation of the flag for the NQCs application to antisym modes"""
    # in particular, if this is not correctly done, the following test yields a 2,2 antisymmetric
    # mode with zeros only (checked as well), which in turn fails the function
    # EOBCalculateNQCCoefficients_freeattach_asym when it comes to the calculation of the solution
    # of Q*coeffs = amps (singular matrix error).

    dict_internal_configuration = _generate_dynamics_array(a1=0.99, a2=0.99)

    m1 = dict_internal_configuration["m1"]
    m2 = dict_internal_configuration["m2"]
    q = dict_internal_configuration["q"]
    chiA = dict_internal_configuration["chiA"]
    chiB = dict_internal_configuration["chiB"]
    omega0 = dict_internal_configuration["omega0"]
    t_attach = dict_internal_configuration["t_attach"]
    t_array = dict_internal_configuration["t_array"]
    dyn_EOB = dict_internal_configuration["dyn_EOB"]
    settings = dict_internal_configuration["settings"]

    new_dyn_array = get_all_dynamics(dyn_EOB, t_array, m1, m2)

    np.testing.assert_almost_equal(
        new_dyn_array["Sigma_n"], np.zeros_like(new_dyn_array["Sigma_n"])
    )

    anti_symmetric_modes = compute_asymmetric_PN(
        dyn=new_dyn_array,
        mA=m1,
        mB=m2,
        modes_to_compute=settings["antisymmetric_modes"],
        nlo22=True,
    )

    # the 22 mode should be zero in this case
    assert (2, 2) in settings["antisymmetric_modes"]

    np.testing.assert_allclose(np.abs(anti_symmetric_modes[(2, 2)]), 0, atol=1e-12)

    params_for_fit_asym = get_params_for_fit(
        dyn_all=new_dyn_array,
        t=t_array,
        mA=m1,
        mB=m2,
        q=q,
        t_attach=t_attach,
    )

    # run the fits
    ivs_asym, mrd_ivs = fits_iv_mrd_antisymmetric(
        params_for_fits=params_for_fit_asym,
        nu=m1 * m2,
        modes_to_compute=settings["antisymmetric_modes"],
    )

    assert ivs_asym is not None
    assert mrd_ivs is not None

    # finally we get to the point where we compute the flag indicating if the NQC should
    # be applied: this is ultimately what we wanted to check
    nqc_flags = apply_antisym_factorized_correction(
        antisym_modes=anti_symmetric_modes,
        v_orb=new_dyn_array["v"],
        ivs_asym=ivs_asym,
        idx_attach=params_for_fit_asym.idx_attach,
        t=new_dyn_array["t"],
        t_attach=t_attach,
        nu=m1 * m2,
        corr_power=6,
        interpolation_step_back=10,
    )

    # For this configuration, we expect that only the (3,3) flag is True
    assert nqc_flags[2, 2] is False
    assert nqc_flags[3, 3] is True
    assert nqc_flags[4, 4] is False

    # if all the previous is checked, then it means we can run the full model
    # calculation as well:
    with mock.patch(
        "pyseobnr.models.SEOBNRv5HM.apply_nqc_phase_antisymmetric"
    ) as p_nqc_asym:
        _ = generate_modes_opt(
            q,
            chiA,
            chiB,
            omega0,
            approximant="SEOBNRv5PHM",
            settings=settings,
            debug=True,
        )

        p_nqc_asym.assert_called()


def test_get_params_for_fit_returned_content_type():
    """Checks for a regression in the handling of the attachment time in get_params_for_fit"""

    dict_internal_configuration = _generate_dynamics_array(a1=0.99, a2=0.99)

    m1 = dict_internal_configuration["m1"]
    m2 = dict_internal_configuration["m2"]
    q = dict_internal_configuration["q"]

    t_array = dict_internal_configuration["t_array"]
    dyn_EOB = dict_internal_configuration["dyn_EOB"]

    new_dyn_array = get_all_dynamics(dyn_EOB, t_array, m1, m2)

    params_for_fit_asym_tm1 = get_params_for_fit(
        dyn_all=new_dyn_array,
        t=t_array,
        mA=m1,
        mB=m2,
        q=q,
        t_attach=t_array[-1],  # simulating the last point in the dynamic
    )

    for content_key in dataclasses.asdict(params_for_fit_asym_tm1).keys():
        # all float or int values
        assert (
            type(getattr(params_for_fit_asym_tm1, content_key)) is int
            or type(getattr(params_for_fit_asym_tm1, content_key)) is float
        )


def test_get_params_for_fit_t_attach_index():
    """Checks for a regression in the handling of the attachment time in get_params_for_fit"""
    # shouldn't be the same spin values (changing the magnitude here) otherwise the S_n ... will
    # be 0 and the test a bit moot.
    dict_internal_configuration = _generate_dynamics_array(a1=0.99, a2=0.95)

    m1 = dict_internal_configuration["m1"]
    m2 = dict_internal_configuration["m2"]
    q = dict_internal_configuration["q"]

    t_array = dict_internal_configuration["t_array"]
    dyn_EOB = dict_internal_configuration["dyn_EOB"]

    new_dyn_array = get_all_dynamics(dyn_EOB, t_array, m1, m2)

    params_for_fit_asym_tm1 = get_params_for_fit(
        dyn_all=new_dyn_array,
        t=t_array,
        mA=m1,
        mB=m2,
        q=q,
        t_attach=t_array[-1],  # simulating the last point in the dynamic
    )

    params_for_fit_asym_tm1_plusa10 = get_params_for_fit(
        dyn_all=new_dyn_array,
        t=t_array,
        mA=m1,
        mB=m2,
        q=q,
        t_attach=t_array[-1] + 10,
    )

    for content_key in dataclasses.asdict(params_for_fit_asym_tm1).keys():
        assert getattr(params_for_fit_asym_tm1, content_key) == getattr(
            params_for_fit_asym_tm1_plusa10, content_key
        )

    # checks the attachment
    assert params_for_fit_asym_tm1.idx_attach == len(t_array) - 1

    params_for_fit_asym_tm2 = get_params_for_fit(
        dyn_all=new_dyn_array,
        t=t_array,
        mA=m1,
        mB=m2,
        q=q,
        t_attach=t_array[-2],  # second to last point
    )

    assert params_for_fit_asym_tm2.idx_attach == len(t_array) - 2
    for content_key in dataclasses.asdict(params_for_fit_asym_tm1).keys():
        if content_key in ["mA", "mB"]:
            continue
        assert getattr(params_for_fit_asym_tm1, content_key) != getattr(
            params_for_fit_asym_tm2, content_key
        )


def test_apply_antisym_factorized_correction():
    """Check that we are not going beyond the dynamics in the function"""

    # different spin magnitudes for different spin values
    dict_internal_configuration = _generate_dynamics_array(a1=0.99, a2=0.95)

    m1 = dict_internal_configuration["m1"]
    m2 = dict_internal_configuration["m2"]
    q = dict_internal_configuration["q"]
    t_array = dict_internal_configuration["t_array"]
    dyn_EOB = dict_internal_configuration["dyn_EOB"]
    settings = dict_internal_configuration["settings"]

    new_dyn_array = get_all_dynamics(dyn_EOB, t_array, m1, m2)

    anti_symmetric_modes = compute_asymmetric_PN(
        dyn=new_dyn_array,
        mA=m1,
        mB=m2,
        modes_to_compute=settings["antisymmetric_modes"],
        nlo22=True,
    )

    # we check at an attachment time at the boundary of the dynamics (and beyond)
    fake_t_attach = t_array[-1]
    idx_attach = len(t_array) - 1

    params_for_fit_asym = get_params_for_fit(
        dyn_all=new_dyn_array,
        t=t_array,
        mA=m1,
        mB=m2,
        q=q,
        t_attach=fake_t_attach,
    )

    # run the fits
    ivs_asym, mrd_ivs = fits_iv_mrd_antisymmetric(
        params_for_fits=params_for_fit_asym,
        nu=m1 * m2,
        modes_to_compute=settings["antisymmetric_modes"],
    )

    assert ivs_asym is not None
    assert mrd_ivs is not None

    anti_symmetric_modes_tm1 = deepcopy(anti_symmetric_modes)
    nqc_flags_tm1 = apply_antisym_factorized_correction(
        antisym_modes=anti_symmetric_modes_tm1,
        v_orb=new_dyn_array["v"],
        ivs_asym=ivs_asym,
        idx_attach=idx_attach,
        t=new_dyn_array["t"],
        t_attach=fake_t_attach,  # simulating attachment time at last point
        nu=m1 * m2,
        corr_power=6,
        interpolation_step_back=10,
        modes_to_apply=[(2, 2), (3, 3), (4, 4)],
    )

    anti_symmetric_modes_tm1_plus10 = deepcopy(anti_symmetric_modes)
    nqc_flags_tm1_plus10 = apply_antisym_factorized_correction(
        antisym_modes=anti_symmetric_modes_tm1_plus10,
        v_orb=new_dyn_array["v"],
        ivs_asym=ivs_asym,
        idx_attach=idx_attach,
        t=new_dyn_array["t"],
        # simulating an extrapolation of the attachment time
        t_attach=fake_t_attach + 10,
        nu=m1 * m2,
        corr_power=6,
        interpolation_step_back=10,
        modes_to_apply=[(2, 2), (3, 3), (4, 4)],
    )

    assert nqc_flags_tm1.keys() == nqc_flags_tm1_plus10.keys()
    for mode_ell_m in nqc_flags_tm1:
        # we should have the same behaviour wrt. the NQC flags
        assert nqc_flags_tm1[mode_ell_m] == nqc_flags_tm1_plus10[mode_ell_m]
        # the arrays should go through the same corrections
        assert np.array_equal(
            anti_symmetric_modes_tm1[mode_ell_m],
            anti_symmetric_modes_tm1_plus10[mode_ell_m],
        )

    # checks that this is actually doing something on the input/input array
    for mode_ell_m in nqc_flags_tm1:
        assert not np.array_equal(
            anti_symmetric_modes_tm1[mode_ell_m],
            anti_symmetric_modes[mode_ell_m],
        )

    anti_symmetric_modes_tm1_minus_epsilon = deepcopy(anti_symmetric_modes)
    nqc_flags_tm1_minus_epsilon = apply_antisym_factorized_correction(
        antisym_modes=anti_symmetric_modes_tm1_minus_epsilon,
        v_orb=new_dyn_array["v"],
        ivs_asym=ivs_asym,
        idx_attach=idx_attach - 1,
        t=new_dyn_array["t"],
        t_attach=fake_t_attach - 0.1 * (t_array[-1] - t_array[-2]),
        nu=m1 * m2,
        corr_power=6,
        interpolation_step_back=10,
        modes_to_apply=[(2, 2), (3, 3), (4, 4)],
    )

    assert nqc_flags_tm1.keys() == nqc_flags_tm1_minus_epsilon.keys()
    for mode_ell_m in nqc_flags_tm1:
        # here this is not enforced
        assert np.array_equal(
            nqc_flags_tm1[mode_ell_m], nqc_flags_tm1_minus_epsilon[mode_ell_m]
        )

        # the arrays should go through the same corrections
        assert not np.array_equal(
            anti_symmetric_modes_tm1[mode_ell_m],
            anti_symmetric_modes_tm1_minus_epsilon[mode_ell_m],
        )


def test_apply_antisym_factorized_correction_sanity_checks():
    """Checks some error handling in the function"""

    # different spin magnitudes for different spin values
    dict_internal_configuration = _generate_dynamics_array(a1=0.99, a2=0.95)

    m1 = dict_internal_configuration["m1"]
    m2 = dict_internal_configuration["m2"]
    q = dict_internal_configuration["q"]
    t_array = dict_internal_configuration["t_array"]
    dyn_EOB = dict_internal_configuration["dyn_EOB"]
    settings = dict_internal_configuration["settings"]

    new_dyn_array = get_all_dynamics(dyn_EOB, t_array, m1, m2)

    anti_symmetric_modes = compute_asymmetric_PN(
        dyn=new_dyn_array,
        mA=m1,
        mB=m2,
        modes_to_compute=settings["antisymmetric_modes"],
        nlo22=True,
    )

    # we check at an attachment time at the boundary of the dynamics (and beyond)
    fake_t_attach = t_array[-1]
    idx_attach = len(t_array) - 1

    params_for_fit_asym = get_params_for_fit(
        dyn_all=new_dyn_array,
        t=t_array,
        mA=m1,
        mB=m2,
        q=q,
        t_attach=fake_t_attach,
    )

    # run the fits
    ivs_asym, mrd_ivs = fits_iv_mrd_antisymmetric(
        params_for_fits=params_for_fit_asym,
        nu=m1 * m2,
        modes_to_compute=settings["antisymmetric_modes"],
    )

    assert ivs_asym is not None
    assert mrd_ivs is not None

    anti_symmetric_modes_tm1 = deepcopy(anti_symmetric_modes)

    with pytest.raises(AssertionError):
        # duplicate mode
        _ = apply_antisym_factorized_correction(
            antisym_modes=anti_symmetric_modes_tm1,
            v_orb=new_dyn_array["v"],
            ivs_asym=ivs_asym,
            idx_attach=idx_attach,
            t=new_dyn_array["t"],
            t_attach=fake_t_attach,
            nu=m1 * m2,
            corr_power=6,
            interpolation_step_back=10,
            modes_to_apply=[(2, 2), (3, 3), (4, 4), (2, 2)],
        )

    with pytest.raises(AssertionError):
        # index outside of array
        _ = apply_antisym_factorized_correction(
            antisym_modes=anti_symmetric_modes_tm1,
            v_orb=new_dyn_array["v"],
            ivs_asym=ivs_asym,
            idx_attach=len(t_array) + 10,
            t=new_dyn_array["t"],
            t_attach=fake_t_attach,
            nu=m1 * m2,
            corr_power=6,
            interpolation_step_back=10,
            modes_to_apply=[(2, 2), (3, 3), (4, 4), (2, 2)],
        )

    with pytest.raises(AssertionError):
        # index outside of array
        _ = apply_antisym_factorized_correction(
            antisym_modes=anti_symmetric_modes_tm1,
            v_orb=new_dyn_array["v"],
            ivs_asym=ivs_asym,
            idx_attach=len(t_array),
            t=new_dyn_array["t"],
            t_attach=fake_t_attach,
            nu=m1 * m2,
            corr_power=6,
            interpolation_step_back=10,
            modes_to_apply=[(2, 2), (3, 3), (4, 4), (2, 2)],
        )

    with pytest.raises(AssertionError):
        # incorrect modes
        _ = apply_antisym_factorized_correction(
            antisym_modes=anti_symmetric_modes_tm1,
            v_orb=new_dyn_array["v"],
            ivs_asym=ivs_asym,
            idx_attach=len(t_array),
            t=new_dyn_array["t"],
            t_attach=fake_t_attach,  # simulating attachment time at last point
            nu=m1 * m2,
            corr_power=6,
            interpolation_step_back=10,
            modes_to_apply=[(2, 3), (3, 3), (4, 4)],
        )


def test_apply_antisym_factorized_correction_only_on_selected_modes():
    """Checks that computations are done only on the selected modes"""

    # different spin magnitudes for different spin values
    dict_internal_configuration = _generate_dynamics_array(a1=0.99, a2=0.95)

    m1 = dict_internal_configuration["m1"]
    m2 = dict_internal_configuration["m2"]
    q = dict_internal_configuration["q"]
    t_array = dict_internal_configuration["t_array"]
    dyn_EOB = dict_internal_configuration["dyn_EOB"]
    settings = dict_internal_configuration["settings"]
    t_attach = dict_internal_configuration["t_attach"]

    new_dyn_array = get_all_dynamics(dyn_EOB, t_array, m1, m2)

    anti_symmetric_modes_initial = compute_asymmetric_PN(
        dyn=new_dyn_array,
        mA=m1,
        mB=m2,
        modes_to_compute=settings["antisymmetric_modes"],
        nlo22=True,
    )

    params_for_fit_asym = get_params_for_fit(
        dyn_all=new_dyn_array,
        t=t_array,
        mA=m1,
        mB=m2,
        q=q,
        t_attach=t_attach,
    )

    # run the fits
    ivs_asym, mrd_ivs = fits_iv_mrd_antisymmetric(
        params_for_fits=params_for_fit_asym,
        nu=m1 * m2,
        modes_to_compute=settings["antisymmetric_modes"],
    )

    assert ivs_asym is not None
    assert mrd_ivs is not None

    for modes_to_apply in [(2, 2), (3, 3), (4, 4)], [(2, 2)]:
        anti_symmetric_modes_current = deepcopy(anti_symmetric_modes_initial)

        nqc_flags = apply_antisym_factorized_correction(
            antisym_modes=anti_symmetric_modes_current,
            v_orb=new_dyn_array["v"],
            ivs_asym=ivs_asym,
            idx_attach=params_for_fit_asym.idx_attach,
            t=new_dyn_array["t"],
            t_attach=t_attach,
            nu=m1 * m2,
            corr_power=6,
            interpolation_step_back=10,
            modes_to_apply=modes_to_apply,
        )

        assert anti_symmetric_modes_current.keys() == nqc_flags.keys()

        for mode in anti_symmetric_modes_current.keys():
            if mode in modes_to_apply:
                assert not np.array_equal(
                    anti_symmetric_modes_current[mode],
                    anti_symmetric_modes_initial[mode],
                )

            else:
                assert np.array_equal(
                    anti_symmetric_modes_current[mode],
                    anti_symmetric_modes_initial[mode],
                )


def test_apply_antisym_factorized_correction_fit_only_on_selected_modes():
    """Checks that the fits are calculated only on the selected modes"""

    # different spin magnitudes for different spin values
    dict_internal_configuration = _generate_dynamics_array(a1=0.99, a2=0.95)

    m1 = dict_internal_configuration["m1"]
    m2 = dict_internal_configuration["m2"]
    q = dict_internal_configuration["q"]
    t_array = dict_internal_configuration["t_array"]
    dyn_EOB = dict_internal_configuration["dyn_EOB"]
    t_attach = dict_internal_configuration["t_attach"]

    new_dyn_array = get_all_dynamics(dyn_EOB, t_array, m1, m2)

    params_for_fit_asym = get_params_for_fit(
        dyn_all=new_dyn_array,
        t=t_array,
        mA=m1,
        mB=m2,
        q=q,
        t_attach=t_attach,
    )

    # run the fits
    from pyseobnr.eob.fits.antisymmetric_modes import PredictorSparseFits

    original_predict = PredictorSparseFits.predict

    with mock.patch.object(PredictorSparseFits, "predict", autospec=True) as p_predict:
        p_predict.side_effect = original_predict
        ivs_asym, mrd_ivs = fits_iv_mrd_antisymmetric(
            params_for_fits=params_for_fit_asym, nu=m1 * m2, modes_to_compute=[(2, 2)]
        )

        p_predict.assert_called()
        assert p_predict.call_count == 5
        k: Literal["amp", "omega"]
        for k in ivs_asym:
            assert set(ivs_asym[k].keys()) == {(2, 2)}

        p_predict.reset_mock()
        ivs_asym, mrd_ivs = fits_iv_mrd_antisymmetric(
            params_for_fits=params_for_fit_asym, nu=m1 * m2, modes_to_compute=[(3, 3)]
        )

        p_predict.assert_called()
        assert p_predict.call_count == 1
        for k in ivs_asym:
            assert set(ivs_asym[k].keys()) == {(3, 3)}

        p_predict.reset_mock()
        ivs_asym, mrd_ivs = fits_iv_mrd_antisymmetric(
            params_for_fits=params_for_fit_asym, nu=m1 * m2, modes_to_compute=[(4, 4)]
        )

        p_predict.assert_called()
        assert p_predict.call_count == 1
        for k in ivs_asym:
            assert set(ivs_asym[k].keys()) == {(4, 4)}


def test_check_quaternions():
    """Checks the correctness of the rotations quaternions"""

    # different spin magnitudes for different spin values
    dict_internal_configuration = _generate_dynamics_array(a1=0.99, a2=0.95)

    m1 = dict_internal_configuration["m1"]
    m2 = dict_internal_configuration["m2"]
    t_array = dict_internal_configuration["t_array"]
    dyn_EOB = dict_internal_configuration["dyn_EOB"]

    assert "q_copr" in dyn_EOB
    q_norm = [_.norm() for _ in dyn_EOB["q_copr"]]
    assert 1 - min(q_norm) < 1e-7
    assert max(q_norm) <= 1 + 1e-7

    assert "L_N" in dyn_EOB
    L_n_norm = [np.linalg.norm(_) for _ in dyn_EOB["L_N"]]
    assert 1 - min(L_n_norm) < 1e-5
    assert max(L_n_norm) <= 1 + 1e-7

    new_dyn_array = get_all_dynamics(dyn_EOB, t_array, m1, m2)

    assert "n_hat" in new_dyn_array
    assert len(dyn_EOB["L_N"]) == len(new_dyn_array["n_hat"])

    # the vectors should be orthogonal, the test on the inner product is a bit loose on
    # purpose
    inner_prod = np.sum((dyn_EOB["L_N"] * new_dyn_array["n_hat"]), axis=1)
    assert len(inner_prod) == len(dyn_EOB["L_N"])
    assert np.max(np.abs(inner_prod)) < 1e-2


@pytest.mark.skipif(
    "CI_TEST_DYNAMIC_REGRESSIONS" not in os.environ,
    reason="tests are for specific platforms only",
)
def test_fit_values():
    """Some tests on the fits for known input/output"""

    def check_tolerance(current, ground_truth, tolerance=1e-7):
        assert abs(current - ground_truth) < tolerance

    # checking 2,2
    X = np.array(
        [0.25, 0.28507627, 0.0119466, 0.78474606, 0.01604184, -0.67557996, -0.73728673]
    )
    predict_habs_22 = get_predictor_from_fits(
        nb_dimensions=X.shape[0], ell=2, emm=2, quantity="habs"
    ).predict(X)
    np.testing.assert_almost_equal(predict_habs_22, 0.6615414214584113)

    X = np.array(
        [0.25, -0.11312589, 0.26166966, 0.78474606, 0.00481153, -0.01093482, 0.01604184]
    )
    predict_22_omega_peak = get_predictor_from_fits(
        nb_dimensions=X.shape[0], ell=2, emm=2, quantity="omega_peak"
    ).predict(X)
    np.testing.assert_almost_equal(predict_22_omega_peak, 0.19773037345690228)

    X = np.array(
        [0.25, 0.28507627, 0.0119466, 0.78474606, 0.01604184, -0.67557996, -0.73728673]
    )

    predict_omegalm = get_predictor_from_fits(
        nb_dimensions=X.shape[0], ell=2, emm=2, quantity="omegalm"
    ).predict(X)
    np.testing.assert_almost_equal(predict_omegalm, 0.07828906097971013)

    X = np.array([0.25, 0.78474606, 0.01604184, -0.68505705, -0.72848942])
    predict_c1f_22 = get_predictor_from_fits(
        nb_dimensions=X.shape[0], ell=2, emm=2, quantity="c1f"
    ).predict(X)
    np.testing.assert_almost_equal(predict_c1f_22, 0.06657660512733413)

    X = np.array([0.25, 0.06657661, 0.78474606, 0.01604184, -0.68505705, -0.72848942])
    predict_c2f_22 = get_predictor_from_fits(
        nb_dimensions=X.shape[0], ell=2, emm=2, quantity="c2f"
    ).predict(X)
    np.testing.assert_almost_equal(predict_c2f_22, -1.113733705750224)

    # checking 3,3
    X = np.array(
        [0.25, 0.28507627, 0.0119466, 0.78474606, 0.01604184, -0.67557996, -0.73728673]
    )
    predict_omega_33 = get_predictor_from_fits(
        nb_dimensions=X.shape[0], ell=3, emm=3, quantity="omegalm"
    ).predict(X)
    np.testing.assert_almost_equal(predict_omega_33, 0.5070458020832828)

    # checking 4,4
    X = np.array(
        [0.25, 0.28507627, 0.0119466, 0.78474606, 0.01604184, -0.67557996, -0.73728673]
    )
    predict_omega_44 = get_predictor_from_fits(
        nb_dimensions=X.shape[0], ell=4, emm=4, quantity="omegalm"
    ).predict(X)
    np.testing.assert_almost_equal(predict_omega_44, 0.6524849579595458)
