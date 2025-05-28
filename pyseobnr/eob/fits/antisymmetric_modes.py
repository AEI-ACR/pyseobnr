"""Functions for loading the anti-symmetric fits and running the corresponding predictions"""

from __future__ import annotations

import io
import sys
from functools import cache
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

if sys.version_info < (3, 10):
    import importlib_resources  # noqa
else:
    import importlib.resources as importlib_resources


@cache
def get_fits_asymmetries() -> dict:
    """Reads the fits file for the antisymmetric modes

    The returned dictionary is organized by modes and quantities.
    """
    # warning: the returned cached dictionary is mutable
    fits_asym = {}
    for ell, emm in [(2, 2), (3, 3), (4, 4)]:
        fits_asym[(ell, emm)] = {}
        pkg_dir = (
            importlib_resources.files("pyseobnr.eob.fits.asym_fits")
            / f"fits_{ell}{emm}"
        )
        ff: Path
        for ff in pkg_dir.iterdir():

            if not ff.is_file():
                continue

            if ff.suffix not in [".npz"]:
                continue

            content = ff.read_bytes()
            quantity = ff.stem

            fits_asym[(ell, emm)][quantity] = {}

            data = np.load(io.BytesIO(content))

            for key in data.files:
                fits_asym[(ell, emm)][quantity][key] = data[key]

    return fits_asym


class PredictorSparseFits:
    """Helper class to calculate the predictions from the fits

    As the coefficients are sparse, the implementation performs the calculations only
    on the active monomials (coefficients != 0).

    The fitting pipeline is made of 3 stages in order:

    #. polynomial fit on the input data:
    #. scaling: mean and std
    #. matching pursuit: gives the coefficients and the intercept term

    The scaling and matching pursuit will see the coefficients from the output of the
    polynomial, and the bias term of the polynomial will propagate further to the mean/std
    and coefficients of the matching pursuit
    """

    def __init__(
        self,
        monomial_coefficients: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        intercept: float,
        dimension: int,
        max_degree: int,
        polynomials_contain_bias: bool = False,
    ):
        assert monomial_coefficients.ndim == 1
        assert mean.shape == std.shape and mean.shape == monomial_coefficients.shape

        self.polynomials_contain_bias = polynomials_contain_bias
        self.intercept = float(intercept)
        self.dimension = int(dimension)
        self.max_degree = int(max_degree)

        start_index = 1 if self.polynomials_contain_bias else 0
        self.original_monomial_coefficients = monomial_coefficients

        # only a few coefficients are non zero, we index those
        self.indices = np.flatnonzero(self.original_monomial_coefficients[start_index:])

        self.nonzero_coefficients = self.original_monomial_coefficients[start_index:][
            self.indices
        ]
        self.nonzero_coefficients_mean = mean[start_index:][self.indices]
        self.nonzero_coefficients_std = std[start_index:][self.indices]
        self.scaled_nonzero_coefficients = (
            self.nonzero_coefficients / self.nonzero_coefficients_std
        )

        # the monomials never include the bias term (where all powers are 0)
        # the bias term has been discarded above from the coefficients (via start_index)
        # and is included in the global intercept below
        monomials = self.get_monomials_coefficients(
            dimension=self.dimension, max_degree=self.max_degree
        )
        assert len(monomials) == len(self.original_monomial_coefficients) - start_index

        self.monomials: list[dict[int, int]] = [
            monomials[idx_non_zero_monomial] for idx_non_zero_monomial in self.indices
        ]

        # if there is an intercept term in the polynomials, we make it part of the intercept
        # term of the matching pursuit
        if self.polynomials_contain_bias:
            self.intercept += (
                self.original_monomial_coefficients[0] * (1 - mean[0]) / std[0]
            )

        self.actual_max_degree = max(max(_.values()) for _ in self.monomials)

    @staticmethod
    @cache
    def get_monomials_coefficients(
        dimension: int, max_degree: int
    ) -> list[dict[int, int]]:
        """Returns the sequence of monomials coefficients.

        The order follows the same as the one of the ``PolynomialFeatures`` from
        the ``scikit-learn`` package (used for producing the fits). Each monomial
        is described by a dictionary, where the keys are the feature dimension and the values
        are their corresponding power in the monomial: ``{0:3, 2:5}`` would mean
        :math:`x^3 + z^5` if ``x`` identifies the first dimension (index 0) and ``z``
        the third.

        The intercept (constant term) is not returned.
        """
        if max_degree == 0:
            return []

        # list is import to make a copy, otherwise we may mutate the cached values
        feature_combinations = list(
            PredictorSparseFits.get_monomials_coefficients(
                dimension=dimension, max_degree=max_degree - 1
            )
        )

        for i in range(max_degree, max_degree + 1):
            for comb in combinations_with_replacement(range(dimension), i):
                feature_combinations.append(
                    {variable: comb.count(variable) for variable in set(comb)}
                )

        return feature_combinations

    def predict(self, input_features: np.ndarray):
        """Applies the prediction of the fit from the feature vector"""
        assert input_features.shape == (self.dimension,)
        # precomputing the powers
        all_required_powers = [input_features]
        for k in range(2, self.actual_max_degree + 1):
            # we could multiply the last one by input_feature, but this
            # calculation seems to be more accurate
            all_required_powers += [input_features**k]

        all_required_powers = np.vstack(all_required_powers)

        evaluated_monomials = np.ones(len(self.monomials))
        for idx, element in enumerate(self.monomials):

            evaluated_monomials[idx] = np.prod(
                [
                    all_required_powers[power - 1, variable]
                    for variable, power in element.items()
                ]
            )

        ret_value = (
            np.dot(
                self.scaled_nonzero_coefficients,
                evaluated_monomials - self.nonzero_coefficients_mean,
            )
            + self.intercept
        )

        return ret_value


@cache
def get_predictor_from_fits(nb_dimensions: int, ell: int, emm: int, quantity: str):
    """Returns an instance of PredictorSparseFits

    :param nb_dimensions: dimension of the input features for generating the prediction.
    :param ell: the :math:`\\ell` part of the mode to fit
    :param emm: the :math:`m` part of the mode to fit
    :param quantity: the quantity to be fitted.

    .. note ::

        Subsequent calls with the same parameters are cached and should not involve
        any additional computation.

    .. warning::

        The returned object should not be mutated.
    """
    fits = get_fits_asymmetries()[(ell, emm)][quantity]
    monomial = PredictorSparseFits(
        monomial_coefficients=fits["coefficients"],
        mean=fits["means"],
        std=fits["stds"],
        intercept=float(fits["intercept"]),
        dimension=nb_dimensions,
        max_degree=fits["degree"],
        polynomials_contain_bias=bool(fits["include_bias"]),
    )

    return monomial
