from unittest.mock import patch

import lal
import numpy as np

from pyseobnr import GenerateWaveform
from pyseobnr.eob.dynamics.postadiabatic_C import (
    compute_combined_dynamics,
    cumulative_integral,
    fin_diff_coeffs_order_9,
    fin_diff_derivative,
    interpolated_integral_order_3,
    interpolated_integral_order_5,
    interpolated_integral_order_7,
)
from pyseobnr.eob.dynamics.postadiabatic_C_fast import (
    compute_combined_dynamics as compute_combined_dynamics_fast,
)
from pyseobnr.generate_waveform import generate_modes_opt


class TestPAOptimizations:
    @staticmethod
    def _single_deriv(y, h, coeffs):
        """Used from _fin_diff_derivative_slow and only for testing"""
        total = 0.0
        for i in range(9):
            total += coeffs[i] * y[i]
        total /= h
        return total

    @staticmethod
    def _fin_diff_derivative_slow(
        x: np.array,
        y: np.array,
        n=8,
    ):
        """
        Same implementation as fin_diff_derivative but using slow loops. Taken from cython
        and made back to python for comparisons.
        """

        dy_dx = np.zeros(x.size)
        h = abs(x[1] - x[0])
        size = x.shape[0]
        for i in range(size):
            if i == 0:
                dy_dx[i] = TestPAOptimizations._single_deriv(
                    y[0:9], h, fin_diff_coeffs_order_9[0]
                )
            elif i == 1:
                dy_dx[i] = TestPAOptimizations._single_deriv(
                    y[0:9], h, fin_diff_coeffs_order_9[1]
                )
            elif i == 2:
                dy_dx[i] = TestPAOptimizations._single_deriv(
                    y[0:9], h, fin_diff_coeffs_order_9[2]
                )
            elif i == 3:
                dy_dx[i] = TestPAOptimizations._single_deriv(
                    y[0:9], h, fin_diff_coeffs_order_9[3]
                )
            elif i == size - 4:
                dy_dx[i] = TestPAOptimizations._single_deriv(
                    y[-9:], h, fin_diff_coeffs_order_9[5]
                )
            elif i == size - 3:
                dy_dx[i] = TestPAOptimizations._single_deriv(
                    y[-9:], h, fin_diff_coeffs_order_9[6]
                )
            elif i == size - 2:
                dy_dx[i] = TestPAOptimizations._single_deriv(
                    y[-9:], h, fin_diff_coeffs_order_9[7]
                )
            elif i == size - 1:
                dy_dx[i] = TestPAOptimizations._single_deriv(
                    y[-9:], h, fin_diff_coeffs_order_9[8]
                )
            else:
                dy_dx[i] = TestPAOptimizations._single_deriv(
                    y[i - 4 : i + 5], h, fin_diff_coeffs_order_9[4]
                )

        return dy_dx

    def test_finite_differences_derivative(self):
        """Checks the cython implementation of the fin_diff_derivative function"""
        assert np.allclose(
            self._fin_diff_derivative_slow(
                np.arange(100) * 0.1, np.cos(np.arange(100) * 0.1)
            ),
            -np.sin(np.arange(100) * 0.1),
        )

        assert np.allclose(
            fin_diff_derivative(np.arange(100) * 0.1, np.cos(np.arange(100) * 0.1)),
            -np.sin(np.arange(100) * 0.1),
        )

        assert np.allclose(
            fin_diff_derivative(np.arange(100) * 0.1, np.cos(np.arange(100) * 0.1)),
            self._fin_diff_derivative_slow(
                np.arange(100) * 0.1, np.cos(np.arange(100) * 0.1)
            ),
        )

    @staticmethod
    def _cumulative_integral_slow(
        x: np.array,
        y: np.array,
        order=7,
    ):
        """
        Original un-cythonized implementation of the cumulative integrals
        """
        h = x[1] - x[0]

        integral = np.zeros(x.size)

        if order == 3:
            for i in range(x.size - 1):
                if i == 0:
                    z = np.sum(interpolated_integral_order_3[0] * y[:4])
                elif i == x.size - 2:
                    z = np.sum(interpolated_integral_order_3[2] * y[-4:])
                else:
                    z = np.sum(interpolated_integral_order_3[1] * y[i - 1 : i + 3])

                integral[i + 1] = integral[i] + z * h
        elif order == 5:
            for i in range(x.size - 1):
                if i == 0:
                    z = np.sum(interpolated_integral_order_5[0] * y[:6])
                elif i == 1:
                    z = np.sum(interpolated_integral_order_5[1] * y[:6])
                elif i == x.size - 3:
                    z = np.sum(interpolated_integral_order_5[3] * y[-6:])
                elif i == x.size - 2:
                    z = np.sum(interpolated_integral_order_5[4] * y[-6:])
                else:
                    z = np.sum(interpolated_integral_order_5[2] * y[i - 2 : i + 4])

                integral[i + 1] = integral[i] + z * h
        elif order == 7:
            for i in range(x.size - 1):
                if i == 0:
                    z = np.sum(interpolated_integral_order_7[0] * y[:8])
                elif i == 1:
                    z = np.sum(interpolated_integral_order_7[1] * y[:8])
                elif i == 2:
                    z = np.sum(interpolated_integral_order_7[2] * y[:8])
                elif i == x.size - 4:
                    z = np.sum(interpolated_integral_order_7[4] * y[-8:])
                elif i == x.size - 3:
                    z = np.sum(interpolated_integral_order_7[5] * y[-8:])
                elif i == x.size - 2:
                    z = np.sum(interpolated_integral_order_7[6] * y[-8:])
                else:
                    z = np.sum(interpolated_integral_order_7[3] * y[i - 3 : i + 5])

                integral[i + 1] = integral[i] + z * h

        return integral

    def test_cumulative_integral(self):
        """Checks the implementation of the cython cumulative integrals"""

        for order, kwargs in ((3, dict(atol=1e-6)), (5, {}), (7, {})):

            assert np.allclose(
                self._cumulative_integral_slow(
                    np.arange(1000) * 0.1, np.cos(np.arange(1000) * 0.1), order=order
                ),
                np.sin(np.arange(1000) * 0.1),
                **kwargs,
            )

            assert np.allclose(
                cumulative_integral(
                    np.arange(1000) * 0.1, np.cos(np.arange(1000) * 0.1), order=order
                ),
                np.sin(np.arange(1000) * 0.1),
                **kwargs,
            )

            assert np.allclose(
                self._cumulative_integral_slow(
                    np.arange(1000) * 0.1, np.cos(np.arange(1000) * 0.1), order=order
                ),
                cumulative_integral(
                    np.arange(1000) * 0.1, np.cos(np.arange(1000) * 0.1), order=order
                ),
                **kwargs,
            )


class TestPAIntegration:
    def test_pa_functions(self):
        m1 = 50.0
        m2 = 30.0
        Mt = m1 + m2
        dt = 1 / 2048.0
        f_min = 0.0157 / (Mt * np.pi * lal.MTSUN_SI)
        params_dict = {
            "mass1": m1,
            "mass2": m2,
            "spin1x": 0.0,
            "spin1y": 0.0,
            "spin1z": 0.5,
            "spin2x": 0.0,
            "spin2y": 0.0,
            "spin2z": 0.1,
            "deltaT": dt,
            "deltaF": 0.125,
            "f_ref": 20,
            "f22_start": f_min,
            "phi_ref": 0.0,
            "distance": 1.0,
            "inclination": np.pi / 3.0,
            "f_max": 1024.0,
            "approximant": "SEOBNRv5HM",
            "postadiabatic": False,
        }

        with patch(
            "pyseobnr.models.SEOBNRv5HM.compute_combined_dynamics"
        ) as pa_intercept:
            wfm_gen = GenerateWaveform(params_dict)
            _ = wfm_gen.generate_td_polarizations()
            pa_intercept.assert_not_called()

        with patch(
            "pyseobnr.models.SEOBNRv5HM.compute_combined_dynamics"
        ) as pa_intercept:
            wfm_gen = GenerateWaveform(
                params_dict | {"postadiabatic": True, "postadiabatic_type": "numeric"}
            )
            pa_intercept.side_effect = compute_combined_dynamics
            _ = wfm_gen.generate_td_polarizations()
            pa_intercept.assert_called_once()

        with patch(
            "pyseobnr.models.SEOBNRv5HM.compute_combined_dynamics_fast"
        ) as pa_intercept:
            wfm_gen = GenerateWaveform(params_dict)
            _ = wfm_gen.generate_td_polarizations()
            pa_intercept.assert_not_called()

        with patch(
            "pyseobnr.models.SEOBNRv5HM.compute_combined_dynamics_fast"
        ) as pa_intercept:
            wfm_gen = GenerateWaveform(
                params_dict | {"postadiabatic": True, "postadiabatic_type": "analytic"}
            )
            pa_intercept.side_effect = compute_combined_dynamics_fast
            _ = wfm_gen.generate_td_polarizations()
            pa_intercept.assert_called_once()

        # default is analytic
        with patch(
            "pyseobnr.models.SEOBNRv5HM.compute_combined_dynamics_fast"
        ) as pa_intercept:
            wfm_gen = GenerateWaveform(params_dict | {"postadiabatic": True})
            pa_intercept.side_effect = compute_combined_dynamics_fast
            _ = wfm_gen.generate_td_polarizations()
            pa_intercept.assert_called_once()

    def test_adiabatic_implementations_equivalences_non_spinning(self):
        """Checks that in the aligned spin case the calls to the PA dynamics are
        called with the same parameters"""
        q = 3
        chi_1 = 0.8
        chi_2 = 0.3
        omega0 = 0.0137

        class MyException(Exception):
            pass

        with patch(
            "pyseobnr.models.SEOBNRv5HM.compute_combined_dynamics"
        ) as pa_intercept:

            pa_intercept.side_effect = MyException

            try:
                _, _, model = generate_modes_opt(
                    q,
                    chi_1,
                    chi_2,
                    omega0,
                    debug=True,
                    settings={"postadiabatic_type": "numeric", "postadiabatic": True},
                    approximant="SEOBNRv5HM",
                )
            except MyException:
                pass

            pa_intercept.assert_called_once()

            (
                omega0_root,
                H_root,
                RR_root,
                chi_1_root,
                chi_2_root,
                m_1_root,
                m_2_root,
            ) = pa_intercept.call_args.args

            param_root = pa_intercept.call_args.kwargs["params"]
            PA_order_root = pa_intercept.call_args.kwargs["PA_order"]
            r_stop_root = pa_intercept.call_args.kwargs["r_stop"]

        assert param_root is not None
        assert PA_order_root is not None
        assert r_stop_root is not None
        assert H_root is not None
        assert RR_root is not None

        with patch(
            "pyseobnr.models.SEOBNRv5HM.compute_combined_dynamics_fast"
        ) as pa_intercept:
            pa_intercept.side_effect = MyException

            try:
                _, _, model = generate_modes_opt(
                    q,
                    chi_1,
                    chi_2,
                    omega0,
                    debug=True,
                    settings={"postadiabatic_type": "analytic", "postadiabatic": True},
                    approximant="SEOBNRv5HM",
                )
            except MyException:
                pass

            pa_intercept.assert_called_once()

            (
                omega0_analytic,
                H_analytic,
                RR_analytic,
                chi_1_analytic,
                chi_2_analytic,
                m_1_analytic,
                m_2_analytic,
            ) = pa_intercept.call_args.args

            param_analytic = pa_intercept.call_args.kwargs["params"]
            PA_order_analytic = pa_intercept.call_args.kwargs["PA_order"]
            r_stop_analytic = pa_intercept.call_args.kwargs["r_stop"]

        assert param_analytic is not None
        assert PA_order_analytic is not None
        assert r_stop_analytic is not None
        assert H_analytic is not None
        assert RR_analytic is not None

        # the parameters of the call should be the same
        assert omega0_root == omega0_analytic
        assert PA_order_root == PA_order_analytic
        assert r_stop_root == r_stop_analytic
        assert chi_1_root == chi_1_analytic
        assert chi_2_root == chi_2_analytic
        assert m_1_root == m_1_analytic
        assert m_2_root == m_2_analytic
