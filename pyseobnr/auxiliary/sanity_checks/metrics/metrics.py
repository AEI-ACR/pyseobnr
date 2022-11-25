#!/usr/bin/env python3
import logging
from abc import ABC
from typing import Any, Dict, Union

import lal
import numpy as np
from pyseobnr.models.model import Model
from rich.logging import RichHandler
from rich.traceback import install
from scipy.interpolate import CubicSpline

from .default_settings import default_unfaitfulness_mode_by_mode_settings
from .unfaithfulness_mode_by_mode import (
    condition_pycbc_series,
    fast_unfaithfulness_mode_by_mode,
    generate_dominant_mode_pol_LAL,
    get_padded_length,
)

try:
    from waveform_tools.mismatch.auxillary_funcs import *

except ImportError:
    print("waveform_tools is not installed, some metrics not available")


from pycbc.types import FrequencySeries, TimeSeries
from pycbc.waveform import fd_approximants

# Setup the logger to work with rich
logger = logging.getLogger(__name__)
logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))
logger.setLevel("INFO")
# Setup rich to get nice tracebacks
install()


class Metric(ABC):
    def __init__(self):
        pass

    def __call__(self):
        pass


class UnfaithfulnessModeByModeLAL(Metric):
    def __init__(self, settings: Dict[Any, Any] = None):
        self.settings = default_unfaitfulness_mode_by_mode_settings()
        if settings is not None:
            self.settings.update(**settings)

        self.debug = self.settings.get("debug", False)
        self.masses = self.settings.get(
            "masses", np.array([10.0, 50.0, 100.0, 150.0, 200.0])
        )
        self.buffer = self.settings.get("buffer", 1.35)
        self.fmax = self.settings.get("fmax", 2048.0)
        self.name = "UnfaithfulnessModeByModeLAL"

    def __call__(
        self,
        target: Union[str, Model],
        model: Union[str, Model],
        params: Dict[Any, Any] = None,
        ell: int = 2,
        m: int = 2,
    ) -> float:
        return self._evaluate(target, model, params=params, ell=ell, m=m)

    def _evaluate(
        self,
        target: Union[str, Model],
        model: Union[str, Model],
        params: Dict[Any, Any] = None,
        ell: int = 2,
        m: int = 2,
        M=50.0,
    ) -> float:

        # Ensure that the input parameters of everrything are consistent

        logger.debug("Sanity checking inputs")
        # self._sanity_check_inputs(target, model, params)
        logger.debug("Generating waveforms")
        if isinstance(target, Model):
            # This is one of fully wrapped models, don't need to generate things, just grab them
            h1 = TimeSeries(
                np.real(target.waveform_modes[f"{ell},{m}"]), delta_t=target.delta_T
            )
            t1 = target.t
            #omega1 = target.omega0
            # Estimate the actual starting frequency from (2,2) mode
            intrp = CubicSpline(target.t,np.unwrap(np.angle(target.waveform_modes['2,2'])))
            omega1 = np.abs(intrp.derivative()(target.t[0]))/2.
            logger.debug("Target was a Model, grabbed stuff")
        else:
            # Assume that otherwise it's a call to LAL and special care has to be taken
            if params is not None:
                params_target = self._wrap_params(params, target)
            else:
                raise ValueError(
                    "The target specified is not a full Model. You must supply params!"
                )

            h1 = generate_dominant_mode_pol_LAL(params_target, f_max=self.fmax)
            omega1 = params["omega0"]  # By construction

        if isinstance(model, Model):
            # This is one of fully wrapped models, don't need to generate things, just grab them
            h2 = TimeSeries(
                np.real(model.waveform_modes[f"{ell},{m}"]), delta_t=model.delta_T
            )
            t2 = model.t
            omega2 = model.omega0
        else:
            # Assume that otherwise it's a call to LAL and special care has to be taken
            if params is not None:
                params_model = self._wrap_params(params, model)

            else:
                raise ValueError(
                    "The model specified is not a full Model. You must supply params!"
                )

            # Ugly logic to ensure consistency of Fourier-domain waveforms
            if params_model.domain == "FD" and isinstance(h1, TimeSeries):
                # We want to generate and FD waveform that will match the given TD
                # waveform when it's FFTed.
                padsize = get_padded_length(h1)
                padded_duration = h1.delta_t * padsize * M * lal.MTSUN_SI
                params_model.delta_f = 1 / padded_duration
                nyquist_f = 0.5 / (h1.delta_t * M * lal.MTSUN_SI)
                f_max = nyquist_f

            else:
                f_max = self.fmax

            # print(params_model)
            h2 = generate_dominant_mode_pol_LAL(params_model, f_max=f_max)
            omega2 = params["omega0"]  # by construction!

        omega0 = np.max((omega1, omega2))
        flow = self.buffer * m * omega0 / (2 * np.pi)

        # Some convoluted logic for conditioning
        # Situation 1: both inputs are in time domain
        # Figure out the length of the longer waveform
        # We use this to pad the waveforms
        # Situtation 2: one of the inputs is in frequency domain
        # Figure out what we do.
        if isinstance(h1, TimeSeries) and isinstance(h2, TimeSeries):
            N = np.max((len(h1), len(h2)))
            h1 = condition_pycbc_series(h1, n=N)
            h2 = condition_pycbc_series(h2, n=N)
        elif isinstance(h1, TimeSeries) and isinstance(h2, FrequencySeries):
            h1 = condition_pycbc_series(h1)
        elif isinstance(h1, FrequencySeries) and isinstance(h2, TimeSeries):
            h2 = condition_pycbc_series(h2)

        matches = fast_unfaithfulness_mode_by_mode(h1, h2, flow, Ms=self.masses, psd_t = 'aLIGO')
        # Get the highest unfaithulness
        mm = 1 - np.min(matches)
        if self.debug:
            return 1 - matches
        return mm

    def _sanity_check_inputs(self, target, model, params):

        if (
            isinstance(target, Model) and isinstance(model, Model)
        ) and params is not None:
            logger.warning("Ignoring the params dict, as it was not needed!")
            return
        # At this point only one (target,model) can be a Model
        if isinstance(target, Model) and params is not None:
            q_t = target.q
            chi1_t = target.chi1
            chi2_t = target.chi2
            q = params["q"]
            chi1 = params["chi1"]
            chi2 = params["chi1"]
            assert np.allclose(q, q_t)
            # assert np.allclose(chi1_t, chi1)
            # assert np.allclose(chi2_t, chi2)
        elif isinstance(model, Model) and params is not None:
            q_t = model.q
            chi1_t = model.chi1
            chi2_t = model.chi2
            q = params["q"]
            chi1 = params["chi1"]
            chi2 = params["chi1"]
            assert np.allclose(q, q_t)
            # assert np.allclose(chi1_t[-1], chi1)
            # assert np.allclose(chi2_t, chi2)

    def _wrap_params(self, params, approx, M=50.0):
        q = params["q"]
        chi1 = params["chi1"]
        chi2 = params["chi2"]
        omega0 = params["omega0"]
        dt = params["dt"] * M * lal.MTSUN_SI
        df = params["df"]
        domain = self._get_domain(approx)
        if domain == "TD":
            df = None
        else:
            dt = None

        m1 = q / (1 + q)
        m2 = 1 - m1
        f_ref = f_min = omega0 / (M * np.pi * lal.MTSUN_SI)
        wf_pars = waveform_params(
            m1=M * m1,
            m2=M * m2,
            s1x=chi1[0],
            s1y=chi1[1],
            s1z=chi1[2],
            s2x=chi2[0],
            s2y=chi2[1],
            s2z=chi2[2],
            iota=0.0,
            phi=0.0,
            f_ref=f_ref,
            f_min=f_min,
            distance=1.2342710325965468e25,
            delta_t=dt,
            delta_f=df,
            wf_param=None,
            approx=approx,
            domain=domain,
            ecc=0.0,
            mean_anomaly=0.0,
        )
        return wf_pars

    def _get_domain(self, approx):
        fd_approxs = fd_approximants()
        if approx not in fd_approxs or approx == "SEOBNRv4":
            return "TD"
        else:
            return "FD"
