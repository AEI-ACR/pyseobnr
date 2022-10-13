#!/usr/bin/env python3
import logging
import os
from abc import ABC
from typing import Any, Dict

import h5py
import lal
import lalsimulation as lalsim
import numpy as np
import scri
import sxs
from rich.logging import RichHandler
from rich.traceback import install
from scipy.interpolate import CubicSpline, InterpolatedUnivariateSpline

from .default_settings import *
from pyseobnr.models.model import Model

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))
logger.setLevel("INFO")
install()


class NRModel_LVC(Model):
    """Represents a numerical relativity waveform as evaluated through the LAL
    NR interface. Requires data in the LVCNR format.
    """

    def __init__(self, data_file: str, settings: Dict[Any, Any] = None) -> None:
        self.data_file = data_file
        self.settings = default_NR_LVC_settings()
        # If we were given settings, override the defaults
        if settings is not None:
            self.settings.update(**settings)

        # Deal with debug mode
        self.debug = self.settings.get("debug", False)
        if self.debug:
            logger.setLevel("DEBUG")

        with h5py.File(self.data_file, "r") as f:
            # We rescale the masses so that they add to 1!
            self.m_1 = f.attrs["mass1"] / (f.attrs["mass1"] + f.attrs["mass2"])
            self.m_2 = f.attrs["mass2"] / (f.attrs["mass1"] + f.attrs["mass2"])
            # Starting frequency in Hz at 1 solar mass total mass
            Mf_lower = f.attrs["f_lower_at_1MSUN"]

        self.q = self.m_1 / self.m_2
        self.M = self.settings["M"]

        f_lower = Mf_lower / (self.m_1 + self.m_2)
        (
            s1x,
            s1y,
            s1z,
            s2x,
            s2y,
            s2z,
        ) = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
            f_lower,
            1,
            self.data_file,
        )

        params = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertNumRelData(params, self.data_file)

        self.distance = self.settings["distance"]

        self.dt = self.settings["dt"]
        self.delta_T = self.dt / (self.M * lal.MTSUN_SI)

        self.f_start = Mf_lower / self.M
        self.f_ref = Mf_lower / self.M
        self.omega0 = self.f_start * self.M * lal.MTSUN_SI * np.pi

        ell_max = 8
        mode_array = lalsim.SimInspiralCreateModeArray()
        for ell in range(2, ell_max + 1):
            for m in range(-ell, ell + 1):
                mode_array = lalsim.SimInspiralModeArrayActivateMode(mode_array, ell, m)

        _, hlm = lalsim.SimInspiralNRWaveformGetHlms(
            self.dt,
            self.m_1 * self.M * lal.MSUN_SI,
            self.m_2 * self.M * lal.MSUN_SI,
            self.distance,
            self.f_start,
            self.f_ref,
            s1x,
            s1y,
            s1z,
            s2x,
            s2y,
            s2z,
            self.data_file,
            mode_array,
        )
        self.chi1 = np.array([s1x, s1y, s1z])
        self.chi2 = np.array([s2x, s2y, s2z])
        scaling_factor = self.M * lal.MRSUN_SI / self.distance

        self.waveform_modes = {}
        for mode in self.settings["modes"]:
            # Note the  rescaling of the amplitude from SI to geometric units
            self.waveform_modes[f"{mode[0]},{mode[1]}"] = (
                lalsim.SphHarmTimeSeriesGetMode(hlm, mode[0], mode[1]).data.data
                / scaling_factor
            )

        # We are guaranteed that the (2,2) mode is present
        tmp = lalsim.SphHarmTimeSeriesGetMode(hlm, 2, 2)
        # The shift
        shift = 0 * (tmp.epoch.gpsSeconds + tmp.epoch.gpsNanoSeconds / 1e9)
        # Rescale the time
        time_array = (np.arange(0, len(tmp.data.data)) * self.dt + shift) / (
            self.M * lal.MTSUN_SI
        )
        self.t = time_array

        """
        Compute and store the energy and momentum fluxes
        """
        energy_flux_sum = 0
        momentum_flux_sum = 0

        for ell in range(2, ell_max + 1):
            for m in range(-ell, ell + 1):
                h_lm = (
                    lalsim.SphHarmTimeSeriesGetMode(hlm, ell, m).data.data
                    / scaling_factor
                )
                h_lm_interp = CubicSpline(self.t, h_lm)
                dh_lm_dt = h_lm_interp.derivative()(self.t)

                energy_flux_sum += np.abs(dh_lm_dt) ** 2
                momentum_flux_sum += -m * np.imag(np.conjugate(dh_lm_dt) * h_lm)

        self.energy_flux = energy_flux_sum / (16 * np.pi)
        self.momentum_flux = momentum_flux_sum / (16 * np.pi)

    def __call__(self):
        """
        Since the modes are computed at initialization, don't need to do anything here
        """
        pass


class NRModel_SXS(Model):
    """Represents a numerical relativity waveform from SXS catalog.
    This requires the scri package.
    """

    def __init__(self, data_file: str, settings: Dict[Any, Any] = None) -> None:
        self.data_file = data_file
        self.settings = default_NR_SXS_settings()
        # If we were given settings, override the defaults
        if settings is not None:
            self.settings.update(**settings)

        # Deal with debug mode
        self.debug = self.settings.get("debug", False)
        if self.debug:
            logger.setLevel("DEBUG")

        self.delta_T = self.settings.get("dt", 0.1)
        waveform = scri.SpEC.read_from_h5(
            f"{data_file}/Extrapolated_N{self.settings['extrap_order']}.dir"
        )
        t_new = np.arange(waveform.t[0], waveform.t[-1], self.delta_T)
        waveform = waveform.interpolate(t_new)
        # Load the metadata. Note: assume it is in the same directory as the h5 file
        path = os.path.dirname(data_file)
        metadata = sxs.load(f"{path}/metadata.json")

        self.m1 = metadata.reference_mass1
        self.m2 = metadata.reference_mass2
        self.m_1 = self.m1
        self.m_2 = self.m2
        self.M = self.m1 + self.m2
        self.q = self.m1 / self.m2
        self.chi1 = metadata.reference_dimensionless_spin1
        self.chi2 = metadata.reference_dimensionless_spin2
        self.tr = metadata.reference_time

        self.omega0 = np.linalg.norm(metadata.initial_orbital_frequency)
        self.waveform_modes = {}
        ell_max = self.settings.get("ell_max", 8)
        idx_ok = np.where(waveform.t >= self.tr)
        for ell in range(2, ell_max + 1):
            for m in range(-ell, ell + 1):
                idx = waveform.index(ell, m)
                self.waveform_modes[f"{ell},{m}"] = waveform.data[idx_ok][:, idx]

        self.t = waveform.t[idx_ok]

        """
        Compute and store the energy and momentum fluxes
        """
        energy_flux_sum = 0
        momentum_flux_sum = 0

        for ell in range(2, ell_max + 1):
            for m in range(-ell, ell + 1):
                h_lm = self.waveform_modes[f"{ell},{m}"].copy()
                h_lm_interp = CubicSpline(self.t, h_lm)
                dh_lm_dt = h_lm_interp.derivative()(self.t)

                energy_flux_sum += np.abs(dh_lm_dt) ** 2
                momentum_flux_sum += -m * np.imag(np.conjugate(dh_lm_dt) * h_lm)

        self.energy_flux = energy_flux_sum / (16 * np.pi)
        self.momentum_flux = momentum_flux_sum / (16 * np.pi)

    def __call__(self):
        """
        Since the modes are computed at initialization, don't need to do anything here
        """
        pass


class NRModel_RIT(Model):
    """Represents a numerical relativity waveform from RIT catalog.
    This requires the sxs package.
    """

    def __init__(self, data_file: str, settings: Dict[Any, Any] = None) -> None:
        self.data_file = data_file
        self.settings = default_NR_RIT_settings()
        # If we were given settings, override the defaults
        if settings is not None:
            self.settings.update(**settings)

        # Deal with debug mode
        self.debug = self.settings.get("debug", False)
        if self.debug:
            logger.setLevel("DEBUG")

        # Load the metadata. Note: assume it is in the same directory as the h5 file
        path = os.path.dirname(data_file)
        metadata = sxs.load(f"{path}/metadata.json")

        # Note, for RIT m_2>=m_1. But we always have m_1>=m_2
        self.m1 = metadata.relaxed_mass2
        self.m2 = metadata.relaxed_mass1
        self.q = self.m1 / self.m2
        if "relaxed_chi1x" in metadata.keys():
            chi1 = np.array(
                [metadata.relaxed_chi2x, metadata.relaxed_chi2y, metadata.relaxed_chi2z]
            )
            chi2 = np.array(
                [metadata.relaxed_chi1x, metadata.relaxed_chi1y, metadata.relaxed_chi1z]
            )
        else:
            chi1 = np.array([0.0, 0.0, metadata.relaxed_chi2z])
            chi2 = np.array([0.0, 0.0, metadata.relaxed_chi1z])
        self.chi1 = chi1
        self.chi2 = chi2
        # FIXME: this is what was done for SEOBNRv4, but it probably makes more sense
        # to use the relaxed frequency
        self.omega0 = metadata.freq_start_22 * np.pi
        self.waveform_modes = {}
        ell_min, ell_max = 2, 4
        with h5py.File(self.data_file, "r") as f:
            t = f["NRTimes"][:]
            for ell in range(ell_min, ell_max + 1):
                for m in range(-ell, ell + 1):
                    amp = InterpolatedUnivariateSpline(
                        f[f"amp_l{ell}_m{m}/X"][:],
                        f[f"amp_l{ell}_m{m}/Y"][:],
                        k=f[f"amp_l{ell}_m{m}/deg"][()],
                    )(t)
                    phase = InterpolatedUnivariateSpline(
                        f[f"phase_l{ell}_m{m}/X"][:],
                        f[f"phase_l{ell}_m{m}/Y"][:],
                        k=f[f"phase_l{ell}_m{m}/deg"][()],
                    )(t)
                    self.waveform_modes[f"{ell},{m}"] = amp * np.exp(1j * phase)

        self.t = t - t[0]
        self.tr = metadata.relaxed_time

        """
        Compute and store the energy and momentum fluxes
        """
        energy_flux_sum = 0
        momentum_flux_sum = 0

        for ell in range(2, ell_max + 1):
            for m in range(-ell, ell + 1):
                h_lm = self.waveform_modes[f"{ell},{m}"].copy()
                h_lm_interp = CubicSpline(self.t, h_lm)
                dh_lm_dt = h_lm_interp.derivative()(self.t)

                energy_flux_sum += np.abs(dh_lm_dt) ** 2
                momentum_flux_sum += -m * np.imag(np.conjugate(dh_lm_dt) * h_lm)

        self.energy_flux = energy_flux_sum / (16 * np.pi)
        self.momentum_flux = momentum_flux_sum / (16 * np.pi)

    def __call__(self):
        """
        Since the modes are computed at initialization, don't need to do anything here
        """
        pass


class SEOBNRv4HM_LAL(Model):
    """
    Represents an SEOBNRv4HM waveform generated through LAL.
    It generates the following modes:
    (2,2), (3,3), (4,4), (5,5), (2,1)

    Args:
        q (float): Mass ratio m1/m2 >= 1
        chi_1 (float): z component of the dimensionless spin of primary
        chi_2 (float): z component of the dimensionless spin of secondary
        omega0 (float): Initial orbital frequency, in geomtric units
        settings (Dict[Any, Any], optional): The settings. Defaults to None.
    """

    def __init__(
        self,
        q: float,
        chi_1: float,
        chi_2: float,
        omega0: float,
        coeffs: Dict[str, Any] = None,
        settings: Dict[Any, Any] = None,
    ) -> None:
        self.settings = default_SEOBNRv4HM_settings()
        # If we were given settings, override the defaults
        if settings is not None:
            self.settings.update(**settings)

        self.M = self.settings["M"]
        self.q = q
        m_1 = q / (1 + q)
        m_2 = 1 - m_1

        self.chi_1 = chi_1
        self.chi_2 = chi_2

        self.dt = self.settings["dt"]
        self.delta_T = self.dt / (self.M * lal.MTSUN_SI)
        # Starting *orbital* frequency
        self.omega0 = omega0
        self.f0 = self.omega0 / (self.M * lal.MTSUN_SI * np.pi)
        self.distance = self.settings["distance"]

        # Component masses in solar masses
        self.m1_msun = m_1 * self.M
        self.m2_msun = m_2 * self.M

        # NQC coeffs
        self.nqcCoeffsInput = lal.CreateREAL8Vector(10)

        # SEOBNRv4HM is numbered 41
        self.wf_version = 41

        # Debug mode?
        self.debug = self.settings.get("debug", False)

    def __call__(self):
        self._evaluate_model()

    def _evaluate_model(self):
        """
        Evaluate the model
        """
        pars = {
            "phiC": 0.0,
            "deltaT": self.dt,
            "m1_SI": self.m1_msun * lal.MSUN_SI,
            "m2_SI": self.m2_msun * lal.MSUN_SI,
            "distance": self.distance,
            "inclination": 0.0,
            "lambda2Tidal1": 0.0,
            "lambda2Tidal2": 0.0,
            "omega02Tidal1": 0.0,
            "omega02Tidal2": 0.0,
            "lambda3Tidal1": 0.0,
            "lambda3Tidal2": 0.0,
            "omega03Tidal1": 0.0,
            "omega03Tidal2": 0.0,
            "quadparam1": 1.0,
            "quadparam2": 1.0,
        }

        sphtseries, dyn, dynHI = lalsim.SimIMRSpinAlignedEOBModes(
            pars["deltaT"],
            pars["m1_SI"],
            pars["m2_SI"],
            self.f0,
            pars["distance"],
            self.chi_1,
            self.chi_2,
            self.wf_version,
            pars["lambda2Tidal1"],
            pars["lambda2Tidal2"],
            pars["omega02Tidal1"],
            pars["omega02Tidal2"],
            pars["lambda3Tidal1"],
            pars["lambda3Tidal2"],
            pars["omega03Tidal1"],
            pars["omega03Tidal2"],
            pars["quadparam1"],
            pars["quadparam2"],
            self.nqcCoeffsInput,
            0,
        )

        hlm = {}

        ##55 mode
        modeL = sphtseries.l
        modeM = sphtseries.m
        h55 = sphtseries.mode.data.data  # This is h_55
        hlm[f"{modeL},{modeM}"] = -h55

        ##44 mode
        modeL = sphtseries.next.l
        modeM = sphtseries.next.m
        h44 = sphtseries.next.mode.data.data  # This is h_44
        hlm[f"{modeL},{modeM}"] = -h44

        ##21 mode
        modeL = sphtseries.next.next.l
        modeM = sphtseries.next.next.m
        h21 = sphtseries.next.next.mode.data.data  # This is h_21
        hlm[f"{modeL},{modeM}"] = -h21

        ##33 mode
        modeL = sphtseries.next.next.next.l
        modeM = sphtseries.next.next.next.m
        h33 = sphtseries.next.next.next.mode.data.data  # This is h_33
        hlm[f"{modeL},{modeM}"] = -h33

        ##22 mode
        modeL = sphtseries.next.next.next.next.l
        modeM = sphtseries.next.next.next.next.m
        h22 = sphtseries.next.next.next.next.mode.data.data  # This is h_22
        hlm[f"{modeL},{modeM}"] = -h22

        scaling_factor = self.M * lal.MRSUN_SI / self.distance
        shift = 0
        time_array = (np.arange(0, len(h22)) * self.dt + shift) / (
            self.M * lal.MTSUN_SI
        )
        for key in hlm.keys():
            hlm[key] /= scaling_factor

        self.t = time_array

        self.waveform_modes = hlm
        self.dyn = dyn


class NRHybSur3dq8Model(Model):
    """
    NRHybSur3dq8 waveform

    This model includes the following spin-weighted spherical harmonic modes:
    (2,2), (2,1), (2,0), (3,3), (3,2), (3,1), (3,0), (4,4) (4,3), (4,2) and (5,5)

    """

    def __init__(
        self,
        q: float,
        chi_1: float,
        chi_2: float,
        omega0: float,
        settings: Dict[Any, Any] = None,
    ) -> None:
        """
        Args:
            q (float): Mass ratio m1/m2 >= 1
            chi_1 (float): z component of the dimensionless spin of primary
            chi_2 (float): z component of the dimensionless spin of secondary
            omega0 (float): Initial orbital frequency, in geomtric units
        """

        self.settings = default_NRHybSur3dq8_settings()
        if settings is not None:
            self.settings.update(**settings)

        self.M = self.settings["M"]
        self.dt = self.settings["dt"]
        self.delta_T = self.dt
        if q < 1.0:
            q = 1.0

        self.q = q

        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self.omega0 = omega0

    def __call__(self, sur, parameters=None):
        self._evaluate_model(sur)

    def _evaluate_model(self, sur):

        omega = self.omega0
        f_low = omega / np.pi
        t_sur, h, dyn = sur(
            self.q,
            [0, 0, self.chi_1],
            [0, 0, self.chi_2],
            dt=self.dt,
            f_low=f_low,
            f_ref=f_low,
        )

        self.t = t_sur - t_sur[0]

        self.waveform_modes = {}

        self.waveform_modes["2,2"] = -h[(2, 2)]
        self.waveform_modes["2,1"] = -h[(2, 1)]
        self.waveform_modes["2,0"] = -h[(2, 0)]
        self.waveform_modes["3,3"] = -h[(3, 3)]
        self.waveform_modes["3,2"] = -h[(3, 2)]
        self.waveform_modes["3,1"] = -h[(3, 1)]
        self.waveform_modes["3,0"] = -h[(3, 0)]
        self.waveform_modes["4,4"] = -h[(4, 4)]
        self.waveform_modes["4,3"] = -h[(4, 3)]
        self.waveform_modes["4,2"] = -h[(4, 2)]
        self.waveform_modes["5,5"] = -h[(5, 5)]


class NRHybSur2dq15Model(Model):
    """
       NRHybSur2dq15 waveform

       This model includes the following spin-weighted spherical harmonic modes:
       (2,2), (2,1), (3,3), (4,4) and (5,5)

    |  The surrogate has been trained in the range
    |  q \in [1, 15] and chi1z \in [-0.5, 0.5] chi2z = 0

    """

    def __init__(
        self,
        q: float,
        chi_1: float,
        chi_2: float,
        omega0: float,
        settings: Dict[Any, Any] = None,
    ) -> None:
        """
        Args:
            q (float): Mass ratio m1/m2 >= 1
            chi_1 (float): z component of the dimensionless spin of primary
            chi_2 (float): z component of the dimensionless spin of secondary
            omega0 (float): Initial orbital frequency, in geomtric units
        """

        self.settings = default_NRHybSur2dq15_settings()
        if settings is not None:
            self.settings.update(**settings)

        self.M = self.settings["M"]
        self.dt = self.settings["dt"]
        self.delta_T = self.dt
        if q < 1.0:
            q = 1.0

        self.q = q

        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self.omega0 = omega0

    def __call__(self, sur, parameters=None):
        self._evaluate_model(sur)

    def _evaluate_model(self, sur):

        omega = self.omega0
        f_low = omega / np.pi
        t_sur, h, dyn = sur(
            self.q,
            [0, 0, self.chi_1],
            [0, 0, self.chi_2],
            dt=self.dt,
            f_low=f_low,
            f_ref=f_low,
        )

        self.t = t_sur - t_sur[0]

        self.waveform_modes = {}

        self.waveform_modes["2,2"] = -h[(2, 2)]
        self.waveform_modes["2,1"] = -h[(2, 1)]
        self.waveform_modes["3,3"] = -h[(3, 3)]
        self.waveform_modes["4,4"] = -h[(4, 4)]
        self.waveform_modes["5,5"] = -h[(5, 5)]
