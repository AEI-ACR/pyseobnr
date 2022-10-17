from .eob.hamiltonian.Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C import (
    Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C as Ham_aligned_opt,
)
from .eob.waveform.waveform import RR_force, SEOBNRv5RRForce
from .models import SEOBNRv5HM

import lal
import lalsimulation as lalsim
import numpy as np


def generate_modes_opt(
    q, chi1, chi2, omega0, approximant="SEOBNRv5HM", settings=None, debug=False
):
    if approximant == "SEOBNRv5HM":
        RR_f = SEOBNRv5RRForce()
        model = SEOBNRv5HM.SEOBNRv5HM_opt(
            q, chi1, chi2, omega0, Ham_aligned_opt, RR_f, settings=settings
        )
        model()
    if debug:
        return model.t, model.waveform_modes, model
    else:
        return model.t, model.waveform_modes


class GenerateWaveform:
    """
    Class for generating modes, time-domain polarizations and frequency-domain
    polarizations following LAL conventions.
    """

    def __init__(self, parameters):
        """
        Initialize class

        Parameters
        ----------
            Parameter dictionary, it can cointain:

            mass1: Mass of companion 1, in solar masses - Required
            mass2: Mass of companion 2, in solar masses - Required
            spin1x: x-component of dimensionless spin of companion 1 - Required
            spin1y: y-component of dimensionless spin of companion 1 - Required
            spin1z: z-component of dimensionless spin of companion 1 - Required
            spin2x: x-component of dimensionless spin of companion 2 - Required
            spin2y: y-component of dimensionless spin of companion 2 - Required
            spin2z: z-component of dimensionless spin of companion 2 - Required
            distance: Distance to the source, in Mpc - Required
            inclination: Inclination of the source, in radians - Required.
            phi_ref: Orbital phase at the reference frequency, in radians - Required.
            f22_start: Starting waveform generation frequency, in Hz - Required
            deltaT: Time spacing, in seconds - Required
            f_max: Maximum frequency, in Hz - Required
            deltaF: Frequency spacing, in Hz - Required
            approximant: 'SEOBNRv5HM' or 'SEOBNRv5PHM' (not implemented yet).

        """

        self.validate_parameters(parameters)

    def validate_parameters(self, parameters):

        if "mass1" not in parameters:
            raise ValueError("mass1 has to be specified!")
        if "mass2" not in parameters:
            raise ValueError("mass2 has to be specified!")

        mass1 = parameters["mass1"]
        mass2 = parameters["mass2"]

        Mtot = mass1 + mass2
        if Mtot < 0.001 or Mtot > 1e6:
            raise ValueError("Unreasonable value for total mass, aborting.")

        default_params = {
            "spin1x": 0.0,
            "spin1y": 0.0,
            "spin1z": 0.0,
            "spin2x": 0.0,
            "spin2y": 0.0,
            "spin2z": 0.0,
            "distance": 100.0,
            "inclination": 0.0,
            "phi_ref": 0.0,
            "f22_start": 20.0,
            "deltaT": 1.0 / 2048.0,
            "f_max": 1024.0,
            "deltaF": 0.125,
            "approximant": "SEOBNRv5HM",
        }

        for param in default_params.keys():
            if param not in parameters:
                parameters[param] = default_params[param]

        if (
            parameters["approximant"] is "SEOBNRv5HM"
            and (
                np.abs(parameters["spin1x"])
                + np.abs(parameters["spin1y"])
                + np.abs(parameters["spin2x"])
                + np.abs(parameters["spin2y"])
            )
            > 0.0
        ):
            raise ValueError(
                "In-plane spin components must be zero for calling non-precessing approximant."
            )

        if mass2 > mass1:
            self.swap_masses = True
            aux_spin1 = [
                parameters["spin1x"],
                parameters["spin1y"],
                parameters["spin1z"],
            ]
            parameters["spin1x"], parameters["spin1y"], parameters["spin1z"] = [
                parameters["spin2x"],
                parameters["spin2y"],
                parameters["spin2z"],
            ]
            parameters["spin2x"], parameters["spin2y"], parameters["spin2z"] = aux_spin1
            aux_mass1 = parameters["mass1"]
            parameters["mass1"] = parameters["mass2"]
            parameters["mass2"] = aux_mass1
        else:
            self.swap_masses = False

        # TODO: Implement checking of f_min not too high, and f_nyquist
        self.parameters = parameters

    def generate_td_modes(self):
        """
        Generate dictionary of positive and negative m modes in physical units.
        """
        fmin, dt = self.parameters["f22_start"], self.parameters["deltaT"]
        m1, m2 = self.parameters["mass1"], self.parameters["mass2"]
        Mtot = m1 + m2
        chi1 = self.parameters["spin1z"]
        chi2 = self.parameters["spin2z"]
        dist = self.parameters["distance"]
        approx = self.parameters["approximant"]

        omega0 = np.pi * fmin * Mtot * lal.MTSUN_SI
        q = m1 / m2  # Model convention q=m1/m2>=1
        if q < 1.0:
            q = 1 / q

        # Generate internal models, in geometrized units
        if "postadiabatic" in self.parameters:
            settings = {
                "dt": dt,
                "M": Mtot,
                "postadiabatic": self.parameters["postadiabatic"],
            }
        else:
            settings = {"dt": dt, "M": Mtot}
        times, h = generate_modes_opt(
            q, chi1, chi2, omega0, approximant=approx, settings=settings
        )

        # Convert to physical units and LAL convention
        Mpc_to_meters = 3.08567758128e22
        times *= Mtot * lal.MTSUN_SI  # Physical times
        fac = (
            -1 * Mtot * lal.MRSUN_SI / (dist * Mpc_to_meters)
        )  # Minus sign to satisfy LAL convention

        hlm_dict = {}
        for ellm, mode in h.items():
            ell = int(ellm[0])
            emm = int(ellm[2])
            hlm_dict[(ell, emm)] = fac * mode

        if (
            approx == "SEOBNRv5HM"
        ):  # If aligned-spin model, compute negative modes using equatorial symmetry
            for ellm, mode in h.items():
                ell = int(ellm[0])
                emm = int(ellm[2])
                hlm_dict[(ell, -emm)] = pow(-1, ell) * fac * np.conj(mode)

        if self.swap_masses is True:
            for ell, emm in hlm_dict.keys():
                if np.abs(emm) % 2 != 0:
                    hlm_dict[(ell, emm)] *= -1.0

        return times, hlm_dict

    def generate_td_polarizations(self):
        """
        Generate time-domain polarizations, returned as LAL REAL8TimeSeries
        """
        incl = self.parameters["inclination"]
        phi = self.parameters["phi_ref"]

        hpc = 0.0
        times, hlm_dict = self.generate_td_modes()
        for ell, emm in hlm_dict:
            hlm = hlm_dict[(ell, emm)]
            ylm = lal.SpinWeightedSphericalHarmonic(incl, np.pi / 2 - phi, -2, ell, emm)
            hpc += ylm * hlm

        hp = np.real(hpc)
        hc = -np.imag(hpc)
        epoch = lal.LIGOTimeGPS(times[0])

        hp_lal = lal.CreateREAL8TimeSeries(
            "hplus", epoch, 0, self.parameters["deltaT"], lal.DimensionlessUnit, len(hp)
        )
        hc_lal = lal.CreateREAL8TimeSeries(
            "hcross",
            epoch,
            0,
            self.parameters["deltaT"],
            lal.DimensionlessUnit,
            len(hp),
        )

        hp_lal.data.data = hp
        hc_lal.data.data = hc

        return hp_lal, hc_lal

    def generate_td_polarizations_conditioned(self):
        """
        Generate time-domain polarizations, with tappering at the beginning of the waveform,
        returned as LAL REAL8TimeSeries
        """
        hp_lal, hc_lal = self.generate_td_polarizations()
        lalsim.SimInspiralREAL8WaveTaper(hp_lal.data, 1)
        lalsim.SimInspiralREAL8WaveTaper(hc_lal.data, 1)

        return hp_lal, hc_lal

    def generate_fd_polarizations(self):
        """
        Generate Fourier-domain polarizations, returned as LAL COMPLEX16FrequencySeries

        Routine similar to LAL SimInspiralFD, but without modifying starting frequency.
        """

        fmax = self.parameters["f_max"]
        f_nyquist = fmax
        # print(f_nyquist)
        deltaF = 0
        if "deltaF" in self.parameters.keys():
            deltaF = self.parameters["deltaF"]

        if deltaF != 0:
            n = int(np.round(fmax / deltaF))
            if ((n & (n - 1)) == 0) and n != 0:
                chirplen_exp = np.frexp(n)
                f_nyquist = np.ldexp(0.5, chirplen_exp[1]) * deltaF
            # print(f_nyquist)

        deltaT = 0.5 / f_nyquist
        self.parameters["deltaT"] = deltaT

        hp, hc = self.generate_td_polarizations_conditioned()

        if deltaF == 0:
            chirplen = hp.data.length
            chirplen_exp = np.frexp(chirplen)
            chirplen = int(np.ldexp(1, chirplen_exp[1]))
            deltaF = 1.0 / (chirplen * deltaT)

        else:
            chirplen = int(1.0 / (deltaF * deltaT))

        # resize waveforms to the required length
        lal.ResizeREAL8TimeSeries(hp, hp.data.length - chirplen, chirplen)
        lal.ResizeREAL8TimeSeries(hc, hc.data.length - chirplen, chirplen)

        hptilde = lal.CreateCOMPLEX16FrequencySeries(
            "FD H_PLUS",
            hp.epoch,
            0.0,
            deltaF,
            lal.DimensionlessUnit,
            int(chirplen / 2.0 + 1),
        )
        hctilde = lal.CreateCOMPLEX16FrequencySeries(
            "FD H_CROSS",
            hc.epoch,
            0.0,
            deltaF,
            lal.DimensionlessUnit,
            int(chirplen / 2.0 + 1),
        )

        plan = lal.CreateForwardREAL8FFTPlan(chirplen, 0)

        lal.REAL8TimeFreqFFT(hctilde, hc, plan)
        lal.REAL8TimeFreqFFT(hptilde, hp, plan)

        shift = float(hp.deltaT * hp.data.length - hp.epoch)

        for idx in range(hptilde.data.length):
            ff = idx * hptilde.deltaF
            facshift = np.exp(2j * np.pi * shift * ff)
            hptilde.data.data[idx] *= facshift
            hctilde.data.data[idx] *= facshift
        # return FrequencySeries(hpf, frequencies = hpf.sample_frequencies), FrequencySeries(hcf, frequencies = hcf.sample_frequencies)
        return hptilde, hctilde
