import logging
import re
from copy import deepcopy
from typing import Any, Callable, Dict

import lal
import numpy as np
from pygsl import spline
from pyseobnr.eob.dynamics.postadiabatic_C import Kerr_ISCO, compute_combined_dynamics
from pyseobnr.eob.utils.containers import CalibCoeffs, EOBParams
from pyseobnr.eob.waveform.waveform import compute_hlms as compute_hlms_new
from pyseobnr.eob.waveform.waveform import (
    compute_newtonian_prefixes,
    compute_special_coeffs,
)
from pyseobnr.eob.dynamics.integrate_ode import (
    augment_dynamics,
    compute_dynamics_opt,
)
from pyseobnr.eob.fits.fits_Hamiltonian import NR_deltaT, NR_deltaT_NS, a6_NS, dSO
from pyseobnr.eob.fits.GSF_fits import GSF_amplitude_fits
from pyseobnr.eob.fits.IV_fits import InputValueFits
from pyseobnr.eob.hamiltonian import Hamiltonian
from pyseobnr.eob.waveform.compute_hlms import (
    NQC_correction,
    apply_nqc_corrections,
    compute_IMR_modes,
    concatenate_modes,
    interpolate_modes_fast,
)
from pyseobnr.models.model import Model
from rich.logging import RichHandler
from rich.traceback import install
from scipy.interpolate import CubicSpline

# Setup the logger to work with rich
logger = logging.getLogger(__name__)
logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))
logger.setLevel("INFO")
# Setup rich to get nice tracebacks
install()

# List of valid modes
VALID_MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5)]


class SEOBNRv5HM_opt(Model):
    """Represents an aligned-spin SEOBNRv5HM waveform with new MR choices."""

    def __init__(
        self,
        q: float,
        chi_1: float,
        chi_2: float,
        omega0: float,
        H: Hamiltonian,
        RR: Callable,
        settings: Dict[Any, Any] = None,
    ) -> None:
        """Initialize the SEOBNRv5 approximant

        Args:
            q (float): Mass ratio m1/m2 >= 1
            chi_1 (float): z component of the dimensionless spin of primary
            chi_2 (float): z component of the dimensionless spin of secondary
            omega0 (float): Initial orbital frequency, in geomtric units
            settings (Dict[Any, Any], optional): The settings. Defaults to None.
        """

        self.settings = self._default_settings()
        # If we were given settings, override the defaults
        if settings is not None:
            self.settings.update(**settings)

        self.M = self.settings["M"]
        self.q = q

        self.chi_1 = chi_1
        self.chi_2 = chi_2

        self.NR_deltaT = 0

        self.RR = RR

        self.m_1 = q / (1.0 + q)
        self.m_2 = 1.0 / (1 + q)

        # self.nu = self.m_1 * self.m_2 / (self.m_1 + self.m_2) ** 2
        self.nu = q / (1.0 + q) ** 2
        self.omega0 = omega0
        self.f0 = self.omega0 / (self.M * lal.MTSUN_SI * np.pi)
        self.step_back = self.settings.get("step_back", 250.0)
        self.chi_S = (self.chi_1 + self.chi_2) / 2
        self.chi_A = (self.chi_1 - self.chi_2) / 2
        self.ap = self.m_1 * self.chi_1 + self.m_2 * self.chi_2
        self.am = self.m_1 * self.chi_1 - self.m_2 * self.chi_2
        self.dt = self.settings["dt"]
        self.delta_T = self.dt / (self.M * lal.MTSUN_SI)

        self.prefixes = compute_newtonian_prefixes(self.m_1, self.m_2)

        self.tplspin = (1 - 2 * self.nu) * self.chi_S + (self.m_1 - self.m_2) / (
            self.m_1 + self.m_2
        ) * self.chi_A

        self.phys_pars = dict(
            m_1=self.m_1,
            m_2=self.m_2,
            chi_1=self.chi_1,
            chi_2=self.chi_2,
            chi_1x=0.0,
            chi_1y=0.0,
            chi_1z=self.chi_1,
            chi_2x=0.0,
            chi_2y=0.0,
            chi_2z=self.chi_2,
            omega=self.omega0,
            omega_circ=self.omega0,
        )
        # Figure out which modes need to be
        # i) computed
        # ii) returned
        # The situation where those match can be e.g. when the user
        # asks for mixed modes so we must compute all the modes
        # that are needed even if we will not return them

        # All the modes we will need to output
        self.return_modes = self.settings.get("return_modes", None)
        # Check that the modes are valid, i.e. something we
        # can return
        self._validate_modes()
        # Now deal with which mixed modes the user wants, if any

        self.mixed_modes = [(3, 2), (4, 3)]
        self.mixed_modes = [x for x in self.mixed_modes if x in self.return_modes]

        # All the modes we need to compute. This can be a larger list
        # than the returned modes, e.g. when we need certain modes to
        # do mode mixing
        self.computed_modes = deepcopy(self.return_modes)
        # Make sure the array contains what we need
        self._ensure_consistency()

        self._initialize_params(self.phys_pars)
        # Initialize the Hamiltonian
        self.H = H(self.eob_pars)

    def _default_settings(self):
        settings = dict(
            M=50.0,  # Total mass in solar masses
            dt=2.4627455127717882e-05,  # Desired time spacing, *in seconds*
            debug=False,  # Run in debug mode
            postadiabatic=False,  # Use postadiabatic?
            return_modes=[(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5)],
        )
        return settings

    def _validate_modes(self):
        """
        Check that the mode array is sensible, i.e.
        has something and the modes being asked for are valid.
        """

        if not self.return_modes:
            logger.error("The mode list specified is empty!")
            raise ValueError

        for mode in self.return_modes:
            if mode not in VALID_MODES:
                print(f"{mode} is not valid!")
                logger.error(f"The specified mode, {mode} is not available!")
                logger.error(f"The allowed modes are: {VALID_MODES}")
                raise ValueError

    def _ensure_consistency(self):
        """Make sure that the modes contains everything needed to compute mixed modes

        Args:
            modes (list): Current list of modes to compute
        """
        for mode in self.mixed_modes:
            ell, m = mode
            if (m, m) not in self.computed_modes:
                self.computed_modes.append((m, m))

    def _initialize_params(self, phys_pars):
        """
        Re-initialize all parameters to make sure everthing is reset
        """
        self.eob_pars = EOBParams(phys_pars, {}, mode_array=self.computed_modes)
        self.eob_pars.flux_params.rho_initialized = False
        self.eob_pars.flux_params.prefixes = np.array(self.prefixes)
        self.eob_pars.flux_params.prefixes_abs = np.abs(
            self.eob_pars.flux_params.prefixes
        )
        self.eob_pars.flux_params.extra_PN_terms = self.settings.get(
            "extra_PN_terms", True
        )
        self.step_back = self.settings.get("step_back", 250.0)

    def __call__(self):
        # Evaluate the model

        # Initialize the containers
        self._initialize_params(self.phys_pars)

        # Compute the shift from reference point to peak of (2,2) mode
        NR_deltaT_fit = NR_deltaT_NS(self.nu) + NR_deltaT(self.nu, self.ap, self.am)
        self.NR_deltaT = NR_deltaT_fit
        # Set the Hamiltonian coefficients
        self._set_H_coeffs()
        # Set the GSF contributions to the waveform
        gsf_coeffs = GSF_amplitude_fits(self.nu)
        keys = gsf_coeffs.keys()
        # The following is just a fancy way of passing the coeffs
        for key in keys:
            tmp = re.findall("h(\d)(\d)_v(\d+)", key)
            if tmp:
                l, m, v = [int(x) for x in tmp[0]]
                self.eob_pars.flux_params.extra_coeffs[l, m, v] = gsf_coeffs[key]
            else:
                tmp = re.findall("h(\d)(\d)_vlog(\d+)", key)
                if tmp:
                    l, m, v = [int(x) for x in tmp[0]]
                    self.eob_pars.flux_params.extra_coeffs_log[l, m, v] = gsf_coeffs[
                        key
                    ]
        self._evaluate_model()

    def _set_H_coeffs(self):

        # Flags for different PN orders
        flags = {
            "flagNLOSO": 1.0,
            "flagNLOSO2": 1.0,
            "flagNLOSO3": 1.0,
            "flagNLOSS": 1.0,
            "flagNLOSS2": 1.0,
            "flagS3": 1.0,
        }

        coeffs = list(flags.keys())
        coeffs.extend(["dSS", "d5", "dSO", "a6"])
        n = len(coeffs)

        dc = {}
        for key, value in flags.items():
            dc[key] = value

        # Actual coeffs inside the Hamiltonian
        a6_fit = a6_NS(self.nu)
        dSO_fit = dSO(self.nu, self.ap, self.am)
        dc["a6"] = a6_fit
        dc["dSO"] = dSO_fit
        dc["d5"] = 0.0  # Canonical
        dc["dSS"] = 0.0  # Canonical
        cfs = CalibCoeffs(dc)
        self.H.calibration_coeffs = cfs

    def _evaluate_model(self):

        try:
            # Step 1: compute the dynamics
            # This includes both the initial conditions
            # and the integration of the ODEs
            if not self.settings["postadiabatic"]:
                dynamics_low, dynamics_fine = compute_dynamics_opt(
                    self.omega0,
                    self.H,
                    self.RR,
                    self.chi_1,
                    self.chi_2,
                    self.m_1,
                    self.m_2,
                    rtol=1e-10,
                    atol=1e-11,
                    params=self.eob_pars,
                    backend="ode",
                    step_back=self.step_back,
                )
            else:
                dynamics_low, dynamics_fine = compute_combined_dynamics(
                    self.omega0,
                    self.H,
                    self.RR,
                    self.chi_1,
                    self.chi_2,
                    self.m_1,
                    self.m_2,
                    tol=1e-11,
                    params=self.eob_pars,
                    backend="ode",
                    step_back=self.step_back,
                )

            len_fine = dynamics_fine[-1, 0] - dynamics_fine[0, 0]
            if len_fine < self.step_back:
                self.step_back = len_fine

            # Combine the low and high dynamics
            dynamics = np.vstack((dynamics_low, dynamics_fine))
            self.dynamics = dynamics

            t_fine = dynamics_fine[:, 0]

            # Step 2: compute the reference point based on Kerr r_ISCO of remnant
            # with final spin

            r_ISCO, _ = Kerr_ISCO(
                self.chi_1,
                self.chi_2,
                self.m_1,
                self.m_2,
            )

            self.r_ISCO = r_ISCO

            r_fine = dynamics_fine[:, 1]

            if r_ISCO < r_fine[-1]:
                # In some corners of parameter space r_ISCO can be *after*
                # the end of the dynamics. In those cases just use the last
                # point of the dynamics as the reference point
                t_ISCO = t_fine[-1]
                logger.debug("Kerr ISCO after the last r in the dynamics")
            else:
                # Find a time corresponding to r_ISCO
                sp = 0.001
                N = int((t_fine[-1] - t_fine[0]) / sp)
                zoom = np.linspace(t_fine[0], t_fine[-1], N)
                n = len(t_fine)
                intrp_r = spline.cspline(n)
                intrp_r.init(t_fine, r_fine)
                r_zoomed_in = intrp_r.eval_e_vector(zoom)
                idx = (np.abs(r_zoomed_in - r_ISCO)).argmin()
                t_ISCO = zoom[idx]

            # We define the attachment with respect to t_ISCO
            self.t_ISCO = t_ISCO
            t_attach = t_ISCO - self.NR_deltaT
            self.t_attach_predicted = t_attach

            # If the fit for NR_deltaT is too negative and overshoots the end of the
            # dynamics we attach the MR at the last point
            self.attachment_check = 0.0
            if t_attach > t_fine[-1]:
                self.attachment_check = 1.0
                t_attach = t_fine[-1]
                logger.debug(
                    "NR_deltaT too negative, attaching the MR at the last point of the dynamics, careful!"
                )

            self.t_attach = t_attach

            # Step 3: compute the special calibration coefficients to tame zeros in some odd-m modes
            input_value_fits = InputValueFits(
                self.m_1, self.m_2, [0.0, 0.0, self.chi_1], [0.0, 0.0, self.chi_2]
            )
            amp_fits = input_value_fits.hsign()
            # The following values were determined *empirically*
            self.amp_thresholds = {
                (2, 1): 300,
                (4, 3): 200 * self.nu * (1 - 0.8 * self.chi_A),
                (5, 5): 2000,
            }
            if np.abs(self.q - 1) < 1e-14 and np.abs(self.chi_A) < 1e-14:
                pass
            else:
                compute_special_coeffs(
                    dynamics, t_attach, self.eob_pars, amp_fits, self.amp_thresholds
                )

            # Step 4: compute the waveform on finely sampled dynamics
            hlms_fine = compute_hlms_new(dynamics_fine[:, 1:], self.eob_pars)
            omega_orb_fine = dynamics_fine[:, -2]
            # Polar dynamics, r,pr,omega_orb
            polar_dynamics_fine = [
                dynamics_fine[:, 1],
                dynamics_fine[:, 3],
                omega_orb_fine,
            ]

            # Step 5: compute NQCs coeffs

            nqc_coeffs = NQC_correction(
                hlms_fine,
                t_fine,
                polar_dynamics_fine,
                t_ISCO,
                self.NR_deltaT,
                self.m_1,
                self.m_2,
                self.chi_1,
                self.chi_2,
            )
            self.nqc_coeffs = nqc_coeffs
            # Apply NQC corrections to high sampling modes
            apply_nqc_corrections(hlms_fine, nqc_coeffs, polar_dynamics_fine)

            # Step 6: compute the modes in the inspiral
            hlms_low = compute_hlms_new(dynamics_low[:, 1:], self.eob_pars)
            # Apply the NQC corrections to inspiral modes
            omega_orb_low = dynamics_low[:, -2]
            # Polar dynamics, r,pr,omega_orb
            polar_dynamics_low = [dynamics_low[:, 1], dynamics_low[:, 3], omega_orb_low]
            apply_nqc_corrections(hlms_low, nqc_coeffs, polar_dynamics_low)

            # Step 7: Concatenate low and high sampling modes
            hlms_joined = concatenate_modes(hlms_low, hlms_fine)

            # Step 8: interpolate the modes onto the desired spacing
            t_new = np.arange(dynamics[0, 0], dynamics[-1, 0], self.delta_T)

            t_original = dynamics[:, 0]
            phi_orb = dynamics[:, 2]
            hlms_interp = interpolate_modes_fast(
                t_original,
                t_new,
                hlms_joined,
                phi_orb,
            )

            # Step 9: construct the full IMR waveform
            t_full, hlms_full = compute_IMR_modes(
                t_new,
                hlms_interp,
                t_fine,
                hlms_fine,
                self.m_1,
                self.m_2,
                self.chi_1,
                self.chi_2,
                t_attach,
                mixed_modes=self.mixed_modes,
            )

            self.t = t_full
            self.waveform_modes = {}
            # Step 10: fill the final dictionary of modes
            # for key, value in hlms_full.items():
            #    self.waveform_modes[f"{key[0]},{key[1]}"] = value
            for key in self.return_modes:
                self.waveform_modes[f"{key[0]},{key[1]}"] = hlms_full[key]
            self.success = True
        except Exception as e:

            logger.error(
                f"Waveform generation failed for q={self.q},chi_1={self.chi_1},chi_2={self.chi_2},omega0={self.omega0}"
            )
            raise e
