import logging
import re
from copy import deepcopy
from typing import Any, Callable, Dict

import lal
import numpy as np
import quaternion
import scri
from pygsl import spline
from pyseobnr.eob.dynamics.integrate_ode import (augment_dynamics,
                                                 compute_dynamics_opt)
from pyseobnr.eob.dynamics.integrate_ode_prec import \
    compute_dynamics_quasiprecessing
from pyseobnr.eob.dynamics.pn_evolution_opt import ODE_system_RHS_omega_opt
from pyseobnr.eob.dynamics.postadiabatic_C import (Kerr_ISCO,
                                                   compute_combined_dynamics)
from pyseobnr.eob.dynamics.postadiabatic_C_fast import \
    compute_combined_dynamics as compute_combined_dynamics_fast
from pyseobnr.eob.dynamics.postadiabatic_C_prec import \
    compute_combined_dynamics as compute_combined_dynamics_qp
from pyseobnr.eob.fits.fits_Hamiltonian import (NR_deltaT, NR_deltaT_NS, a6_NS,
                                                dSO)
from pyseobnr.eob.fits.GSF_fits import GSF_amplitude_fits
from pyseobnr.eob.fits.IV_fits import InputValueFits
from pyseobnr.eob.hamiltonian import Hamiltonian
from pyseobnr.eob.utils.containers import CalibCoeffs, EOBParams
from pyseobnr.eob.utils.math_ops_opt import my_dot, my_norm
from pyseobnr.eob.utils.utils_precession_opt import (
    SEOBRotatehIlmFromhJlm_opt, custom_swsh, interpolate_quats,
    quat_ringdown_approx_opt)
from pyseobnr.eob.utils.waveform_ops import frame_inv_amp
from pyseobnr.eob.waveform.compute_hlms import (NQC_correction,
                                                apply_nqc_corrections,
                                                compute_IMR_modes,
                                                concatenate_modes,
                                                interpolate_modes_fast)
from pyseobnr.eob.waveform.waveform import compute_hlms as compute_hlms_new
from pyseobnr.eob.waveform.waveform import (compute_newtonian_prefixes,
                                            compute_special_coeffs)
from pyseobnr.models.model import Model
from rich.logging import RichHandler
from rich.traceback import install
from scipy.interpolate import CubicSpline
from scipy.optimize import root
from spherical_functions import SWSH

# Setup the logger to work with rich
logger = logging.getLogger(__name__)
logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))
logger.setLevel("INFO")
# Setup rich to get nice tracebacks
install()

# List of valid modes
VALID_MODES = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5)]


from lalinference.imrtgr import nrutils


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
        # Deal with reference and starting frequencies
        self.f_ref = self.settings.get(
            "f_ref", omega0 / (self.M * lal.MTSUN_SI * np.pi)
        )
        r_min = 10.0
        omega_min = 1/r_min**(3./2)
        if omega0> omega_min:
            logger.warning("Short waveform, changing omega0")
            omega0 = omega_min
        self.f0 = omega0 / (self.M * lal.MTSUN_SI * np.pi)
        self.omega0 = omega0

        if np.abs(self.f_ref - self.f0) > 1e-10:
            # The reference frequency is not the same as the starting frequency
            # If the starting frequency is smaller than the reference frequency,
            # we don't need to adjust anything here, and will account for this
            # with a phase shift of the dynamics.
            # If the reference frequency is _less_ than the starting frequency
            # then just change the starting frequency to the reference frequency
            if self.f_ref < self.f0:
                self.omega0 = self.f_ref * (self.M * lal.MTSUN_SI * np.pi)
                self.f0 = self.omega0 / (self.M * lal.MTSUN_SI * np.pi)

        self.step_back = self.settings.get("step_back", 250.0)
        self.chi_S = (self.chi_1 + self.chi_2) / 2
        self.chi_A = (self.chi_1 - self.chi_2) / 2
        self.ap = self.m_1 * self.chi_1 + self.m_2 * self.chi_2
        self.am = self.m_1 * self.chi_1 - self.m_2 * self.chi_2
        self.dt = self.settings["dt"]
        self.delta_T = self.dt / (self.M * lal.MTSUN_SI)
        #print(f"In SI units, dt = {self.dt}. In geometric units, with M={self.M}, delta_T={self.delta_T}")
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
            a1 = self.chi_1,
            a2 = self.chi_2,
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
        if self.settings.get("postadiabatic",False):
            self.PA_style = self.settings.get("PA_style","analytic")
            self.PA_order = self.settings.get("PA_order",8)

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

        dc = {}
        # Actual coeffs inside the Hamiltonian
        a6_fit = a6_NS(self.nu)
        dSO_fit = dSO(self.nu, self.ap, self.am)
        dc["a6"] = a6_fit
        dc["dSO"] = dSO_fit

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
                    rtol=1e-11,
                    atol=1e-12,
                    params=self.eob_pars,
                    backend="ode",
                    step_back=self.step_back,

                )
            else:
                if self.PA_style=="root":
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
                        PA_order=self.PA_order
                    )
                elif self.PA_style =="analytic":
                    dynamics_low, dynamics_fine = compute_combined_dynamics_fast(
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
                        PA_order=self.PA_order
                    )

            len_fine = dynamics_fine[-1, 0] - dynamics_fine[0, 0]
            if len_fine < self.step_back:
                self.step_back = len_fine

            # Combine the low and high dynamics
            dynamics = np.vstack((dynamics_low, dynamics_fine))

            self.dynamics = dynamics
            if np.abs(self.f_ref - self.f0) > 1e-10:
                # Reference frequency is not the same as starting frequency
                # To account for the LAL conventions, shift things so that
                # the orbital phase is 0 at f_ref
                omega_orb = dynamics[:, -2]
                t_d = dynamics[:, 0]
                # Approximate
                f_22 = omega_orb / (self.M * lal.MTSUN_SI * np.pi)
                if self.f_ref > f_22[-1]:
                    logger.error(
                        "f_ref is larger than the highest frequency in the inspiral!"
                    )
                    raise ValueError
                # Solve for time when f_22 = f_ref
                intrp = CubicSpline(t_d, f_22)
                guess = t_d[np.argmin(np.abs(f_22 - self.f_ref))]
                rhs = lambda x: np.abs(intrp(x) - self.f_ref)
                res = root(rhs, guess)

                t_correct = res.x
                if not res.success:
                    logger.error(
                        "Failed to find the time corresponding to requested f_ref."
                    )
                    raise ValueError
                phase = dynamics[:, 2]
                intrp_phase = CubicSpline(t_d, phase)
                phase_shift = intrp_phase(t_correct)
                # Shift the phase for all dynamics arrays
                self.dynamics[:, 2] -= phase_shift
                dynamics_low[:, 2] -= phase_shift
                dynamics_fine[:, 2] -= phase_shift

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
            del hlms_joined
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


class SEOBNRv5PHM_opt(Model):
    """Represents an precessing SEOBNRv5PHM waveform with new MR choices."""

    def __init__(
        self,
        q: float,
        chi1_x: float,
        chi1_y: float,
        chi1_z: float,
        chi2_x: float,
        chi2_y: float,
        chi2_z: float,
        omega0: float,
        H: Hamiltonian,
        RR: Callable,
        settings: Dict[Any, Any] = None,
    ) -> None:
        """Initialize the SEOBNRv5PHM approximant

        Args:
            q (float): Mass ratio m1/m2 >= 1
            chi_1 (float): z component of the dimensionless spin of primary
            chi_2 (float): z component of the dimensionless spin of secondary
            omega0 (float): Initial orbital frequency, in geomtric units
            coeffs (Dict[str, Any], optional): Calibration coefficient. Defaults to None.
            settings (Dict[Any, Any], optional): The settings. Defaults to None.
        """

        self.settings = self._default_settings()
        # If we were given settings, override the defaults
        if settings is not None:
            self.settings.update(**settings)

        self.M = self.settings["M"]
        self.q = q
        self.ell_max = self.settings.get("ell_max", 5)

        self.chi_1 = np.array([chi1_x, chi1_y, chi1_z])
        self.chi_2 = np.array([chi2_x, chi2_y, chi2_z])

        self.a1 = my_norm(self.chi_1)
        self.a2 = my_norm(self.chi_2)

        self.NR_deltaT = 0

        self.RR = RR

        self.m_1 = q / (1.0 + q)
        self.m_2 = 1.0 / (1.0 + q)

        self.nu = self.m_1 * self.m_2 / (self.m_1 + self.m_2) ** 2

        self.omega0 = omega0

        # Deal with reference and starting frequencies
        self.f_ref = self.settings.get(
            "f_ref", omega0 / (self.M * lal.MTSUN_SI * np.pi)
        )
        self.omega_ref = self.f_ref * (self.M * lal.MTSUN_SI * np.pi)
        self.f0 = omega0 / (self.M * lal.MTSUN_SI * np.pi)
        self.omega0 = omega0

        if np.abs(self.f_ref - self.f0) > 1e-10:
            # The reference frequency is not the same as the starting frequency
            # If the starting frequency is smaller than the reference frequency,
            # we don't need to adjust anything here, and will account for this
            # with a phase shift of the dynamics.
            # If the reference frequency is _less_ than the starting frequency
            # then just change the starting frequency to the reference frequency
            if self.f_ref < self.f0:
                self.omega0 = self.f_ref * (self.M * lal.MTSUN_SI * np.pi)
                self.f0 = self.omega0 / (self.M * lal.MTSUN_SI * np.pi)

        self.step_back = self.settings.get("step_back", 250.0)
        # print(f"f0={self.f0},f_ref={self.f_ref}")
        self.beta_approx = self.settings["beta_approx"]
        self.backend = self.settings.get("backend", "dopri5")
        # z-component because LN_0 = [0,0,1]
        self.chi_S = (chi1_z + chi2_z) / 2.0
        self.chi_A = (chi1_z - chi2_z) / 2.0

        self.dt = self.settings["dt"]
        self.delta_T = self.dt / (self.M * lal.MTSUN_SI)

        self.prefixes = compute_newtonian_prefixes(self.m_1, self.m_2)

        self.tplspin = (1 - 2 * self.nu) * self.chi_S + (self.m_1 - self.m_2) / (
            self.m_1 + self.m_2
        ) * self.chi_A

        self.phys_pars = dict(
            m_1=self.m_1,
            m_2=self.m_2,
            chi_1=chi1_z,
            chi_2=chi2_z,
            chi_1x=chi1_x,
            chi_1y=chi1_y,
            chi_1z=chi1_z,
            chi_2x=chi2_x,
            chi_2y=chi2_y,
            chi_2z=chi2_z,
            a1 = self.a1,
            a2 = self.a2,
            omega=self.omega0,
            omega_circ=self.omega0,
        )

        self._initialize_params(self.phys_pars)

        # Initialize the Hamiltonian
        self.H = H(self.eob_pars)
        # Uncomment and comment the line above to make the python Hamiltonian work
        # self.H = H()

        self.modes_list = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5)]
        self.mixed_modes = [(3, 2), (4, 3)]

    def _default_settings(self):
        settings = dict(
            M=50.0,  # Total mass in solar masses
            dt=2.4627455127717882e-05,  # Desired time spacing, *in seconds*
            debug=False,  # Run in debug mode
            postadiabatic=False,  # Use postadiabatic?
            return_modes=[(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5)],
            polarizations_from_coprec=False,  # True for computing directly polarizations
        )
        return settings

    def _initialize_params(self, phys_pars):
        """
        Re-initialize all parameters to make sure everthing is reset
        """
        self.eob_pars = EOBParams(phys_pars, {})
        self.eob_pars.flux_params.rho_initialized = False
        self.eob_pars.flux_params.prefixes = np.array(self.prefixes)
        self.eob_pars.flux_params.prefixes_abs = np.abs(
            self.eob_pars.flux_params.prefixes
        )
        self.step_back = self.settings.get("step_back", 250.0)

        self.eob_pars.flux_params.extra_PN_terms = self.settings.get(
            "extra_PN_terms", True
        )

        self.eob_pars.aligned = False

    def _set_H_coeffs(self):

        dc = {}
        # Actual coeffs inside the Hamiltonian
        a6_fit = a6_NS(self.nu)
        # dSO_fit = dSO(self.nu, self.ap, self.am)
        dc["a6"] = a6_fit
        # dc["dSO"] = dSO_fit

        cfs = CalibCoeffs(dc)
        self.H.calibration_coeffs = cfs

    # Uncomment this, and comment the function above to make the python Hamiltonian working again
    """
    def _set_H_coeffs(self):
        keys = ["a6", "dSO"]
        n = len(keys)
        dtype = list(zip(keys, n * ["f8"]))
        calibration_coeffs = np.zeros(dtype=dtype, shape=(1,))
        # Actual coeffs inside the Hamiltonian
        a6_fit = a6_NS(self.nu)
        calibration_coeffs["a6"] = a6_fit
        # Note that dSO is set inside the actual dynamics evolution,
        # since it varies as the spins evolve.
        # cfs = CalibCoeffs(dc)
        self.H.calibration_coeffs = calibration_coeffs
    """

    def __call__(self, parameters=None):
        # Evaluate the model

        # Initialize the containers
        self._initialize_params(self.phys_pars)

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

    def _unpack_scri(self, w):
        result = {}
        for key in w.LM:
            result[f"{key[0]},{key[1]}"] = 1 * w.data[:, w.index(key[0], key[1])]
        return result

    def _add_negative_m_modes(self, waveform_modes):
        """Add the negative m modes using the usual

        Note:
            This should only be called on *co-precessing frame* modes

        Args:
            waveform_modes (Dict[Any,Any]): Dictionary of modes with m>0

        Returns:
            Dict[Any,Any]: Dictionary of modes with |m| !=0
        """
        result = deepcopy(waveform_modes)
        for key in waveform_modes.keys():
            ell, m = key
            result[(ell, -m)] = (-1) ** ell * np.conjugate(waveform_modes[key])
        return result

    def _package_modes(self, waveform_modes, ell_min=2, ell_max=5):
        keys = waveform_modes.keys()
        shape = waveform_modes[(2, 2)].shape
        result = np.zeros((shape[0], 32), dtype=np.complex128)
        i = 0
        for ell in range(ell_min, ell_max + 1):
            for m in range(-ell, ell + 1):

                if (ell, m) in keys:
                    result[:, i] = waveform_modes[(ell, m)]
                i += 1
        return result

    def _evaluate_model(self):
        try:
            # TODO: implement PA dynamics

            # Generate PN and EOB dynamics
            if not self.settings["postadiabatic"]:
                self.ODE_system_RHS = ODE_system_RHS_omega_opt
                (
                    dynamics_low,
                    dynamics_fine,
                    t_PN,
                    dyn_PN,
                    chi1L_spline,
                    chi2L_spline,
                    chi1LN_spline,
                    chi2LN_spline,
                    splines,
                    dynamics,
                ) = compute_dynamics_quasiprecessing(
                    self.omega_ref,
                    self.omega0,
                    self.H,
                    self.RR,
                    self.m_1,
                    self.m_2,
                    self.chi_1,
                    self.chi_2,
                    rtol=1e-9,  # 1e-11,
                    atol=1e-9,  # 1e-12,
                    backend=self.backend,
                    params=self.eob_pars,
                    step_back=self.step_back,
                )
            else:
                self.ODE_system_RHS = ODE_system_RHS_omega_opt

                (

                    dynamics_low,
                    dynamics_fine,
                    t_PN,
                    dyn_PN,
                    chi1L_spline,
                    chi2L_spline,
                    chi1LN_spline,
                    chi2LN_spline,
                    splines,
                    dynamics,
                ) = compute_combined_dynamics_qp(
                    self.omega_ref,
                    self.omega0,
                    self.H,
                    self.RR,
                    self.m_1,
                    self.m_2,
                    self.chi_1,
                    self.chi_2,
                    self.ODE_system_RHS,
                    tol=1e-12,
                    backend=self.backend,
                    params=self.eob_pars,
                    step_back=self.step_back,
                )


            tmp_LN = dyn_PN[:, :3]
            self.LNhat = tmp_LN

            self.pn_dynamics = np.c_[t_PN, dyn_PN]
            self.dynamics = dynamics_low

            omega_orb_fine = dynamics_fine[:, 6]
            omega_EOB_low = dynamics_low[:, 6]

            tmp_LN_low = splines["L_N"](omega_EOB_low)
            tmp_LN_fine = splines["L_N"](omega_orb_fine)

            tmp_LN_low_norm = np.sqrt(np.einsum("ij,ij->i", tmp_LN_low, tmp_LN_low))

            tmp_LN_low = (tmp_LN_low.T / tmp_LN_low_norm).T

            tmp_LN_fine_norm = np.sqrt(np.einsum("ij,ij->i", tmp_LN_fine, tmp_LN_fine))

            tmp_LN_fine = (tmp_LN_fine.T / tmp_LN_fine_norm).T
            self.dynamics = dynamics

            if np.abs(self.f_ref - self.f0) > 1e-10:
                # Reference frequency is not the same as starting frequency
                # To account for the LAL conventions, shift things so that
                # the orbital phase is 0 at f_ref
                omega_orb = dynamics[:, 6]
                t_d = dynamics[:, 0]
                # Approximate
                f_22 = omega_orb / (self.M * lal.MTSUN_SI * np.pi)
                if self.f_ref > f_22[-1]:
                    logger.error(
                        "f_ref is larger than the highest frequency in the inspiral!"
                    )
                    raise ValueError
                # Solve for time when f_22 = f_ref
                intrp = CubicSpline(t_d, f_22)
                guess = t_d[np.argmin(np.abs(f_22 - self.f_ref))]
                rhs = lambda x: np.abs(intrp(x) - self.f_ref)
                res = root(rhs, guess)

                t_correct = res.x
                if not res.success:
                    logger.error(
                        "Failed to find the time corresponding to requested f_ref."
                    )
                    raise ValueError
                phase = dynamics[:, 2]
                intrp_phase = CubicSpline(t_d, phase)
                phase_shift = intrp_phase(t_correct)
                # Shift the phase for all dynamics arrays
                self.dynamics[:, 2] -= phase_shift
                dynamics_low[:, 2] -= phase_shift
                dynamics_fine[:, 2] -= phase_shift

            self.splines = splines

            # Step 2, i): Determine the spins at r=10 M
            # We need this so that we can compute our reference point,
            # which is r_ISCO. Note that by definition, r_ISCO < 10

            t_fine = dynamics_fine[:, 0]

            irt = CubicSpline(1.0 / dynamics[:, 1], dynamics[:, 0])
            r_ref = 1.0 / 10.0
            t_r10M = irt(r_ref)

            iom = CubicSpline(dynamics[:, 0], dynamics[:, 6])
            om_r10M = iom(t_r10M)
            chi1_spline = splines["chi1"]
            chi2_spline = splines["chi2"]
            LN_spline = splines["L_N"]

            chi1_om_r10M = chi1_spline(om_r10M)
            chi2_om_r10M = chi2_spline(om_r10M)
            LN_om_r10M = LN_spline(om_r10M)
            chi1L_om_r10M = np.dot(chi1_om_r10M, LN_om_r10M)
            chi2L_om_r10M = np.dot(chi2_om_r10M, LN_om_r10M)

            # Step 2, ii): compute the reference point based on Kerr r_ISCO of remnant
            # with final spin

            r_ISCO, _ = Kerr_ISCO(
                chi1L_om_r10M,
                chi2L_om_r10M,
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
                intrp_r = CubicSpline(t_fine, r_fine)
                zoom = np.arange(t_fine[0], t_fine[-1], 0.001)
                r_zoomed_in = intrp_r(zoom)
                idx = (np.abs(r_zoomed_in - r_ISCO)).argmin()
                t_ISCO = zoom[idx]

            om_rISCO = iom(t_ISCO)
            chi1_om_rISCO = chi1_spline(om_rISCO)
            chi2_om_rISCO = chi2_spline(om_rISCO)
            LN_om_rISCO = LN_spline(om_rISCO)
            chi1L_om_rISCO = np.dot(chi1_om_rISCO, LN_om_rISCO)
            chi2L_om_rISCO = np.dot(chi2_om_rISCO, LN_om_rISCO)

            ap = (
                chi1L_om_rISCO * self.eob_pars.p_params.X_1
                + chi2L_om_rISCO * self.eob_pars.p_params.X_2
            )
            am = (
                chi1L_om_rISCO * self.eob_pars.p_params.X_1
                - chi2L_om_rISCO * self.eob_pars.p_params.X_2
            )

            # Compute the timeshift from the reference point
            self.NR_deltaT = NR_deltaT_NS(self.nu) + NR_deltaT(self.nu, ap, am)

            self.t_ISCO = t_ISCO
            self.omega_rISCO = om_rISCO
            t_attach = t_ISCO - self.NR_deltaT

            self.t_attach_predicted = t_attach

            # If the fit for NR_deltaT is too negative and overshoots the end of the
            # dynamics, we attach the MR at the last point
            self.attachment_check = 0.0
            if t_attach > t_fine[-1]:
                self.attachment_check = 1.0
                t_attach = t_fine[-1]
                logger.debug(
                    "NR_deltaT too negative, attaching the MR at the last point of the dynamics, careful!"
                )

            self.t_attach = t_attach

            # For the following steps, we also need spins at the attachment point
            # First compute the orbital frequency at that point
            iom_orb_fine = CubicSpline(dynamics_fine[:, 0], dynamics_fine[:, 6])

            omega_orb_attach = iom_orb_fine(t_attach)
            self.omega_attach = omega_orb_attach

            # Now find the spins at attachment
            LN_attach = LN_spline(omega_orb_attach)
            chi1_attach = chi1_spline(omega_orb_attach)
            chi2_attach = chi2_spline(omega_orb_attach)

            chi1L_attach = my_dot(chi1_attach, LN_attach)
            chi2L_attach = my_dot(chi2_attach, LN_attach)

            Lvec_attach = splines["Lvec"](omega_orb_attach)
            Jf_attach = (
                chi1_attach * self.m_1 * self.m_1
                + chi2_attach * self.m_2 * self.m_2
                + Lvec_attach
            )

            Lvec_hat_attach = Lvec_attach / my_norm(Lvec_attach)
            Jfhat_attach = Jf_attach / my_norm(Jf_attach)

            # print(f"tattach : Lvec = {Lvec_attach}, s1 = {chi1_attach*self.m_1*self.m_1}, s2 = {chi2_attach*self.m_2*self.m_2}, Jf bo = {Jf}")

            self.eob_pars.p_params.chi_1 = chi1L_attach
            self.eob_pars.p_params.chi_2 = chi2L_attach

            # Remember to update _all_ derived spin quantites as well
            self.eob_pars.p_params.update_spins(
                self.eob_pars.p_params.chi_1, self.eob_pars.p_params.chi_2
            )

            # Step 3: compute the special calibration coefficients to tame zeros in some odd-m modes
            input_value_fits = InputValueFits(
                self.m_1, self.m_2, [0.0, 0.0, chi1L_attach], [0.0, 0.0, chi2L_attach]
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
                chi1L_attach,
                chi2L_attach,
            )

            self.nqc_coeffs = nqc_coeffs

            # Apply NQC corrections to high sampling modes
            apply_nqc_corrections(hlms_fine, nqc_coeffs, polar_dynamics_fine)

            # Compute the final state quantities
            # Following the same approach as in SEOBNRv4PHM,
            # evaluate the final state formulas using the spins at
            # r=10 M

            # Spin magnitudes
            a1 = my_norm(chi1_om_r10M)
            a2 = my_norm(chi2_om_r10M)

            # Tilt angles w.r.t L_N
            if a1 != 0:
                angle = chi1L_om_r10M / a1
                if angle < -1:
                    tilt1 = np.pi
                elif angle > 1:
                    tilt1 = 0.0
                else:
                    tilt1 = np.arccos(angle)
            else:
                tilt1 = 0.0

            if a2 != 0:
                angle = chi2L_om_r10M / a2
                if angle < -1:
                    tilt2 = np.pi
                elif angle > 1:
                    tilt2 = 0.0
                else:
                    tilt2 = np.arccos(angle)
            else:
                tilt2 = 0.0

            # Angle between the inplane spin components
            chi1_perp_r10M = chi1_om_r10M - chi1L_om_r10M * LN_om_r10M
            chi2_perp_r10M = chi2_om_r10M - chi2L_om_r10M * LN_om_r10M

            if a1 == 0 or a2 == 0:
                phi12 = np.pi / 2.0

            else:
                phi12 = np.arccos(np.dot(chi1_perp_r10M / a1, chi2_perp_r10M / a2))

            # Get the final mass
            final_mass = nrutils.bbh_final_mass_non_precessing_UIB2016(
                self.m_1, self.m_2, chi1L_om_r10M, chi2L_om_r10M
            )

            # Call non-precessing HBR fit to get the *sign* of the final spin
            final_spin_nonprecessing = nrutils.bbh_final_spin_non_precessing_HBR2016(
                self.m_1, self.m_2, chi1L_om_r10M, chi2L_om_r10M, version="M3J4"
            )
            sign_final_spin = final_spin_nonprecessing / np.linalg.norm(
                final_spin_nonprecessing
            )

            # Compute the magnitude of the final spin using the precessing fit
            final_spin = nrutils.bbh_final_spin_precessing_HBR2016(
                self.m_1, self.m_2, a1, a2, tilt1, tilt2, phi12, version="M3J4"
            )
            # Flip sign if needed
            final_spin *= sign_final_spin

            # Double check all is well
            if np.isnan(dynamics_low[:, 1:]).any():
                raise ValueError

            # Step 6: compute the modes in the inspiral
            hlms_low = compute_hlms_new(dynamics_low[:, 1:], self.eob_pars)

            # Apply the NQC corrections to inspiral modes
            t_low = dynamics_low[:, 0]
            omega_orb_low = dynamics_low[:, 6]

            # Polar dynamics, r,pr,omega_orb
            polar_dynamics_low = [dynamics_low[:, 1], dynamics_low[:, 3], omega_orb_low]
            apply_nqc_corrections(hlms_low, nqc_coeffs, polar_dynamics_low)

            # Step 7: Concatenate low and high sampling modes
            hlms_joined = concatenate_modes(hlms_low, hlms_fine)

            t_new = np.arange(dynamics[0, 0], dynamics[-1, 0], self.delta_T)
            # Step 8: interpolate the modes onto the desired spacing
            hlms_interp = interpolate_modes_fast(
                dynamics[:, 0],
                t_new,
                hlms_joined,
                dynamics[:, 2],
            )

            # Step 9: construct the full IMR waveform
            t_full, imr_full = compute_IMR_modes(
                t_new,
                hlms_interp,
                t_fine,
                hlms_fine,
                self.m_1,
                self.m_2,
                chi1L_attach,
                chi2L_attach,
                t_attach,
                mixed_modes=self.mixed_modes,
                final_state=[final_mass, final_spin],
            )

            t_full -= t_full[0]

            # Step 10: twist up the co-precessing modes
            # Package modes for scri
            # i) Add negative m modes

            amp_inv = frame_inv_amp(imr_full)
            idx_max = np.argmax(amp_inv)
            # ii) Set to zero all missing modes in co-precessing frame

            self.imr_full = imr_full

            idx = np.where(t_full < t_attach)[0]

            # Quaternion representing the rotation to the frame where L_N is
            # along z
            qt = quaternion.as_quat_array(np.zeros((len(t_full), 4)))

            self.idx = idx
            self.t_low = t_low
            self.t_fine = t_fine
            self.tmp_LN_low = tmp_LN_low
            self.tmp_LN_fine = tmp_LN_fine
            self.final_spin = final_spin
            self.final_mass = final_mass
            self.t_attach = t_attach
            self.omega_orb_attach = omega_orb_attach
            self.splines = splines

            (
                t_dyn,
                quatJ2P_dyn,
                quat_postMerger,
                alphaI2J,
                betaI2J,
                gammaI2J,
            ) = quat_ringdown_approx_opt(
                self.nu,
                self.m_1,
                self.m_2,
                idx,
                t_full,
                t_low,
                t_fine,
                tmp_LN_low,
                tmp_LN_fine,
                final_spin,
                final_mass,
                t_attach,
                omega_orb_attach,  # omega_peak,
                Lvec_hat_attach,
                Jfhat_attach,
                splines,
                self.settings.get("rd_approx", True),
                beta_approx=self.beta_approx,
            )

            # Construct the full rotation , by concatinating inspiral
            # and merger-ringdown
            self.quatJ2P_dyn = quatJ2P_dyn
            self.t_intrp = t_dyn
            self.t_forres = t_full[idx]
            # qt[idx] = quaternion.squad(quatJ2P_dyn, t_dyn, t_full[idx])
            qt[idx] = interpolate_quats(quatJ2P_dyn, t_dyn, t_full[idx])
            qt[idx[-1] :] = quat_postMerger

            if self.settings["polarizations_from_coprec"] == False:

                imr_full = self._add_negative_m_modes(imr_full)
                # We only need to package modes here
                imr_full = self._package_modes(imr_full)
                # iii) Create a co-precessing frame scri waveform
                w = scri.WaveformModes(
                    dataType=scri.h,
                    t=t_full,
                    data=imr_full,
                    ell_min=2,
                    ell_max=5,
                    frameType=scri.Coprecessing,
                    r_is_scaled_out=True,
                    m_is_scaled_out=True,
                )
                # Before rotating, store the complete co-precessing frame modes, if asked
                if self.settings.get("return_coprec", False):
                    self.w = deepcopy(w)
                    self.coprecessing_modes = self._unpack_scri(w)
                # Rotate to the P->J. This is a time-dependent rotation
                w.frame = qt
                w_modes = w.to_inertial_frame()

                # Store the time array

                self.t = t_full - t_full[idx_max]
                # Store the I2J quatnerion
                self.quaternion = quaternion.as_float_array(qt)

                # Store all the angles
                # For posterity store the rotation as Euler angles
                anglesJ2P = quaternion.as_euler_angles(qt).T
                self.anglesI2J = [alphaI2J, betaI2J, gammaI2J]
                self.anglesJ2P = anglesJ2P

                modes_lmax = self.ell_max

                # Rotate J->I. This is a rotation by constant angles
                w_I = SEOBRotatehIlmFromhJlm_opt(
                    w_modes,  # hJlm time series, complex values on fixed sampling
                    modes_lmax,  # Input: maximum value of l in modes (l,m)
                    alphaI2J,  # Input: Euler angle alpha I->J
                    betaI2J,  # Input: Euler angle beta I->J
                    gammaI2J,  # Input: Euler angle gamma I->J
                )

                # Unpack the modes
                waveform_modesI = self._unpack_scri(w_I)
                self.waveform_modes = waveform_modesI

                self.success = True
            else:
                # Construct full rotation
                qJ2P = qt.conj()
                qI2J = quaternion.from_euler_angles(-gammaI2J, -betaI2J, -alphaI2J)
                qI02I = quaternion.from_euler_angles(
                    self.settings["phiref"], self.settings["inclination"], 0.0
                )

                qTot = qJ2P * qI2J * qI02I

                alphaTot, betaTot, gammaTot = quaternion.as_euler_angles(qTot).T

                sYlm = custom_swsh(betaTot, alphaTot)
                # Construct polarizations
                hpc = np.zeros(imr_full[(2, 2)].size, dtype=np.complex)
                for ell, emm in imr_full.keys():
                    # sYlm = SWSH(qTot,-2,[ell,emm])
                    hpc += (
                        sYlm[ell, emm] * imr_full[(ell, emm)]
                        + sYlm[ell, -emm] * pow(-1, ell) * imr_full[(ell, emm)].conj()
                    )

                hpc *= np.exp(2j * gammaTot)

                # self.hpc = hpcIfromP
                self.t = t_full - t_full[idx_max]
                self.hpc = hpc
                # self.coprec = imr_full_2
                # self.anglesTot = [alphaTot, betaTot, gammaTot]
                self.success = True
        except Exception as e:

            logger.error(
                f"Waveform generation failed for q={self.q},chi_1={self.chi_1},chi_2={self.chi_2},omega_ref={self.omega_ref},omega0 = {self.omega0}"
            )
            raise e
