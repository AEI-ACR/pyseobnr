from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any, Callable, Dict, Final, cast, get_args

import lal
import numpy as np
import quaternion
import scri
from pygsl_lite import spline
from scipy.interpolate import CubicSpline
from scipy.optimize import root

from pyseobnr.eob.dynamics.integrate_ode import compute_dynamics_opt
from pyseobnr.eob.dynamics.integrate_ode_prec import (
    InitialConditionPostadiabaticTypes as InitialConditionPostadiabaticTypesPrecessing,
)
from pyseobnr.eob.dynamics.integrate_ode_prec import (
    InitialConditionTypes as InitialConditionTypesPrecessing,
)
from pyseobnr.eob.dynamics.integrate_ode_prec import compute_dynamics_quasiprecessing
from pyseobnr.eob.dynamics.postadiabatic_C import Kerr_ISCO, compute_combined_dynamics
from pyseobnr.eob.dynamics.postadiabatic_C_fast import (
    compute_combined_dynamics as compute_combined_dynamics_fast,
)
from pyseobnr.eob.dynamics.postadiabatic_C_prec import (
    compute_combined_dynamics_exp_v1,
    precessing_final_spin,
)
from pyseobnr.eob.fits.EOB_fits import compute_QNM
from pyseobnr.eob.fits.fits_Hamiltonian import NR_deltaT, NR_deltaT_NS, a6_NS, dSO
from pyseobnr.eob.fits.GSF_fits import GSF_amplitude_fits
from pyseobnr.eob.fits.IV_fits import InputValueFits
from pyseobnr.eob.hamiltonian import Hamiltonian
from pyseobnr.eob.utils.containers import EOBParams
from pyseobnr.eob.utils.math_ops_opt import my_norm
from pyseobnr.eob.utils.nr_utils import bbh_final_mass_non_precessing_UIB2016
from pyseobnr.eob.utils.utils import estimate_time_max_amplitude
from pyseobnr.eob.utils.utils_precession_opt import (
    SEOBRotatehIlmFromhJlm_opt_v1,
    custom_swsh,
    inspiral_merger_quaternion_angles,
    interpolate_quats,
    seobnrv4P_quaternionJ2P_postmerger_extension,
)
from pyseobnr.eob.utils.waveform_ops import frame_inv_amp
from pyseobnr.eob.waveform.compute_antisymmetric import (
    apply_antisym_factorized_correction,
    apply_nqc_phase_antisymmetric,
    compute_asymmetric_PN,
    compute_time_for_asym,
    fits_iv_mrd_antisymmetric,
    get_all_dynamics,
    get_params_for_fit,
)
from pyseobnr.eob.waveform.compute_hlms import (
    NQC_correction,
    apply_nqc_corrections,
    compute_IMR_modes,
    concatenate_modes,
    interpolate_modes_fast,
)
from pyseobnr.eob.waveform.waveform import compute_hlms as compute_hlms_new
from pyseobnr.eob.waveform.waveform import (
    compute_newtonian_prefixes,
    compute_special_coeffs,
)
from pyseobnr.models.model import Model

from .common import VALID_MODES
from .SEOBNRv5Base import SEOBNRv5ModelBaseWithpSEOBSupport

logger = logging.getLogger(__name__)


class SEOBNRv5HM_opt(Model, SEOBNRv5ModelBaseWithpSEOBSupport):
    """Represents an aligned-spin SEOBNRv5HM waveform with new MR choices."""

    def __init__(
        self,
        q: float,
        chi_1: float,
        chi_2: float,
        omega0: float,
        H: type[Hamiltonian],
        RR: Callable,
        settings: Dict[Any, Any] = None,
    ) -> None:
        """Initialize the SEOBNRv5 approximant

        Args:
            q (float): Mass ratio :math:`m1/m2 \\ge 1`
            chi_1 (float): z component of the dimensionless spin of primary
            chi_2 (float): z component of the dimensionless spin of secondary
            omega0 (float): Initial orbital frequency, in geometric units
            H (Hamiltonian): Hamiltonian class
            RR (Callable): RR force
            settings (Dict[Any, Any], optional): The settings. Defaults to None.
        """

        super().__init__()

        self.settings = self._default_settings()
        # If we were given settings, override the defaults
        if settings is not None:
            self.settings.update(**settings)

        self.M = self.settings["M"]
        self.q = q

        self.chi_1 = chi_1
        self.chi_2 = chi_2

        self.chi1_v = np.array([0.0, 0.0, self.chi_1])
        self.chi2_v = np.array([0.0, 0.0, self.chi_2])

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
        omega_min = 1 / r_min ** (3.0 / 2)
        if omega0 > omega_min:
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
                logger.warning(
                    f"f_ref < f_start: f_start={self.f0} changed to f_ref={self.f_ref}"
                )
                self.omega0 = self.f_ref * (self.M * lal.MTSUN_SI * np.pi)
                self.f0 = self.omega0 / (self.M * lal.MTSUN_SI * np.pi)

        # The choice of step-back is determined by the range of
        # NR_deltaT in the parameter space of application.
        # The largest value is reached for maximum q and
        # maximum negative spins. The default choice of 250
        # is valid for q<=100 and spins between -1 and 1
        self.step_back = self.settings.get("step_back", 250.0)
        self.chi_S = (self.chi_1 + self.chi_2) / 2
        self.chi_A = (self.chi_1 - self.chi_2) / 2
        self.ap = self.m_1 * self.chi_1 + self.m_2 * self.chi_2
        self.am = self.m_1 * self.chi_1 - self.m_2 * self.chi_2
        self.dt = self.settings["dt"]
        self.delta_T = self.dt / (self.M * lal.MTSUN_SI)
        self.f_nyquist = 0.5 / self.delta_T

        # print(f"In SI units, dt = {self.dt}. In geometric units, with M={self.M}, delta_T={self.delta_T}")
        self.prefixes = compute_newtonian_prefixes(self.m_1, self.m_2)

        self.tplspin = (1 - 2 * self.nu) * self.chi_S + (self.m_1 - self.m_2) / (
            self.m_1 + self.m_2
        ) * self.chi_A

        self.phys_pars = dict(
            m_1=self.m_1,
            m_2=self.m_2,
            chi_1=self.chi_1,
            chi_2=self.chi_2,
            a1=abs(self.chi_1),
            a2=abs(self.chi_2),
            chi1_v=self.chi1_v,
            chi2_v=self.chi2_v,
            H_val=0.0,  # Only used in SEOBNRv5PHM
            lN=np.array([0.0, 0.0, 1.0]),  # Only used in SEOBNRv5PHM
            omega=self.omega0,
            omega_circ=self.omega0,
        )
        # Figure out which modes need to be
        # i) computed
        # ii) returned
        # The situation where those match can be e.g. when the user
        # asks for mixed modes so we must compute all the modes
        # that are needed even if we will not return them

        # Check that the modes are valid, i.e. something we
        # can return

        # All the modes we will need to output
        self.return_modes = self.settings.get("return_modes", None)
        self.max_ell_returned = self._validate_modes(
            settings
        )  # we need to pass the user settings here
        self.lmax_nyquist: int = int(
            self.settings.get("lmax_nyquist", self.max_ell_returned)
        )

        # Now deal with which mixed modes the user wants, if any

        self.mixed_modes = [(3, 2), (4, 3)]
        self.mixed_modes = [x for x in self.mixed_modes if x in self.return_modes]

        # All the modes we need to compute. This can be a larger list
        # than the returned modes, e.g. when we need certain modes to
        # do mode mixing
        self.computed_modes = deepcopy(self.return_modes)
        # Make sure the array contains what we need
        self._ensure_consistency()

        self._initialize_params(phys_pars=self.phys_pars)
        # Initialize the Hamiltonian
        self.H = H(self.eob_pars)

        self.settings["postadiabatic_type"] = self.settings.get(
            "postadiabatic_type", "analytic"
        )
        if self.settings["postadiabatic_type"] not in ["numeric", "analytic"]:
            raise ValueError("Incorrect value for postadiabatic_type")
        self.PA_order: Final = self.settings.get("PA_order", 8)

        # Whether one is sampling over the deltaT parameter that determines the merger-ringdown attachment.
        # This does not allow attaching the merger-ringdown at the last point of the dynamics.
        self.deltaT_sampling = self.settings.get("deltaT_sampling", False)

    def _default_settings(self) -> dict[str, Any]:

        M_default: Final = 50

        # dt is set equal to 0.1M for a system of 10 solar masses.
        dt: Final = (
            M_default * lal.MTSUN_SI / 10
        )  # = 5 * 4.925490947641266978197229498498379006e-06 = 2.4627454738206332e-05

        settings = dict(
            M=M_default,  # Total mass in solar masses
            dt=dt,  # Desired time spacing, *in seconds*
            debug=False,  # Run in debug mode
            postadiabatic=False,  # Use postadiabatic?
            return_modes=[_ for _ in VALID_MODES if _ != (5, 5)],
        )
        return settings

    def _set_H_coeffs(self):
        # Actual coeffs inside the Hamiltonian
        a6_fit = a6_NS(self.nu) + self.da6
        dSO_fit = dSO(self.nu, self.ap, self.am)

        self.H.calibration_coeffs.a6 = a6_fit
        self.H.calibration_coeffs.dSO = dSO_fit + self.ddSO

        assert self.eob_pars.c_coeffs.a6 == a6_fit
        assert self.eob_pars.c_coeffs.dSO == dSO_fit + self.ddSO

    def __call__(self):
        # Evaluate the model

        # Initialize the containers
        self._initialize_params(phys_pars=self.phys_pars)

        assert id(self.H.eob_params) == id(self.eob_pars)

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
            tmp = re.findall(r"h(\d)(\d)_v(\d+)", key)
            if tmp:
                l, m, v = [int(x) for x in tmp[0]]
                self.eob_pars.flux_params.extra_coeffs[l, m, v] = gsf_coeffs[key]
            else:
                tmp = re.findall(r"h(\d)(\d)_vlog(\d+)", key)
                if tmp:
                    l, m, v = [int(x) for x in tmp[0]]
                    self.eob_pars.flux_params.extra_coeffs_log[l, m, v] = gsf_coeffs[
                        key
                    ]
        self._evaluate_model()

    def _evaluate_model(self):
        r_ISCO, _ = Kerr_ISCO(
            self.chi_1,
            self.chi_2,
            self.m_1,
            self.m_2,
        )
        if self.NR_deltaT > 0:
            r_stop = 0.98 * r_ISCO
        else:
            r_stop = -1
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
                    rtol=self.rtol_ode,
                    atol=self.atol_ode,
                    params=self.eob_pars,
                    backend="ode",
                    step_back=self.step_back,
                    r_stop=r_stop,
                )
            else:
                if self.settings["postadiabatic_type"] == "numeric":
                    dynamics_low, dynamics_fine = compute_combined_dynamics(
                        self.omega0,
                        self.H,
                        self.RR,
                        self.chi_1,
                        self.chi_2,
                        self.m_1,
                        self.m_2,
                        tol=self.tol_PA,
                        rtol_ode=self.rtol_ode,
                        atol_ode=self.atol_ode,
                        params=self.eob_pars,
                        backend="ode",
                        step_back=self.step_back,
                        PA_order=self.PA_order,
                        r_stop=r_stop,
                    )
                else:
                    assert (
                        self.settings["postadiabatic_type"] == "analytic"
                    )  # already checked in the constructor
                    dynamics_low, dynamics_fine = compute_combined_dynamics_fast(
                        self.omega0,
                        self.H,
                        self.RR,
                        self.chi_1,
                        self.chi_2,
                        self.m_1,
                        self.m_2,
                        tol=self.tol_PA,
                        rtol_ode=self.rtol_ode,
                        atol_ode=self.atol_ode,
                        params=self.eob_pars,
                        backend="ode",
                        step_back=self.step_back,
                        PA_order=self.PA_order,
                        r_stop=r_stop,
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
                        "Internal function call failed: Input domain error. "
                        "f_ref is larger than the highest frequency in the inspiral!"
                    )
                    raise ValueError(
                        "Internal function call failed: Input domain error. "
                        "f_ref is larger than the highest frequency in the inspiral!"
                    )
                # Solve for time when f_22 = f_ref
                intrp = CubicSpline(t_d, f_22)
                guess = t_d[np.argmin(np.abs(f_22 - self.f_ref))]
                res = root(lambda x: np.abs(intrp(x) - self.f_ref), guess)

                t_correct = res.x
                if not res.success:
                    logger.error(
                        "Failed to find the time corresponding to requested f_ref."
                    )
                    raise ValueError(
                        "Failed to find the time corresponding to requested f_ref."
                    )
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
            # --------------------------------------
            # NOTE:
            # self.NR_deltaT = - delta t^{22}_{ISCO}
            # --------------------------------------

            # ----------------------------------------------------------------------------
            # NOTE: in v5, the t^{22}_{peak} is defined differently relative to the v4
            # Cf. Eqs. (42) and (43) of 2303.18039.
            # Here we are modifying the calibration parameters Delta t^{22}_{ISCO},
            # which is *different* from the one in pSEOBNRv4HM_PA.
            # Structure above:
            #
            # t_match = t_peak + ( nrDeltaT  + dTpeak - extra )
            # ----------------------------------------------------------------------------

            self.NR_deltaT = self.NR_deltaT + self.dTpeak
            t_attach = t_ISCO - self.NR_deltaT
            self.t_attach_predicted = t_attach

            # If the fit for NR_deltaT is too negative and overshoots the end of the
            # dynamics we attach the MR at the last point
            self.attachment_check = 0.0
            if t_attach > t_fine[-1]:
                if self.deltaT_sampling is True:
                    raise ValueError(
                        "Error: NR_deltaT too negative, attaching the MR at the last point of the dynamics "
                        "is not allowed for calibration."
                    )
                else:
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

            # same condition as in compute_rholm_single otherwise we get NaNs from a division with
            # 0 in compute_special_coeffs
            if np.abs(self.nu - 0.25) < 1e-14 and np.abs(self.chi_A) < 1e-14:
                pass
            else:
                compute_special_coeffs(
                    dynamics, t_attach, self.eob_pars, amp_fits, self.amp_thresholds
                )

            # Step 4: compute the waveform on finely sampled dynamics
            hlms_fine = compute_hlms_new(dynamics_fine[:, 1:], self.eob_pars)
            omega_orb_fine = dynamics_fine[:, -2]
            # Polar dynamics, r,pr,omega_orb
            polar_dynamics_fine = (
                dynamics_fine[:, 1],
                dynamics_fine[:, 3],
                omega_orb_fine,
            )

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
                dA_dict=self.dA_dict,
                dw_dict=self.dw_dict,
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
                m_max=self.max_ell_returned,  # m is never above ell
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
                self.f_nyquist,
                self.lmax_nyquist,
                mixed_modes=self.mixed_modes,
                dw_dict=self.dw_dict,
                domega_dict=self.domega_dict,
                dtau_dict=self.dtau_dict,
            )

            amp_inv = frame_inv_amp(hlms_full, ell_max=self.max_ell_returned)
            # Shift the time so that the peak of the frame-invariant amplitude is at t=0
            self.t = t_full - estimate_time_max_amplitude(
                time=t_full, amplitude=amp_inv, delta_t=self.delta_T, precision=0.001
            )
            self.waveform_modes = {}

            # Step 10: fill the final dictionary of modes
            for key in self.return_modes:
                self.waveform_modes[f"{key[0]},{key[1]}"] = hlms_full[key]
            self.success = True

        except Exception as e:
            logger.error(
                f"Waveform generation failed for q={self.q},chi_1={self.chi_1},"
                f"chi_2={self.chi_2},omega0={self.omega0}"
            )
            raise e


class SEOBNRv5PHM_opt(Model, SEOBNRv5ModelBaseWithpSEOBSupport):
    """Represents a precessing SEOBNRv5PHM waveform with new MR choices."""

    def __init__(
        self,
        q: float,
        chi1_x: float,
        chi1_y: float,
        chi1_z: float,
        chi2_x: float,
        chi2_y: float,
        chi2_z: float,
        omega_start: float,
        H: type[Hamiltonian],
        RR: Callable,
        omega_ref: float = None,
        settings: Dict[Any, Any] = None,
    ) -> None:
        """Initialize the SEOBNRv5PHM approximant

        Args:
            q (float): Mass ratio :math:`m1/m2 \\ge 1`
            chi1_x (float): x-component of the dimensionless spin of the primary
            chi1_y (float): y-component of the dimensionless spin of the primary
            chi1_z (float): z-component of the dimensionless spin of the primary
            chi2_x (float): x-component of the dimensionless spin of the secondary
            chi2_y (float): y-component of the dimensionless spin of the secondary
            chi2_z (float): z-component of the dimensionless spin of the secondary
            omega_start (float): Initial orbital frequency, in geometric units
            H (Hamiltonian): Hamiltonian class
            RR (Callable): RR force
            settings (Dict[Any, Any], optional): The settings. Defaults to None.
            omega_ref (float): Reference orbital frequency at which the spins are defined, in geometric units
        """

        super().__init__()

        self.settings = self._default_settings()
        # If we were given settings, override the defaults
        if settings is not None:
            self.settings.update(**settings)

        self.M = self.settings["M"]
        self.q = q
        # self.ell_max = self.settings.get("ell_max", 5)

        self.chi1_v = np.array([chi1_x, chi1_y, chi1_z])
        self.chi2_v = np.array([chi2_x, chi2_y, chi2_z])

        self.a1 = my_norm(self.chi1_v)
        self.a2 = my_norm(self.chi2_v)

        self.NR_deltaT = 0

        self.RR = RR

        self.m_1 = q / (1.0 + q)
        self.m_2 = 1.0 / (1.0 + q)

        self.nu = self.m_1 * self.m_2 / (self.m_1 + self.m_2) ** 2
        assert (
            self.nu - 0.25
        ) < 1e-12, (
            f"nu is above 0.25 by {(self.nu - 0.25)}, please check your configuration!"
        )
        self.nu = min(self.nu, 0.25)

        self.omega_start = omega_start

        # Do not use a omega_start which implies r0< 10.5, if that is
        # the case change starting frequency to correspond to r0=10.5M
        if self.omega_start ** (-2.0 / 3.0) < 10.5:
            self.omega_start = 10.5 ** (-3.0 / 2.0)

        self.f_start = omega_start / (self.M * lal.MTSUN_SI * np.pi)

        # Deal with reference and starting frequencies
        if omega_ref:
            self.omega_ref = omega_ref
            self.f_ref = omega_ref / (self.M * lal.MTSUN_SI * np.pi)
        else:
            self.omega_ref = omega_start
            self.f_ref = self.f_start

        if np.abs(self.f_ref - self.f_start) > 1e-10:
            # The reference frequency is not the same as the starting frequency
            # If the starting frequency is smaller than the reference frequency,
            # we don't need to adjust anything here, and will account for this
            # with a phase shift of the dynamics.
            # If the reference frequency is _less_ than the starting frequency
            # then just change the starting frequency to the reference frequency
            if self.f_ref < self.f_start:
                logger.warning(
                    f"f_ref < f_start: f_start={self.f_start} changed to f_ref={self.f_ref}"
                )
                self.omega_start = self.f_ref * (self.M * lal.MTSUN_SI * np.pi)
                self.f_start = self.omega_start / (self.M * lal.MTSUN_SI * np.pi)

        self.step_back = self.settings.get("step_back", 250.0)
        self.beta_approx = self.settings["beta_approx"]
        self.rd_approx = self.settings["rd_approx"]
        self.rd_smoothing = self.settings["rd_smoothing"]

        self.backend = self.settings.get("backend", "dopri5")
        # z-component because LN_0 = [0,0,1]
        self.chi_S = (chi1_z + chi2_z) / 2.0
        self.chi_A = (chi1_z - chi2_z) / 2.0

        self.dt = self.settings["dt"]
        self.delta_T = self.dt / (self.M * lal.MTSUN_SI)
        self.f_nyquist = 0.5 / self.delta_T

        self.prefixes = compute_newtonian_prefixes(self.m_1, self.m_2)

        self.tplspin = (1 - 2 * self.nu) * self.chi_S + (self.m_1 - self.m_2) / (
            self.m_1 + self.m_2
        ) * self.chi_A

        self.phys_pars = dict(
            m_1=self.m_1,
            m_2=self.m_2,
            chi_1=chi1_z,
            chi_2=chi2_z,
            chi1_v=self.chi1_v,
            chi2_v=self.chi2_v,
            a1=self.a1,
            a2=self.a2,
            H_val=0.0,  # initialize value of the Hamiltonian to zero
            lN=np.array(
                [0.0, 0.0, 1.0]
            ),  # Initialize the value of the initial Newtonian orbital angular momentum
            omega=self.omega_start,
            omega_circ=self.omega_start,
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
        self.max_ell_returned = self._validate_modes(user_settings=settings)
        self.lmax_nyquist = self.settings.get("lmax_nyquist", self.max_ell_returned)
        # Now deal with which mixed modes the user wants, if any

        self.mixed_modes = [(3, 2), (4, 3)]
        self.mixed_modes = [x for x in self.mixed_modes if x in self.return_modes]

        # All the modes we need to compute. This can be a larger list
        # than the returned modes, e.g. when we need certain modes to
        # do mode mixing
        self.computed_modes = deepcopy(self.return_modes)
        # Make sure the array contains what we need
        self._ensure_consistency()
        self._validate_antisymmetric_parameters()
        self._initialize_params(phys_pars=self.phys_pars)

        # Initialize the Hamiltonian
        self.H = H(self.eob_pars)

        # Uncomment and comment the line above to make the python Hamiltonian work
        # self.H = H()

        if self.settings["postadiabatic_type"] not in ["numeric", "analytic"]:
            raise ValueError("Incorrect value for postadiabatic_type")

        if self.settings["initial_conditions"] not in get_args(
            InitialConditionTypesPrecessing
        ):
            raise ValueError("Incorrect value for initial_conditions")

        if self.settings["initial_conditions_postadiabatic_type"] not in get_args(
            InitialConditionPostadiabaticTypesPrecessing
        ):
            raise ValueError(
                "Incorrect value for initial_conditions_postadiabatic_type"
            )

        # sign of the final spin
        self._sign_final_spin: int | None = None

    def _default_settings(self) -> dict[str, Any]:
        M_default: Final = 50
        # dt is set equal to 0.1M for a system of 10 solar masses.
        dt: Final = (
            M_default * lal.MTSUN_SI / 10
        )  # = 5 * 4.925490947641266978197229498498379006e-06 = 2.4627454738206332e-05
        settings = dict(
            M=M_default,  # Total mass in solar masses
            dt=dt,  # Desired time spacing, *in seconds*
            debug=False,  # Run in debug mode
            initial_conditions="adiabatic",
            initial_conditions_postadiabatic_type="analytic",
            postadiabatic=False,  # Use postadiabatic?
            postadiabatic_type="analytic",
            return_modes=[_ for _ in VALID_MODES if _ != (5, 5)],
            polarizations_from_coprec=False,  # True for computing directly polarizations
            beta_approx=0,
            rd_approx=True,
            rd_smoothing=False,
        )
        return settings

    def _validate_antisymmetric_parameters(self) -> None:
        # parameters for anti-symmetries computations
        valid_antisymmetric_modes: Final[set] = {(2, 2), (3, 3), (4, 4)}

        enable_antisymmetric_modes = self.settings.get(
            "enable_antisymmetric_modes", False
        )
        if enable_antisymmetric_modes:
            antisymmetric_modes = self.settings.get("antisymmetric_modes", [])
            if not antisymmetric_modes:
                antisymmetric_modes = [(2, 2)]
            else:
                if len(set(antisymmetric_modes)) != len(antisymmetric_modes):
                    raise RuntimeError(
                        "Redundant modes in 'antisymmetric_modes' settings"
                    )
                elif not set(antisymmetric_modes) <= valid_antisymmetric_modes:
                    raise RuntimeError(
                        "Incorrect modes in 'antisymmetric_modes' settings: "
                        f"{sorted(set(antisymmetric_modes) - valid_antisymmetric_modes)}"
                        " is not "
                        f"in the set of valid modes {sorted(valid_antisymmetric_modes)}"
                    )
            self.settings["antisymmetric_modes"] = antisymmetric_modes
        elif not enable_antisymmetric_modes:
            # if enable_antisymmetric_modes is False, then no other anti symmetric
            # related parameter should be passed
            for current_antisym_param in [
                "antisymmetric_modes",
                "ivs_mrd",
                "antisymmetric_modes_hm",
            ]:
                if current_antisym_param in self.settings:
                    raise ValueError(
                        f"Setting '{current_antisym_param}' provided while anti-symmetric modes "
                        "calculations not enabled"
                    )

        # check fits parameters override
        if "ivs_mrd" in self.settings and self.settings["ivs_mrd"] is not None:
            if not isinstance(self.settings["ivs_mrd"], dict) or set(
                self.settings["ivs_mrd"].keys()
            ) != {"ivs_asym", "mrd_ivs"}:
                raise ValueError("Incorrect 'ivs_mrd' parameter provided")

        return

    def _initialize_params(
        self, *, phys_pars: dict | None, eob_pars: EOBParams | None = None
    ):
        """
        Re-initialize all parameters to make sure everything is reset
        """
        assert eob_pars is None
        super()._initialize_params(phys_pars=phys_pars)

        # Whether one is including QNM deviations in the precession rate computation
        self.omega_prec_deviation = self.settings.get("omega_prec_deviation", True)

        self.eob_pars.aligned = False

    def _set_H_coeffs(self):
        # Actual coeffs inside the Hamiltonian
        a6_fit = a6_NS(self.nu) + self.da6
        self.eob_pars.c_coeffs.a6 = a6_fit
        self.eob_pars.c_coeffs.ddSO = self.ddSO
        assert self.H.calibration_coeffs.a6 == a6_fit
        assert self.H.calibration_coeffs.ddSO == self.ddSO

    def __call__(self, parameters=None):
        # Evaluate the model

        # Initialize the containers
        self._initialize_params(phys_pars=self.phys_pars)

        # Set the Hamiltonian coefficients
        self._set_H_coeffs()
        # Set the GSF contributions to the waveform
        gsf_coeffs = GSF_amplitude_fits(self.nu)
        keys = gsf_coeffs.keys()
        # The following is just a fancy way of passing the coeffs
        for key in keys:
            tmp = re.findall(r"h(\d)(\d)_v(\d+)", key)
            if tmp:
                l, m, v = [int(x) for x in tmp[0]]
                self.eob_pars.flux_params.extra_coeffs[l, m, v] = gsf_coeffs[key]
            else:
                tmp = re.findall(r"h(\d)(\d)_vlog(\d+)", key)
                if tmp:
                    l, m, v = [int(x) for x in tmp[0]]
                    self.eob_pars.flux_params.extra_coeffs_log[l, m, v] = gsf_coeffs[
                        key
                    ]
        self._evaluate_model()

    def _unpack_scri(self, w):
        result = {}
        for key in w.LM:
            result[f"{key[0]},{key[1]}"] = np.copy(w.data[:, w.index(key[0], key[1])])
        return result

    def _add_negative_m_modes(self, waveform_modes: dict[tuple[int, int], Any], fac=1):
        """Add the negative m modes using the usual

        Note:
            This should only be called on *co-precessing frame* modes

        Args:
            waveform_modes (Dict[Any,Any]): Dictionary of modes with m>0

        Returns:
            Dict[Any,Any]: Dictionary of modes with :math:`|m| != 0`
        """
        result = deepcopy(waveform_modes)
        for key in waveform_modes.keys():
            ell, m = key
            result[(ell, -m)] = fac * (-1) ** ell * np.conjugate(waveform_modes[key])
        return result

    def _package_modes(self, waveform_modes, ell_min=2, ell_max=5):
        keys = waveform_modes.keys()
        shape = waveform_modes[(2, 2)].shape
        n_elem = (ell_max + 3) * (ell_max - 1)
        # result = np.empty((shape[0], n_elem), dtype=np.complex128)
        result = []
        zero_array = None
        i = 0
        for ell in range(ell_min, ell_max + 1):
            for m in range(-ell, ell + 1):
                if (ell, m) in keys:
                    # result[:, i] = waveform_modes[(ell, m)]
                    result += [waveform_modes[(ell, m)]]
                else:
                    if zero_array is None:
                        # instanciate this array only once
                        zero_array = np.zeros(shape[0], dtype=np.complex128)
                    # result[:, i] = np.zeros(shape[0], dtype=np.complex128)
                    result += [zero_array]
                i += 1

        assert i == n_elem
        return np.column_stack(result)

    def _compute_full_rotation(
        self, qt: np.ndarray, quat_i2j: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        # warning: parenthesis on the right of qt.conj() as
        # we want to multiply qt (big) only once on the right

        q_tot = qt.conj() * (  # qJ2P
            quat_i2j.conj()
            * quaternion.from_euler_angles(  # qI02I
                self.settings["phiref"], self.settings["inclination"], 0.0
            )
        )

        return quaternion.as_euler_angles(q_tot).T

    def _evaluate_model(self):
        try:
            # print(
            #     f"Waveform parameters: q={self.q},chi_1={self.chi1_v},chi_2={self.chi2_v},"
            #     f"omega_ref={self.omega_ref}, omega_start = {self.omega_start}, Mt = {self.M}"
            # )
            # Generate PN and EOB dynamics
            if not self.settings["postadiabatic"]:
                (
                    dynamics_low,
                    dynamics_fine,
                    t_PN,
                    dyn_PN,
                    splines,
                    dynamics,
                    idx_restart,
                ) = compute_dynamics_quasiprecessing(
                    self.omega_ref,
                    self.omega_start,
                    self.H,
                    self.RR,
                    self.m_1,
                    self.m_2,
                    self.chi1_v,
                    self.chi2_v,
                    self.eob_pars,
                    rtol=1e-8,  # 1e-11,
                    atol=1e-8,  # 1e-12,
                    step_back=self.step_back,
                    initial_conditions=cast(
                        InitialConditionTypesPrecessing,
                        self.settings["initial_conditions"],
                    ),
                    initial_conditions_postadiabatic_type=cast(
                        InitialConditionPostadiabaticTypesPrecessing,
                        self.settings["initial_conditions_postadiabatic_type"],
                    ),
                )
            else:
                (
                    dynamics_low,
                    dynamics_fine,
                    t_PN,
                    dyn_PN,
                    splines,
                    dynamics,
                ) = compute_combined_dynamics_exp_v1(
                    self.omega_ref,
                    self.omega_start,
                    self.H,
                    self.RR,
                    self.m_1,
                    self.m_2,
                    self.chi1_v,
                    self.chi2_v,
                    tol=1e-12,
                    backend=self.backend,
                    params=self.eob_pars,
                    step_back=self.step_back,
                    postadiabatic_type=self.settings["postadiabatic_type"],
                )
                idx_restart = len(dynamics_low)

            self.idx_restart = idx_restart
            self.pn_dynamics = np.c_[t_PN, dyn_PN]
            self.dynamics_fine = dynamics_fine
            self.dynamics_low = dynamics_low

            omega_orb_fine = dynamics_fine[:, 6]
            # omega_EOB_low = dynamics_low[:, 6]

            self.dynamics = dynamics
            t_correct = None
            if np.abs(self.f_ref - self.f_start) > 1e-10:
                # Reference frequency is not the same as starting frequency
                # To account for the LAL conventions, shift things so that
                # the orbital phase is 0 at f_ref
                omega_orb = dynamics[:, 6]
                t_d = dynamics[:, 0]
                # Approximate
                f_22 = omega_orb / (self.M * lal.MTSUN_SI * np.pi)
                if self.f_ref > f_22[-1]:
                    logger.error(
                        "Internal function call failed: Input domain error. f_ref is larger than the highest "
                        "frequency in the inspiral!"
                    )
                    raise ValueError(
                        "Internal function call failed: Input domain error. f_ref is larger than the highest "
                        "frequency in the inspiral!"
                    )
                # Solve for time when f_22 = f_ref
                intrp = CubicSpline(t_d, f_22)
                guess = t_d[np.argmin(np.abs(f_22 - self.f_ref))]
                res = root(lambda x: np.abs(intrp(x) - self.f_ref), guess)

                t_correct = res.x
                self.t_ref = t_correct
                if not res.success:
                    logger.error(
                        "Failed to find the time corresponding to requested f_ref."
                    )
                    raise ValueError(
                        "Failed to find the time corresponding to requested f_ref."
                    )
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

            # Use restart index to interpolate only in the region typically r>6
            # This avoids interpolation errors at close separations where drdt can be > 0
            u_rlow = 1.0 / dynamics_low[:, 1]  # [:idx_restart]
            t_rlow = dynamics_low[:, 0]  # [:idx_restart]
            irt = CubicSpline(u_rlow, t_rlow)
            r_ref = 1.0 / 10.0
            t_r10M = irt(r_ref)

            t_dyn = dynamics[:, 0]
            omega_dyn = dynamics[:, 6]
            iom = CubicSpline(t_dyn, omega_dyn)
            om_r10M = iom(t_r10M)
            tmp = splines["everything"](om_r10M)

            chi1LN_om_r10M = tmp[0]
            chi2LN_om_r10M = tmp[1]
            chi1_om_r10M = tmp[4:7]
            chi2_om_r10M = tmp[7:10]
            LN_om_r10M = tmp[10:13]

            # Step 2, ii): compute the reference point based on Kerr r_ISCO of remnant
            # with final spin
            r_ISCO, _ = Kerr_ISCO(
                chi1LN_om_r10M,
                chi2LN_om_r10M,
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
            tmp = splines["everything"](om_rISCO)
            chi1LN_om_rISCO = tmp[0]
            chi2LN_om_rISCO = tmp[1]
            # chi1_om_rISCO = tmp[4:7]
            # chi2_om_rISCO = tmp[7:10]
            # LN_om_rISCO = tmp[10:13]

            ap = (
                chi1LN_om_rISCO * self.eob_pars.p_params.X_1
                + chi2LN_om_rISCO * self.eob_pars.p_params.X_2
            )
            am = (
                chi1LN_om_rISCO * self.eob_pars.p_params.X_1
                - chi2LN_om_rISCO * self.eob_pars.p_params.X_2
            )

            # Compute the timeshift from the reference point
            self.NR_deltaT = NR_deltaT_NS(self.nu) + NR_deltaT(self.nu, ap, am)
            self.t_ISCO = t_ISCO
            self.omega_rISCO = om_rISCO

            self.NR_deltaT = self.NR_deltaT + self.dTpeak
            t_attach = t_ISCO - self.NR_deltaT

            self.t_attach_predicted = t_attach

            # If the fit for NR_deltaT is too negative and overshoots the end of the
            # dynamics, we attach the MR at the last point
            self.attachment_check = 0.0
            if t_attach > t_fine[-1]:
                if self.deltaT_sampling is True:
                    raise ValueError(
                        "Error: NR_deltaT too negative, attaching the MR at the last point of the dynamics "
                        "is not allowed for calibration."
                    )
                else:
                    self.attachment_check = 1.0
                    t_attach = t_fine[-1]
                    logger.debug(
                        "NR_deltaT too negative, attaching the MR at the last point of the dynamics, careful!"
                    )

            self.t_attach = t_attach
            # For the following steps, we also need spins at the attachment point
            # First compute the orbital frequency at that point
            # iom_orb_fine = CubicSpline(dynamics_fine[:, 0], dynamics_fine[:, 6])

            omega_orb_attach = iom(t_attach)  # iom_orb_fine(t_attach)
            self.omega_attach = omega_orb_attach

            # Now find the spins at attachment
            tmp = splines["everything"](omega_orb_attach)

            chi1LN_attach = tmp[0]
            chi2LN_attach = tmp[1]

            chi1_attach = tmp[4:7]
            chi2_attach = tmp[7:10]
            Lvec_attach = tmp[13:16]

            Jf_attach = (
                chi1_attach * self.m_1 * self.m_1
                + chi2_attach * self.m_2 * self.m_2
                + Lvec_attach
            )

            Lvec_hat_attach = Lvec_attach / my_norm(Lvec_attach)
            Jfhat_attach = Jf_attach / my_norm(Jf_attach)

            self.Lvec_hat_attach = Lvec_hat_attach
            self.Jfhat_attach = Jfhat_attach

            self.eob_pars.p_params.chi_1 = chi1LN_attach
            self.eob_pars.p_params.chi_2 = chi2LN_attach

            # Remember to update _all_ derived spin quantities as well
            self.eob_pars.p_params.update_spins(
                self.eob_pars.p_params.chi_1, self.eob_pars.p_params.chi_2
            )

            # Step 3: compute the special calibration coefficients to tame zeros in some odd-m modes
            input_value_fits = InputValueFits(
                self.m_1, self.m_2, [0.0, 0.0, chi1LN_attach], [0.0, 0.0, chi2LN_attach]
            )

            amp_fits = input_value_fits.hsign()
            # The following values were determined *empirically*
            self.amp_thresholds = {
                (2, 1): 300,
                (4, 3): 200 * self.nu * (1 - 0.8 * self.chi_A),
                (5, 5): 2000,
            }

            # same condition as in compute_rholm_single otherwise we get NaNs from a division with
            # 0 in compute_special_coeffs
            if np.abs(self.nu - 0.25) < 1e-14 and np.abs(self.chi_A) < 1e-14:
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
                chi1LN_attach,
                chi2LN_attach,
                dA_dict=self.dA_dict,
                dw_dict=self.dw_dict,
            )

            self.nqc_coeffs = nqc_coeffs

            # Apply NQC corrections to high sampling modes
            apply_nqc_corrections(hlms_fine, nqc_coeffs, polar_dynamics_fine)

            # Compute the final state quantities
            # Following the same approach as in SEOBNRv4PHM,
            # evaluate the final state formulas using the spins at
            # r=10 M

            # Get the final mass
            final_mass = bbh_final_mass_non_precessing_UIB2016(
                self.m_1, self.m_2, chi1LN_om_r10M, chi2LN_om_r10M
            )

            # Get the final spin
            final_spin, sign_final_spin = precessing_final_spin(
                chi1LN_om_r10M,
                chi2LN_om_r10M,
                chi1_om_r10M,
                chi2_om_r10M,
                LN_om_r10M,
                self.m_1,
                self.m_2,
            )
            self._sign_final_spin = int(sign_final_spin)

            # Double check all is well
            if np.isnan(dynamics_low[:, 1:]).any():
                raise ValueError

            # Step 6: compute the modes in the inspiral
            hlms_low = compute_hlms_new(dynamics_low[:, 1:], self.eob_pars)

            # Apply the NQC corrections to inspiral modes
            # t_low = dynamics_low[:, 0]
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
                m_max=self.max_ell_returned,  # m is never above ell
            )

            # Sep 8.9) Compute the quaternions necessary to rotate the inspiral part of the waveform
            #          as well as the Euler angles at the attachment point

            (
                t_dyn,
                quatJ2P_dyn,
                quatI2J,
                euler_angles_attach,
                euler_angles_derivative_attach,
                # not using the flip from this function, see
                # https://git.ligo.org/waveforms/software/pyseobnr/-/merge_requests/70
                _,
            ) = inspiral_merger_quaternion_angles(
                dynamics[:, 0],
                dynamics[:, 6],
                t_attach,
                Lvec_hat_attach,
                Jfhat_attach,
                splines,
                t_ref=t_correct,
            )

            # Evaluate the term necessary to rotate the QNM consistently
            # in the co-precessing frame as described in arXiv:2301.06558
            betaJ2P_attach = euler_angles_attach[1]
            cosbetaJ2P_attach = np.cos(betaJ2P_attach)

            sigmaQNM220 = compute_QNM(2, 2, 0, final_spin, final_mass).conjugate()
            sigmaQNM210 = compute_QNM(2, 1, 0, final_spin, final_mass).conjugate()

            if self.omega_prec_deviation is True:
                # We include fractional deviations to the J-frame QNM frequencies
                # also in the precession rate computation (Eq. 13 in arXiv:2301.06558)
                omegaQNM220 = sigmaQNM220.real * (1 + self.domega_dict["2,2"])
                omegaQNM210 = sigmaQNM210.real * (1 + self.domega_dict["2,1"])
            else:
                omegaQNM220 = sigmaQNM220.real
                omegaQNM210 = sigmaQNM210.real

            precRate = omegaQNM220 - omegaQNM210

            # Multiply by the sign of the final spin for retrograde cases
            precRate *= sign_final_spin

            qnm_rotation = (1.0 - abs(cosbetaJ2P_attach)) * precRate

            # Step 9: construct the full IMR waveform
            t_full, imr_full = compute_IMR_modes(
                t_new,
                hlms_interp,
                t_fine,
                hlms_fine,
                self.m_1,
                self.m_2,
                chi1LN_attach,
                chi2LN_attach,
                t_attach,
                self.f_nyquist,
                self.lmax_nyquist,
                mixed_modes=self.mixed_modes,
                final_state=[final_mass, final_spin],
                qnm_rotation=qnm_rotation,
                dw_dict=self.dw_dict,
                domega_dict=self.domega_dict,
                dtau_dict=self.dtau_dict,
            )

            t_full -= t_full[0]

            # Step 10: twist up the co-precessing modes
            # Package modes for scri
            # i) Add negative m modes

            # ii) Set to zero all missing modes in co-precessing frame

            self.imr_full = imr_full

            idx = np.where(t_full < t_attach)[0]

            self.idx = idx
            self.final_spin = final_spin
            self.final_mass = final_mass
            self.t_attach = t_attach
            self.omega_orb_attach = omega_orb_attach
            self.splines = splines

            # Now:
            # 1) Compute the time-dependent quaternions from the P-frame to the J-frame using LN_hat
            # 2) Compute the time-dependent quaternions at ringdown in the J-frame assuming simple
            #    precession around the final J
            # 3) Compute the time-independent Euler angles from the J-frame to the I-frame
            self.euler_angles_attach = euler_angles_attach
            # Compute ringdown approximation of the Euler angles in the J-frame
            quat_postMerger = seobnrv4P_quaternionJ2P_postmerger_extension(
                t_full,
                precRate,
                euler_angles_attach,
                euler_angles_derivative_attach,
                t_attach,
                idx,
                self.rd_approx,
                self.rd_smoothing,
                beta_approx=self.beta_approx,
            )

            # Construct the full rotation , by concatenating inspiral
            # and merger-ringdown
            self.quatJ2P_dyn = quatJ2P_dyn
            self.t_intrp = t_dyn
            self.t_forres = t_full[idx]
            # qt[idx] = quaternion.squad(quatJ2P_dyn, t_dyn, t_full[idx])

            # Quaternion representing the rotation to the frame where L_N is
            # along z. Empty init as will be filled out completely in the next steps.
            qt = quaternion.as_quat_array(np.empty((len(t_full), 4)))
            # Interpolate the quaternions from P to J-frame to the finer time grid of the waveform modes
            qt[idx] = interpolate_quats(quatJ2P_dyn, t_dyn, t_full[idx])
            qt[idx[-1] + 1 :] = quat_postMerger

            # Routine to compute and include  spin-precessing anti-symmetric modes
            if self.settings.get("enable_antisymmetric_modes", False):
                # First compute needed quantities to evaluate the PN modes

                # Rotation from co-precessing to inertial frame
                q_copr = quatI2J * quatJ2P_dyn

                # Individual spins and Newtonian orbital angular momentum (normalized)
                dyn_PN_EOB = self.splines["everything"](self.dynamics[:, 6])
                chi1_EOB = dyn_PN_EOB[:, 4:7]
                chi2_EOB = dyn_PN_EOB[:, 7:10]
                L_N_EOB = dyn_PN_EOB[:, 10:13]
                L_N_EOB /= np.linalg.norm(L_N_EOB, axis=-1)[:, None]

                orbital_phase = self.dynamics[:, 2]

                dyn_EOB = {
                    "chiA": chi1_EOB,
                    "chiB": chi2_EOB,
                    "L_N": L_N_EOB,
                }

                # We need to resolve the orbital time-scale to evaluate the anti-symmetric modes
                t_for_asym, orb_phase_asym = compute_time_for_asym(
                    self.dynamics[:, 0], orbital_phase
                )

                # we first interpolate on an array where all columns have been
                # concatenated, then we unpack each column on the target mode
                # note: order is preserved in the dictionary dyn_EOB
                interpolated_dynamics = CubicSpline(
                    self.dynamics[:, 0], np.column_stack(tuple(dyn_EOB.values()))
                )(t_for_asym)
                idx = 0
                for key, current_array in dyn_EOB.items():
                    dyn_EOB[key] = interpolated_dynamics[
                        :, idx : idx + current_array.shape[1]
                    ]
                    idx += current_array.shape[1]

                dyn_EOB["q_copr"] = interpolate_quats(
                    q_copr, self.dynamics[:, 0], t_for_asym
                )
                dyn_EOB["orbphase"] = orb_phase_asym
                dyn_EOB["exp_orbphase"] = np.exp(-1j * orb_phase_asym)

                # Get intermediate dynamical quantities and spins for evaluating anti-symmetric modes
                dyn_EOB = get_all_dynamics(
                    dyn=dyn_EOB, t=t_for_asym, mA=self.m_1, mB=self.m_2
                )
                self.dyn_full_EOB = dyn_EOB

                # Parameters for evaluating IV and MRD fits
                interpolation_step_back: Final = 10
                params_for_fit_asym = get_params_for_fit(
                    dyn_all=dyn_EOB,
                    t=t_for_asym,
                    mA=self.m_1,
                    mB=self.m_2,
                    q=self.q,
                    t_attach=t_attach,
                    interpolation_step_back=interpolation_step_back,
                )

                self.params_for_fit_asym = params_for_fit_asym

                # Compute PN asymmetries
                anti_symmetric_modes = compute_asymmetric_PN(
                    dyn=dyn_EOB,
                    mA=self.m_1,
                    mB=self.m_2,
                    modes_to_compute=self.settings["antisymmetric_modes"],
                    nlo22=True,
                )
                self.pn_anti_symmetric_modes = deepcopy(anti_symmetric_modes)

                # Evaluate IVs and MRD fits
                if self.settings.get("ivs_mrd", None) is not None:
                    ivs_asym = self.settings["ivs_mrd"]["ivs_asym"]
                    mrd_ivs = self.settings["ivs_mrd"]["mrd_ivs"]
                else:
                    ivs_asym, mrd_ivs = fits_iv_mrd_antisymmetric(
                        params_for_fits=params_for_fit_asym,
                        nu=self.nu,
                        modes_to_compute=self.settings["antisymmetric_modes"],
                    )

                self.ivs_asym = ivs_asym
                self.mrd_ivs = mrd_ivs

                # Apply amplitude correction factor
                corr_power = self.settings.get("fac_corr_power", 6)
                nqc_flags = apply_antisym_factorized_correction(
                    antisym_modes=anti_symmetric_modes,
                    v_orb=dyn_EOB["v"],
                    ivs_asym=ivs_asym,
                    idx_attach=params_for_fit_asym.idx_attach,
                    t=dyn_EOB["t"],
                    t_attach=t_attach,
                    nu=self.nu,
                    corr_power=corr_power,
                    interpolation_step_back=interpolation_step_back,
                )

                # Apply NQC corrections - only for phase

                # todo: put the names of the columns
                polar_dynamics_full = CubicSpline(
                    self.dynamics[:, 0], self.dynamics[:, (1, 3, 6)]
                )(t_for_asym).T

                apply_nqc_phase_antisymmetric(
                    anti_symmetric_modes,
                    t_for_asym,
                    polar_dynamics_full,
                    t_attach,
                    ivs_asym,
                    nqc_flags,
                )

                self.anti_symmetric_modes = anti_symmetric_modes
                self.t_asym = t_for_asym

                # Compute anti-symmetric MRD
                # we do one interpolation with all the relevant columns, values and keys
                # are in the same order.
                interpolated_packed = CubicSpline(
                    t_for_asym,
                    np.column_stack(tuple(anti_symmetric_modes.values())),
                )(t_new)

                hlms_interp_asym = {}
                for key, column in zip(anti_symmetric_modes, interpolated_packed.T):
                    hlms_interp_asym[key] = column

                t_full_asym, imr_asym = compute_IMR_modes(
                    t_new,
                    hlms_interp_asym,
                    self.t_asym,
                    anti_symmetric_modes,
                    self.m_1,
                    self.m_2,
                    chi1LN_attach,
                    chi2LN_attach,
                    t_attach,
                    self.f_nyquist,
                    self.lmax_nyquist,
                    mixed_modes=[],
                    final_state=[final_mass, final_spin],
                    qnm_rotation=qnm_rotation,
                    ivs_mrd=mrd_ivs,
                    dtau_22_asym=self.dtau_dict["2,2"],
                )

                # Construct full co-precessing modes (symm + asymm)
                imr_full = self._add_negative_m_modes(imr_full)
                imr_asym = self._add_negative_m_modes(imr_asym, fac=-1)

                self.symmetric_modes_full = deepcopy(imr_full)
                self.antisymmetric_modes_full = deepcopy(imr_asym)

                for key in imr_asym.keys():
                    ell, m = key
                    imr_full[(ell, m)] += imr_asym[(ell, m)]

                # Compute frame-invariant amplitude to set t=0
                # With asymmetries, we want to employ positive and negative-m modes
                amp_inv = frame_inv_amp(
                    imr_full, ell_max=self.max_ell_returned, use_symm=False
                )

            else:  # if enable_antisymmetric_modes

                # Compute frame-invariant amplitude to set t=0
                # Without asymmetries, we want to employ only positive-m modes
                amp_inv = frame_inv_amp(
                    imr_full, ell_max=self.max_ell_returned, use_symm=True
                )

                # Add negative-m modes to mode dict
                imr_full = self._add_negative_m_modes(imr_full)

            # end "if enable_antisymmetric_modes"

            # Compute t=0 from peak of frame-invariant amplitude
            # Store the time array and shift it with an interpolated
            # guess of the max of the frame invariant amplitude
            self.t = t_full - estimate_time_max_amplitude(
                time=t_full, amplitude=amp_inv, delta_t=self.delta_T, precision=0.001
            )

            if self.settings["polarizations_from_coprec"] is False:

                # We only need to package modes here
                imr_full = self._package_modes(imr_full, ell_max=self.max_ell_returned)
                # iii) Create a co-precessing frame scri waveform
                w = scri.WaveformModes(
                    dataType=scri.h,
                    t=t_full,
                    data=imr_full,
                    ell_min=2,
                    ell_max=self.max_ell_returned,
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
                self.wavefom_modesJ = self._unpack_scri(w_modes)

                # Store the I2J quaternion
                self.quaternion = quaternion.as_float_array(qt)

                # Store all the angles
                # For posterity store the rotation as Euler angles
                anglesJ2P = quaternion.as_euler_angles(qt).T
                # self.anglesI2J = [alphaI2J, betaI2J, gammaI2J]
                self.quatI2J = quatI2J
                self.anglesJ2P = anglesJ2P

                modes_lmax = self.max_ell_returned

                # Rotate J->I. This is a rotation by constant angles
                w_I = SEOBRotatehIlmFromhJlm_opt_v1(
                    w_modes,  # hJlm time series, complex values on fixed sampling
                    modes_lmax,  # Input: maximum value of l in modes (l,m)
                    quatI2J,
                )

                # Unpack the modes
                waveform_modesI = self._unpack_scri(w_I)
                self.waveform_modes = waveform_modesI

                self.success = True
            else:
                # Construct full rotation
                alphaTot, betaTot, gammaTot = self._compute_full_rotation(qt, quatI2J)

                sYlm = custom_swsh(betaTot, alphaTot, self.max_ell_returned)
                # Construct polarizations
                hpc = np.zeros(imr_full[(2, 2)].size, dtype=complex)
                for ell, emm in imr_full.keys():
                    # sYlm = SWSH(qTot,-2,[ell,emm])
                    hpc += sYlm[ell, emm] * imr_full[(ell, emm)]

                hpc *= np.exp(2j * gammaTot)
                self.hpc = hpc
                self.success = True

        except Exception as e:
            logger.error(
                f"Waveform generation failed for q={self.q},chi_1={self.chi1_v},chi_2={self.chi2_v},"
                f"omega_ref={self.omega_ref}, omega_start = {self.omega_start}, Mt = {self.M}"
            )
            raise e
