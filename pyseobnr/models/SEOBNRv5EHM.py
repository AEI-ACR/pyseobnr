from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any, Dict, Final, cast, get_args

import lal
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks

from ..eob.dynamics.integrate_ode_ecc import (
    IntegratorType,
    compute_background_qc_dynamics,
    compute_dynamics_ecc_backwards_opt,
    compute_dynamics_ecc_opt,
    compute_dynamics_ecc_secular_opt,
)
from ..eob.dynamics.postadiabatic_C import Kerr_ISCO
from ..eob.fits.fits_Hamiltonian import NR_deltaT, NR_deltaT_NS, a6_NS, dSO
from ..eob.fits.GSF_fits import GSF_amplitude_fits
from ..eob.fits.IV_fits import InputValueFits
from ..eob.hamiltonian.hamiltonian import Hamiltonian
from ..eob.utils.containers import EOBParams
from ..eob.utils.utils import estimate_time_max_amplitude
from ..eob.utils.utils_eccentric import (
    compute_attachment_time_qc,
    compute_ref_values,
    dot_phi_omega_avg_e_z,
    interpolate_background_qc_dynamics,
    r_omega_avg_e_z,
)
from ..eob.utils.waveform_ops import frame_inv_amp
from ..eob.waveform.compute_hlms import (
    NQC_correction,
    apply_nqc_corrections,
    compute_IMR_modes,
    concatenate_modes,
    interpolate_modes_fast,
)
from ..eob.waveform.waveform import SEOBNRv5RRForce, compute_newtonian_prefixes
from ..eob.waveform.waveform_ecc import (
    RadiationReactionForceEcc,
    compute_hlms_ecc,
    compute_special_coeffs_ecc,
)
from .common import VALID_MODES_ECC
from .model import Model
from .SEOBNRv5Base import SEOBNRv5ModelBase

logger = logging.getLogger(__name__)


class SEOBNRv5EHM_opt(Model, SEOBNRv5ModelBase):
    """
    Represents an aligned-spin eccentric waveform model, whose
    eccentricity-zero limit is the SEOBNRv5HM model.

    [Gamboa2024a]_ , [Gamboa2024b]_
    """

    model_valid_modes = VALID_MODES_ECC

    def __init__(
        self,
        q: float,
        chi_1: float,
        chi_2: float,
        omega_start: float,
        eccentricity: float,
        rel_anomaly: float,
        H: type[Hamiltonian],
        RR: RadiationReactionForceEcc,
        settings: Dict[Any, Any] = None,
    ) -> None:
        """
        Initialize the SEOBNRv5EHM approximant.

        Args:
            q (float): Mass ratio :math:`m1/m2 \\ge 1`.
            chi_1 (float): z-component of the dimensionless spin of primary.
            chi_2 (float): z-component of the dimensionless spin of secondary.
            omega_start (float): Initial orbital frequency, in geometric units.
            eccentricity (float): Initial eccentricity of the orbit.
            rel_anomaly (float): Initial radial phase parametrizing the orbit
                (0 for periastron and pi for apastron).
            H (Hamiltonian): Hamiltonian class
            RR (Callable): RR force
            settings (Dict[Any, Any], optional): The settings. Defaults to None.
        """

        super().__init__()

        self.settings = self._default_settings()
        # If we were given settings, override the defaults
        if settings is not None:
            self.settings.update(**settings)

        # Mass and spin parameters
        self.M = self.settings["M"]
        self.q = q
        self.m_1 = q / (1.0 + q)
        self.m_2 = 1.0 / (1 + q)
        self.nu = q / (1.0 + q) ** 2
        self.chi_1 = chi_1
        self.chi_2 = chi_2
        self.chi_A = 0.5 * (chi_1 - chi_2)
        self.chi_S = 0.5 * (chi_1 + chi_2)
        self.chi1_v = np.array([0.0, 0.0, self.chi_1])
        self.chi2_v = np.array([0.0, 0.0, self.chi_2])
        self.ap = self.m_1 * self.chi_1 + self.m_2 * self.chi_2
        self.am = self.m_1 * self.chi_1 - self.m_2 * self.chi_2
        self.tplspin = (1 - 2 * self.nu) * self.chi_S + (self.m_1 - self.m_2) / (
            self.m_1 + self.m_2
        ) * self.chi_A

        # Parameters employed in the eccentric model
        self.eccentricity = eccentricity
        self.rel_anomaly = rel_anomaly
        self.omega_start = omega_start
        self.f_start = omega_start / (self.M * lal.MTSUN_SI * np.pi)
        self._radiation_reaction: RadiationReactionForceEcc = RR
        self._radiation_reaction_is_initialized = False
        self.EccIC = self.settings["EccIC"]
        self.flags_ecc = (
            dict(
                flagPN12=1,
                flagPN1=1,
                flagPN32=1,
                flagPN2=1,
                flagPN52=1,
                flagPN3=1,
                flagPA=1,
                flagPA_modes=1,
                flagTail=1,
                flagMemory=1,
            )
            | self.settings["flags_ecc"]
        )
        self.atol = self.settings["atol"]
        self.dissipative_ICs = self.settings["dissipative_ICs"]
        self.e_stop = self.settings["e_stop"]
        self.h_0 = self.settings["h_0"]
        self.IC_messages = self.settings["IC_messages"]
        self.inspiral_modes = self.settings["inspiral_modes"]
        self.nqc_method = self.settings["nqc_method"]
        self.r_min = self.settings["r_min"]
        self.r_start_min = self.settings["r_start_min"]
        self.rtol = self.settings["rtol"]
        self.secular_bwd_int = self.settings["secular_bwd_int"]
        self.t_backwards = self.settings["t_backwards"]
        self.t_bwd_secular = self.settings["t_bwd_secular"]
        self.t_max = self.settings["t_max"]
        self.warning_bwd_int = self.settings["warning_bwd_int"]
        self.warning_secular_bwd_int = self.settings["warning_secular_bwd_int"]
        self.y_init = self.settings["y_init"]
        assert self.settings["integrator"] in get_args(IntegratorType)
        self.integrator: IntegratorType = cast(
            IntegratorType, self.settings["integrator"]
        )

        # Dictionary with parameters
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
            omega=self.omega_start,
            omega_circ=self.omega_start,
            dissipative_ICs=self.dissipative_ICs,
            eccentricity=self.eccentricity,
            EccIC=self.EccIC,
            e_stop=self.e_stop,
            flags_ecc=self.flags_ecc,
            IC_messages=self.IC_messages,
            rel_anomaly=self.rel_anomaly,
            r_min=self.r_min,
            t_backwards=self.t_backwards,
            t_max=self.t_max,
        )

        # Time spacing
        self.dt = self.settings["dt"]
        self.delta_T = self.dt / (self.M * lal.MTSUN_SI)
        self.f_nyquist = 0.5 / self.delta_T

        # Figure out which modes need to be
        # i) computed
        # ii) returned
        # The situation where those match can be e.g. when the user
        # asks for mixed modes so we must compute all the modes
        # that are needed even if we will not return them

        # All the modes we will need to output
        self.return_modes = self.settings.get("return_modes", None)

        # Check that the modes are valid, i.e. something we can return
        self.max_ell_returned = self._validate_modes(
            settings
        )  # we need to pass the user settings here
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

        # Initialize parameters
        self.prefixes = compute_newtonian_prefixes(self.m_1, self.m_2)
        self._initialize_params(phys_pars=self.phys_pars)

        # Initialize the Hamiltonian
        self.H = H(self.eob_pars)

    @property
    def RR(self) -> RadiationReactionForceEcc:
        if not self._radiation_reaction_is_initialized:
            self._radiation_reaction.initialize(self.eob_pars)
            self._radiation_reaction_is_initialized = True
        return self._radiation_reaction

    @staticmethod
    def _default_settings():

        M_default: Final = 50
        # dt is set equal to 0.1M for a system of 10 solar masses.
        dt: Final = (
            M_default * lal.MTSUN_SI / 10
        )  # = 5 * 4.925490947641266978197229498498379006e-06 = 2.4627454738206332e-05

        settings = dict(
            M=M_default,  # Total mass in solar masses
            dt=dt,  # Desired time spacing, *in seconds*
            debug=False,  # Run in debug mode
            return_modes=VALID_MODES_ECC,
            atol=1e-12,
            dissipative_ICs="root",
            EccIC=1,
            e_stop=0.9,
            flags_ecc={},
            h_0=1.0,
            IC_messages=False,
            inspiral_modes=False,
            integrator="rk8pd",
            nqc_method="qc",
            r_min=0.0,
            r_start_min=10.0,
            rtol=1e-11,
            t_backwards=0.0,
            t_bwd_secular=0.0,
            secular_bwd_int=True,
            t_max=1e9,
            warning_bwd_int=True,
            warning_secular_bwd_int=True,
            y_init=None,
        )

        return settings

    def _initialize_params(
        self, *, phys_pars: dict | None, eob_pars: EOBParams | None = None
    ):
        """
        Re-initialize all parameters to make sure everything is reset.
        """
        super()._initialize_params(
            phys_pars=None,
            eob_pars=EOBParams(
                phys_pars, {}, mode_array=list(self.computed_modes), ecc_model=True
            ),
        )

        # The choice of step-back is determined by the range of
        # NR_deltaT in the parameter space of application.
        # The largest value is reached for maximum q and
        # maximum negative spins. The default choice of 250
        # is valid for q <= 100 and spins between -1 and 1.
        # In the eccentric model, the attachment time is computed
        # with the background QC dynamics, so the default choice of
        # step_back = 250 is kept
        self.step_back = self.settings.get("step_back", 250.0)

    def _compute_starting_values(self):
        """
        Compute the starting values of the orbit-averaged orbital frequency,
        the instantaneous orbital frequency, and the starting separation of
        the system.
        """

        flags = self.flags_ecc.copy()
        flags["flagPA"] = 0  # This is employed in conservative equations
        if self.EccIC == 0:
            self.omega_inst = self.omega_start
            self.RR.evolution_equations.compute(
                e=self.eccentricity,
                omega=self.omega_start,
                z=self.rel_anomaly,
            )
            self.x_avg = self.RR.evolution_equations.get("xavg_omegainst")
            self.omega_avg = self.x_avg**1.5
        elif self.EccIC == 1:
            self.omega_avg = self.omega_start
            self.x_avg = self.omega_avg ** (2.0 / 3.0)
            self.omega_inst = dot_phi_omega_avg_e_z(
                self.omega_avg,
                self.eccentricity,
                self.rel_anomaly,
                self.nu,
                self.m_1 - self.m_2,
                self.chi_A,
                self.chi_S,
                flags,
            )
        else:
            raise NotImplementedError(
                "Select a supported value of 'EccIC': "
                "0 (for starting instantaneous orbital frequency) "
                "or 1 (for starting orbit-averaged orbital frequency)."
            )
        flags["flagPA"] = 1
        self.r_start = r_omega_avg_e_z(
            self.omega_avg,
            self.eccentricity,
            self.rel_anomaly,
            self.nu,
            self.m_1 - self.m_2,
            self.chi_A,
            self.chi_S,
            flags,
        )
        self.eob_pars.ecc_params.omega_avg = self.omega_avg
        self.eob_pars.ecc_params.x_avg = self.x_avg
        self.eob_pars.ecc_params.omega_inst = self.omega_inst
        self.eccentricity_ref = self.eccentricity
        self.rel_anomaly_ref = self.rel_anomaly
        self.omega_avg_ref = self.omega_avg

    def __call__(self):
        """
        Waveform model evaluation.
        """

        # Initialize the parameters of the model
        self._initialize_params(phys_pars=self.phys_pars)
        assert id(self.H.eob_params) == id(self.eob_pars)
        self._compute_starting_values()

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

    def _set_H_coeffs(self):
        """
        Compute the actual coeffs inside the Hamiltonian calibrated
        to quasicircular NR waveforms.
        """

        dc = {"a6": a6_NS(self.nu), "dSO": dSO(self.nu, self.ap, self.am)}
        self.H.calibration_coeffs.a6 = dc["a6"]
        self.H.calibration_coeffs.dSO = dc["dSO"]
        assert self.H.eob_params.c_coeffs.a6 == dc["a6"]
        assert self.H.eob_params.c_coeffs.dSO == dc["dSO"]
        assert self.eob_pars.c_coeffs.a6 == dc["a6"]
        assert self.eob_pars.c_coeffs.dSO == dc["dSO"]

    def _evaluate_model(self):
        """
        Evaluation of the model.
        """

        self.r_ISCO, _ = Kerr_ISCO(
            self.chi_1,
            self.chi_2,
            self.m_1,
            self.m_2,
        )

        if self.NR_deltaT > 0:
            r_stop = float(0.98 * self.r_ISCO)
        else:
            r_stop = None

        self.eob_pars.ecc_params.r_ISCO = self.r_ISCO
        self.eob_pars.ecc_params.NR_deltaT = self.NR_deltaT

        try:
            # Step 1: Check if the initial separation of the binary is
            # larger than 10 M. If not, perform a backwards evolution of
            # a set of secular evolution equations

            if not self.secular_bwd_int:
                self.eob_pars.ecc_params.r_min = self.r_start_min

            if self.r_start < self.r_start_min and self.secular_bwd_int:
                level = (
                    logging.WARNING if self.warning_secular_bwd_int else logging.DEBUG
                )
                logger.log(
                    level,
                    "The predicted starting separation of the system "
                    f"(r_start = {self.r_start}) "
                    "is below the minimum starting separation allowed by "
                    f"the model (r_start_min = {self.r_start_min}). "
                    f"Waveform parameters are q = {self.q}, M = {self.M}, "
                    f"chi_1 = {self.chi_1}, chi_2 = {self.chi_2}, "
                    f"omega_avg = {self.omega_avg}, "
                    f"omega_inst = {self.omega_inst}, "
                    f"eccentricity = {self.eccentricity}, "
                    f"rel_anomaly = {self.rel_anomaly}. "
                    "Integrating backwards in time a set of secular "
                    "evolution equations to obtain a prediction "
                    "for the starting separation larger than this minimum. "
                    "This will change the starting input values of the system.",
                )
                try:
                    self.RR.initialize_secular_evolution_equations(self.eob_pars)
                    (
                        self.t_bwd_secular,
                        self.eccentricity,
                        self.rel_anomaly,
                        self.omega_avg,
                    ) = compute_dynamics_ecc_secular_opt(
                        eccentricity=self.eccentricity,
                        rel_anomaly=self.rel_anomaly,
                        omega_avg=self.omega_avg,
                        RR=self.RR,
                        params=self.eob_pars,
                        rtol=self.rtol,
                        atol=self.atol,
                        integrator=self.integrator,
                        r_stop=self.r_start_min,
                        h_0=self.h_0,
                    )
                    logger.log(
                        level,
                        "The new starting input values of the system are: "
                        f"eccentricity = {self.eccentricity}, "
                        f"rel_anomaly = {self.rel_anomaly}, "
                        f"omega_avg = {self.omega_avg}.",
                    )
                except Exception:
                    error_message = (
                        "Internal function call failed: Input domain error. "
                        "The backwards integration of the secular dynamics "
                        "failed. If the given eccentricity is high "
                        f"(e = {self.eccentricity}) "
                        "and/or the associated separation is small "
                        f"(r_start = {self.r_start} M), "
                        "then it is likely that we are exiting the valid "
                        "regime of the model. Please, review the physical "
                        "sense of the input parameters."
                    )
                    logger.error(error_message)
                    raise ValueError(error_message)
                self.x_avg = self.omega_avg ** (2.0 / 3.0)
                flags = self.flags_ecc.copy()
                flags["flagPA"] = 0  # This is employed in conservative equations
                self.omega_inst = dot_phi_omega_avg_e_z(
                    self.omega_avg,
                    self.eccentricity,
                    self.rel_anomaly,
                    self.nu,
                    self.m_1 - self.m_2,
                    self.chi_A,
                    self.chi_S,
                    flags,
                )
                self.eob_pars.ecc_params.omega_avg = self.omega_avg
                self.eob_pars.ecc_params.x_avg = self.x_avg
                self.eob_pars.ecc_params.omega_inst = self.omega_inst
                self.eob_pars.ecc_params.validate_separation = False

            # Step 2: Compute the EOB eccentric dynamics. This includes both
            # the initial conditions and the integration of the ODEs

            # Backward evolution of the full EOB equations, if requested
            self.delta_t_backwards = abs(self.t_backwards) - abs(self.t_bwd_secular)
            dyn_backwards = None
            if self.delta_t_backwards > 0:
                level = logging.WARNING if self.warning_bwd_int else logging.DEBUG
                logger.log(
                    level,
                    "Integrating backwards in time the full EOB equations of "
                    "motion by the specified amount of time "
                    f"(|t| = {abs(self.t_backwards)} M) with respect to the "
                    "reference values. For a total mass of "
                    f"M = {self.M} M_sun, this corresponds to "
                    f"|t| = {abs(self.t_backwards) * self.M * lal.MTSUN_SI} "
                    "seconds. The starting parameters of the system will "
                    "change accordingly.",
                )
                try:
                    y0, dyn_backwards = compute_dynamics_ecc_backwards_opt(
                        m_1=self.m_1,
                        m_2=self.m_2,
                        chi_1=self.chi_1,
                        chi_2=self.chi_2,
                        eccentricity=self.eccentricity,
                        rel_anomaly=self.rel_anomaly,
                        H=self.H,
                        RR=self.RR,
                        params=self.eob_pars,
                        integrator=self.integrator,
                        atol=self.atol,
                        rtol=self.rtol,
                        y_init=self.y_init,
                        e_stop=self.e_stop,
                        t_stop=self.delta_t_backwards,
                    )
                except Exception:
                    error_message = (
                        "Internal function call failed: Input domain error. "
                        "The backwards integration failed. "
                        "If the given eccentricity is high "
                        f"(e = {self.eccentricity}) "
                        "and/or the associated separation is small "
                        f"(r = {self.eob_pars.ecc_params.r_start_ICs} M), "
                        "then it is likely that we "
                        "are exiting the valid regime of the model. "
                        "Please, review the physical sense "
                        "of the input parameters."
                    )
                    logger.error(error_message)
                    raise ValueError(error_message)
                # Time and phase shift to align with forward evolution
                self.t_shift = dyn_backwards[0, 0]
                self.phi_shift = dyn_backwards[0, 2]
                dyn_backwards[:, 0] -= self.t_shift
                self.y_init = y0

            # Forward evolution
            (
                dynamics_low,
                dynamics_fine,
                dynamics,
                _,
            ) = compute_dynamics_ecc_opt(
                m_1=self.m_1,
                m_2=self.m_2,
                chi_1=self.chi_1,
                chi_2=self.chi_2,
                eccentricity=self.eccentricity,
                rel_anomaly=self.rel_anomaly,
                H=self.H,
                RR=self.RR,
                params=self.eob_pars,
                integrator=self.integrator,
                atol=self.atol,
                rtol=self.rtol,
                h_0=self.h_0,
                y_init=self.y_init,
                r_stop=r_stop,
                step_back=self.step_back,
            )

            # If backwards evolution of the full EOMs was requested, then
            # stack the backward dynamics into the forward dynamics
            # and perform a time and phase shift
            if self.delta_t_backwards > 0:
                dynamics_low[:, 0] -= self.t_shift
                dynamics_fine[:, 0] -= self.t_shift
                dynamics[:, 0] -= self.t_shift
                dynamics_low = np.vstack((dyn_backwards, dynamics_low))
                dynamics = np.vstack((dyn_backwards, dynamics))

            # If there was a secular backwards integration, then shift
            # the phase to have phi = 0 at the reference values
            if abs(self.t_bwd_secular):
                if self.eccentricity_ref == 0:
                    # If e = 0, use omega to compute the shift
                    phi_arr = dynamics[:, 2]
                    omega_avg_arr = dynamics[:, -1]
                    if self.omega_avg_ref <= dynamics[-1, -1]:
                        phi_interp = CubicSpline(omega_avg_arr, phi_arr)
                        self.phi_shift_secular = phi_interp(self.omega_avg_ref)
                    else:
                        error_message = (
                            "Internal function call failed: Input domain error. "
                            "The reference frequency is larger than the "
                            "highest frequency in the inspiral."
                        )
                        logger.error(error_message)
                        raise ValueError(error_message)
                else:
                    # If e != 0, use eccentricity to compute the shift
                    phi_arr = np.flip(dynamics[:, 2], axis=0)
                    e_arr = np.flip(dynamics[:, 5], axis=0)
                    if self.eccentricity_ref >= dynamics[-1, 5]:
                        phi_interp = CubicSpline(e_arr, phi_arr)
                        self.phi_shift_secular = phi_interp(self.eccentricity_ref)
                    else:
                        error_message = (
                            "Internal function call failed: Input domain error. "
                            "The reference eccentricity is smaller than the "
                            "lowest eccentricity in the inspiral."
                        )
                        logger.error(error_message)
                        raise ValueError(error_message)
                dynamics_low[:, 2] -= self.phi_shift_secular
                dynamics_fine[:, 2] -= self.phi_shift_secular
                dynamics[:, 2] -= self.phi_shift_secular

            # Step 3: Compute the background QC dynamics

            duration_ecc = dynamics[-1, 0].item()
            dyn_qc, dyn_fine_qc = compute_background_qc_dynamics(
                duration_ecc=duration_ecc,
                m_1=self.m_1,
                m_2=self.m_2,
                chi_1=self.chi_1,
                chi_2=self.chi_2,
                H=self.H,
                RR=SEOBNRv5RRForce(),  # QC RR force
                params=self.eob_pars,
                r_stop=r_stop,
            )
            r_final_qc = dyn_qc[-1, 1]

            # Step 4: Compute the time of attachment of the merger-ringdown

            # Reference values at which the eccentric and background
            # QC dynamics are going to be aligned
            t_ref, r_ref = compute_ref_values(
                dynamics_fine=dynamics_fine,
                r_final_qc=r_final_qc,
            )
            t_ecc_low_aligned = dynamics_low[:, 0] - t_ref
            t_ecc_fine_aligned = dynamics_fine[:, 0] - t_ref

            # Compute attachment time of the QC dynamics and use that
            # information for the attachment in the eccentric dynamics
            t_qc_aligned, delta_t_attach, delta_t_ISCO = compute_attachment_time_qc(
                r_ref=r_ref,
                dynamics_qc=dyn_qc,
                dynamics_fine_qc=dyn_fine_qc,
                params=self.eob_pars,
            )
            self.t_ISCO_ecc = t_ref + delta_t_ISCO
            self.t_attach_ecc = t_ref + delta_t_attach

            # Checks to avoid overshooting
            if self.t_ISCO_ecc > dynamics_fine[-1, 0]:
                self.t_ISCO_ecc = dynamics_fine[-1, 0]
                self.t_attach_ecc = self.t_ISCO_ecc - self.NR_deltaT

            if self.t_attach_ecc > dynamics_fine[-1, 0]:
                self.t_attach_ecc = dynamics_fine[-1, 0]

            if self.t_attach_ecc < dynamics_fine[0, 0]:
                self.t_attach_ecc = dynamics_fine[0, 0]

            # Step 5: Prepare the dynamical quantities entering in the
            # computation of the NQCs

            # Interpolate the QC dynamics into the eccentric time grid
            (
                dynamics_low,
                dynamics_fine,
                dynamics,
                dynamics_low_qc,
                dynamics_fine_qc,
            ) = interpolate_background_qc_dynamics(
                t_ecc_low_aligned=t_ecc_low_aligned,
                t_ecc_fine_aligned=t_ecc_fine_aligned,
                t_qc_aligned=t_qc_aligned,
                dynamics_low=dynamics_low,
                dynamics_fine=dynamics_fine,
                dynamics=dynamics,
                dyn_qc=dyn_qc,
            )
            self.dynamics = dynamics
            t_fine = dynamics_fine[:, 0]

            # Dynamical quantities employed in the computation of NQCs
            if self.nqc_method == "qc":
                r_av_low, pr_av_low, omega_av_low = (
                    dynamics_low_qc[:, 1],
                    dynamics_low_qc[:, 2],
                    dynamics_low_qc[:, 3],
                )
                r_av_fine, pr_av_fine, omega_av_fine = (
                    dynamics_fine_qc[:, 1],
                    dynamics_fine_qc[:, 2],
                    dynamics_fine_qc[:, 3],
                )

            elif self.nqc_method == "no_nqc":
                (r_av_low, omega_av_low, pr_av_low) = (
                    dynamics_low[:, 1],
                    dynamics_low[:, -1],
                    dynamics_low[:, 3],
                )
                (r_av_fine, omega_av_fine, pr_av_fine) = (
                    dynamics_fine[:, 1],
                    dynamics_fine[:, -1],
                    dynamics_fine[:, 3],
                )
            else:
                error_message = (
                    "Select a valid method for the computation of NQCs. "
                    "Available methods are: 'qc' and 'no_nqc'."
                )
                logger.error(error_message)
                raise ValueError(error_message)

            # Step 6: Compute the special calibration coefficients to
            # tame zeros in some odd-m modes

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

            if np.abs(self.nu - 0.25) < 1e-14 and np.abs(self.chi_A) < 1e-14:
                pass
            else:
                compute_special_coeffs_ecc(
                    dynamics,
                    self.t_attach_ecc,
                    self.eob_pars,
                    amp_fits,
                    self.amp_thresholds,
                )

            # Step 7: Compute the waveform on finely sampled dynamics

            hlms_fine = compute_hlms_ecc(
                dynamics_fine[:, 1:],
                self.RR.instance_hlm,
                self.eob_pars,
            )

            if self.inspiral_modes:
                self.hlms_fine_before_nqc = {
                    k: np.copy(v) for k, v in hlms_fine.items()
                }
                self.t_before_nqc = t_fine

            # Step 8: Compute NQCs coeffs for the high sampling modes

            if not self.nqc_method == "no_nqc":
                polar_dynamics_fine = (
                    r_av_fine,
                    pr_av_fine,
                    omega_av_fine,
                )
                # Compute the NQCs with the background QC dynamics
                nqc_coeffs = NQC_correction(
                    hlms_fine,
                    t_fine,
                    polar_dynamics_fine,
                    self.t_ISCO_ecc,
                    self.NR_deltaT,
                    self.m_1,
                    self.m_2,
                    self.chi_1,
                    self.chi_2,
                )
                self.nqc_coeffs = nqc_coeffs
                # Apply NQC corrections to high sampling eccentric modes
                apply_nqc_corrections(hlms_fine, nqc_coeffs, polar_dynamics_fine)

            # Step 9: Compute the modes in the inspiral

            hlms_low = compute_hlms_ecc(
                dynamics_low[:, 1:],
                self.RR.instance_hlm,
                self.eob_pars,
            )

            if self.inspiral_modes:
                self.hlms_low_no_nqc = {k: np.copy(v) for k, v in hlms_low.items()}
                self.t_low_no_nqc = dynamics_low[:, 0]

            if not self.nqc_method == "no_nqc":
                # Apply the NQC corrections to inspiral modes
                # Polar dynamics (r, pr, omega_orb)
                polar_dynamics_low = [
                    r_av_low,
                    pr_av_low,
                    omega_av_low,
                ]
                apply_nqc_corrections(hlms_low, nqc_coeffs, polar_dynamics_low)

            # Step 10: Concatenate low and high sampling modes

            hlms_joined = concatenate_modes(hlms_low, hlms_fine)

            # If producing the inspiral-only modes construct inspiral
            # modes from the coarse and fine grid modes
            if self.inspiral_modes:
                hlms_inspiral_joined = concatenate_modes(
                    self.hlms_low_no_nqc, self.hlms_fine_before_nqc
                )

            # Step 11: Interpolate the modes onto the desired spacing

            t_new = np.arange(dynamics[0, 0], dynamics[-1, 0], self.delta_T)
            hlms_interp = interpolate_modes_fast(
                dynamics[:, 0],  # t_original
                t_new,
                hlms_joined,
                dynamics[:, 2],  # phi_orb
            )
            del hlms_joined

            # If producing the inspiral-only modes interpolate on the
            # inspiral dynamics
            if self.inspiral_modes:
                self.hlms_inspiral_interp = interpolate_modes_fast(
                    dynamics[:, 0],  # t_original
                    t_new,
                    hlms_inspiral_joined,
                    dynamics[:, 2],  # phi_orb
                )
                del hlms_inspiral_joined
                self.t_inspiral = t_new

            # Step 12: Construct the full IMR waveform

            t_full, hlms_full = compute_IMR_modes(
                t_new,
                hlms_interp,
                t_fine,
                hlms_fine,
                self.m_1,
                self.m_2,
                self.chi_1,
                self.chi_2,
                self.t_attach_ecc,
                self.f_nyquist,
                self.lmax_nyquist,
                mixed_modes=self.mixed_modes,
                align=False,
            )

            # Shift the time so that t = 0 corresponds to the last peak
            # of the frame-invariant amplitude, given a certain threshold
            amp_inv = frame_inv_amp(hlms_full, ell_max=self.max_ell_returned)
            try:
                self.t = t_full - estimate_time_max_amplitude(
                    time=t_full,
                    amplitude=amp_inv,
                    delta_t=self.delta_T,
                    precision=0.001,
                    peak_time_method=lambda _: int(
                        # 0.1 is the threshold for the peak, [0] is the array
                        # of the peaks
                        find_peaks(_, height=np.max(_ * 0.1))[0][-1]
                    ),
                )

            except Exception:
                error_message = (
                    "Internal function call failed: Input domain error. "
                    "Error in the localization of the maximum of the frame "
                    "invariant amplitude. If the eccentricity or the "
                    "frequency are high, then it is likely that we are exiting "
                    "the valid regime of the model. Please, review the "
                    "physical sense of the input parameters."
                )
                logger.error(error_message)
                raise ValueError(error_message)

            if self.inspiral_modes:
                amp_inv = frame_inv_amp(
                    self.hlms_inspiral_interp, ell_max=self.max_ell_returned
                )

                self.t = self.t_inspiral - estimate_time_max_amplitude(
                    time=self.t_inspiral,
                    amplitude=amp_inv,
                    delta_t=self.delta_T,
                    precision=0.001,
                    peak_time_method=lambda _: int(
                        # 0.1 is the threshold for the peak
                        find_peaks(_, height=np.max(_ * 0.1))[0][-1]
                    ),
                )

            # Step 13: Fill the final dictionary of modes

            self.waveform_modes = {}
            for key in self.return_modes:
                if not self.inspiral_modes:
                    self.waveform_modes[f"{key[0]},{key[1]}"] = hlms_full[key]
                else:
                    self.waveform_modes[f"{key[0]},{key[1]}"] = (
                        self.hlms_inspiral_interp[key]
                    )

            self.success = True

        except Exception as e:
            error_message = (
                f"Waveform generation failed for q = {self.q}, "
                f"chi_1 = {self.chi_1}, chi_2 = {self.chi_2}, "
                f"omega_avg = {self.omega_avg}, "
                f"omega_inst = {self.omega_inst}, "
                f"eccentricity = {self.eccentricity}, "
                f"rel_anomaly = {self.rel_anomaly}. "
            )
            if self.eccentricity != self.eccentricity_ref:
                error_message += (
                    "Reference values: "
                    f"eccentricity_ref = {self.eccentricity_ref}, "
                    f"rel_anomaly_ref = {self.rel_anomaly_ref}, "
                    f"omega_avg_ref = {self.omega_avg_ref}."
                )
            logger.error(error_message)
            raise e
