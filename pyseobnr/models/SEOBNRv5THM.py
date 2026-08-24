from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any, Callable, Final

import lal
import numpy as np
from pygsl_lite import spline
from scipy.interpolate import CubicSpline
from scipy.optimize import root

from ..eob.dynamics.integrate_ode_tidal import compute_dynamics_opt_tidal
from ..eob.dynamics.postadiabatic_C import Kerr_ISCO
from ..eob.dynamics.postadiabatic_C_fast import (
    compute_combined_dynamics as compute_combined_dynamics_fast,
)
from ..eob.fits import GSF_amplitude_fits, NR_deltaT, NR_deltaT_NS, a6_NS, dSO
from ..eob.fits.fits_Hamiltonian import NR_deltaT_Tidal
from ..eob.hamiltonian import Hamiltonian
from ..eob.utils.containers import EOBParams
from ..eob.utils.universal_relations import (
    NSStoppingConditions,
    SmoothTransitionFunction,
    UniversalRelationlambda3TidalVSlambda2Tidal,
    UniversalRelationomega02TidalVSlambda2Tidalv2,
    UniversalRelationomega03TidalVSlambda3Tidal,
    UniversalRelationQuadMonVSlambda2Tidal,
    UniversalRelationSpinInducedHexadecupoleVSSpinInducedQuadrupole,
    UniversalRelationSpinInducedOctupoleVSSpinInducedQuadrupole,
    UniversalRelationSpinShiftomega02VSlambda2Tidal,
    UniversalRelationSpinShiftomega03VSlambda2Tidal,
)
from ..eob.utils.waveform_ops import frame_inv_amp
from ..eob.waveform.compute_hlms import (
    BNS_NQC_correction,
    apply_nqc_corrections,
    compute_tidal_tapering_v5T,
    concatenate_modes,
    interpolate_modes_fast,
)
from ..eob.waveform.stationary_phase_plus_fft import (
    compute_fd_polarizations_via_spa_plus_fft,
    get_SPA_spline_fast,
)
from ..eob.waveform.waveform import compute_hlms as compute_hlms_new
from ..eob.waveform.waveform import compute_newtonian_prefixes
from .model import Model
from .SEOBNRv5Base import SEOBNRv5ModelBase

logger = logging.getLogger(__name__)


class SEOBNRv5THM_opt(Model, SEOBNRv5ModelBase):  # noqa: N801
    """Represents an aligned-spin SEOBNRv5HM waveform with new MR choices and tidal effects."""

    def __init__(
        self,
        q: float,
        chi_1: float,
        chi_2: float,
        omega0: float,
        H: Hamiltonian,  # noqa: N806
        RR: Callable,  # noqa: N806
        settings: dict[Any, Any] = None,
        lambda2Tidal1: float = 0.0,  # noqa: N806
        lambda2Tidal2: float = 0.0,  # noqa: N806
        omega02Tidal1: float | None = None,  # noqa: N806
        omega02Tidal2: float | None = None,  # noqa: N806
        spinshiftomega02Tidal1: float | None = None,  # noqa: N806
        spinshiftomega02Tidal2: float | None = None,  # noqa: N806
        lambda3Tidal1: float | None = None,  # noqa: N806
        lambda3Tidal2: float | None = None,  # noqa: N806
        omega03Tidal1: float | None = None,  # noqa: N806
        omega03Tidal2: float | None = None,  # noqa: N806
        spinshiftomega03Tidal1: float | None = None,  # noqa: N806
        spinshiftomega03Tidal2: float | None = None,  # noqa: N806
        CES21: float | None = None,  # noqa: N806
        CES22: float | None = None,  # noqa: N806
        CBS31: float | None = None,  # noqa: N806
        CBS32: float | None = None,  # noqa: N806
        CES41: float | None = None,  # noqa: N806
        CES42: float | None = None,  # noqa: N806
        BBH_model=None,
    ) -> None:
        """
        Initialize the aligned-spin SEOBNRv5THM (tidal) approximant.

        :param q: Mass ratio :math:`m1/m2 >= 1`.
        :param chi_1: Dimensionless z-component of the primary spin.
        :param chi_2: Dimensionless z-component of the secondary spin.
        :param omega0: Initial orbital frequency (geometric units, M=1).
        :param H: Hamiltonian class to use.
        :param RR: Radiation-reaction force to use.
        :param settings: Model settings (e.g. M, dt, postadiabatic, postadiabatic_type,
            return_modes, spinshift, spinmultipoles, ...).
        :param lambda2Tidal1: Quadrupolar adiabatic tidal deformability of primary in units of
            :math:`m1 = (2/3 k2/C^5)`.
        :param lambda2Tidal2: Quadrupolar adiabatic tidal deformability of secondary in units of
            :math:`m2 = (2/3 k2/C^5)`.
        :param omega02Tidal1: Quadrupolar f-mode resonance frequency of primary in units of ``m1``,
            ``None`` (default) for determination via quasi-universal relation, 0. for adiabatic tides.
        :param omega02Tidal2: Quadrupolar f-mode resonance frequency of secondary in units of ``m2``,
            ``None`` (default) for determination via quasi-universal relation, 0. for adiabatic tides.
        :param spinshiftomega02Tidal1: Spin-induced shift of the quadrupolar f-mode resonance of
            primary in units of ``m1``,``None`` (default) for determination via quasi-universal relation,
            0. for ignorance of this effect.
        :param spinshiftomega02Tidal2: Spin-induced shift of the quadrupolar f-mode resonance of secondary
            in units of ``m2``, ``None`` (default) for determination via quasi-universal relation,
            0. for ignorance of this effect.
        :param lambda3Tidal1: Octupolar adiabatic tidal deformability of primary in units of ``m1``,
            ``None`` (default) for determination from lambda2Tidal1 via quasi-universal relation,
            0. to switch off octupolar adiabatic tides.
        :param lambda3Tidal2: Octupolar adiabatic tidal deformability of secondary in units of ``m2``,
            ``None`` (default) for determination from lambda2Tidal2 via quasi-universal relation, 0. to switch
            off octupolar adiabatic tides.
        :param omega03Tidal1: Octupolar f-mode resonance frequency of primary in units of ``m1``,
            ``None`` (default) for determination via quasi-universal relation, 0. for adiabatic tides.
        :param omega03Tidal2: Octupolar f-mode resonance frequency of secondary in units of ``m2``,
            ``None`` for determination via quasi-universal relation, 0. for adiabatic tides.
        :param spinshiftomega03Tidal1: Spin-induced shift of the octupolar f-mode resonance of primary
            in units of ``m1``, ``None`` (default) for determination via quasi-universal relation,
            0. for ignorance of this effect
        :param spinshiftomega03Tidal2: Spin-induced shift of the octupolar f-mode resonance of
            secondary in units of ``m2``, ``None`` (default) for determination via quasi-universal relation,
            0. for ignorance of this effect
        :param CES21: Spin-induced quadrupole-monopole coefficient (C_ES2, kappa_1) of the primary,
            ``None`` (default) for determination via quasi-universal relation, 1. for the black-hole value
        :param CES22: Spin-induced quadrupole-monopole coefficient (C_ES2, kappa_2) of the secondary,
            ``None`` (default) for determination via quasi-universal relation, 1. for the black-hole value
        :param CBS31: Spin-induced octupole coefficient (C_BS3) of the primary,
            ``None`` (default) for determination via quasi-universal relation, 1. for the black-hole value
        :param CBS32: Spin-induced octupole coefficient (C_BS3) of the secondary,
            ``None`` (default) for determination via quasi-universal relation, 1. for the black-hole value
        :param CES41: Spin-induced hexadecapole coefficient (C_ES4) of the primary,
            ``None`` (default) for determination via quasi-universal relation, 1. for the black-hole value
        :param CES42: Spin-induced hexadecapole coefficient (C_ES4) of the secondary,
            ``None`` (default) for determination via quasi-universal relation, 1. for the black-hole value
        :param BBH_model=None: Model used for the computation of the NQCs.
        """

        super().__init__()

        self.settings = self._default_settings()
        # If we were given settings, override the defaults
        if settings is not None:
            self.settings.update(**settings)

        self.use_nqcs = self.settings["use_nqcs"]

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

        if BBH_model is not None:
            self.BBH_NQCs = BBH_model.nqc_coeffs
            self.nqc_coeffs = BBH_model.nqc_coeffs

        # Using Universal relations if necessary
        self._apply_universal_relations(
            lambda2Tidal1,
            lambda2Tidal2,
            omega02Tidal1,
            omega02Tidal2,
            spinshiftomega02Tidal1,
            spinshiftomega02Tidal2,
            lambda3Tidal1,
            lambda3Tidal2,
            omega03Tidal1,
            omega03Tidal2,
            spinshiftomega03Tidal1,
            spinshiftomega03Tidal2,
            CES21,
            CES22,
            CBS31,
            CBS32,
            CES41,
            CES42,
            self.chi_1,
            self.chi_2,
        )

        # For the input value fits we also need the (not rescaled) kappa^T_2
        # parameter that is often used in the literature
        self.kappaT = (
            3
            * self.nu
            * (self.m_1**3 * self.lambda2Tidal1 + self.m_2**3 * self.lambda2Tidal2)
        )

        # Rescaling the dimensionless quantities originally normalized as e.g.
        # m_A * omega_{0l,A} to the EOB definition in terms of the total mass
        # M * omega_{0l,A} (and similarly for all other quantites)
        self.lambda2Tidal1 = self.lambda2Tidal1 * pow(self.m_1, 5)
        self.lambda2Tidal2 = self.lambda2Tidal2 * pow(self.m_2, 5)
        self.omega02Tidal1 = self.omega02Tidal1 / self.m_1
        self.omega02Tidal2 = self.omega02Tidal2 / self.m_2
        self.lambda3Tidal1 = self.lambda3Tidal1 * pow(self.m_1, 7)
        self.lambda3Tidal2 = self.lambda3Tidal2 * pow(self.m_2, 7)
        self.omega03Tidal1 = self.omega03Tidal1 / self.m_1
        self.omega03Tidal2 = self.omega03Tidal2 / self.m_2
        if self.settings["spinshift"]:
            self.spinshiftomega02Tidal1 = self.spinshiftomega02Tidal1 / self.m_1
            self.spinshiftomega02Tidal2 = self.spinshiftomega02Tidal2 / self.m_2
            self.spinshiftomega03Tidal1 = self.spinshiftomega03Tidal1 / self.m_1
            self.spinshiftomega03Tidal2 = self.spinshiftomega03Tidal2 / self.m_2
        else:
            self.spinshiftomega02Tidal1 = 0.0
            self.spinshiftomega02Tidal2 = 0.0
            self.spinshiftomega03Tidal1 = 0.0
            self.spinshiftomega03Tidal2 = 0.0
        if not self.settings["spinmultipoles"]:
            self.CES21 = 1.0
            self.CES22 = 1.0
            self.CBS31 = 1.0
            self.CBS32 = 1.0
            self.CES41 = 1.0
            self.CES42 = 1.0

        # Calculating the small parameter $\epsilon$ as in eq. 5a of
        # http://arxiv.org/abs/1812.08643 - there's no need to recompute it later
        self.sqrtepsilon2Tidal1 = np.sqrt(
            256.0
            * self.nu
            / 5
            * ((self.omega02Tidal1 + self.spinshiftomega02Tidal1) / 2) ** (5 / 3)
        )
        self.sqrtepsilon2Tidal2 = np.sqrt(
            256.0
            * self.nu
            / 5
            * ((self.omega02Tidal2 + self.spinshiftomega02Tidal2) / 2) ** (5 / 3)
        )

        self.sqrtepsilon3Tidal1 = np.sqrt(
            256.0
            * self.nu
            / 5
            * ((self.omega03Tidal1 + self.spinshiftomega03Tidal1) / 3) ** (5 / 3)
        )
        self.sqrtepsilon3Tidal2 = np.sqrt(
            256.0
            * self.nu
            / 5
            * ((self.omega03Tidal2 + self.spinshiftomega03Tidal2) / 3) ** (5 / 3)
        )

        # Deal with reference and starting frequencies
        self.f_ref = self.settings.get(
            "f_ref", omega0 / (self.M * lal.MTSUN_SI * np.pi)
        )

        # Chosen arbitrarily as a point where a lot of the NR simulations start
        omega_min = 0.015
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
            lambda2Tidal1=self.lambda2Tidal1,
            lambda2Tidal2=self.lambda2Tidal2,
            omega02Tidal1=self.omega02Tidal1,
            omega02Tidal2=self.omega02Tidal2,
            spinshiftomega02Tidal1=self.spinshiftomega02Tidal1,
            spinshiftomega02Tidal2=self.spinshiftomega02Tidal2,
            lambda3Tidal1=self.lambda3Tidal1,
            lambda3Tidal2=self.lambda3Tidal2,
            omega03Tidal1=self.omega03Tidal1,
            omega03Tidal2=self.omega03Tidal2,
            spinshiftomega03Tidal1=self.spinshiftomega03Tidal1,
            spinshiftomega03Tidal2=self.spinshiftomega03Tidal2,
            sqrtepsilon2Tidal1=self.sqrtepsilon2Tidal1,
            sqrtepsilon2Tidal2=self.sqrtepsilon2Tidal2,
            sqrtepsilon3Tidal1=self.sqrtepsilon3Tidal1,
            sqrtepsilon3Tidal2=self.sqrtepsilon3Tidal2,
            omega=self.omega0,
            omega_circ=self.omega0,
            CES21=self.CES21,
            CES22=self.CES22,
            CBS31=self.CBS31,
            CBS32=self.CBS32,
            CES41=self.CES41,
            CES42=self.CES42,
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
        self.max_ell_returned = self._validate_modes(settings)
        self.lmax_nyquist = self.settings.get("lmax_nyquist", self.max_ell_returned)
        # Now deal with which mixed modes the user wants, if any

        # self.mixed_modes = [(3, 2), (4, 3)]
        # self.mixed_modes = [x for x in self.mixed_modes if x in self.return_modes]

        # All the modes we need to compute. This can be a larger list
        # than the returned modes, e.g. when we need certain modes to
        # do mode mixing
        self.computed_modes = deepcopy(self.return_modes)
        # Make sure the array contains what we need
        # self._ensure_consistency()

        self._initialize_params(phys_pars=self.phys_pars)
        # Initialize the Hamiltonian
        self.H = H(self.eob_pars)

        self.settings["postadiabatic_type"] = self.settings.get(
            "postadiabatic_type", "analytic"
        )
        if self.settings["postadiabatic_type"] not in ["analytic"]:
            raise ValueError("Incorrect value for postadiabatic_type")
        self.PA_order: Final = self.settings.get("PA_order", 8)

    def _default_settings(self) -> dict[str, Any]:
        settings = dict(
            M=3.0,  # Total mass in solar masses
            dt=1.4776472842923804e-05,  # Desired time spacing, *in seconds*
            debug=False,  # Run in debug mode
            postadiabatic=True,  # Use postadiabatic?
            return_modes=[(2, 2)],  # (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5)],
            # For now we only try to model (2,2)
            use_nqcs=True,  # Use NQCs?
            spinshift=True,  # Use spinshift in keff?
            spinmultipoles=True,
        )
        return settings

    def _initialize_params(
        self, *, phys_pars: dict | None, eob_pars: EOBParams | None = None
    ):
        """
        Re-initialize all parameters to make sure everything is reset
        """
        assert eob_pars is None
        super()._initialize_params(
            phys_pars=phys_pars,
            eob_pars=EOBParams(
                phys_pars,
                {},
                mode_array=list(self.computed_modes),
                tidal_model=True,
            ),
        )

        assert self.eob_pars.tidal_params is not None

        self.eob_pars.flux_params.extra_PN_terms = self.settings.get(
            "extra_PN_terms", True
        )
        self.eob_pars.flux_params.extra_tidal_terms = self.settings.get(
            "extra_tidal_terms", True
        )
        self.eob_pars.flux_params.dynamic_mode_flux = self.settings.get(
            "dynamic_mode_flux", False
        )

    def __call__(self):
        # Evaluate the model

        # Initialize the containers
        self._initialize_params(phys_pars=self.phys_pars)

        # Compute the shift from reference point to peak of (2,2) mode
        NR_deltaT_fit_BBH = NR_deltaT_NS(self.nu) + NR_deltaT(  # noqa: N806
            self.nu, self.ap, self.am
        )
        self.NR_deltaT_BBH = NR_deltaT_fit_BBH

        self.NR_deltaT = self.settings.get(
            "DeltaT",
            (
                NR_deltaT_Tidal(self.nu, self.kappaT)
                + NR_deltaT(self.nu, self.ap, self.am)
            ),
        )

        if self.NR_deltaT > 300:
            self.step_back += self.NR_deltaT
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

    def _apply_universal_relations(
        self,
        lambda2Tidal1,  # noqa: N806
        lambda2Tidal2,  # noqa: N806
        omega02Tidal1,  # noqa: N806
        omega02Tidal2,  # noqa: N806
        spinshiftomega02Tidal1,  # noqa: N806
        spinshiftomega02Tidal2,  # noqa: N806
        lambda3Tidal1,  # noqa: N806
        lambda3Tidal2,  # noqa: N806
        omega03Tidal1,  # noqa: N806
        omega03Tidal2,  # noqa: N806
        spinshiftomega03Tidal1,  # noqa: N806
        spinshiftomega03Tidal2,  # noqa: N806
        CES21,  # noqa: N806
        CES22,  # noqa: N806
        CBS31,  # noqa: N806
        CBS32,  # noqa: N806
        CES41,  # noqa: N806
        CES42,  # noqa: N806
        chi_1,
        chi_2,
    ):

        self.lambda2Tidal1 = lambda2Tidal1
        self.lambda2Tidal2 = lambda2Tidal2

        if lambda2Tidal1 > 0.0:
            if omega02Tidal1 is None:
                self.omega02Tidal1 = UniversalRelationomega02TidalVSlambda2Tidalv2(
                    lambda2Tidal1
                )
            else:
                self.omega02Tidal1 = omega02Tidal1

            if spinshiftomega02Tidal1 is None:
                self.spinshiftomega02Tidal1 = (
                    UniversalRelationSpinShiftomega02VSlambda2Tidal(
                        lambda2Tidal1, chi_1, self.omega02Tidal1
                    )
                )
            else:
                self.spinshiftomega02Tidal1 = spinshiftomega02Tidal1

            if lambda3Tidal1 is None:
                self.lambda3Tidal1 = UniversalRelationlambda3TidalVSlambda2Tidal(
                    lambda2Tidal1
                )
            else:
                self.lambda3Tidal1 = lambda3Tidal1

            if omega03Tidal1 is None:
                self.omega03Tidal1 = UniversalRelationomega03TidalVSlambda3Tidal(
                    self.lambda3Tidal1
                )
            else:
                self.omega03Tidal1 = omega03Tidal1

            if spinshiftomega03Tidal1 is None:
                self.spinshiftomega03Tidal1 = (
                    UniversalRelationSpinShiftomega03VSlambda2Tidal(
                        self.lambda2Tidal1, chi_1, self.omega03Tidal1
                    )
                )
            else:
                self.spinshiftomega03Tidal1 = spinshiftomega03Tidal1

            if CES21 is None:
                self.CES21 = UniversalRelationQuadMonVSlambda2Tidal(lambda2Tidal1)
                CES21 = self.CES21  # noqa: N806
            else:
                self.CES21 = CES21

            if CBS31 is None:
                self.CBS31 = (
                    UniversalRelationSpinInducedOctupoleVSSpinInducedQuadrupole(CES21)
                )
            else:
                self.CBS31 = CBS31

            if CES41 is None:
                self.CES41 = (
                    UniversalRelationSpinInducedHexadecupoleVSSpinInducedQuadrupole(
                        CES21
                    )
                )
            else:
                self.CES41 = CES41
        else:
            self.omega02Tidal1 = 0.0
            self.spinshiftomega02Tidal1 = 0.0
            self.lambda3Tidal1 = 0.0
            self.omega03Tidal1 = 0.0
            self.spinshiftomega03Tidal1 = 0.0
            self.CES21 = 1.0
            self.CBS31 = 1.0
            self.CES41 = 1.0

        if lambda2Tidal2 > 0.0:
            if omega02Tidal2 is None:
                self.omega02Tidal2 = UniversalRelationomega02TidalVSlambda2Tidalv2(
                    lambda2Tidal2
                )
            else:
                self.omega02Tidal2 = omega02Tidal2
            if spinshiftomega02Tidal2 is None:
                self.spinshiftomega02Tidal2 = (
                    UniversalRelationSpinShiftomega02VSlambda2Tidal(
                        lambda2Tidal2, chi_2, self.omega02Tidal2
                    )
                )
            else:
                self.spinshiftomega02Tidal2 = spinshiftomega02Tidal2

            if lambda3Tidal2 is None:
                self.lambda3Tidal2 = UniversalRelationlambda3TidalVSlambda2Tidal(
                    lambda2Tidal2
                )
            else:
                self.lambda3Tidal2 = lambda3Tidal2

            if omega03Tidal2 is None:
                self.omega03Tidal2 = UniversalRelationomega03TidalVSlambda3Tidal(
                    self.lambda3Tidal2
                )
            else:
                self.omega03Tidal2 = omega03Tidal2

            if spinshiftomega03Tidal2 is None:
                self.spinshiftomega03Tidal2 = (
                    UniversalRelationSpinShiftomega03VSlambda2Tidal(
                        self.lambda2Tidal2, chi_2, self.omega03Tidal2
                    )
                )
            else:
                self.spinshiftomega03Tidal2 = spinshiftomega03Tidal2

            if CES22 is None:
                self.CES22 = UniversalRelationQuadMonVSlambda2Tidal(lambda2Tidal2)
                CES22 = self.CES22  # noqa: N806
            else:
                self.CES22 = CES22

            if CBS32 is None:
                self.CBS32 = (
                    UniversalRelationSpinInducedOctupoleVSSpinInducedQuadrupole(CES22)
                )
            else:
                self.CBS32 = CBS32

            if CES42 is None:
                self.CES42 = (
                    UniversalRelationSpinInducedHexadecupoleVSSpinInducedQuadrupole(
                        CES22
                    )
                )
            else:
                self.CES42 = CES42
        else:
            self.omega02Tidal2 = 0.0
            self.spinshiftomega02Tidal2 = 0.0
            self.lambda3Tidal2 = 0.0
            self.omega03Tidal2 = 0.0
            self.spinshiftomega03Tidal2 = 0.0
            self.CES22 = 1.0
            self.CBS32 = 1.0
            self.CES42 = 1.0
        return

    def _set_H_coeffs(self):  # noqa: N806
        # Actual coeffs inside the Hamiltonian
        a6_fit = a6_NS(self.nu)
        dSO_fit = dSO(self.nu, self.ap, self.am)  # noqa: N806

        self.H.calibration_coeffs.a6 = a6_fit
        self.H.calibration_coeffs.dSO = dSO_fit

    def _evaluate_model(self):
        r_ISCO, _ = Kerr_ISCO(  # noqa: N806
            self.chi_1,
            self.chi_2,
            self.m_1,
            self.m_2,
        )

        if self.NR_deltaT > 0:
            r_stop = 0.98 * r_ISCO
        else:
            r_stop = -1

        # We also use stopping conditions based on the peak 22 orbital frequency
        omega_stop_NR, omega_stop_resonance = NSStoppingConditions(  # noqa: N806
            self.eob_pars
        )

        # Step 1: compute the dynamics
        # This includes both the initial conditions
        # and the integration of the ODEs
        try:

            if not self.settings["postadiabatic"]:
                # print('Compute dynamics')
                dynamics_low, dynamics_fine = compute_dynamics_opt_tidal(
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
                    r_stop=r_stop,
                    omega_stop_NR=omega_stop_NR,
                    omega_stop_resonance=omega_stop_resonance,
                )
            else:
                assert self.settings["postadiabatic_type"] == "analytic"

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
                    PA_order=self.PA_order,
                    r_stop=r_stop,
                    Tidal=True,
                    omega_stop_NR=omega_stop_NR,
                    omega_stop_resonance=omega_stop_resonance,
                )
                # print(np.shape(dynamics_low))

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
                        "Internal function call failed: Input domain error. f_ref is larger than the highest "
                        "frequency in the inspiral!"
                    )
                    raise ValueError
                intrp = CubicSpline(t_d, f_22)
                guess = t_d[np.argmin(np.abs(f_22 - self.f_ref))]
                res = root(lambda x: np.abs(intrp(x) - self.f_ref), guess)

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

            # Step 2: compute the reference point based on 2 * Kerr r_ISCO of remnant
            # with final spin. Note that this is in contrast to how it is done in
            # the BBH case!

            # Also compute the BBH t_attach and transition to BBH

            self.r_ISCO = 2 * r_ISCO
            r_ISCO = 2 * r_ISCO  # noqa: N806

            self.NR_deltaT = self.settings.get(
                "DeltaT",
                (
                    NR_deltaT_Tidal(self.nu, self.kappaT)
                    + NR_deltaT(self.nu, self.ap, self.am)
                ),
            )

            r_fine = dynamics_fine[:, 1]

            if r_ISCO < r_fine[-1]:
                # In some corners of parameter space r_ISCO can be *after*
                # the end of the dynamics. In those cases just use the last
                # point of the dynamics as the reference point
                t_ISCO = t_fine[-1]  # noqa: N806
                logger.debug("2 * Kerr ISCO after the last r in the dynamics")
            else:
                # Find a time corresponding to r_ISCO
                sp = 0.001
                N = int((t_fine[-1] - t_fine[0]) / sp)  # noqa: N806
                zoom = np.linspace(t_fine[0], t_fine[-1], N)
                n = len(t_fine)
                intrp_r = spline.cspline(n)
                intrp_r.init(t_fine, r_fine)
                r_zoomed_in = intrp_r.eval_e_vector(zoom)
                idx = (np.abs(r_zoomed_in - r_ISCO)).argmin()
                t_ISCO = zoom[idx]  # noqa: N806

            # We define the attachment with respect to t_ISCO
            self.t_ISCO = t_ISCO
            t_attach = t_ISCO - self.NR_deltaT
            self.t_attach_predicted = t_attach
            self.t_attach = t_attach

            # If the fit for NR_deltaT is too negative and overshoots the end of the
            # dynamics we attach the MR at the last point
            self.attachment_check = 0.0
            if t_attach > t_fine[-1]:
                self.attachment_check = 1.0
                t_attach = t_fine[-1]
                logger.debug(
                    "NR_deltaT too negative, attaching the MR at the last point of the dynamics, careful!"
                )

            if self.kappaT < 20:
                # BBH case

                r_ISCO_BBH = r_ISCO / 2  # noqa: N806

                r_fine = dynamics_fine[:, 1]

                if r_ISCO_BBH < r_fine[-1]:
                    # In some corners of parameter space r_ISCO can be *after*
                    # the end of the dynamics. In those cases just use the last
                    # point of the dynamics as the reference point
                    t_ISCO_BBH = t_fine[-1]  # noqa: N806
                    logger.debug("Kerr ISCO after the last r in the dynamics")
                else:
                    # Find a time corresponding to r_ISCO
                    sp = 0.001
                    N = int((t_fine[-1] - t_fine[0]) / sp)  # noqa: N806
                    zoom = np.linspace(t_fine[0], t_fine[-1], N)
                    n = len(t_fine)
                    intrp_r = spline.cspline(n)
                    intrp_r.init(t_fine, r_fine)
                    r_zoomed_in = intrp_r.eval_e_vector(zoom)
                    idx = (np.abs(r_zoomed_in - r_ISCO_BBH)).argmin()
                    t_ISCO_BBH = zoom[idx]  # noqa: N806

                # We define the attachment with respect to t_ISCO
                self.t_ISCO_BBH = t_ISCO_BBH
                t_attach_BBH = t_ISCO_BBH - self.NR_deltaT_BBH  # noqa: N806
                self.t_attach_predicted_BBH = t_attach_BBH

                # If the fit for NR_deltaT is too negative and overshoots the end of the
                # dynamics we attach the MR at the last point
                self.attachment_check_BBH = 0.0
                if t_attach_BBH > t_fine[-1]:
                    self.attachment_check_BBH = 1.0
                    t_attach_BBH = t_fine[-1]  # noqa: N806
                    logger.debug(
                        "NR_deltaT too negative, attaching the MR at the last point of the dynamics, careful!"
                    )

                self.t_attach_BBH = t_attach_BBH

                # Transition between BNS and BBH case
                scale = SmoothTransitionFunction(self.kappaT, a=10.0, b=20.0, flip=True)
                t_attach = self.t_attach_BBH * scale + self.t_attach_predicted * (
                    1 - scale
                )
                self.t_attach = t_attach

            # We have seen this once, so just to guard against that
            if t_attach < dynamics_fine[0, 0]:
                t_attach = dynamics_fine[10, 0]

            # # Step 3: compute the special calibration coefficients to tame zeros in some odd-m modes
            # input_value_fits = InputValueFits(
            #     self.m_1, self.m_2, [0.0, 0.0, self.chi_1], [0.0, 0.0, self.chi_2]
            # )
            # amp_fits = input_value_fits.hsign()
            # # The following values were determined *empirically*
            # self.amp_thresholds = {
            #     (2, 1): 300,
            #     (4, 3): 200 * self.nu * (1 - 0.8 * self.chi_A),
            #     (5, 5): 2000,
            # }
            # if np.abs(self.q - 1) < 1e-14 and np.abs(self.chi_A) < 1e-14:
            #     pass
            # else:
            #     compute_special_coeffs(
            #         dynamics, t_attach, self.eob_pars, amp_fits, self.amp_thresholds
            #     )

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
            # Else we use the NQCs similarly to v4T as computed from the BBH limit
            self.nqc_coeffs, self.fits_dict = BNS_NQC_correction(
                hlms_fine,
                t_fine,
                polar_dynamics_fine,
                t_attach,
                0.0,
                self.m_1,
                self.m_2,
                self.chi_1,
                self.chi_2,
                self.kappaT,
            )

            # Apply NQC corrections to high sampling modes
            apply_nqc_corrections(hlms_fine, self.nqc_coeffs, polar_dynamics_fine)

            # Step 6: compute the modes in the inspiral
            hlms_low = compute_hlms_new(dynamics_low[:, 1:], self.eob_pars)

            # Apply the NQC corrections to inspiral modes
            omega_orb_low = dynamics_low[:, -2]
            # Polar dynamics, r,pr,omega_orb
            polar_dynamics_low = [dynamics_low[:, 1], dynamics_low[:, 3], omega_orb_low]

            apply_nqc_corrections(hlms_low, self.nqc_coeffs, polar_dynamics_low)

            # Step 7: Concatenate low and high sampling modes
            hlms_joined = concatenate_modes(hlms_low, hlms_fine)

            if not self.settings.get("stationary_phase_approximation", False):
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
                self.tau_phase_factor = self.settings.get("tau_phase", 3.5)

                t_full, hlms_full = compute_tidal_tapering_v5T(  # compute_non_tapered(
                    t_new,
                    hlms_interp,
                    t_fine,
                    hlms_fine,
                    self.m_1,
                    self.m_2,
                    self.chi_1,
                    self.chi_2,
                    self.kappaT,
                    t_attach,
                    self.fits_dict,
                    self.tau_phase_factor,
                )

                self.t = t_full
                self.waveform_modes = {}
                # Shift the time so that the peak of the frame-invariant amplitude is at t=0
                # TODO Marcus check the proper time shifting
                amp_inv = frame_inv_amp(hlms_full, ell_max=self.max_ell_returned)

                # Step 10: fill the final dictionary of modes
                for key in self.return_modes:
                    self.waveform_modes[f"{key[0]},{key[1]}"] = hlms_full[key]
                self.success = True

            else:
                # Step 8: Perform the SPA as far as we can
                frequencies = self.settings["frequency_array"]
                frequencies_size = np.size(frequencies)

                self.result_SPA, self.start_idx_fft = get_SPA_spline_fast(
                    t_old=dynamics[:, 0],
                    f_new=frequencies,
                    modes_dict=hlms_joined,
                    phi_orb=dynamics[:, 2],
                    adiabacity_epsilon=0.01,
                    t_min=t_attach - 500,
                    t_attachment=t_attach,
                )

                # It can happen for low-mass systems, that the region where
                # an FFT becomes necessary is outside the detector band.
                # In that case we just end the waveform generation here
                minimum_frequency = np.min(
                    ([np.size(self.result_SPA[ellm]) for ellm in self.result_SPA])
                )
                if minimum_frequency == frequencies_size:
                    self.waveform_modes = {}
                    for ell, m in self.return_modes:
                        self.waveform_modes[f"{ell},{m}"] = self.result_SPA[(ell, m)]
                    self.t = 0.0
                    self.success = True

                else:
                    taper_length = 500

                    minimum_start_idx = np.min(
                        [self.start_idx_fft[ellm] for ellm in self.start_idx_fft]
                    )
                    # Step 9: interpolate the late-inspiral-modes onto the desired spacing
                    FFT_start_idx = np.searchsorted(  # noqa: N806
                        dynamics[:, 0], (dynamics[minimum_start_idx, 0] - taper_length)
                    )
                    if dynamics[FFT_start_idx, 0] > (
                        dynamics[minimum_start_idx, 0] - taper_length
                    ):
                        FFT_start_idx -= 1

                    # Compute the fine grid
                    t_new = np.arange(
                        dynamics[FFT_start_idx, 0], dynamics[-1, 0], self.delta_T
                    )

                    hlms_old = {
                        key: hlms_joined[key][FFT_start_idx:] for key in hlms_joined
                    }

                    t_original = dynamics[FFT_start_idx:, 0]
                    phi_orb = dynamics[FFT_start_idx:, 2]
                    hlms_interp = interpolate_modes_fast(
                        t_original,
                        t_new,
                        hlms_old,
                        phi_orb,
                    )

                    # Step 10: construct the IMR waveform for the late inspiral
                    self.tau_phase_factor = self.settings.get("tau_phase", 3.5)

                    t_full, hlms_full = (
                        compute_tidal_tapering_v5T(  # compute_non_tapered(
                            t_new,
                            hlms_interp,
                            t_fine,
                            hlms_fine,
                            self.m_1,
                            self.m_2,
                            self.chi_1,
                            self.chi_2,
                            self.kappaT,
                            t_attach,
                            self.fits_dict,
                            self.tau_phase_factor,
                        )
                    )

                    # Last, we need to compute all the FFT'd modes and stitch
                    # everything together
                    self.waveform_modes = compute_fd_polarizations_via_spa_plus_fft(
                        t_full,
                        hlms_full,
                        t_new,
                        self.delta_T,
                        taper_length,
                        t_attach,
                        frequencies,
                        self.result_SPA,
                        self.start_idx_fft,
                        dynamics,
                    )

                    self.t = 0.0
                    self.success = True

        except Exception as e:
            logger.exception(e)

            logger.error(
                f"Waveform generation failed for q={self.q},chi_1={self.chi_1},"
                f"chi_2={self.chi_2},omega0={self.omega0},lambda1={self.lambda2Tidal1},"
                f"lambda2={self.lambda2Tidal2}"
            )
            raise ValueError(
                f"Input domain error : Waveform generation failed for failed for q={self.q},"
                f"chi_1={self.chi_1},"
                f"chi_2={self.chi_2},omega0={self.omega0},lambda1={self.lambda2Tidal1},"
                f"lambda2={self.lambda2Tidal2}"
            )
