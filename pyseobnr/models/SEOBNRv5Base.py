from __future__ import annotations

import logging

import numpy as np

from ..eob.utils.containers import EOBParams
from .common import VALID_MODES

logger = logging.getLogger(__name__)


class SEOBNRv5ModelBase:
    """Provides a set of common functions for the family of models in SEOBNRv5"""

    #: Sets the valid modes of the model. Override in a subclass
    #: if supported modes are different than the default :py:attr:`~.common.VALID_MODES`
    model_valid_modes = VALID_MODES

    def __init__(self):
        self.settings = None
        self.computed_modes = None
        self.mixed_modes = None
        self.prefixes = None
        self.H = None

        self.rtol_ode = None
        self.atol_ode = None
        self.tol_PA = None

    def _validate_modes(self, user_settings: dict) -> int:
        """
        Check that the mode array is sensible, i.e.
        has something and the modes being asked for are valid.
        """

        # consistency check on the passed settings
        if (
            user_settings
            # this is settings and not self.settings which contains the default
            and "lmax" in user_settings
            and user_settings["lmax"] is not None
        ):
            # some sanity checks
            if user_settings["lmax"] < 1 or user_settings["lmax"] > max(
                _[0] for _ in self.model_valid_modes
            ):
                raise ValueError(
                    f"Incorrect value for lmax={user_settings['lmax']}: "
                    f"the condition 1 <= lmax <= {max(_[0] for _ in self.model_valid_modes)} "
                    f"is not satisfied"
                )
            selected_modes_by_lmax = [
                (ell, emm)
                for ell, emm in self.model_valid_modes
                if ell <= self.settings["lmax"]
            ]

            if "return_modes" in user_settings and set(selected_modes_by_lmax) != set(
                user_settings["return_modes"]
            ):
                missing_modes = sorted(
                    set(selected_modes_by_lmax).symmetric_difference(
                        set(user_settings["return_modes"])
                    )
                )
                raise ValueError(
                    f"Setting lmax={user_settings['lmax']} together with "
                    f"the selection of modes {user_settings['return_modes']} "
                    f"yields inconsistencies for the following modes: {missing_modes}"
                )

            else:
                self.return_modes = selected_modes_by_lmax

        if not self.return_modes:
            logger.error("The mode list specified is empty!")
            raise ValueError

        ell_mx = 2
        for mode in self.return_modes:
            ell, m = mode

            if mode not in self.model_valid_modes:
                logger.error(f"The specified mode, {mode} is not available!")
                logger.error(f"The allowed modes are: { self.model_valid_modes}")
                raise ValueError

            if ell > ell_mx:
                ell_mx = ell

        return ell_mx

    def _ensure_consistency(self):
        """Make sure that the modes contains everything needed to compute mixed modes

        Args:
            modes (list): Current list of modes to compute
        """
        for mode in self.mixed_modes:
            ell, m = mode
            if (m, m) not in self.computed_modes:
                self.computed_modes.append((m, m))

    def _initialize_params(
        self, *, phys_pars: dict | None, eob_pars: EOBParams | None = None
    ):
        """
        Re-initialize all parameters to make sure everything is reset
        """
        if eob_pars is not None:
            self.eob_pars = eob_pars
        else:
            self.eob_pars = EOBParams(
                phys_pars, {}, mode_array=list(self.computed_modes)
            )
        self.eob_pars.flux_params.rho_initialized = False
        self.eob_pars.flux_params.prefixes = np.array(self.prefixes)
        self.eob_pars.flux_params.prefixes_abs = np.abs(
            self.eob_pars.flux_params.prefixes
        )
        self.eob_pars.flux_params.extra_PN_terms = self.settings.get(
            "extra_PN_terms", True
        )
        self.step_back = self.settings.get("step_back", 250.0)

        # PA/ODE integration tolerances
        self.tol_PA = self.settings.get("tol_PA", 1e-11)
        self.rtol_ode = self.settings.get("rtol_ode", 1e-11)
        self.atol_ode = self.settings.get("atol_ode", 1e-12)

        # points the hamiltonian's parameter to the current class parameters
        if self.H is not None:
            self.H.eob_params = self.eob_pars


class SEOBNRv5ModelBaseWithpSEOBSupport(SEOBNRv5ModelBase):
    """Subclass of SEOBNRv5 models that support the pSEOB extensions"""

    def __init__(self):
        super().__init__()
        self.dA_dict = None
        self.dw_dict = None
        self.domega_dict = None
        self.dtau_dict = None
        self.dTpeak = None
        self.da6 = None
        self.ddSO = None

    def _initialize_params(
        self, *, phys_pars: dict | None, eob_pars: EOBParams | None = None
    ):
        super()._initialize_params(phys_pars=phys_pars, eob_pars=eob_pars)
        default_deviation_dict = {
            f"{ell},{emm}": 0.0 for ell, emm in self.model_valid_modes
        }

        # Plunge-merger deviations
        self.dA_dict = default_deviation_dict | self.settings.get("dA_dict", {})
        self.dw_dict = default_deviation_dict | self.settings.get("dw_dict", {})

        self.dTpeak = self.settings.get("dTpeak", 0.0)

        # EOB Hamiltonian deviation
        self.da6 = self.settings.get("da6", 0.0)
        self.ddSO = self.settings.get("ddSO", 0.0)

        # QNM deviations
        self.domega_dict = default_deviation_dict | self.settings.get("domega_dict", {})
        self.dtau_dict = default_deviation_dict | self.settings.get("dtau_dict", {})
        for value in self.dtau_dict.values():
            if value <= -1:
                raise ValueError(
                    "dtau must be larger than -1, otherwise the remnant rings up instead "
                    "of ringing down."
                )

        # Whether one is sampling over the deltaT parameter that determines the merger-ringdown attachment.
        # This does not allow attaching the merger-ringdown at the last point of the dynamics.
        self.deltaT_sampling = self.settings.get("deltaT_sampling", False)
