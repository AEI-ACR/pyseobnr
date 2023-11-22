from __future__ import annotations

from typing import Any, Dict, Final, Literal, Tuple, Union, cast, get_args

import lal
import lalsimulation as lalsim
import numpy as np

from .eob.hamiltonian.Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C import (
    Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C as Ham_aligned_opt,
)
from .eob.hamiltonian.Ham_AvgS2precess_simple_cython_PA_AD import (
    Ham_AvgS2precess_simple_cython_PA_AD as Ham_prec_pa_cy,
)
from .eob.waveform.waveform import SEOBNRv5RRForce
from .models import SEOBNRv5HM
from .models.model import Model

#: Supported approximants
SupportedApproximants = Literal["SEOBNRv5HM", "SEOBNRv5PHM"]


def generate_prec_hpc_opt(
    q: float,
    chi1: np.ndarray,
    chi2: np.ndarray,
    omega_start: float,
    omega_ref: float | None = None,
    settings: Dict[Any, Any] | None = None,
    debug: bool = False,
) -> tuple[np.array, np.array] | tuple[np.array, np.array, Model]:
    """Generate the GW wave polarizations for precessing model in an optimised way. In particular,
    do not compute the inertial frame modes, instead write the polarizations directly summing
    over the co-precessing frame modes

    Args:
        q (float): Mass ratio >=1
        chi1 (np.ndarray): Dimensionless spin of the primary
        chi2 (np.ndarray): Dimensionless spin of the secondary
        omega_start (float): Starting orbital frequency, in geometric units
        omega_ref (float, optional): Reference orbital frequency, in geometric units.
                                     Defaults to None, in which case equals omega_start.
        settings (Dict[Any,Any], optional): Any additional settings to pass to model. Defaults to None.
        debug (bool, optional): Run in debug mode and return the model object. Defaults to False.

    Returns:
        Tuple: Either (time,polarization) or (time,polarizations,model) if debug=True
    """
    if settings is None:
        settings = {"polarizations_from_coprec": True}
    else:
        settings.update(polarizations_from_coprec=True)

    if q < 1.0:
        raise ValueError("mass-ratio has to be positive and with convention q>=1")
    if omega_start < 0:
        raise ValueError("omega_start has to be positive")

    if np.linalg.norm(chi1) > 1 or np.linalg.norm(chi2) > 1:
        raise ValueError("chi1 and chi2 have to respect Kerr limit (|chi|<=1)")

    RR_f = SEOBNRv5RRForce()
    model = SEOBNRv5HM.SEOBNRv5PHM_opt(
        q,
        *chi1,
        *chi2,
        omega_start,
        Ham_prec_pa_cy,
        RR_f,
        omega_ref=omega_ref,
        settings=settings,
    )

    model()
    if debug:
        return model.t, model.hpc, model
    else:
        return model.t, model.hpc


def _check_spins(
    chi1: Union[
        float,
        np.ndarray,
        list,
        tuple[
            float,
            float,
            float,
        ],
    ],
    chi2: Union[
        float,
        np.ndarray,
        list,
        tuple[
            float,
            float,
            float,
        ],
    ],
    approximant: SupportedApproximants,
) -> tuple[float, float, float, float, float, float]:
    chi1 = np.asarray(chi1, dtype=np.float64)
    chi2 = np.asarray(chi2, dtype=np.float64)
    if len(chi1.shape) == 0:
        chi1 = np.array([0, 0, chi1], dtype=np.float64)
    elif chi1.shape[0] > 3:
        raise ValueError("Incorrect number of spin elements.")
    if len(chi2.shape) == 0:
        chi2 = np.array([0, 0, chi2], dtype=np.float64)
    elif chi2.shape[0] > 3:
        raise ValueError("Incorrect number of spin elements.")

    if approximant == "SEOBNRv5HM" and (
        (np.abs(chi1[:2]).max() > 1e-10) or (np.abs(chi2[:2]).max() > 1e-10)
    ):
        raise ValueError(
            "In-plane spin components must be zero for calling non-precessing approximant."
        )
    if np.linalg.norm(chi1) > 1 or np.linalg.norm(chi2) > 1:
        # raise ValueError("chi1 and chi2 have to respect Kerr limit (|chi|<=1)")
        raise ValueError("Dimensionless spin magnitudes cannot be greater than 1!")

    return cast(tuple[float, float, float], tuple(float(_) for _ in chi1)) + cast(
        tuple[float, float, float], tuple(float(_) for _ in chi2)
    )


def generate_modes_opt(
    q: float,
    chi1: Union[float, np.ndarray],
    chi2: Union[float, np.ndarray],
    omega_start: float,
    omega_ref: float = None,
    approximant: SupportedApproximants = "SEOBNRv5HM",
    settings: Dict[Any, Any] = None,
    debug: bool = False,
) -> Tuple:
    """Compute the GW waveform modes for the given configuration and approximant.

    Args:
        q (float): Mass ratio >=1
        chi1 (Union[float,np.ndarray]): Dimensionless spin of primary.
                                        If ``float``, interpreted as z component
        chi2 (Union[float,np.ndarray]): Dimensionless spin of secondary.
                                        If ``float``, interpreted as z component
        omega_start (float): Starting orbital frequency, in geometric units
        omega_ref (float, optional): Reference orbital frequency, in geometric units.
                                     Defaults to None, in which case equals omega_start.
        approximant (SupportedApproximants, optional): The approximant to use. Defaults to "SEOBNRv5HM".
        settings (Dict[Any,Any], optional): Additional settings to pass to model. Defaults to None.
        debug (bool, optional): Run in debug mode . Defaults to False.

    Raises:
        ValueError: If input parameters are not physical
        NotImplementedError: If the approximant requested does not exist

    Returns:
        Tuple: Either (time,dictionary of modes) or (time, dictionary of modes, model) if
        ``debug=True``
    """
    if q < 1.0:
        raise ValueError("mass-ratio has to be positive and with convention q>=1")
    if omega_start < 0:
        raise ValueError("omega_start has to be positive")

    chi1_x, chi1_y, chi1_z, chi2_x, chi2_y, chi2_z = _check_spins(
        chi1, chi2, approximant=approximant
    )

    if approximant == "SEOBNRv5HM":
        RR_f = SEOBNRv5RRForce()
        model = SEOBNRv5HM.SEOBNRv5HM_opt(
            q,
            chi_1=chi1_z,
            chi_2=chi2_z,
            omega0=omega_start,
            H=Ham_aligned_opt,
            RR=RR_f,
            settings=settings,
        )
        model()

    elif approximant == "SEOBNRv5PHM":
        RR_f = SEOBNRv5RRForce()
        model = SEOBNRv5HM.SEOBNRv5PHM_opt(
            q,
            chi1_x=chi1_x,
            chi1_y=chi1_y,
            chi1_z=chi1_z,
            chi2_x=chi2_x,
            chi2_y=chi2_y,
            chi2_z=chi2_z,
            omega_start=omega_start,
            H=Ham_prec_pa_cy,
            RR=RR_f,
            omega_ref=omega_ref,
            settings=settings,
        )
        model()

    else:
        raise NotImplementedError(f"Approximant '{approximant}' is not available")
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
        Initialize the class with the given ``parameters``.

        ``parameters`` is a dictionary which keys are defined as follow:

        Parameters
        ----------

        float mass1:
            Mass of companion 1, in solar masses - Required
        float mass2:
            Mass of companion 2, in solar masses - Required
        float spin1x:
            x-component of dimensionless spin of companion 1 - Default: 0
        float spin1y:
            y-component of dimensionless spin of companion 1 - Default: 0
        float spin1z:
            z-component of dimensionless spin of companion 1 - Default: 0
        float spin2x:
            x-component of dimensionless spin of companion 2 - Default: 0
        float spin2y:
            y-component of dimensionless spin of companion 2 - Default: 0
        float spin2z:
            z-component of dimensionless spin of companion 2 - Default: 0
        float distance:
            Distance to the source, in Mpc - Default: 100 Mpc
        float inclination:
            Inclination of the source, in radians - Default: 0
        float phi_ref:
            Orbital phase at the reference frequency, in radians - Default: 0
        float f22_start:
            Starting waveform generation frequency, in Hz - Default: 20 Hz
        float f_ref:
            The reference frequency, in Hz - Default: ``f22_start``
        float deltaT:
            Time spacing, in seconds - Default: 1/2048 seconds
        float f_max:
            Maximum frequency, in Hz - Default: 1024 Hz
        float deltaF:
            Frequency spacing, in Hz - Default: 0.125
        list ModeArray:
        list mode_array:
            Mode content (only positive must be specified, e.g ``[(2,2),(2,1)]``).
            Defaults to ``None`` (all modes, see notes below).

        dict domega_dict:
            The non-GR fractional deviation to the frequencies for each mode.
        dict dA_dict:
            The non-GR fractional deviation to the merger amplitudes for each mode.
        dict dw_dict:
            The non-GR fractional deviation to the merger frequencies.
        dict dtau_dict:
            The non-GR fractional deviation to the damping times for each mode.
            Values should be :math:`> -1`.
        float dTpeak:
            The non-GR additive deviation to the amplitude's peak time.
        float da6:
            The non-GR additive deviation to the ``a6`` calibration parameter.
        float ddSO:
            The non-GR additive deviation to the dSO calibration parameter.
        bool deltaT_sampling:
            If set to ``True``, throws an error if the the attachment time
            induced by negative values of the deviation ``dTpeak`` is beyond the last
            point calculated from the dynamics. In those cases, the attachment time is
            set to be the last point of the dynamics.
            Setting the parameter to ``True`` prevents having incorrect posteriors when
            sampling over ``dTpeak``, as those parameters would be rejected.
            Needs to be set to ``True`` if ``dTpeak`` is non-zero. Defaults to ``False``.
        bool omega_prec_deviation:
            If ``True`` (default), the fractional deviations to the J-frame QNM frequencies
            are included into the precession rate computation (Eq. 13 in
            `arXiv:2301.06558 <https://arxiv.org/abs/2301.06558>`_ ).

        str initial_conditions:
            Possible values are ``adiabatic`` (default) and ``postadiabatic``. Used
            only for ``approximant="SEOBNRv5PHM"`` and ``postadiabatic`` is ``False``.
        str initial_conditions_postadiabatic_type:
            Possible values are ``analytic`` (default) and ``numeric``. Used together
            with ``initial_conditions``.
        bool postadiabatic:
            Defaults to ``True``.
        str postadiabatic_type:
            Either ``analytic`` (default) or ``numeric``. Used only when ``postadiabatic``
            is ``True``.
        float tol_PA:
            Tolerance for the root finding routine in case ``postadiabatic_type="numeric"``
            is used. Defaults to 1e-11.

        str approximant:

            * ``SEOBNRv5HM`` (default)
            * ``SEOBNRv5PHM``

        float rtol_ode:
            Relative tolerance of the ODE integrators. Defaults to 1e-11.
        float atol_ode:
            Absolute tolerance of the ODE integrators. Defaults to 1e-12.

        Note
        ----

        The default modes are ``(2, 2)``, ``(2, 1)``, ``(3, 3)``, ``(3, 2)``,
        ``(4, 4)`` and ``(4, 3)``. In particular ``(5, 5)`` is not included
        and should be explicitly set through ``ModeArray``.

        All GR deviations default to 0.
        For the dictionaries ``domega_dict``, ``dA_dict``, ``dw_dict``, ``dtau_dict``,
        keys are the modes ``l,m`` as a string, for :math:`\\ell > 0`.
        """

        self.swap_masses: bool = False
        self.parameters: dict[str, Any] | None = None
        self._validate_parameters(parameters)

    @property
    def model(self):
        if not hasattr(self, "_model"):
            raise ValueError("A model object has not been created!")
        return self._model

    def _validate_parameters(self, parameters):
        if "mass1" not in parameters:
            raise ValueError("mass1 has to be specified!")
        if "mass2" not in parameters:
            raise ValueError("mass2 has to be specified!")

        if parameters["mass1"] > 0 and parameters["mass2"] > 0:
            mass1 = parameters["mass1"]
            mass2 = parameters["mass2"]
        else:
            raise ValueError("Masses have to be positive!")

        Mtot = mass1 + mass2
        if Mtot < 0.001 or Mtot > 1e12:
            raise ValueError("Unreasonable value for total mass, aborting.")

        if mass1 * mass2 / Mtot**2 < 100.0 / (1 + 100) ** 2:
            raise ValueError(
                "Internal function call failed: Input domain error. Model is only valid for systems with "
                "mass-ratio up to 100."
            )

        default_params: Final = {
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
            "deltaF": 0.0,
            "ModeArray": None,
            "mode_array": None,
            "approximant": "SEOBNRv5HM",
            "conditioning": 2,
            "polarizations_from_coprec": True,
            "initial_conditions": "adiabatic",
            "initial_conditions_postadiabatic_type": "analytic",
            "postadiabatic": True,
            "postadiabatic_type": "analytic",
            "r_size_input": 12,
            "dA_dict": {
                "2,2": 0.0,
                "2,1": 0.0,
                "3,3": 0.0,
                "3,2": 0.0,
                "4,4": 0.0,
                "4,3": 0.0,
                "5,5": 0.0,
            },
            "dw_dict": {
                "2,2": 0.0,
                "2,1": 0.0,
                "3,3": 0.0,
                "3,2": 0.0,
                "4,4": 0.0,
                "4,3": 0.0,
                "5,5": 0.0,
            },
            "dTpeak": 0.0,
            "da6": 0.0,
            "ddSO": 0.0,
            "domega_dict": {
                "2,2": 0.0,
                "2,1": 0.0,
                "3,3": 0.0,
                "3,2": 0.0,
                "4,4": 0.0,
                "4,3": 0.0,
                "5,5": 0.0,
            },
            "dtau_dict": {
                "2,2": 0.0,
                "2,1": 0.0,
                "3,3": 0.0,
                "3,2": 0.0,
                "4,4": 0.0,
                "4,3": 0.0,
                "5,5": 0.0,
            },
            "tol_PA": 1e-11,
            "rtol_ode": 1e-11,
            "atol_ode": 1e-12,
            "deltaT_sampling": False,
            "omega_prec_deviation": True,
        }

        # fills the provided parameters over the default ones
        parameters = default_params | parameters

        if "f_ref" not in parameters.keys():
            parameters["f_ref"] = parameters["f22_start"]

        if parameters["approximant"] not in get_args(SupportedApproximants):
            raise ValueError("Approximant not implemented!")

        # Disable direct polarizations for aligned-spin model
        if parameters["approximant"] == "SEOBNRv5HM":
            parameters["polarizations_from_coprec"] = False

        if "f_max" not in parameters.keys():
            parameters["f_max"] = 0.5 / parameters["deltaT"]

        for param in [
            "spin1x",
            "spin1y",
            "spin1z",
            "spin2x",
            "spin2y",
            "spin2z",
            "distance",
            "inclination",
            "phi_ref",
            "f22_start",
            "f_ref",
            "deltaT",
            "f_max",
            "deltaF",
        ]:
            if not isinstance(parameters[param], float) and not isinstance(
                parameters[param], int
            ):
                raise ValueError(f"{param} has to be a real number!")

        (
            parameters["spin1x"],
            parameters["spin1y"],
            parameters["spin1z"],
            parameters["spin2x"],
            parameters["spin2y"],
            parameters["spin2z"],
        ) = _check_spins(
            chi1=(
                parameters["spin1x"],
                parameters["spin1y"],
                parameters["spin1z"],
            ),
            chi2=(
                parameters["spin2x"],
                parameters["spin2y"],
                parameters["spin2z"],
            ),
            approximant=parameters["approximant"],
        )

        for param in ["f22_start", "f_ref", "f_max", "deltaT", "deltaF", "distance"]:
            if parameters[param] < 0:
                raise ValueError(f"{param} has to be positive!")

        if mass2 > mass1:
            self.swap_masses = True
            aux_spin1 = [
                parameters["spin1x"],
                parameters["spin1y"],
                parameters["spin1z"],
            ]
            parameters["spin1x"], parameters["spin1y"], parameters["spin1z"] = [
                -parameters["spin2x"],
                -parameters["spin2y"],
                parameters["spin2z"],
            ]
            parameters["spin2x"], parameters["spin2y"], parameters["spin2z"] = [
                -aux_spin1[0],
                -aux_spin1[1],
                aux_spin1[2],
            ]
            aux_mass1 = parameters["mass1"]
            parameters["mass1"] = parameters["mass2"]
            parameters["mass2"] = aux_mass1

        if parameters["initial_conditions"] not in ["adiabatic", "postadiabatic"]:
            raise ValueError("Unrecognised setting for initial conditions.")

        if parameters["initial_conditions_postadiabatic_type"] not in [
            "numeric",
            "analytic",
        ]:
            raise ValueError(
                "Unrecognised setting for initial conditions postadiabatic type."
            )

        if parameters["postadiabatic_type"] not in ["numeric", "analytic"]:
            raise ValueError("Unrecognised setting for dynamics postadiabatic type.")

        if parameters["ModeArray"] is not None and parameters["mode_array"] is not None:
            raise ValueError(
                "Only one setting can be specified between ModeArray and mode_array."
            )

        self.parameters = parameters

    def generate_td_modes(self):
        """
        Generate dictionary of positive and negative m modes in physical units.
        """
        fmin, dt = self.parameters["f22_start"], self.parameters["deltaT"]
        f_ref = self.parameters.get("f_ref")
        m1, m2 = self.parameters["mass1"], self.parameters["mass2"]
        Mtot = m1 + m2
        dist = self.parameters["distance"]
        approx: SupportedApproximants = cast(
            SupportedApproximants, self.parameters["approximant"]
        )

        if approx == "SEOBNRv5HM":
            chi1 = self.parameters["spin1z"]
            chi2 = self.parameters["spin2z"]
        elif approx == "SEOBNRv5PHM":
            chi1 = np.array(
                [
                    self.parameters["spin1x"],
                    self.parameters["spin1y"],
                    self.parameters["spin1z"],
                ]
            )
            chi2 = np.array(
                [
                    self.parameters["spin2x"],
                    self.parameters["spin2y"],
                    self.parameters["spin2z"],
                ]
            )

        omega_start = np.pi * fmin * Mtot * lal.MTSUN_SI
        omega_ref = np.pi * f_ref * Mtot * lal.MTSUN_SI
        q = m1 / m2  # Model convention q=m1/m2>=1
        if q < 1.0:
            q = 1 / q

        # Generate internal models, in geometrized units
        settings = {
            "dt": dt,
            "M": Mtot,
            "beta_approx": None,
            "polarizations_from_coprec": False,
        }
        if "postadiabatic" in self.parameters:
            settings.update(postadiabatic=self.parameters["postadiabatic"])

            if "postadiabatic_type" in self.parameters:
                settings.update(
                    postadiabatic_type=self.parameters["postadiabatic_type"]
                )

        if "r_size_input" in self.parameters:
            settings.update(r_size_input=self.parameters["r_size_input"])

        if "initial_conditions" in self.parameters:
            settings.update(initial_conditions=self.parameters["initial_conditions"])

            if "initial_conditions_postadiabatic_type" in self.parameters:
                settings.update(
                    initial_conditions_postadiabatic_type=self.parameters[
                        "initial_conditions_postadiabatic_type"
                    ]
                )

        # Select mode array
        if self.parameters["mode_array"] is not None:
            settings["return_modes"] = self.parameters["mode_array"]
        if self.parameters["ModeArray"] is not None:
            settings["return_modes"] = self.parameters["ModeArray"]

        if "lmax_nyquist" in self.parameters:
            settings.update(lmax_nyquist=self.parameters["lmax_nyquist"])

        if "dA_dict" in self.parameters:
            settings.update(dA_dict=self.parameters["dA_dict"])
        if "dw_dict" in self.parameters:
            settings.update(dw_dict=self.parameters["dw_dict"])
        if "dTpeak" in self.parameters:
            settings.update(dTpeak=self.parameters["dTpeak"])
        if "da6" in self.parameters:
            settings.update(da6=self.parameters["da6"])
        if "ddSO" in self.parameters:
            settings.update(ddSO=self.parameters["ddSO"])
        if "domega_dict" in self.parameters:
            settings.update(domega_dict=self.parameters["domega_dict"])
        if "dtau_dict" in self.parameters:
            settings.update(dtau_dict=self.parameters["dtau_dict"])
        if "tol_PA" in self.parameters:
            settings.update(tol_PA=self.parameters["tol_PA"])
        if "rtol_ode" in self.parameters:
            settings.update(rtol_ode=self.parameters["rtol_ode"])
        if "atol_ode" in self.parameters:
            settings.update(atol_ode=self.parameters["atol_ode"])
        if "deltaT_sampling" in self.parameters:
            settings.update(deltaT_sampling=self.parameters["deltaT_sampling"])
        if "omega_prec_deviation" in self.parameters:
            settings.update(
                omega_prec_deviation=self.parameters["omega_prec_deviation"]
            )

        settings.update(f_ref=self.parameters["f_ref"])
        times, h, self._model = generate_modes_opt(
            q,
            chi1,
            chi2,
            omega_start,
            approximant=approx,
            omega_ref=omega_ref,
            settings=settings,
            debug=True,
        )

        # Convert to physical units and LAL convention
        Mpc_to_meters = lal.PC_SI * 1e6
        times *= Mtot * lal.MTSUN_SI  # Physical times
        fac = (
            -1 * Mtot * lal.MRSUN_SI / (dist * Mpc_to_meters)
        )  # Minus sign to satisfy LAL convention

        hlm_dict = {}
        for ellm, mode in h.items():
            ell = int(ellm[0])
            if ellm[2] == "-":
                emm = -int(ellm[3])
            else:
                emm = int(ellm[2])
            hlm_dict[(ell, emm)] = fac * mode

        # If aligned-spin model, compute negative modes using equatorial symmetry
        if approx == "SEOBNRv5HM":
            for ellm, mode in h.items():
                ell = int(ellm[0])
                emm = int(ellm[2])
                hlm_dict[(ell, -emm)] = pow(-1, ell) * fac * np.conj(mode)

        # If masses are swapped to satisfy the m1/m2>=1 convention, this implies a
        # pi rotation on the orbital plane, which translates into a minus sign for the odd modes.
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

        if self.parameters["polarizations_from_coprec"] is False:
            hpc = 0.0
            times, hlm_dict = self.generate_td_modes()
            for ell, emm in hlm_dict:
                hlm = hlm_dict[(ell, emm)]
                ylm = lal.SpinWeightedSphericalHarmonic(
                    incl, np.pi / 2 - phi, -2, ell, emm
                )
                hpc += ylm * hlm

        else:
            fmin, dt = self.parameters["f22_start"], self.parameters["deltaT"]
            f_ref = self.parameters.get("f_ref")
            m1, m2 = self.parameters["mass1"], self.parameters["mass2"]
            Mtot = m1 + m2
            chi1 = np.array(
                [
                    self.parameters["spin1x"],
                    self.parameters["spin1y"],
                    self.parameters["spin1z"],
                ]
            )
            chi2 = np.array(
                [
                    self.parameters["spin2x"],
                    self.parameters["spin2y"],
                    self.parameters["spin2z"],
                ]
            )
            dist = self.parameters["distance"]
            omega_start = np.pi * fmin * Mtot * lal.MTSUN_SI
            omega_ref = np.pi * f_ref * Mtot * lal.MTSUN_SI
            q = m1 / m2
            # Generate internal polarizations, in geometrized units
            settings = {"dt": dt, "M": Mtot, "beta_approx": None}
            if "postadiabatic" in self.parameters:
                settings.update(postadiabatic=self.parameters["postadiabatic"])

                if "postadiabatic_type" in self.parameters:
                    settings.update(
                        postadiabatic_type=self.parameters["postadiabatic_type"]
                    )
            if "initial_conditions" in self.parameters:
                settings.update(
                    initial_conditions=self.parameters["initial_conditions"]
                )

                if "initial_conditions_postadiabatic_type" in self.parameters:
                    settings.update(
                        initial_conditions_postadiabatic_type=self.parameters[
                            "initial_conditions_postadiabatic_type"
                        ]
                    )

            if "r_size_input" in self.parameters:
                settings.update(r_size_input=self.parameters["r_size_input"])

            # Select mode array
            if self.parameters["mode_array"] is not None:
                settings["return_modes"] = self.parameters["mode_array"]
            if self.parameters["ModeArray"] is not None:
                settings["return_modes"] = self.parameters["ModeArray"]

            if "lmax_nyquist" in self.parameters:
                settings.update(lmax_nyquist=self.parameters["lmax_nyquist"])

            if "dA_dict" in self.parameters:
                settings.update(dA_dict=self.parameters["dA_dict"])
            if "dw_dict" in self.parameters:
                settings.update(dw_dict=self.parameters["dw_dict"])
            if "dTpeak" in self.parameters:
                settings.update(dTpeak=self.parameters["dTpeak"])
            if "da6" in self.parameters:
                settings.update(da6=self.parameters["da6"])
            if "ddSO" in self.parameters:
                settings.update(ddSO=self.parameters["ddSO"])
            if "domega_dict" in self.parameters:
                settings.update(domega_dict=self.parameters["domega_dict"])
            if "dtau_dict" in self.parameters:
                settings.update(dtau_dict=self.parameters["dtau_dict"])
            if "tol_PA" in self.parameters:
                settings.update(tol_PA=self.parameters["tol_PA"])
            if "rtol_ode" in self.parameters:
                settings.update(rtol_ode=self.parameters["rtol_ode"])
            if "atol_ode" in self.parameters:
                settings.update(atol_ode=self.parameters["atol_ode"])
            if "deltaT_sampling" in self.parameters:
                settings.update(deltaT_sampling=self.parameters["deltaT_sampling"])
            if "omega_prec_deviation" in self.parameters:
                settings.update(
                    omega_prec_deviation=self.parameters["omega_prec_deviation"]
                )

            settings.update(f_ref=self.parameters["f_ref"])
            Mpc_to_meters = lal.PC_SI * 1e6
            fac = (
                -1 * Mtot * lal.MRSUN_SI / (dist * Mpc_to_meters)
            )  # Minus sign to satisfy LAL convention

            # inclination and phiref for polarizations
            settings.update(inclination=incl)
            # If masses are swapped to satisfy the m1/m2>=1 convention,
            # this implies a pi rotation on the orbital plane.
            if self.swap_masses:
                phi += np.pi
            settings.update(phiref=np.pi / 2 - phi)
            times, hpc, self._model = generate_prec_hpc_opt(
                q,
                chi1,
                chi2,
                omega_start,
                omega_ref=omega_ref,
                settings=settings,
                debug=True,
            )
            hpc *= fac
            times *= Mtot * lal.MTSUN_SI

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

    # Procedure as in v4PHM in SimInspiralFD
    def generate_td_polarizations_conditioned_1(self):
        """
        Generate time-domain polarizations, with tappering at the beginning of the waveform,
        returned as LAL REAL8TimeSeries
        """
        hp_lal, hc_lal = self.generate_td_polarizations()
        lalsim.SimInspiralREAL8WaveTaper(hp_lal.data, 1)
        lalsim.SimInspiralREAL8WaveTaper(hc_lal.data, 1)

        return hp_lal, hc_lal

    # General SimInspiralFD procedure, with extra time at the beginning
    def generate_td_polarizations_conditioned_2(self):
        """
        Generate conditioned time-domain polarizations as in SimInspiralTDfromTD routine
        """

        extra_time_fraction = (
            0.1  # fraction of waveform duration to add as extra time for tapering
        )
        extra_cycles = (
            3.0  # more extra time measured in cycles at the starting frequency
        )

        f_min = self.parameters["f22_start"]
        m1 = self.parameters["mass1"]
        m2 = self.parameters["mass2"]
        S1z = self.parameters["spin1z"]
        S2z = self.parameters["spin2z"]
        original_f_min = f_min

        fisco = 1.0 / (pow(9.0, 1.5) * np.pi * (m1 + m2) * lal.MTSUN_SI)
        if f_min > fisco:
            f_min = fisco

        # upper bound on the chirp time starting at f_min
        tchirp = lalsim.SimInspiralChirpTimeBound(
            f_min, m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, S1z, S2z
        )
        # upper bound on the final black hole spin */
        spinkerr = lalsim.SimInspiralFinalBlackHoleSpinBound(S1z, S2z)
        # upper bound on the final plunge, merger, and ringdown time */
        tmerge = lalsim.SimInspiralMergeTimeBound(
            m1 * lal.MSUN_SI, m2 * lal.MSUN_SI
        ) + lalsim.SimInspiralRingdownTimeBound((m1 + m2) * lal.MSUN_SI, spinkerr)

        # extra time to include for all waveforms to take care of situations where the
        # frequency is close to merger (and is sweeping rapidly): this is a few cycles
        # at the low frequency
        textra = extra_cycles / f_min
        # compute a new lower frequency
        fstart = lalsim.SimInspiralChirpStartFrequencyBound(
            (1.0 + extra_time_fraction) * tchirp + tmerge + textra,
            m1 * lal.MSUN_SI,
            m2 * lal.MSUN_SI,
        )

        self.parameters["f22_start"] = fstart
        hp_lal, hc_lal = self.generate_td_polarizations()
        self.parameters["f22_start"] = original_f_min

        # condition the time domain waveform by tapering in the extra time at the
        # beginning and high-pass filtering above original f_min
        lalsim.SimInspiralTDConditionStage1(
            hp_lal, hc_lal, extra_time_fraction * tchirp + textra, original_f_min
        )

        # final tapering at the beginning and at the end to remove filter transients
        # waveform should terminate at a frequency >= Schwarzschild ISCO
        # so taper one cycle at this frequency at the end; should not make
        # any difference to IMR waveforms */
        fisco = 1.0 / (pow(6.0, 1.5) * np.pi * (m1 + m2) * lal.MTSUN_SI)
        lalsim.SimInspiralTDConditionStage2(hp_lal, hc_lal, f_min, fisco)

        return hp_lal, hc_lal

    def generate_fd_polarizations(self):
        """
        Generate Fourier-domain polarizations, returned as LAL COMPLEX16FrequencySeries

        Routine similar to LAL SimInspiralFD.
        """
        # Adjust deltaT depending on sampling rate
        fmax = self.parameters["f_max"]
        f_nyquist = fmax
        deltaF = 0
        if "deltaF" in self.parameters.keys():
            deltaF = self.parameters["deltaF"]

        if deltaF != 0:
            n = int(np.round(fmax / deltaF))
            if n & (n - 1):
                chirplen_exp = np.frexp(n)
                f_nyquist = np.ldexp(1, chirplen_exp[1]) * deltaF

        deltaT = 0.5 / f_nyquist
        self.parameters["deltaT"] = deltaT

        # Generate conditioned TD polarizations
        if self.parameters["conditioning"] == 2:
            hp, hc = self.generate_td_polarizations_conditioned_2()
        else:
            hp, hc = self.generate_td_polarizations_conditioned_1()

        # Adjust signal duration
        if deltaF == 0:
            chirplen = hp.data.length
            chirplen_exp = np.frexp(chirplen)
            chirplen = int(np.ldexp(1, chirplen_exp[1]))
            deltaF = 1.0 / (chirplen * deltaT)
            self.parameters["deltaF"] = deltaF

        else:
            chirplen = int(1.0 / (deltaF * deltaT))

        # resize waveforms to the required length
        lal.ResizeREAL8TimeSeries(hp, hp.data.length - chirplen, chirplen)
        lal.ResizeREAL8TimeSeries(hc, hc.data.length - chirplen, chirplen)

        # FFT - Using LAL routines
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

        return hptilde, hctilde
