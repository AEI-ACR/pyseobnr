"""
Test that the aligned-spin eccentric model SEOBNRv5EHM.

"""

import os
from pathlib import Path
from unittest import mock

import lal
import numpy as np
import pandas as pd
from pycbc.filter.matchedfilter import match
from pycbc.psd.analytical import aLIGOZeroDetHighPowerGWINC
from scipy.optimize import RootResults, root, root_scalar

import pytest

from pyseobnr.eob.dynamics.initial_conditions_aligned_ecc_opt import IC_diss_ecc
from pyseobnr.eob.dynamics.integrate_ode_ecc import (
    ColsEccDyn,
    compute_dynamics_ecc_opt,
    compute_dynamics_ecc_secular_opt,
)
from pyseobnr.eob.dynamics.rhs_aligned_ecc import compute_x
from pyseobnr.eob.utils.utils import estimate_time_max_amplitude
from pyseobnr.eob.utils.utils_eccentric import dot_phi_omega_avg_e_z
from pyseobnr.eob.waveform.waveform_ecc import SEOBNRv5RRForceEcc
from pyseobnr.eob.waveform.waveform_ecc import (
    compute_special_coeffs_ecc as compute_special_coeffs_ecc_original,
)
from pyseobnr.generate_waveform import GenerateWaveform, generate_modes_opt

from .helpers import compare_frames, create_eob_params, get_hp_hc

folder_data = Path(__file__).parent.parent / "data"


class TestEccentric:
    def test_smoke(self):
        """Checks the model can actually run without errors"""

        q = 1.5
        chi_1 = [0.0, 0.0, 0.0]
        chi_2 = [0.0, 0.0, 0.0]
        omega0 = 0.01
        omega_start = omega0
        eccentricity = 0.3
        rel_anomaly = 0.0

        # This gives the new eccentric model, but with the v5EHM infrastructure
        settings = {}
        _, _, model_ehm = generate_modes_opt(
            q,
            chi_1[2],
            chi_2[2],
            omega0,
            omega_start,
            eccentricity,
            rel_anomaly,
            approximant="SEOBNRv5EHM",
            settings=settings,
            debug=True,
        )
        assert model_ehm.dynamics is not None

    def test_smoke_with_backward_evolution(self):
        """Checks the model can actually run without errors"""
        q = 1
        chi_1z = 0.1
        chi_2z = 0.3
        omega0 = 0.03
        rel_anomaly = 0.0
        eccentricity = 0.3

        chi_1 = [0.0, 0.0, chi_1z]
        chi_2 = [0.0, 0.0, chi_2z]
        omega_start = omega0

        # This gives the new eccentric model, but with the v5EHM infrastructure
        settings = {}
        with mock.patch(
            "pyseobnr.generate_waveform.SEOBNRv5EHM.compute_dynamics_ecc_secular_opt"
        ) as p_compute_dynamics_ecc_secular_opt:
            p_compute_dynamics_ecc_secular_opt.side_effect = (
                compute_dynamics_ecc_secular_opt
            )
            _, _, model_ehm = generate_modes_opt(
                q,
                chi_1[2],
                chi_2[2],
                omega0,
                omega_start,
                eccentricity,
                rel_anomaly,
                approximant="SEOBNRv5EHM",
                settings=settings,
                debug=True,
            )
        assert model_ehm.dynamics is not None
        p_compute_dynamics_ecc_secular_opt.assert_called_once()

    @pytest.mark.skipif(
        "CI_TEST_DYNAMIC_REGRESSIONS" not in os.environ,
        reason="regressions on dynamics are for specific systems only",
    )
    def test_regression_full_model(self):
        """Regression tests on the full model"""
        q = 1.5
        chi_1 = [0.0, 0.0, 0.0]
        chi_2 = [0.0, 0.0, 0.0]
        omega0 = 0.01
        omega_start = omega0
        eccentricity = 0.3
        rel_anomaly = 0.0

        # This gives the new eccentric model, but with the v5EHM infrastructure
        settings = {}
        _, _, model_ehm = generate_modes_opt(
            q,
            chi_1[2],
            chi_2[2],
            omega0,
            omega_start,
            eccentricity,
            rel_anomaly,
            approximant="SEOBNRv5EHM",
            settings=settings,
            debug=True,
        )

        frame_ehm_reference = pd.read_csv(folder_data / "frame_ehm.csv.gz")
        frame_ehm = pd.DataFrame(
            columns="t_e, r_e, phi_e, prstar_e, pphi_e, e_e, z_e, x_e, H_e, Omega_e".replace(
                " ", ""
            ).split(
                ","
            ),
            data=list(model_ehm.dynamics),
        )

        # we rename t_e to t to remain compatible with the compare_frames function
        frame_ehm_reference = frame_ehm_reference.rename(columns={"t_e": "t"})
        frame_ehm = frame_ehm.rename(columns={"t_e": "t"})

        # we generate the same number of elements in the dynamic
        assert (
            len(frame_ehm_reference.columns) == model_ehm.dynamics.shape[1]
        )  # ( == frame_ehm_reference.shape[1])

        known_differences_percentage = {
            "r_e": 1,
            "phi_e": 1,
            "prstar_e": 3,  # differences python3.9 / python3.10+
            "pphi_e": 1,
            "e_e": 1,
            "z_e": 1,
            "x_e": 1,
            "H_e": 1,
            "Omega_e": 1,
        }

        compare_frames(
            test_frame=frame_ehm,
            reference_frame=frame_ehm_reference,
            known_differences_percentage=known_differences_percentage,
            time_tolerance_percent=1,
        )

    def test_initial_conditions_dissipative_part_not_converging(self):
        """Checks the fallback to the second root finding in the initial condition dissipative part"""

        q = 1.5
        chi_1 = [0.0, 0.0, 0.0]
        chi_2 = [0.0, 0.0, 0.0]
        omega0 = 0.01
        omega_start = omega0
        eccentricity = 0.3
        rel_anomaly = 0.0

        settings = {}
        with mock.patch(
            "pyseobnr.eob.dynamics.initial_conditions_aligned_ecc_opt.root_scalar"
        ) as p_root_scalar, mock.patch(
            "pyseobnr.eob.dynamics.initial_conditions_aligned_ecc_opt.root"
        ) as p_root:

            def root_scalar_fake(f, *args, **kwargs):
                if f is IC_diss_ecc:

                    res = RootResults(
                        root=None, iterations=0, function_calls=0, flag=0, method="test"
                    )
                    res.converged = False
                    return res

                return root_scalar(f, *args, **kwargs)

            p_root_scalar.side_effect = root_scalar_fake
            p_root.side_effect = root

            _, _, model_ehm = generate_modes_opt(
                q,
                chi_1[2],
                chi_2[2],
                omega0,
                omega_start,
                eccentricity,
                rel_anomaly,
                approximant="SEOBNRv5EHM",
                settings=settings | {"dissipative_ICs": "root"},
                debug=True,
            )
            p_root_scalar.assert_called_once()
            p_root.assert_called()
            assert IC_diss_ecc in [_.args[0] for _ in p_root.call_args_list]

            assert model_ehm.dynamics is not None

    def test_estimate_t_for_max_amplitude_EHM(self):
        """Checks that the model is calling the proper methods for shifting to t=0"""

        q = 1
        chi_1z = 0.1
        chi_2z = 0.3
        omega0 = 0.03
        rel_anomaly = 0.0
        eccentricity = 0.3

        chi_1 = [0.0, 0.0, chi_1z]
        chi_2 = [0.0, 0.0, chi_2z]
        omega_start = omega0

        all_t0 = {}
        all_ts = {}
        all_frame_inv_amplitudes = {}

        # This gives the new eccentric model, but with the v5EHM infrastructure
        settings = {}
        with mock.patch(
            "pyseobnr.models.SEOBNRv5EHM.estimate_time_max_amplitude"
        ) as p_estimate_time_max_amplitude:

            def m_estimate_time_max_amplitude(*args, **kwargs):
                ret = estimate_time_max_amplitude(*args, **kwargs)
                # gets the outside freq
                all_t0[freq] = ret
                all_frame_inv_amplitudes[freq] = kwargs["amplitude"]
                return ret

            p_estimate_time_max_amplitude.side_effect = m_estimate_time_max_amplitude

            # 2**11, 2**12 ....
            freqs_to_check = np.logspace(10, 14, base=2, num=14 - 10 + 1)
            for freq in freqs_to_check:
                print(freq)
                _, _, model_ehm = generate_modes_opt(
                    q,
                    chi_1[2],
                    chi_2[2],
                    omega0,
                    omega_start,
                    eccentricity,
                    rel_anomaly,
                    approximant="SEOBNRv5EHM",
                    settings=settings | {"dt": 1 / float(freq), "M": 100},
                    debug=True,
                )
                p_estimate_time_max_amplitude.assert_called_once()
                p_estimate_time_max_amplitude.reset_mock()

                _, _, model_ehm = generate_modes_opt(
                    q,
                    chi_1[2],
                    chi_2[2],
                    omega0,
                    omega_start,
                    eccentricity,
                    rel_anomaly,
                    approximant="SEOBNRv5EHM",
                    settings=settings
                    | {"dt": 1 / float(freq), "M": 100, "inspiral_modes": True},
                    debug=True,
                )
                p_estimate_time_max_amplitude.assert_called()
                assert p_estimate_time_max_amplitude.call_count == 2
                p_estimate_time_max_amplitude.reset_mock()

                all_ts[freq] = model_ehm.t

        # we calculate the value of the amplitude at 10M
        inter_at_10M = []
        for freq in freqs_to_check:
            inter_at_10M += [
                {
                    "freq": freq,
                    "t_0": all_t0[freq],
                    # 10 -> this is 10M
                    "val": np.interp(10, all_ts[freq], all_frame_inv_amplitudes[freq]),
                }
            ]

        inter_at_10M = pd.DataFrame(inter_at_10M)

        # the estimated t_0 does in fact in this case vary so that stddev = 0.4108767856285839
        # assert inter_at_10M["t_0"].std() < 1e-2

        # the estimated amplitude at 10M does not vary too much
        # without the fix, we obtain 0.02275764373726919 for this specific setup
        print(inter_at_10M)
        assert inter_at_10M["val"].std() / inter_at_10M["val"].max() < 1e-3


def test_ehm_very_long_waveform_message():
    """Simple smoke test with spin case yielding an unsupported long waveform"""

    with pytest.raises(
        ValueError,
        match="Very long waveform for the input parameters: .*\\. Please, "
        "review the physical sense of the input parameters\\.",
    ):
        _ = generate_modes_opt(
            q=1.5,
            chi1=1,
            chi2=0,
            omega_start=0.00001,
            approximant="SEOBNRv5EHM",
        )


@pytest.fixture
def basic_ecc_parameters():
    m1 = 50.0
    m2 = 30.0
    params_dict = {
        "mass1": m1,
        "mass2": m2,
        "spin1x": 0,
        "spin1y": 0,
        "spin1z": 0.5,
        "spin2x": 0,
        "spin2y": 0,
        "spin2z": 0.1,
        "deltaT": 1 / 2048.0,
        "deltaF": 0.125,
        "f22_start": 0.0157 / ((m1 + m2) * np.pi * lal.MTSUN_SI),
        "phi_ref": 0.0,
        "distance": 1.0,
        "inclination": np.pi / 3.0,
        "f_max": 1024.0,
        "approximant": "SEOBNRv5EHM",
        "postadiabatic": False,
    }
    return params_dict


def test_parameters_ecc_through_waveform_interface(basic_ecc_parameters):
    minimal_params_dict = {
        "mass1": 50,
        "mass2": 30,
        "approximant": "SEOBNRv5EHM",
    }
    for ecc in [0.1, 0.2, 0.9]:
        # does not throw
        GenerateWaveform(minimal_params_dict | {"eccentricity": ecc})

    for ecc in [-0.1, 1.1, -1000]:
        with pytest.raises(ValueError):
            GenerateWaveform(basic_ecc_parameters | {"eccentricity": ecc})

    with pytest.raises(ValueError):
        GenerateWaveform(basic_ecc_parameters | {"conditioning": 3})

    for approximant in ["SEOBNRv5HM", "SEOBNRv5PHM"]:
        with pytest.raises(ValueError):
            GenerateWaveform(
                basic_ecc_parameters | {"eccentricity": 0.1, "approximant": approximant}
            )


def test_compute_dynamics_ecc():
    """Testing direct calls to the dynamic computation for correctness"""
    chi_1 = 0.1
    chi_2 = 0.1
    m_1 = 0.6
    m_2 = 0.4
    eccentricity = 0
    rel_anomaly = np.pi

    eob_params, hamiltonian = create_eob_params(
        m_1=m_1,
        m_2=m_2,
        chi_1=chi_1,
        chi_2=chi_2,
        omega=1 / 50,
        omega_circ=1 / 20,
        omega_avg=1 / 50,
        omega_instant=1 / 20,
        x_avg=0,
        eccentricity=eccentricity,
        rel_anomaly=rel_anomaly,
    )

    eob_params.ecc_params.omega_inst = dot_phi_omega_avg_e_z(
        omega_avg=eob_params.ecc_params.omega_avg,
        e=eob_params.ecc_params.eccentricity,
        z=eob_params.ecc_params.rel_anomaly,
        nu=eob_params.p_params.nu,
        delta=(m_1 - m_2) / (m_1 + m_2),
        chi_A=eob_params.p_params.chi_A,
        chi_S=eob_params.p_params.chi_S,
        flags_ecc=dict(
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
        ),
    )

    radiation_reaction_force = SEOBNRv5RRForceEcc("Ecc")
    radiation_reaction_force.initialize(eob_params)

    dynamics_ = compute_dynamics_ecc_opt(
        H=hamiltonian,
        RR=radiation_reaction_force,
        chi_1=chi_1,
        chi_2=chi_2,
        m_1=m_1,
        m_2=m_2,
        eccentricity=eccentricity,
        rel_anomaly=rel_anomaly,
        params=eob_params,
        integrator="rk8pd",
        step_back=250.0,
        y_init=None,
        r_stop=2.3241767533912303,
    )

    assert dynamics_ is not None


@pytest.mark.parametrize(
    "dict_params, mm_mean_calculated_in_notebook",
    [
        (
            dict(
                q=19.63211619423976017629,
                m1=23.36677130379361955193,
                m2=1.19023191756829760379,
                chi1z=0.98773810880722356931,
                chi2z=0.64438631556856262872,
                om=0.01381519044943839945,
                iota_s=1.52838651198559216660,
                e=0.42252935567880500756,
                z=3.92717452844127112854,
            ),
            1e-4,
        ),
        (
            dict(
                q=6.84546638516955940901,
                m1=49.63230261003650412022,
                m2=7.25039023163753793000,
                chi1z=-0.08112979979029000255,
                chi2z=0.27848911003169274370,
                om=0.01266671809025760737,
                iota_s=2.08574734100383940572,
                e=0.08550035506935330099,
                z=1.25046608065544151422,
            ),
            1e-5,
        ),
    ],
)
def test_perturbation(dict_params, mm_mean_calculated_in_notebook):
    """Perturbation test extraction"""
    # this is the worst mismatch in the perturbation test, put here
    # for ease of debugging. One of the reasons of the high mismatch is due to
    # different stopping conditions for both systems.

    q = dict_params["q"]
    m1 = dict_params["m1"]
    m2 = dict_params["m2"]
    chi1z = dict_params["chi1z"]
    chi2z = dict_params["chi2z"]
    om = dict_params["om"]
    iota_s = dict_params["iota_s"]
    e = dict_params["e"]
    z = dict_params["z"]

    phi_s = 0.0
    distance = 1e6 * lal.PC_SI

    ell_max = 5
    delta_t = 1.0 / 16384.0
    mode_list = ["2,2", "2,1", "3,3", "3,2", "4,4", "4,3"]
    mode_list_ints = list(
        (int(k[0]), int(k[1])) for element in mode_list if (k := element.split(","))
    )

    settings = {"ell_max": ell_max, "beta_approx": None, "dt": delta_t} | {
        "h_0": 1.0,
    }

    assert abs(q - m1 / m2) < 1e-5

    time_1, modes_non_pert = generate_modes_opt(
        q=m1 / m2,
        chi1=chi1z,
        chi2=chi2z,
        omega_start=om,
        eccentricity=e,
        rel_anomaly=z,
        approximant="SEOBNRv5EHM",
        settings=settings
        | {
            "return_modes": mode_list_ints,
            "M": m1 + m2,
        },
    )

    time_2, modes_pert = generate_modes_opt(
        q=m1 * (1 + 1e-15) / m2,
        chi1=chi1z,
        chi2=chi2z,
        omega_start=om,
        eccentricity=e,
        rel_anomaly=z,
        approximant="SEOBNRv5EHM",
        settings=settings
        | {"return_modes": mode_list_ints, "M": m1 * (1 + 1e-15) + m2},
    )

    hp, hc = get_hp_hc(
        current_modes=modes_non_pert,
        total_mass=m1 + m2,
        delta_t=delta_t,
        distance=distance,
        iota_s=iota_s,
        phi_s=phi_s,
    )
    hp_pert, hc_pert = get_hp_hc(
        current_modes=modes_pert,
        total_mass=m1 * (1 + 1e-15) + m2,
        delta_t=delta_t,
        distance=distance,
        iota_s=iota_s,
        phi_s=phi_s,
    )

    f_min = om / (np.pi * (m1 + m2) * lal.MTSUN_SI)

    f_low_phys = f_min
    f_high_phys = 2048.0

    psd = aLIGOZeroDetHighPowerGWINC(len(hp), hp.delta_f, f_low_phys)

    mm_hp = match(
        hp,
        hp_pert,
        psd,
        low_frequency_cutoff=f_low_phys,
        high_frequency_cutoff=f_high_phys,
    )[0]

    mm_hc = match(
        hc,
        hc_pert,
        psd,
        low_frequency_cutoff=f_low_phys,
        high_frequency_cutoff=f_high_phys,
    )[0]

    mm_mean = 1.0 - np.mean([mm_hp, mm_hc])
    assert abs(mm_mean - mm_mean_calculated_in_notebook) < 1e-2


def test_compute_x():
    """Checks the behaviour of the x computation and the arrays content returned by the dynamics"""

    m_1 = 49.63230261003650412022
    m_2 = 7.25039023163753793000
    omega_start = 0.012666718090257607
    x_avg = omega_start ** (2.0 / 3.0)

    eob_params_call1, hamiltonian1 = create_eob_params(
        m_1=m_1,
        m_2=m_2,
        chi_1=-0.08112979979029,
        chi_2=0.27848911003169274,
        omega=omega_start,
        omega_circ=omega_start,
        omega_avg=omega_start,
        omega_instant=0.013442463859018835,
        x_avg=x_avg,
        eccentricity=0.1,
        rel_anomaly=0,
    )

    # adjust omega_inst
    eob_params_call1.ecc_params.omega_inst = dot_phi_omega_avg_e_z(
        omega_avg=eob_params_call1.ecc_params.omega_avg,
        e=eob_params_call1.ecc_params.eccentricity,
        z=eob_params_call1.ecc_params.rel_anomaly,
        nu=eob_params_call1.p_params.nu,
        delta=(m_1 - m_2) / (m_1 + m_2),
        chi_A=eob_params_call1.p_params.chi_A,
        chi_S=eob_params_call1.p_params.chi_S,
        flags_ecc=dict(
            flagPN12=1,
            flagPN1=1,
            flagPN32=1,
            flagPN2=1,
            flagPN52=1,
            flagPN3=1,
            flagPA=0,
            flagPA_modes=1,
            flagTail=1,
            flagMemory=1,
        ),
    )

    radiation_reaction_force = SEOBNRv5RRForceEcc("Ecc")
    radiation_reaction_force.initialize(eob_params_call1)

    dyn_coarse, dyn_fine, dynamics_full, dynamics_raw = compute_dynamics_ecc_opt(
        H=hamiltonian1,
        RR=radiation_reaction_force,
        chi_1=eob_params_call1.p_params.chi_1,
        chi_2=eob_params_call1.p_params.chi_2,
        m_1=m_1,
        m_2=m_2,
        eccentricity=eob_params_call1.ecc_params.eccentricity,
        rel_anomaly=eob_params_call1.ecc_params.rel_anomaly,
        params=eob_params_call1,
        integrator="rk8pd",
        step_back=250.0,
        y_init=None,
        r_stop=2.3241767533912303,
    )

    for current_dynamics in (dyn_coarse, dyn_fine, dynamics_full):

        dynamics_x = compute_x(
            e=current_dynamics[:, ColsEccDyn.e],
            z=current_dynamics[:, ColsEccDyn.z],
            omega=current_dynamics[:, ColsEccDyn.omega],
            RR=radiation_reaction_force,
        )

        # check what the function compute_dynamics_ecc_opt is doing
        assert (current_dynamics[:, ColsEccDyn.x] == dynamics_x).all()

        # check what the function compute_x is doing
        for i in range(current_dynamics.shape[0]):

            radiation_reaction_force.evolution_equations.compute(
                e=current_dynamics[i, ColsEccDyn.e],
                z=current_dynamics[i, ColsEccDyn.z],
                omega=current_dynamics[i, ColsEccDyn.omega],
            )

            assert (
                radiation_reaction_force.evolution_equations.get("xavg_omegainst")
                == current_dynamics[i, ColsEccDyn.x]
            )


def test_compute_special_coeffs_ecc_not_called():
    """Checks the conditions under which compute_special_coeffs_ecc is called"""
    # see test_0_spins in test_cython_containers

    # this test checks that the conditions inside cython
    # if np.abs(self.nu - 0.25) < 1e-14 and np.abs(self.chi_A) < 1e-14:
    # and the python code are equivalent.
    # If they are not equivalent, the computations generate division by 0.

    q = 0.5000000057439491 / 0.49999999425605085
    omega = 0.012761920876346968
    eccentricity = 0.14745898920231795
    rel_anomaly = 0

    with mock.patch(
        "pyseobnr.generate_waveform.SEOBNRv5EHM.compute_special_coeffs_ecc"
    ) as p_compute_special_coeffs_ecc:
        # we have 0 spin, q is close to 1, we should not call compute_special_coeffs_ecc
        # as this would yield NaNs (0 division)
        generate_modes_opt(
            q=q,
            chi1=0,
            chi2=0,
            omega_start=omega,
            omega_ref=omega,
            eccentricity=eccentricity,
            rel_anomaly=rel_anomaly,
            approximant="SEOBNRv5EHM",
            settings={},
            debug=True,
        )

        p_compute_special_coeffs_ecc.assert_not_called()

    with mock.patch(
        "pyseobnr.generate_waveform.SEOBNRv5EHM.compute_special_coeffs_ecc"
    ) as p_compute_special_coeffs_ecc:
        # we have now almost spin, q is close to 1, we should be able to call compute_special_coeffs_ecc
        # without creating any NaNs

        array_fcoeffs = None

        def track_params(*args, **kwargs):
            nonlocal array_fcoeffs
            compute_special_coeffs_ecc_original(*args, **kwargs)
            array_fcoeffs = np.array(args[2].flux_params.f_coeffs).copy()

        p_compute_special_coeffs_ecc.side_effect = track_params

        generate_modes_opt(
            q=q,
            chi1=2.3602683827876965e-06,
            chi2=2.3505263976724266e-06,
            omega_start=omega,
            omega_ref=omega,
            eccentricity=eccentricity,
            rel_anomaly=rel_anomaly,
            approximant="SEOBNRv5EHM",
            settings={},
            debug=True,
        )

        p_compute_special_coeffs_ecc.assert_called_once()
        assert array_fcoeffs is not None
        assert bool(np.any(np.isnan(array_fcoeffs))) is False


def test_regression_nan_captured_in_dynamics_and_hamiltonians():
    """Test a regression introduced by cpow=True not triggering exceptions on domain error"""
    # see https://git.ligo.org/waveforms/software/pyseobnr/-/issues/25
    # the fix is to check the values taken by variables cast to double after exponentiation:
    # they should not be any NaN (which was the behaviour of python: "python pow" of negative
    # number yield complex values that cannot be cast to double. The cast to double was triggering
    # the exception).

    _ = generate_modes_opt(
        q=1.4731099610683323,
        chi1=-0.8693840187604053,
        chi2=-0.510689577500817,
        omega_start=0.025294844481864975,
        eccentricity=0.0,
        rel_anomaly=0.0,
        approximant="SEOBNRv5EHM",
    )
