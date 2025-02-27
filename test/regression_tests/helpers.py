import re

import lal
import numpy as np
import pandas as pd
from pycbc.filter import make_frequency_series
from pycbc.types import TimeSeries
from pycbc.waveform import taper_timeseries

from pyseobnr.eob.fits import GSF_amplitude_fits, a6_NS, dSO
from pyseobnr.eob.hamiltonian.Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C import (
    Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C as Ham_aligned_opt,
)
from pyseobnr.eob.utils.containers import CalibCoeffs, EOBParams
from pyseobnr.eob.waveform.waveform import compute_newtonian_prefixes


def compare_frames(
    test_frame, reference_frame, known_differences_percentage, time_tolerance_percent=1
):
    # sanity checks
    # all returned values are numbers
    assert test_frame.isna().any(axis=None).item() is False
    # all time steps are unique
    assert test_frame.t.is_unique and reference_frame.t.is_unique
    # time starts with 0
    assert test_frame.t[0] == 0
    # time is all positive
    assert (test_frame.t[1:] > 0).all()
    # time is ordered and strictly increasing
    assert (test_frame.t.diff()[1:] > 0).all()
    # time steps are decreasing non monotonically
    # this is not true
    # assert (frame_ehm.t.diff()[1:].diff().diff() < 0).all()

    # we generate the same number of elements in the dynamic
    assert len(reference_frame.columns) == reference_frame.shape[1]

    # dynamics stop at more or less the same time
    # this is a very rough check that values are not completely crazy
    t1_end = test_frame[["t"]].max().item()
    t2_end = reference_frame[["t"]].max().item()

    assert abs(t1_end - t2_end) / max(t1_end, t2_end) < (time_tolerance_percent / 100)

    # we create a new index composite of both index, keep unique values
    new_index_t = np.array(
        sorted(pd.concat((reference_frame.t, test_frame.t)).unique())
    )

    # we now inject the time steps of the reference frame into the current frame
    test_frame_missing_values = test_frame.set_index("t").reindex(new_index_t)
    reference_frame_missing_values = reference_frame.set_index("t").reindex(new_index_t)

    # we interpolate the missing values (method is important, index takes the time
    # point for performing the interpolation but is not precise enough)
    test_frame_missing_values_interpolated = test_frame_missing_values.interpolate(
        # method="index"
        method="cubic",
        order=3,
        limit_area="inside",
    )

    reference_frame_missing_values_interpolated = reference_frame_missing_values.interpolate(
        # method="index"
        method="cubic",
        order=3,
        limit_area="inside",
    )

    # we keep the section common to both
    test_frame_projected = test_frame_missing_values_interpolated[
        test_frame_missing_values_interpolated.index <= min(t1_end, t2_end)
    ]
    reference_frame_projected = reference_frame_missing_values_interpolated[
        reference_frame_missing_values_interpolated.index <= min(t1_end, t2_end)
    ]

    # all timesteps of both reference and test frames have now a proper value and they align
    assert test_frame_projected.isna().any(axis=None).item() is False
    assert reference_frame_projected.isna().any(axis=None).item() is False
    assert (test_frame_projected.index == reference_frame_projected.index).all()

    for idx_column, column in enumerate(reference_frame_projected.columns):
        percentage_tolerance = known_differences_percentage[column]

        diffs_column = (
            reference_frame_projected[[column]] - test_frame_projected[[column]]
        ).abs()

        fractional_diffs_column = (
            100
            * diffs_column[column]
            / pd.concat(
                (reference_frame_projected[column], test_frame_projected[column]),
                axis=1,
            ).max(axis=1)
        )

        assert float(fractional_diffs_column.max()) < percentage_tolerance, (
            f"column {column} exceeds fractional tolerance: "
            f"%g > {percentage_tolerance} at index %d and time %g "
            "(reference=%g, test=%g)"
        ) % (
            float(fractional_diffs_column.max()),
            fractional_diffs_column.argmax(),
            fractional_diffs_column.idxmax(),
            reference_frame_projected.iloc[fractional_diffs_column.argmax()][
                [column]
            ].item(),
            test_frame_projected.iloc[fractional_diffs_column.argmax()][
                [column]
            ].item(),
        )


def combine_modes(
    iota: float, phi: float, modes_dict: dict
) -> tuple[np.array, np.array]:
    """Combine modes to compute the waveform polarizations in the direction
    (iota,np.pi/2-phi)

    Args:
        iota (float): Inclination angle (rad)
        phi (float): Azimuthal angle(rad)
        modes_dict (Dict): Dictionary containing the modes, either time of frequency-domain

    Returns:
        np.array: Waveform in the given direction
    """
    sm = 0.0
    for key in modes_dict.keys():
        # print(key)
        ell, m = [int(x) for x in key.split(",")]
        Ylm0 = lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phi, -2, ell, m)
        sm += Ylm0 * modes_dict[key]

    return np.real(sm), -np.imag(sm)


def ampNRtoPhysicalTD(ampNR, mt: float, distance: float):
    return ampNR * (lal.C_SI * mt * lal.MTSUN_SI) / distance


def get_hp_hc(
    current_modes,
    total_mass,
    iota_s: float,
    distance: float,
    phi_s: float,
    delta_t: float,
):
    hp_NR, hc_NR = combine_modes(iota_s, phi_s, current_modes)

    hp = ampNRtoPhysicalTD(hp_NR, total_mass, distance)
    hc = ampNRtoPhysicalTD(hc_NR, total_mass, distance)

    # Taper
    hp_td = TimeSeries(hp, delta_t=delta_t)
    hc_td = TimeSeries(hc, delta_t=delta_t)
    hp_td = taper_timeseries(hp_td, tapermethod="startend")
    hc_td = taper_timeseries(hc_td, tapermethod="startend")

    N = max(len(hp_td), len(hc_td))
    pad = int(2 ** (np.floor(np.log2(N)) + 2))
    hp_td.resize(pad)
    hc_td.resize(pad)

    # Perform the Fourier Transform
    hp = make_frequency_series(hp_td)
    hc = make_frequency_series(hc_td)

    return hp, hc


def create_eob_params(
    m_1,
    m_2,
    chi_1,
    chi_2,
    omega,
    omega_circ,
    omega_avg,
    omega_instant,
    x_avg,
    eccentricity,
    rel_anomaly,
):

    # to get the exact same nu as in the models computations (a6 is very
    # sensitive to this), we perform the exact same calculations for
    # q, nu, m_1 and m_2
    q = m_1 / m_2
    nu = q / (1.0 + q) ** 2

    m_1 = q / (1.0 + q)
    m_2 = 1.0 / (1.0 + q)

    eob_params_call1 = EOBParams(
        {
            "m_1": m_1,
            "m_2": m_2,
            "chi_1": chi_1,
            "chi_2": chi_2,
            "a1": abs(chi_1),
            "a2": abs(chi_2),
            "chi1_v": np.array([0.0, 0.0, chi_1]),
            "chi2_v": np.array([0.0, 0.0, chi_2]),
            "lN": np.array([0.0, 0.0, 1.0]),
            "omega": omega,
            "omega_circ": omega_circ,
            "H_val": 0.0,
            # ecc params
            "flags_ecc": dict(
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
            "IC_messages": False,
            "dissipative_ICs": "root",
            "eccentricity": eccentricity,
            "rel_anomaly": rel_anomaly,
            "EccIC": 1,
            "t_max": 1e9,
            "r_min": 7.0,
        },
        {},
        mode_array=[(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)],
        ecc_model=True,
    )
    eob_params_call1.flux_params.rho_initialized = False
    eob_params_call1.flux_params.prefixes = np.array(
        compute_newtonian_prefixes(m_1, m_2)
    )
    eob_params_call1.flux_params.prefixes_abs = np.abs(
        eob_params_call1.flux_params.prefixes
    )
    eob_params_call1.flux_params.extra_PN_terms = True

    eob_params_call1.ecc_params.omega_avg = omega_avg
    eob_params_call1.ecc_params.omega_inst = omega_instant
    eob_params_call1.ecc_params.x_avg = x_avg

    eob_params_call1.dynamics.p_circ = np.array([0.0, 0.0])

    assert eob_params_call1.aligned is True
    assert eob_params_call1.c_coeffs is not None

    assert (np.array(eob_params_call1.flux_params.delta_coeffs) == 0).all()
    assert (np.array(eob_params_call1.flux_params.delta_coeffs_vh) == 0).all()
    assert (np.array(eob_params_call1.flux_params.deltalm) == 0).all()
    # extra_coeffs, extra_coeffs_log not 0

    gsf_coeffs = GSF_amplitude_fits(eob_params_call1.p_params.nu)
    keys = gsf_coeffs.keys()
    for key in keys:
        tmp = re.findall(r"h(\d)(\d)_v(\d+)", key)
        if tmp:
            l, m, v = [int(x) for x in tmp[0]]
            eob_params_call1.flux_params.extra_coeffs[l, m, v] = gsf_coeffs[key]
        elif tmp := re.findall(r"h(\d)(\d)_vlog(\d+)", key):
            l, m, v = [int(x) for x in tmp[0]]
            eob_params_call1.flux_params.extra_coeffs_log[l, m, v] = gsf_coeffs[key]

    hamiltonian = Ham_aligned_opt(eob_params_call1)

    hamiltonian.eob_params.c_coeffs = CalibCoeffs(
        {
            "a6": a6_NS(nu),
            "dSO": dSO(
                nu,
                m_1 * chi_1 + m_2 * chi_2,  # ap
                m_1 * chi_1 - m_2 * chi_2,  # am
            ),
        }
    )

    return eob_params_call1, hamiltonian
