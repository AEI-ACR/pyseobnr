from __future__ import annotations

from typing import Any

import numpy as np
from pygsl_lite import spline
from scipy.fft import fftfreq, fft
from scipy.interpolate import CubicSpline


def get_SPA_spline_fast(
    t_old: np.ndarray,
    f_new: np.ndarray,
    modes_dict: dict[tuple[int, int], Any],
    phi_orb: np.ndarray,
    adiabacity_epsilon: np.float64,
    t_min: np.float64,
    t_attachment: np.float64,
) -> dict[tuple[int, int], Any]:
    """A way to generate the SPA FD waveform for each mode based on the sparse
    dynamics array, while checking some proxys for the accuracy of the SPA on
    the fly and terminating it appropriately to be then combined with the FFT

    Args:
        t_old (np.ndarray): The times of the sparse dynamics grid
        f_new (np.ndarray): The dense frequency grid on which we want to
            evaluate the FD waveform
        modes_dict (dict[tuple[int, int], Any]): The modes evaluated on the
            sparse dynamics grid, corresponding to t_old
        phi_orb (np.ndarray): The unwrapped orbital phase of the sparse
            dynamics grid
        adiabacity_epsilon (np.float64): The threshold for the phase and
            amplitude adiabacity conditions
        t_min (np.float64): The maximal time until which we would evaluate the
            waveform to have the pre-merger model be contained in the FFT part
        t_attachment (np.float64): The time of merger/peak of the 22-mode

    Returns:
        dict[tuple[int, int], Any]: _description_
    """

    num_sparse_dynamics = np.size(t_old)

    Pio4 = np.pi / 4.0
    twoPi = 2 * np.pi

    # Setup orbital phase interpolation as carrier signal
    intrp_orb = spline.cspline(num_sparse_dynamics)
    intrp_orb.init(t_old, phi_orb)

    # Setup first frequency derivative to be used later for omega
    dphi_orb_interp = intrp_orb.eval_deriv_e_vector(t_old)

    intrp_amp = spline.cspline(num_sparse_dynamics)
    intrp_phase = spline.cspline(num_sparse_dynamics)

    # We start with the 22-mode, as this is where we actually compute the
    # adiabacity parameter and make the cut for the SPA

    # The FD waveform points of each mode
    result_SPA = {}
    # The frequencies on which to evaluate each mode in FD
    frequencies_SPA = {}
    # The idx from which we start the FFT for each mode
    start_idx_fft = {}

    # The stop_idx is set by either the adiabacity parameter becoming too big
    # or us reaching the region which we would like to compute via the FFT
    # due to the phenomenological pre-merger model
    # searchsorted gives us the idx where t_old[stop_idx_time] >= t_min
    stop_idx_time = np.searchsorted(t_old, t_min)
    if t_min < t_old[stop_idx_time]:
        stop_idx_time -= 1

    absolute_start_time = t_old[0]

    # The sorted(modes_dict, key=lambda x: x != (2,2)) makes the for-loop go over
    # the 22-mode first (if included) which is needed because we want to change
    # the absolute_start_time to be when the 22-mode hits f_new[0] = f22-start
    for ell, m in sorted(modes_dict, key=lambda x: x != (2, 2)):
        item = modes_dict[(ell, m)]

        # This is our carried mode
        tmp = item * np.exp(1j * m * phi_orb)
        amp_old = np.abs(tmp)
        phase_old = np.unwrap(np.angle(tmp))

        # It can happen under special conditions that the amplitude of the
        # waveform mode crosses zero, which makes the phase jump by pi if not
        # taken into account for through naive use of just np.abs
        # So here we make the phase continuous again by checking the maximum
        # and make the amplitude be negative
        # Assumption: Usually the carried phase is on the order of ~1.5 rad,
        # so a jump would leave an imprint of ~1.5 rad
        if phase_old[-1] - phase_old[0] > 1.5:
            # Assumption: We could see a discontinuity at the 2.5 rad level
            if np.any(phase_old[1:] - phase_old[:-1] > 2.5):
                # Assumption: This happens only once
                idx = np.argmax(phase_old[1:] - phase_old[:-1] > 2.5)
                amp_old[idx + 1 :] *= -1
                phase_old[idx + 1 :] -= np.pi

        # Step 1: Evaluate the splines at the original points for the
        # First and second derivatives, which we need for the adiabacity
        intrp_amp.init(t_old, amp_old)
        intrp_phase.init(t_old, phase_old)
        result_frequency = (
            -intrp_phase.eval_deriv_e_vector(t_old) + m * dphi_orb_interp
        ) / twoPi
        result_dfrequency = (
            -intrp_phase.eval_deriv2_e_vector(t_old)
            + m * intrp_orb.eval_deriv2_e_vector(t_old)
        ) / twoPi

        stop_idx = stop_idx_time
        # SPA also stops where the mode frequency stops rising (df/dt <= 0);
        # one of the SPA-stop conditions of Eq. (12) in arXiv:2606.02690.
        if not np.all(result_dfrequency[3:stop_idx] > 0):
            stop_idx = np.where(result_dfrequency[3:stop_idx] <= 0)[0][0] + 1
            # The +1 is because we started from index 3 and want to be
            # conservative and want to stop before we run into issues

        # Characteristic width (one std-dev, sigma = 1/sqrt(2*pi*df/dt)) of the SPA
        # Gaussian integrand; sets how much extra we must keep for the FFT. Cf. the
        # 3-sigma (~99% volume) SPA support in Eq. (11) of arXiv:2606.02690 (here the
        # prefactor is 2, not 3).
        integrand_support = 2.0 / np.sqrt(twoPi * result_dfrequency[stop_idx])

        # If the merger enters the SPA integrand support the SPA waveform degrades,
        # so step back by ~3 support widths. This is the "support reaches the merger"
        # SPA-stop condition, cf. Eq. (12) of arXiv:2606.02690.
        if (t_attachment - t_old[stop_idx]) < integrand_support:
            ref = t_old[stop_idx]
            stop_idx = np.searchsorted(t_old, (ref - 3 * integrand_support))
            integrand_support = 2.0 / np.sqrt(twoPi * result_dfrequency[stop_idx])

        # Start the FFT segment a few support widths before the SPA stop so the
        # transition frequency is faithfully resolved; cf. the FFT-start choice in
        # Eq. (21) of arXiv:2606.02690 (an additional fixed taper in M is applied
        # later, in the model / compute_tidal_tapering_v5T).
        start_idx_fft[(ell, m)] = np.searchsorted(
            t_old, (t_old[stop_idx] - 5 * integrand_support)
        )

        # Define the maximum frequency of the SPA as the waveform frequency
        # on the coarse dynamics grid where we reach our threshold
        f_max_SPA = result_frequency[stop_idx]

        # The start frequency is the frequency in the coarse dynamics array
        # which is just below the required minimum frequency array
        # This has to be contained in the dynamics due to the safe-guard in the
        # f_22_start, but we try to cut two data points to have better derivatives
        # in the spline
        start_idx = np.max((np.searchsorted(result_frequency, f_new[0]) - 2, 0))
        start_ellm = 0

        # The m > 2 modes will start at a higher frequency than f_new[0]
        # So we will pad with zeros in the beginning and try to replicate
        # the conditioning
        if result_frequency[start_idx] > f_new[0]:
            f_start_ellm = (
                -intrp_phase.eval_deriv_e_vector([absolute_start_time])
                + m * intrp_orb.eval_deriv_e_vector([absolute_start_time])
            )[0] / twoPi
            df_start_ellm = (
                -intrp_phase.eval_deriv2_e_vector([absolute_start_time])
                + m * intrp_orb.eval_deriv2_e_vector([absolute_start_time])
            )[0] / twoPi

            # We need a positive frequency derivative for the SPA conditioning
            if df_start_ellm < 1e-20:
                df_start_ellm = 1e-20

            start_ellm = np.searchsorted(f_new, f_start_ellm)
            start_idx = np.max((np.searchsorted(t_old, absolute_start_time) - 2, 0))
            if f_new[start_ellm] < f_start_ellm:
                start_ellm += 1

            # Making a guess for the conditioned frequencies using the SPA
            time_window = 3.0 * 2 / f_new[0] + 0.1 * (
                t_old[-1] - absolute_start_time
            )  # 3 cycles of the starting frequency

            f_conditioning_start_ellm = f_start_ellm - df_start_ellm * time_window
            amp_start_ellm = intrp_amp.eval_e_vector([absolute_start_time])[0]
            phase_start_ellm = (
                -intrp_phase.eval_e_vector([absolute_start_time])[0]
                + m * intrp_orb.eval_e_vector([absolute_start_time])[0]
            )

            conditioning_start_ellm = np.searchsorted(f_new, f_conditioning_start_ellm)
            if f_new[conditioning_start_ellm] < f_conditioning_start_ellm:
                conditioning_start_ellm += 1

            if conditioning_start_ellm < start_ellm:
                f_eval_conditioning = f_new[conditioning_start_ellm:start_ellm]
                time_f_conditioning = (
                    f_eval_conditioning - f_start_ellm
                ) / df_start_ellm
                time_f_conditioning_phase = time_f_conditioning

                amp_conditioning_SPA = (
                    amp_start_ellm
                    / 2
                    / np.sqrt(df_start_ellm)
                    * (1.0 + np.cos(np.pi / time_window * time_f_conditioning))
                )
                phase_conditioning_SPA = (
                    Pio4
                    + phase_start_ellm
                    + 2
                    * np.pi
                    * (
                        f_start_ellm * time_f_conditioning_phase
                        + 0.5 * df_start_ellm * time_f_conditioning_phase**2
                    )
                    - 2
                    * np.pi
                    * f_eval_conditioning
                    * (time_f_conditioning_phase + absolute_start_time - t_attachment)
                )

                conditioning_SPA = amp_conditioning_SPA * np.exp(
                    1j * phase_conditioning_SPA
                )

                frequencies_zero_ellm = np.pad(
                    conditioning_SPA, (conditioning_start_ellm, 0)
                )

            else:
                frequencies_zero_ellm = np.zeros(start_ellm, np.complex128)

        if start_idx >= stop_idx:
            result_SPA[(ell, m)] = []

            # Guard against the case where we don't perform an SPA at all
            continue

        # We try to extend by two data points to have better
        # derivatives in the spline
        stop_idx = np.min((stop_idx + 2, num_sparse_dynamics))

        # This is how many points we'll be using for the interpolation
        n_old = stop_idx - start_idx

        # start_ellm + ind_max_SPA is the index of the new frequency array until
        # which we compute the SPA, so
        # f_new[start_ellm + ind_max_SPA] <= f_max_SPA
        # which is therefore the start of the FFT and our time reference
        # SPA is on [f_new[start_ellm], f_new[start_ellm + ind_max_SPA]]
        # (including f_new[start_ellm + ind_max_SPA])
        # FFT is on (f_new[start_ellm + ind_max_SPA], f_new[-1]]
        # (excluding f_new[start_ellm + ind_max_SPA])
        ind_max_SPA = np.searchsorted(f_new[start_ellm:], f_max_SPA)

        frequencies_SPA[(ell, m)] = f_new[start_ellm : start_ellm + ind_max_SPA]

        # Step 2: Interpolate all quantities to new frequency grid
        intrp_freq_time = spline.cspline(int(n_old))
        intrp_freq_time.init(
            result_frequency[start_idx:(stop_idx)], t_old[start_idx:(stop_idx)]
        )

        t_new = intrp_freq_time.eval_e_vector(frequencies_SPA[(ell, m)])

        phi_orb_interp_new = intrp_orb.eval_e_vector(t_new)
        result_phase_new = -intrp_phase.eval_e_vector(t_new) + m * phi_orb_interp_new
        result_amp_new = intrp_amp.eval_e_vector(t_new)
        ddphi_orb_interp_new = intrp_orb.eval_deriv2_e_vector(t_new)
        result_dfrequency_new = (
            -intrp_phase.eval_deriv2_e_vector(t_new) + m * ddphi_orb_interp_new
        ) / twoPi

        # Step 3: Actually compute the SPA
        # Our t = 0 is when t_new = t_attachment
        psi_f = (
            result_phase_new
            + Pio4
            - twoPi * frequencies_SPA[(ell, m)] * (t_new - t_attachment)
        )
        ampli_f = result_amp_new / np.sqrt(abs(result_dfrequency_new))

        if start_ellm > 0:
            result_SPA[(ell, m)] = np.concatenate(
                (
                    frequencies_zero_ellm,
                    ampli_f * np.exp(1j * (psi_f)),
                )
            )
            if conditioning_start_ellm < start_ellm:
                frequencies_SPA[(ell, m)] = f_new[: start_ellm + ind_max_SPA]
        else:
            result_SPA[(ell, m)] = ampli_f * np.exp(1j * (psi_f))

        if (ell, m) == (2, 2):
            # This is the time where f22(t_new[0]) = f_new[0]
            absolute_start_time = t_new[0]

    return result_SPA, start_idx_fft


def compute_fd_polarizations_via_spa_plus_fft(
    t_full,
    hlms_full,
    t_new,
    delta_T,
    taper_length,
    t_attach,
    frequencies,
    result_SPA,
    start_idx_fft,
    dynamics,
):
    """The function to combine the SPA of the inspiral with the FFT of the
    late-inspiral, pre-merger and post-merger tapering for SEOBNRv5THM,
    performed mode-by-mode.

    Args:
        t_full (np.ndarray): The equidistant time array of the tapered pre-merger waveform
        hlms_full (dict): The full waveform modes of the tapered pre-merger waveform (corresponding to t_full)
        t_new (np.ndarray): The fine equidistant dynamics time grid
        delta_T (float): The time step of both t_full and t_new
        taper_length (float): The length of the tapering for the conditioning of the FFT
        t_attach (float): The attachment time of SEOBNRv5THM
        frequencies (np.ndarray): The frequencies on which we want to evaluate the final FD waveform
        result_SPA (dict): The FD waveform modes of the SPA evaluated on the frequencies up to the transition-frequency
        start_idx_fft (dict): The starting indices for the FFT regions of each mode as a function of frequency
        dynamics (np.ndarray): The dynamics array on the fine grid t_new

    Returns:
        waveform_modes (dict): The FD waveform modes using SPA and FFT combined, evaluated on frequencies
    """
    # Pre-compute the rising half-Hann taper applied to the start of each FFT
    # segment (length `taper_length`, in M). This is the Hanning window of Eq. (14)
    # in arXiv:2606.02690; the 2000 M, includes the Hann-taper of length
    # `taper_length` (500 M from the model) as well as some extra time to have a
    # smooth transition and avoid edge effects in the FFT
    waveform_modes = {}
    num_fine_waveform = np.size(t_full)
    window_length = int(np.floor(taper_length / delta_T))
    window = 0.5 - 0.5 * np.cos(np.pi / window_length * np.arange(window_length))
    for ell, m in hlms_full:

        # We only need to do something if the FFT region
        # is non-empty
        if np.size(frequencies) > np.size(result_SPA[(ell, m)]):

            t_FFT_start = dynamics[start_idx_fft[ell, m], 0] - taper_length
            start_idx_fft_ellm = np.searchsorted(t_new, t_FFT_start)
            if t_new[start_idx_fft_ellm] > (
                dynamics[start_idx_fft[ell, m], 0] - taper_length
            ):
                start_idx_fft_ellm -= 1

            mode = hlms_full[(ell, m)][start_idx_fft_ellm:]
            N_filled = np.size(result_SPA[(ell, m)])

            mode[:window_length] = mode[:window_length] * window

            num_fine_waveform_ellm = num_fine_waveform - start_idx_fft_ellm

            # Next we want to pad to a power of two to get free resolution and
            # have a correct f_max
            # Compute next power of two
            next_pow2 = 1 << int(num_fine_waveform_ellm - 1).bit_length()
            # Compute padding size
            pad_width = next_pow2 - num_fine_waveform_ellm
            # Pad with zeros at the end
            tapered_mode = np.pad(mode, (0, pad_width), mode="constant")

            # plt.plot(tapered_mode)
            # plt.show()

            frequencies_ellm = fftfreq(next_pow2, delta_T)

            t_shift = t_attach - t_new[start_idx_fft_ellm]
            time_shift_fft = np.exp(1j * 2 * np.pi * t_shift * frequencies_ellm)

            # Due to our definition of the mode orientation
            # the FFT will be interesting to us for negative
            # frequencies
            mode_fd = delta_T * fft((tapered_mode.conj()), next_pow2) * time_shift_fft

            only_pos_freqs = frequencies_ellm[: int(next_pow2 / 2)]
            only_pos_fd = mode_fd[: int(next_pow2 / 2)]
            only_pos_fd_full = mode_fd.copy()
            only_pos_fd_full[int(next_pow2 / 2) :] = 0.0

            # plt.plot(frequencies_ellm, mode_fd)
            # plt.plot(only_pos_freqs, only_pos_fd)
            # plt.show()

            # td_full = ifft(mode_fd)
            # plt.plot(np.real(td_full) / self.delta_T)
            # plt.show()

            frequencies_to_evaluate = frequencies[N_filled:]
            start_ellm = np.max(
                (
                    np.searchsorted(only_pos_freqs, frequencies[N_filled]) - 2,
                    0,
                )
            )

            # amp = CubicSpline(only_pos_freqs[start_ellm:], np.abs(only_pos_fd[start_ellm:]))(frequencies_to_evaluate)

            # phase = CubicSpline(only_pos_freqs[start_ellm:], np.unwrap(np.angle(only_pos_fd[start_ellm:])))(frequencies_to_evaluate)

            Re = CubicSpline(
                only_pos_freqs[start_ellm:],
                np.real(only_pos_fd[start_ellm:]),
            )(frequencies_to_evaluate)

            Im = CubicSpline(
                only_pos_freqs[start_ellm:],
                np.imag(only_pos_fd[start_ellm:]),
            )(frequencies_to_evaluate)

            # plt.plot(only_pos_freqs[start_ellm:], np.real(only_pos_fd[start_ellm:]),label='Re FFT')
            # plt.plot(frequencies_to_evaluate, Re, ls='--', label='Re interp')
            # plt.show()

            # plt.plot(only_pos_freqs[start_ellm:], np.imag(only_pos_fd[start_ellm:]),label='Im FFT')
            # plt.plot(frequencies_to_evaluate, Im, ls='--', label='Im interp')
            # plt.legend()
            # plt.show()

            # self.waveform_modes[f"{ell},{m}"] = np.concatenate((self.result_SPA[(ell, m)],
            #                                             amp*np.exp(1j*phase)))

            waveform_modes[f"{ell},{m}"] = np.concatenate(
                (result_SPA[(ell, m)], Re + 1j * Im)
            )

            # plt.plot(only_pos_freqs, np.real(only_pos_fd), label='FFT')
            # plt.plot(frequencies[:N_filled], np.real(self.result_SPA[(ell, m)]), label='SPA')
            # plt.plot(frequencies, self.waveform_modes[f"{ell},{m}"], '--', label='combined')
            # plt.xlim(0., frequencies[-1])
            # plt.legend()
            # plt.xlabel('$Mf$')
            # # plt.axvline()
            # plt.show()

        else:
            waveform_modes[f"{ell},{m}"] = result_SPA[(ell, m)]
    return waveform_modes
