import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import os
import importlib
import argparse

from scipy.interpolate import InterpolatedUnivariateSpline
from bilby.core.utils import check_directory_exists_and_if_not_mkdir
from pathos.multiprocessing import ProcessingPool as Pool
from pyseobnr.generate_waveform import generate_modes_opt

mode_list_v5HM = [(2, 2), (2, 1), (3, 3), (4, 4), (5, 5), (4, 3), (3, 2)]


def plots_smoothness_in_q(N: int, mode):
    """
    Plots of the (ell,m) mode amplitude and frequency for equal-spins varying q to show smoothness
    """

    colors = plt.cm.inferno(
        np.linspace(0, 1, N + 15)
    )  # adjust based on how many curves are plotted

    q_arr = np.linspace(1.0, 100.0, N)
    omega0 = 0.015

    ell, m = mode
    for chi in [
        -0.995,
        -0.5,
        0.0,
        0.5,
        0.995,
    ]:

        # amplitude plots
        for i, q in enumerate(q_arr):
            if not (m % 2 == 1 and q < 1.01):
                _, _, model = generate_modes_opt(q, chi, chi, omega0, debug=True)
                tm = model.t[np.argmax(np.abs(model.waveform_modes[f"2,2"]))]
                plt.plot(
                    model.t - tm,
                    np.abs(model.waveform_modes[f"{ell},{m}"]),
                    color=colors[i],
                )
        plt.xlim(-300, 100)
        plt.ylim(1e-5, 1e0)
        plt.yscale("log")
        plt.xlabel("$t/M$")
        plt.ylabel(f"$|h_{{{ell}{m}}}|$")
        plt.title(f"$\chi_1 = \chi_2 = {chi}$")
        legend_elements = [
            Line2D([0], [0], color=colors[0], lw=2, label=f"$q={q_arr[0]}$"),
            Line2D(
                [0], [0], color=colors[len(q_arr) - 1], lw=2, label=f"$q={q_arr[-1]}$"
            ),
        ]
        plt.legend(handles=legend_elements, loc="best")
        plt.savefig(
            f"./plots_smoothness/smoothness_{ell}{m}_amp_in_q_chi_{chi}.png", dpi=150
        )
        plt.close()

        # frequency plots
        for i, q in enumerate(q_arr):
            if not (m % 2 == 1 and q < 1.01):
                _, _, model = generate_modes_opt(q, chi, chi, omega0, debug=True)
                tm = model.t[np.argmax(np.abs(model.waveform_modes[f"2,2"]))]
                phi = -np.unwrap(np.angle(model.waveform_modes[f"{ell},{m}"]))
                iphi = InterpolatedUnivariateSpline(model.t, phi)
                omega_ellm = iphi.derivative()(model.t)
                plt.plot(model.t - tm, omega_ellm, color=colors[i])
        plt.xlim(-300, 100)
        id = np.abs(model.t - 100).argmin()
        lim = omega_ellm[id] + 0.2
        plt.ylim(0.0, lim)
        plt.xlabel("$t/M$")
        plt.ylabel(f"$\omega_{{{ell}{m}}}$")
        plt.title(f"$\chi_1 = \chi_2 = {chi}$")
        legend_elements = [
            Line2D([0], [0], color=colors[0], lw=2, label=f"$q={q_arr[0]}$"),
            Line2D(
                [0], [0], color=colors[len(q_arr) - 1], lw=2, label=f"$q={q_arr[-1]}$"
            ),
        ]
        plt.legend(handles=legend_elements, loc="best")
        plt.savefig(
            f"./plots_smoothness/smoothness_{ell}{m}_om_in_q_chi_{chi}.png", dpi=150
        )
        plt.close()


def plots_smoothness_in_chi(N: int, mode):
    """
    Plots of the (ell,m) mode amplitude and frequency for equal-spins varying chi to show smoothness
    """

    colors = plt.cm.inferno(
        np.linspace(0, 1, N + 15)
    )  # adjust based on how many curves are plotted

    chi_arr = np.linspace(-0.995, 0.995, N)
    omega0 = 0.015

    ell, m = mode
    #for q in [1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0]:
    for q in [1.0, 2.0, 8.0, 20.0]:

        # amplitude plots
        for i, chi in enumerate(chi_arr):
            if not (m % 2 == 1 and q < 1.01):
                _, _, model = generate_modes_opt(q, chi, chi, omega0, debug=True)
                tm = model.t[np.argmax(np.abs(model.waveform_modes[f"2,2"]))]
                plt.plot(
                    model.t - tm,
                    np.abs(model.waveform_modes[f"{ell},{m}"]),
                    color=colors[i],
                )
        plt.xlim(-300, 100)
        plt.ylim(1e-5, 1e0)
        plt.yscale("log")
        plt.xlabel("$t/M$")
        plt.ylabel(f"$|h_{{{ell}{m}}}|$")
        plt.title(f"$q = {q}$")
        legend_elements = [
            Line2D(
                [0], [0], color=colors[0], lw=2, label=f"$\chi_1 = \chi_2={chi_arr[0]}$"
            ),
            Line2D(
                [0],
                [0],
                color=colors[len(chi_arr) - 1],
                lw=2,
                label=f"$\chi_1 = \chi_2={chi_arr[-1]}$",
            ),
        ]
        plt.legend(handles=legend_elements, loc="best")
        plt.savefig(
            f"./plots_smoothness/smoothness_{ell}{m}_amp_in_chi_q_{q}.png", dpi=150
        )
        plt.close()

        # frequency plots
        for i, chi in enumerate(chi_arr):
            if not (m % 2 == 1 and q < 1.01):
                _, _, model = generate_modes_opt(q, chi, chi, omega0, debug=True)
                tm = model.t[np.argmax(np.abs(model.waveform_modes[f"2,2"]))]
                phi = -np.unwrap(np.angle(model.waveform_modes[f"{ell},{m}"]))
                iphi = InterpolatedUnivariateSpline(model.t, phi)
                omega_ellm = iphi.derivative()(model.t)
                plt.plot(model.t - tm, omega_ellm, color=colors[i])
                plt.xlim(-300, 100)
                id = np.abs(model.t - 100).argmin()
                lim = omega_ellm[id] + 0.2
                plt.ylim(0.0, lim)
        plt.xlabel("$t/M$")
        plt.ylabel(f"$\omega_{{{ell}{m}}}$")
        plt.title(f"$q = {q}$")
        legend_elements = [
            Line2D(
                [0], [0], color=colors[0], lw=2, label=f"$\chi_1 = \chi_2={chi_arr[0]}$"
            ),
            Line2D(
                [0],
                [0],
                color=colors[len(chi_arr) - 1],
                lw=2,
                label=f"$\chi_1 = \chi_2={chi_arr[-1]}$",
            ),
        ]

        plt.legend(handles=legend_elements, loc="best")
        plt.savefig(
            f"./plots_smoothness/smoothness_{ell}{m}_om_in_chi_q_{q}.png", dpi=150
        )
        plt.close()


def plot_one_mode(input):
    N, mode, quantity = input
    if quantity == "q":
        plots_smoothness_in_q(N, mode)
    elif quantity == "chi":
        plots_smoothness_in_chi(N, mode)


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="Make plots to show smoothness varying physical parameters "
    )
    p.add_argument("--points", type=int, help="Number of points", default=41)
    p.add_argument("--n-cpu", type=int, help="Number of cpus to use", default=14)
    args = p.parse_args()

    check_directory_exists_and_if_not_mkdir("./plots_smoothness")

    lst_q = [(args.points, mode, "q") for mode in mode_list_v5HM]
    lst_chi = [(args.points, mode, "chi") for mode in mode_list_v5HM]
    lst = lst_q + lst_chi
    pool = Pool(args.n_cpu)
    pool.map(plot_one_mode, lst)
