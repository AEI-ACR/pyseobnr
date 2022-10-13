import argparse
import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils import check_directory_exists_and_if_not_mkdir
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.interpolate import CubicSpline

from pyseobnr.auxiliary.sanity_checks.metrics import *


from pyseobnr.auxiliary.sanity_checks.parameters import parameters_random, parameters_random_fast
from pyseobnr.auxiliary.sanity_checks.single_waveform_tests import monotonic_quantity_test
from pyseobnr.generate_waveform import generate_modes_opt


def h22_monotonicity(
    q: float,
    chi1: float,
    chi2: float,
    plots: bool = True,
    omega0: float = 0.015,
    threshold: float = 1.0e-8,
):

    """
    Generate a SEOBNRv5 waveform with parameters (q, chi1, chi2)
    and run monotonicity check for the (2,2) mode amplitude

    Return (0,0,0) if the test passed, (q, chi1, chi2) otherwise
    """

    try:

        _, _, model = generate_modes_opt(q, chi1, chi2, omega0,debug=True)
        # Monotonicity of the (2,2) mode amplitude until merger
        h22 = np.abs(model.waveform_modes["2,2"])
        t_h = model.t
        t_merger = t_h[np.argmax(h22)]

        try:
            monotonic_quantity_test(t_h, h22, t_merger, threshold)
            return (0, 0, 0)
        except AssertionError:
            print(
                f"|h22| non monotonic for the following parameters: q = {q}, chi1 = {chi1}, chi2 = {chi2}"
            )
            if plots == True:
                label = f"q{np.round(q,2)}_s{np.round(chi1,3)}_s{np.round(chi2,3)}"
                plt_dir = "./plots_checks_h22"
                check_directory_exists_and_if_not_mkdir(plt_dir)
                plt.plot(t_h, h22)
                plt.axvline(t_merger, ls="--")
                plt.xlim(t_merger - 30, t_merger + 30)
                plt.xlabel("$t/M$")
                plt.ylabel("$|h_{22}|$")
                plt.title(f"{label}")
                plt.savefig(f"{plt_dir}/check_h22_{label}.png", dpi=150)
                plt.close()
            return (q, chi1, chi2)
    except IndexError:
        print(
            f"Error for the following parameters: q = {q}, chi1 = {chi1}, chi2 = {chi2}"
        )


def omega22_monotonicity(
    q: float,
    chi1: float,
    chi2: float,
    plots: bool = True,
    omega0: float = 0.015,
    threshold: float = 1.0e-8,
):

    """
    Generate a SEOBNRv5 waveform with parameters (q, chi1, chi2)
    and run monotonicity check for the (2,2) mode frequency

    Return (0,0,0) if the test passed, (q, chi1, chi2) otherwise
    """

    try:

        _, _, model = generate_modes_opt(q, chi1, chi2, omega0,debug=True)

        # Monotonicity of the (2,2) mode frequency until merger
        phi22 = -np.unwrap(np.angle(model.waveform_modes["2,2"]))
        t_h = model.t
        intphi22 = CubicSpline(t_h, phi22)
        omega22 = intphi22.derivative()(t_h)

        h22 = np.abs(model.waveform_modes["2,2"])
        t_merger = t_h[np.argmax(h22)]

        try:
            monotonic_quantity_test(t_h, omega22, t_merger, threshold)
            return (0, 0, 0)
        except AssertionError:
            print(
                f"Omega22 non monotonic for the following parameters: q = {q}, chi1 = {chi1}, chi2 = {chi2}"
            )
            if plots == True:
                label = f"q{np.round(q,2)}_s{np.round(chi1,3)}_s{np.round(chi2,3)}"
                plt_dir = "./plots_checks_omega22"
                check_directory_exists_and_if_not_mkdir(plt_dir)
                plt.plot(t_h, omega22)
                plt.axvline(t_merger, ls="--")
                plt.xlim(t_merger - 30, t_merger + 30)
                plt.xlabel("$t/M$")
                plt.ylabel("$\Omega_{22}$")
                plt.title(f"{label}")
                plt.savefig(f"{plt_dir}/check_om22_{label}.png", dpi=150)
                plt.close()
            return (q, chi1, chi2)
    except IndexError:
        print(
            f"Error for the following parameters: q = {q}, chi1 = {chi1}, chi2 = {chi2}"
        )


def process_one_case_h(input):
    q, chi1, chi2, plots = input
    (
        q_bad_h,
        chi1_bad_h,
        chi2_bad_h,
    ) = h22_monotonicity(q, chi1, chi2, plots)
    return np.array([q_bad_h, chi1_bad_h, chi2_bad_h])


def process_one_case_om(input):
    q, chi1, chi2, plots = input
    (
        q_bad_h,
        chi1_bad_h,
        chi2_bad_h,
    ) = omega22_monotonicity(q, chi1, chi2, plots)
    return np.array([q_bad_h, chi1_bad_h, chi2_bad_h])


if __name__ == "__main__":

    p = argparse.ArgumentParser(description="Check monotonicity across parameter space")
    p.add_argument("--points", type=int, help="Number of points", default=10000)
    p.add_argument("--q-min", type=float, help="Lower limit on q", default="1.0")
    p.add_argument("--q-max", type=float, help="Upper limit on q", default="100.0")
    p.add_argument(
        "--chi-max", type=float, help="Upper limit on |chi|", default="0.995"
    )
    p.add_argument(
        "--name",
        type=str,
        help="Name of the output file",
        default="monotonicity_test",
    )
    p.add_argument(
        "--quantity",
        type=str,
        help="Quanitiy of which to check monotonicity",
        default="amplitude",
        choices=["amplitude", "frequency"],
    )
    p.add_argument(
        "--plots", action="store_true", help="Make diagnostic plots", default=True
    )
    p.add_argument("--n-cpu", type=int, help="Number of cores to use", default=64)

    args = p.parse_args()

    qarr, chi1arr, chi2arr = parameters_random_fast(
        args.points,
        args.q_min,
        args.q_max,
        -args.chi_max,
        args.chi_max,
        -args.chi_max,
        args.chi_max,
    )

    lst = [(a, b, c, args.plots) for a, b, c in zip(qarr, chi1arr, chi2arr)]
    pool = Pool(args.n_cpu)

    if args.quantity == "amplitude":
        all_means = pool.map(process_one_case_h, lst)
    if args.quantity == "frequency":
        all_means = pool.map(process_one_case_om, lst)
    # """
    all_means = np.array(all_means)

    np.savetxt(f"{args.name}_{args.quantity}.dat", all_means)

    ### Plots ###

    if args.plots:

        plt_dir = f"./plots_monotonicity"
        check_directory_exists_and_if_not_mkdir(plt_dir)

        res_path = f"{args.name}_{args.quantity}.dat"
        res = np.loadtxt(res_path)

        q = res[:, 0]
        idx = np.where(q > 0.0)[0]

        q = q[idx]
        chi1 = res[:, 1][idx]
        chi2 = res[:, 2][idx]

        nu = q / (1 + q) ** 2
        m1 = q / (1 + q)
        m2 = 1 / (1 + q)
        ap = m1 * chi1 + m2 * chi2
        am = m1 * chi1 - m2 * chi2

        if len(q) > 0:
            # Cases where |h22| is not monotonic across parameter space - (q, chieff)
            plt.scatter(q, ap)
            plt.ylabel("$\chi_{\mathrm{eff}}$")
            plt.xlabel("$q$")
            if args.quantity == "amplitude":
                plt.title("Cases where $|h_{22}|$ is not monotonic")
                plt.savefig(
                    f"{plt_dir}/monotonicity_h22.png", bbox_inches="tight", dpi=300
                )
            if args.quantity == "frequency":
                plt.title("Cases where $\Omega_{22}$ is not monotonic")
                plt.savefig(
                    f"{plt_dir}/monotonicity_om22.png", bbox_inches="tight", dpi=300
                )
            plt.close()
        else:
            print("Test passed for all cases")
