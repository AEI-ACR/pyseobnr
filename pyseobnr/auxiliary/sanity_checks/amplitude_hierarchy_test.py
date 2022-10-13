import argparse

import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils import check_directory_exists_and_if_not_mkdir
from pathos.multiprocessing import ProcessingPool as Pool


from pyseobnr.generate_waveform import generate_modes_opt
from pyseobnr.auxiliary.sanity_checks.parameters import parameters_random_fast
from pyseobnr.auxiliary.sanity_checks.single_waveform_tests import amplitude_hierarchy_test


def amplitude_hierarchy(q: float, chi1: float, chi2: float, plots: bool = True):

    """
    Generate a SEOBNRv5 waveform with parameters (q, chi1, chi2) and run
    amplitude_hierarchy_test to check that the amplitude of the subdominant
    modes is less than the amplitude of the (2,2) mode until the peak of the (2,2) modes

    Return (0,0,0) if the test passed, (q, chi1, chi2) otherwise
    """

    omega0 = 0.015

    try:

        _, _, model = generate_modes_opt(q, chi1, chi2, omega0,debug=True)
        t_h = model.t
        hlms_dict = model.waveform_modes
        habs22 = np.abs(model.waveform_modes["2,2"])
        t_merger = t_h[np.argmax(habs22)]

        try:
            amplitude_hierarchy_test(t_h, hlms_dict)
            return (0, 0, 0)
        except AssertionError:
            print(
                f"Amplitude of one subdominant mode is higher than the (2,2) mode the following parameters: q = {q}, chi1 = {chi1}, chi2 = {chi2}"
            )
            if plots:
                label = f"q{np.round(q,2)}_s{np.round(chi1,3)}_s{np.round(chi2,3)}"
                plt_dir = "./plots_hierarchy"
                check_directory_exists_and_if_not_mkdir(plt_dir)

                for key in model.modes_list:
                    plt.plot(
                        t_h, np.abs(hlms_dict[f"{key[0]},{key[1]}"]), label=f"[{key}]"
                    )

                plt.xlabel("t/M")
                plt.legend(loc="best")
                plt.axvline(t_merger, ls="--")
                plt.xlim(t_merger - 30, t_merger + 30)
                plt.xlabel("$t/M$")
                plt.ylabel("$|h_{\ell m}|$")
                plt.title(f"{label}")
                plt.savefig(f"{plt_dir}/check_amplitude_hierarchy_{label}.png", dpi=150)
                plt.close()
            return (q, chi1, chi2)
    except IndexError:
        print(
            f"Error for the following parameters: q = {q}, chi1 = {chi1}, chi2 = {chi2}"
        )


def process_one_case(input):
    q, chi1, chi2, plots = input
    (
        q_bad,
        chi1_bad,
        chi2_bad,
    ) = amplitude_hierarchy(q, chi1, chi2, plots)
    return np.array([q_bad, chi1_bad, chi2_bad])


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="Check amplitude hierarchy across parameter space"
    )
    p.add_argument("--points", type=int, help="Number of points", default="5000")
    p.add_argument("--q-max", type=float, help="Upper limit on q", default="100.0")
    p.add_argument(
        "--chi-max", type=float, help="Upper limit on |chi|", default="0.995"
    )
    p.add_argument(
        "--name",
        type=str,
        help="Name of the output file",
        default="amplitude_hierarchy.dat",
    )
    p.add_argument(
        "--plots", action="store_true", help="Make diagnostic plots", default=True
    )

    p.add_argument("--n-cpu", type=int, help="Number of cores to use", default=64)
    args = p.parse_args()

    qarr, chi1arr, chi2arr = parameters_random_fast(
        args.points,
        1.0,
        args.q_max,
        -args.chi_max,
        args.chi_max,
        -args.chi_max,
        args.chi_max,
    )

    lst = [(a, b, c, args.plots) for a, b, c in zip(qarr, chi1arr, chi2arr)]

    pool = Pool(args.n_cpu)

    all_means = pool.map(process_one_case, lst)

    all_means = np.array(all_means)
    np.savetxt(args.name, all_means)

    ### Plots ###

    if args.plots:

        plt_dir = "./plots_hierarchy"
        check_directory_exists_and_if_not_mkdir(plt_dir)

        res_path = args.name
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
            # Cases where modes amplitude hierarchy is not satisfied across parameter space - (q, chieff)
            plt.scatter(q, ap)
            plt.ylabel("$\chi_{\mathrm{eff}}$")
            plt.xlabel("$q$")

            plt.title("Cases where modes amplitude_hierarchy is not satisfied")
            plt.savefig(
                f"{plt_dir}/amplitude_hierarchy.png", bbox_inches="tight", dpi=300
            )
            plt.close()
        else:
            print("Test passed for all cases")
