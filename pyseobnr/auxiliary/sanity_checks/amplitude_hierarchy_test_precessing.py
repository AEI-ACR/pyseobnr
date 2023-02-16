import argparse

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import importlib
import os,sys

from pathos.multiprocessing import ProcessingPool as Pool

from pyseobnr.generate_waveform import generate_modes_opt
from pyseobnr.auxiliary.sanity_checks.parameters import parameters_random_fast
from pyseobnr.auxiliary.sanity_checks.single_waveform_tests import amplitude_hierarchy_test



def amplitude_hierarchy(q: float, chi1x: float,  chi1y: float,  chi1z: float,  chi2x: float,  chi2y: float,  chi2z: float,  plots: bool = True):

    """
    Generate a SEOBNRv5 waveform with parameters (q, chi1, chi2) and run
    amplitude_hierarchy_test to check that the amplitude of the subdominant
    modes is less than the amplitude of the (2,2) mode until the peak of the (2,2) modes

    Return (0,0,0) if the test passed, (q, chi1, chi2) otherwise
    """

    omega0 = 0.015
    omega_start = omega0
    mt = 60.
    delta_t = 1./8192.

    chi_1 = [chi1x, chi1y, chi1z]
    chi_2 = [chi2x, chi2y, chi2z]


    try:

        ell_max = 5
        settings = {'ell_max':ell_max,'beta_approx':None,'M':mt,"dt":delta_t,"return_coprec":True,
        "postadiabatic": True,
        "postadiabatic_type": "analytic",
        "initial_conditions":"adiabatic",
        "initial_conditions_postadiabatic_type":"analytic"}
        _, _, model = generate_modes_opt(q,chi_1,chi_2,omega0,approximant='SEOBNRv5PHM',
                                   debug=True,settings=settings)

        t_h = model.t
        hlms_cop = model.coprecessing_modes

        # Specify the modes so that the negative m modes are not included
        modes_to_test = ["2,2","2,1","3,3","3,2","4,4","4,3","5,5"]
        hlms_dict ={}
        for lm in modes_to_test:
            hlms_dict[lm] = hlms_cop[lm]

        habs22 = np.abs(hlms_cop["2,2"])
        t_merger = t_h[np.argmax(habs22)]

        try:
            amplitude_hierarchy_test(t_h, hlms_dict)
            return (0, 0, 0, 0, 0, 0, 0)
        except AssertionError:
            print(
                f"Amplitude of one subdominant mode is higher than the (2,2) mode the following parameters: q = {q}, chi1x = {chi1x}, chi1y = {chi1y}, chi1z = {chi1z}, chi2x = {chi2x}, chi2y = {chi2y}, chi2z = {chi2z}"
            )
            if plots:
                label = f"q{np.round(q,2)}__s{np.round(chi1x,3)}_{np.round(chi1y,3)}_{np.round(chi1z,3)}__s{np.round(chi2x,3)}_{np.round(chi2y,3)}_{np.round(chi2z,3)}"
                plt_dir = "./plots_hierarchy"
                Path(plt_dir).mkdir(parents=True, exist_ok=True)

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
            return (q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z)
    except IndexError:
        print(
            f"Error for the following parameters: q = {q}, chi1x = {chi1x}, chi1y = {chi1y}, chi1z = {chi1z}, chi2x = {chi2x}, chi2y = {chi2y}, chi2z = {chi2z}"
        )


def process_one_case_prec(input):
    q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, plots = input

    (
        q_bad,
        chi1x_bad,
        chi1y_bad,
        chi1z_bad,
        chi2x_bad,
        chi2y_bad,
        chi2z_bad
    ) = amplitude_hierarchy(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, plots)
    return np.array([q_bad, chi1x_bad, chi1y_bad, chi1z_bad, chi2x_bad, chi2y_bad, chi2z_bad])

def parameters_random_fast_prec(
    seed: int,
    N: int,
    qmin: float,
    qmax: float,
    a1min: float,
    a1max: float,
    a2min: float,
    a2max: float,
):
    """
    Generate random parameters for q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z

    Parameters
    ----------
    N:
        Number of parameters to generate
    qmin, qmax:
        Bounds on the q parameter
    a1min, a1max, a2min, a2max:
        Bounds on the a parameter

    Return parameters of the waveforms
    """


    np.random.seed(seed)
    q = np.random.uniform(qmin, qmax, N)

    np.random.seed(seed+1)
    a1 = np.random.uniform(a1min, a1max, N)
    np.random.seed(seed+2)
    a2 = np.random.uniform(a2min, a2max, N)


    np.random.seed(seed+3)
    theta1 = np.random.uniform(0,np.pi,N)
    np.random.seed(seed+4)
    theta2 = np.random.uniform(0,np.pi,N)

    np.random.seed(seed+5)
    phi1 = np.random.uniform(0,2*np.pi,N)
    np.random.seed(seed+6)
    phi2 = np.random.uniform(0,2*np.pi,N)

    chi1x = a1*np.sin(theta1)*np.cos(phi1)
    chi1y = a1*np.sin(theta1)*np.sin(phi1)
    chi1z = a1*np.cos(theta1)

    chi2x = a2*np.sin(theta2)*np.cos(phi2)
    chi2y = a2*np.sin(theta2)*np.sin(phi2)
    chi2z = a2*np.cos(theta2)

    return (
        np.array(q),
        np.array(chi1x),
        np.array(chi1y),
        np.array(chi1z),
        np.array(chi2x),
        np.array(chi2y),
        np.array(chi2z),
    )


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="Check amplitude hierarchy across parameter space"
    )
    p.add_argument("--points", type=int, help="Number of points", default="5000")
    p.add_argument("--q-max", type=float, help="Upper limit on q", default="100.0")
    p.add_argument(
        "--a-max", type=float, help="Upper limit on |chi|", default="0.995"
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
    p.add_argument(
        "--wrapper-path", type=str, help="The path to the wrapper, including name"
    )
    p.add_argument("--n-cpu", type=int, help="Number of cores to use", default=64)
    p.add_argument("--seed", type=int, help="Seed for random generation use", default=150914)
    args = p.parse_args()

    qarr, chi1x_arr, chi1y_arr, chi1z_arr, chi2x_arr, chi2y_arr, chi2z_arr = parameters_random_fast_prec(
        args.seed,
        args.points,
        1.0,
        args.q_max,
        0.,
        args.a_max,
        0.,
        args.a_max,
    )

    lst = [(a, b, c, d, e, f, g, args.plots) for a, b, c, d, e, f, g in zip(qarr, chi1x_arr, chi1y_arr, chi1z_arr, chi2x_arr, chi2y_arr, chi2z_arr)]
    module_name = os.path.basename(args.wrapper_path).split(".")[0]
    # The following hacks things so that the wrapper can be loaded from the specified file
    file_path = args.wrapper_path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    globals().update(
        {n: getattr(module, n) for n in module.__all__}
        if hasattr(module, "__all__")
        else {k: v for (k, v) in module.__dict__.items() if not k.startswith("_")}
    )

    pool = Pool(args.n_cpu)

    all_means = pool.map(process_one_case_prec, lst)
    all_means = np.array(all_means)

    np.savetxt(args.name, all_means)

    ### Plots ###

    if args.plots:

        plt_dir = "./plots_hierarchy"
        Path(plt_dir).mkdir(parents=True, exist_ok=True)

        res_path = args.name
        res = np.loadtxt(res_path)

        q = res[:, 0]
        idx = np.where(q > 0.0)[0]

        q = q[idx]
        chi1_x = res[:, 1][idx]
        chi1_y = res[:, 2][idx]
        chi1_z = res[:, 3][idx]

        chi2_x = res[:, 4][idx]
        chi2_y = res[:, 5][idx]
        chi2_z = res[:, 6][idx]


        nu = q / (1. + q) ** 2
        m1 = q / (1. + q)
        m2 = 1. / (1. + q)
        ap = m1 * chi1_z + m2 * chi2_z
        am = m1 * chi1_z - m2 * chi2_z

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
