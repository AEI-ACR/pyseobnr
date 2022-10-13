import argparse
import importlib
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils import check_directory_exists_and_if_not_mkdir
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.interpolate import CubicSpline

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../calibration")
)
from metrics import *
from models import SEOBNRv5PHM


sys.path.append(os.environ['EOB_DEVEL_PATH']+'/calibration/calibration_parameter_fits/model_0509/')
sys.path.append(os.environ['EOB_DEVEL_PATH']+'/auxiliary/sanity_checks/')

from parameters import parameters_random_fast_prec
from single_waveform_tests import amplitude_hierarchy_test

from SEOBNRv5PHM_wrapper_0509 import  SEOBNRv5PHMWrapper
import lal
from dataclasses import dataclass

v5wrap = SEOBNRv5PHMWrapper()


@dataclass
class WaveformParams:
    m1: float
    m2: float
    s1x: float
    s1y: float
    s1z: float
    s2x: float
    s2y: float
    s2z: float
    f_min: float
    f_ref: float
    delta_t: float
    approx: str = "SEOBNRv5PHM"
    augm_option: int = 4
    alpha: float = 0.

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

    f_ref = omega0/(np.pi * mt * lal.MTSUN_SI)
    f_min = omega_start/(np.pi * mt * lal.MTSUN_SI)

    m1 = q/(1.+q)*mt
    m2 = 1./(1.+q)*mt

    s1x, s1y, s1z = chi1x, chi1y, chi1z
    s2x, s2y, s2z = chi2x, chi2y, chi2z


    p = WaveformParams(m1 = m1, m2 = m2, s1x = s1x, s1y = s1y, s1z = s1z,
                       s2x = s2x, s2y = s2y, s2z = s2z, f_min = f_min, f_ref = f_ref, delta_t = 1./16384. )


    try:

        ell_max = 5

        time, modes, model = v5wrap.get_EOB_modes(p,
                                                        ell_max=ell_max)

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
    args = p.parse_args()


    qarr, chi1x_arr, chi1y_arr, chi1z_arr, chi2x_arr, chi2y_arr, chi2z_arr = parameters_random_fast_prec(
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
        check_directory_exists_and_if_not_mkdir(plt_dir)

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
