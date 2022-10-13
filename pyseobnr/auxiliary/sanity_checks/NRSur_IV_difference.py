import argparse
import importlib
import os
import sys

import gwsurrogate
import numpy as np
from bilby.core.utils import check_directory_exists_and_if_not_mkdir
from scipy.interpolate import InterpolatedUnivariateSpline
from pathos.multiprocessing import ProcessingPool as Pool

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../calibration")
)
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../MR_pipeline")
)
sys.path.append(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../../toys/hamiltonian_prototype"
    )
)
from metrics import *
from models import *
from IV_fits import *

from parameters import parameters_random, parameters_random_2D


mode_list_v5HM = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5)]
mode_list_v4HM = [(2, 2), (2, 1), (3, 3), (4, 4), (5, 5)]


sur_3dq8 = gwsurrogate.LoadSurrogate("NRHybSur3dq8")
sur_2dq15 = gwsurrogate.LoadSurrogate("NRHybSur2dq15")


def NRSur_IV(modes, t, ell, m):
    if (
        (ell == 2 and m == 2)
        or (ell == 3 and m == 3)
        or (ell == 2 and m == 1)
        or (ell == 4 and m == 4)
        or (ell == 3 and m == 2)
        or (ell == 4 and m == 3)
    ):
        tpeak = t[np.argmax(abs(modes["2,2"]))]
    elif ell == 5 and m == 5:
        tpeak = t[np.argmax(abs(modes["2,2"]))] - 10
    else:
        return NotImplementedError

    hlm_spline = InterpolatedUnivariateSpline(t, abs(modes[f"{ell},{m}"]))
    hlmdot_spline = hlm_spline.derivative()
    hlmdotdot_spline = hlm_spline.derivative(2)
    philm_spline = InterpolatedUnivariateSpline(
        t, (np.unwrap(np.angle(modes[f"{ell},{m}"])))
    )
    omegalm_spline = philm_spline.derivative()
    omegalmdot_spline = philm_spline.derivative(2)

    hlm = hlm_spline(tpeak)
    hlm_dot = hlmdot_spline(tpeak)
    hlm_dotdot = hlmdotdot_spline(tpeak)
    omegalm = omegalm_spline(tpeak)
    omegalm_dot = omegalmdot_spline(tpeak)

    return hlm, hlm_dot, hlm_dotdot, omegalm, omegalm_dot


def IV_difference_NRHybSur(
    q: float,
    chi1: float,
    chi2: float,
    model_name: str,
):
    if model_name == "NRHybSur2dq15":
        target_model = NRHybSur2dq15Model(
            q,
            chi1,
            chi2,
            0.015,
        )
        sur = sur_2dq15
        mode_list = mode_list_v4HM
    elif model_name == "NRHybSur3dq8":
        target_model = NRHybSur3dq8Model(
            q,
            chi1,
            chi2,
            0.015,
        )
        sur = sur_3dq8
        mode_list = mode_list_v5HM
    target_model(sur)

    m1 = q / (1 + q)
    m2 = 1 - m1
    nu = q/(1 + q)**2
    fits_IV = InputValueFits(m1, m2, [0.0, 0.0, chi1], [0.0, 0.0, chi2])

    hlm_diff = []
    hlm_dot_diff = []
    hlm_dotdot_diff = []
    omegalm_diff = []
    omegalm_dot_diff = []
    for mode in mode_list:
        ell, m = mode
        hlm, hlm_dot, hlm_dotdot, omegalm, omegalm_dot = NRSur_IV(
            target_model.waveform_modes, target_model.t, ell=ell, m=m
        )
        hlm_diff.append(hlm/nu - fits_IV.habs()[ell, m])
        hlm_dot_diff.append(hlm_dot/nu - fits_IV.hdot()[ell, m])
        hlm_dotdot_diff.append(hlm_dotdot/nu - fits_IV.hdotdot()[ell, m])
        omegalm_diff.append(omegalm - fits_IV.omega()[ell, m])
        omegalm_dot_diff.append(omegalm_dot - fits_IV.omegadot()[ell, m])

    return (
        target_model.q,
        target_model.chi_1,
        target_model.chi_2,
        np.array(hlm_diff),
        np.array(hlm_dot_diff),
        np.array(hlm_dotdot_diff),
        np.array(omegalm_diff),
        np.array(omegalm_dot_diff),
    )


def process_one_case(input):
    q, chi1, chi2, model_name = input
    (
        q,
        chi1,
        chi2,
        hlm_diff,
        hlm_dot_diff,
        hlm_dotdot_diff,
        omegalm_diff,
        omegalm_dot_diff,
    ) = IV_difference_NRHybSur(q, chi1, chi2, model_name)
    return hlm_diff, hlm_dot_diff, hlm_dotdot_diff, omegalm_diff, omegalm_dot_diff


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compute input values difference against NRSur"
    )
    p.add_argument("--points", type=int, help="Number of points", default="5000")
    p.add_argument("--chi-max", type=float, help="Maximum spin", default="0.8")
    p.add_argument("--q-max", type=float, help="Maximum mass-ratio", default="8.0")
    p.add_argument(
        "--name", type=str, help="Name of the output file", default="iv_difference"
    )
    p.add_argument(
        "--plots", action="store_true", help="Make diagnostic plots", default=True
    )
    p.add_argument(
        "--model-name",
        type=str,
        help="Name of the surrogate model",
        default="NRHybSur3dq8",
    )
    p.add_argument(
        "--plots-only",
        action="store_true",
        help="Only generate plots, don't recompute things",
    )
    p.add_argument("--n-cpu", type=int, help="Number of cores to use", default=64)
    p.add_argument(
        "--include-all",
        action="store_true",
        help="Include odd m modes close to equal mass",
        default=False,
    )
    args = p.parse_args()

    if args.model_name == "NRHybSur2dq15":
        mode_list = mode_list_v4HM
    elif args.model_name == "NRHybSur3dq8":
        mode_list = mode_list_v5HM

    if not args.plots_only:

        if args.model_name == "NRHybSur2dq15":
            qarr, chi1arr = parameters_random_2D(
                args.points, 1.0, args.q_max, -args.chi_max, args.chi_max
            )
            lst = [(q, chi1, 0.0, args.model_name) for q, chi1 in zip(qarr, chi1arr)]
            chi2arr = np.zeros(args.points)
        elif args.model_name == "NRHybSur3dq8":
            qarr, chi1arr, chi2arr = parameters_random(
                args.points,
                1.0,
                args.q_max,
                -args.chi_max,
                args.chi_max,
                -args.chi_max,
                args.chi_max,
            )
            lst = [
                (q, chi1, chi2, args.model_name)
                for q, chi1, chi2 in zip(qarr, chi1arr, chi2arr)
            ]

        pool = Pool(args.n_cpu)
        all_means = pool.map(process_one_case, lst)

        all_means = np.array(all_means)
        # This is an array with shape n cases x m quantities x p modes
        hlm_diffs = {}
        hlm_dot_diffs = {}
        hlm_dotdot_diffs = {}
        omegalm_diffs = {}
        omegalm_dot_diffs = {}

        for mode in mode_list:
            hlm_diffs[mode] = []
            hlm_dot_diffs[mode] = []
            hlm_dotdot_diffs[mode] = []
            omegalm_diffs[mode] = []
            omegalm_dot_diffs[mode] = []
        for i in range(len(all_means)):
            [
                hlm_for_case,
                hlmdot_for_case,
                hlmddot_for_case,
                omega_for_case,
                omegadot_for_case,
            ] = all_means[i]
            for j, mode in enumerate(mode_list):
                hlm_diffs[mode].append(hlm_for_case[j])
                hlm_dot_diffs[mode].append(hlmdot_for_case[j])
                hlm_dotdot_diffs[mode].append(hlmddot_for_case[j])
                omegalm_diffs[mode].append(omega_for_case[j])
                omegalm_dot_diffs[mode].append(omegadot_for_case[j])

        for mode in mode_list:
            ell, m = mode
            np.savetxt(f"{args.name}_{args.model_name}_h{ell}{m}.dat", hlm_diffs[mode])
            np.savetxt(
                f"{args.name}_{args.model_name}_hdot{ell}{m}.dat", hlm_dot_diffs[mode]
            )
            np.savetxt(
                f"{args.name}_{args.model_name}_hddot{ell}{m}.dat",
                hlm_dotdot_diffs[mode],
            )
            np.savetxt(
                f"{args.name}_{args.model_name}_omega{ell}{m}.dat", omegalm_diffs[mode]
            )
            np.savetxt(
                f"{args.name}_{args.model_name}_omegadot{ell}{m}.dat",
                omegalm_dot_diffs[mode],
            )

        np.savetxt(f"parameters_{args.model_name}.dat", np.c_[qarr, chi1arr, chi2arr])

    ### Plots ###

    if args.plots:

        import matplotlib
        import matplotlib.pyplot as plt

        plt_dir = "./plots"
        check_directory_exists_and_if_not_mkdir(plt_dir)

        res_path = args.name
        params = np.genfromtxt(f"parameters_{args.model_name}.dat")
        q = params[:, 0]
        chi1 = params[:, 1]
        chi2 = params[:, 2]

        nu = q / (1 + q) ** 2
        m1 = q / (1 + q)
        m2 = 1 / (1 + q)
        ap = m1 * chi1 + m2 * chi2
        am = m1 * chi1 - m2 * chi2
        for mode in mode_list:
            ell, m = mode
            h_diff = np.loadtxt(f"{args.name}_{args.model_name}_h{ell}{m}.dat")
            hd_diff = np.loadtxt(f"{args.name}_{args.model_name}_hdot{ell}{m}.dat")
            hdd_diff = np.loadtxt(f"{args.name}_{args.model_name}_hddot{ell}{m}.dat")
            om_diff = np.loadtxt(f"{args.name}_{args.model_name}_omega{ell}{m}.dat")
            omd_diff = np.loadtxt(f"{args.name}_{args.model_name}_omegadot{ell}{m}.dat")

            if not args.include_all:
                if m % 2 == 1:
                    idx_notEMES = np.where((q > 1.01) | (np.abs(chi1 - chi2) > 0.01))[0]
                    q_pl = q[idx_notEMES]
                    ap_pl = ap[idx_notEMES]
                    h_diff = h_diff[idx_notEMES]
                    hd_diff = hd_diff[idx_notEMES]
                    hdd_diff = hdd_diff[idx_notEMES]
                    om_diff = om_diff[idx_notEMES]
                    omd_diff = omd_diff[idx_notEMES]

                else:
                    q_pl = q
                    ap_pl = ap
            else:
                q_pl = q
                ap_pl = ap

            # |h|

            # Histogram
            plt.hist(
                h_diff,
                bins=np.logspace(start=np.log10(0.00001), stop=np.log10(1.0), num=50),
                alpha=0.4,
                label=args.model_name + " - SEOBNRV5HM",
            )
            rmse = np.sqrt(np.sum(h_diff ** 2) / len(h_diff))
            plt.legend(loc="best")
            plt.gca().set_xscale("log")
            plt.xlabel(f"$\Delta |h{ell}{m}|$")
            plt.title("RMSE = " + f"{np.round(rmse,6)}")
            plt.savefig(
                f"{plt_dir}/hist_{args.model_name}_h{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # Difference across parameter space
            h_diff_s, q_s, ap_s = map(list, zip(*sorted(zip(h_diff, q_pl, ap_pl))))
            plt.scatter(
                q_s,
                ap_s,
                c=h_diff_s,
                linewidths=1,
                # norm=matplotlib.colors.LogNorm()
            )
            plt.ylabel("$\chi_{\mathrm{eff}}$")
            plt.xlabel("$q$")
            cbar = plt.colorbar()
            cbar.set_label(f"$\Delta |h{ell}{m}|$")
            plt.savefig(
                f"{plt_dir}/scatter_{args.model_name}_h{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # |\dot h|

            # Histogram
            plt.hist(
                hd_diff,
                bins=np.logspace(start=np.log10(0.00001), stop=np.log10(1.0), num=50),
                alpha=0.4,
                label=args.model_name + " - SEOBNRV5HM",
            )
            rmse = np.sqrt(np.sum(hd_diff ** 2) / len(hd_diff))
            plt.legend(loc="best")
            plt.gca().set_xscale("log")
            plt.xlabel(f"$\Delta d|h{ell}{m}|/dt$")
            plt.title("RMSE = " + f"{np.round(rmse,6)}")
            plt.savefig(
                f"{plt_dir}/hist_{args.model_name}_hdot{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # Difference across parameter space
            hd_diff_s, q_s, ap_s = map(list, zip(*sorted(zip(hd_diff, q_pl, ap_pl))))
            plt.scatter(
                q_s,
                ap_s,
                c=hd_diff_s,
                linewidths=1,
                # norm=matplotlib.colors.LogNorm()
            )
            plt.ylabel("$\chi_{\mathrm{eff}}$")
            plt.xlabel("$q$")
            cbar = plt.colorbar()
            cbar.set_label(f"$\Delta d|h{ell}{m}|/dt$")
            plt.savefig(
                f"{plt_dir}/scatter_{args.model_name}_hdot{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # |\ddot h|

            # Histogram
            plt.hist(
                hdd_diff,
                bins=np.logspace(start=np.log10(0.00001), stop=np.log10(1.0), num=50),
                alpha=0.4,
                label=args.model_name + " - SEOBNRV5HM",
            )
            rmse = np.sqrt(np.sum(hdd_diff ** 2) / len(hdd_diff))
            plt.legend(loc="best")
            plt.gca().set_xscale("log")
            plt.xlabel(f"$\Delta d^2|h{ell}{m}|/dt^2$")
            plt.title("RMSE = " + f"{np.round(rmse,6)}")
            plt.savefig(
                f"{plt_dir}/hist_{args.model_name}_hddot{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # Difference across parameter space
            hdd_diff_s, q_s, ap_s = map(list, zip(*sorted(zip(hdd_diff, q_pl, ap_pl))))
            plt.scatter(
                q_s,
                ap_s,
                c=hdd_diff_s,
                linewidths=1,
                # norm=matplotlib.colors.LogNorm()
            )
            plt.ylabel("$\chi_{\mathrm{eff}}$")
            plt.xlabel("$q$")
            cbar = plt.colorbar()
            cbar.set_label(f"$\Delta d^2|h{ell}{m}|/dt^2$")
            plt.savefig(
                f"{plt_dir}/scatter_{args.model_name}_hddot{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # \omega

            # Histogram
            plt.hist(
                om_diff,
                bins=np.logspace(start=np.log10(0.00001), stop=np.log10(1.0), num=50),
                alpha=0.4,
                label=args.model_name + " - SEOBNRV5HM",
            )
            rmse = np.sqrt(np.sum(om_diff ** 2) / len(om_diff))
            plt.legend(loc="best")
            plt.gca().set_xscale("log")
            plt.xlabel(f"$\Delta \omega{ell}{m}$")
            plt.title("RMSE = " + f"{np.round(rmse,6)}")
            plt.savefig(
                f"{plt_dir}/hist_{args.model_name}_omega{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # Difference across parameter space
            om_diff_s, q_s, ap_s = map(list, zip(*sorted(zip(om_diff, q_pl, ap_pl))))
            plt.scatter(
                q_s,
                ap_s,
                c=om_diff_s,
                linewidths=1,
                # norm=matplotlib.colors.LogNorm()
            )
            plt.ylabel("$\chi_{\mathrm{eff}}$")
            plt.xlabel("$q$")
            cbar = plt.colorbar()
            cbar.set_label(f"$\Delta \omega{ell}{m}$")
            plt.savefig(
                f"{plt_dir}/scatter_{args.model_name}_omega{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # \dot \omega

            # Histogram
            plt.hist(
                omd_diff,
                bins=np.logspace(start=np.log10(0.00001), stop=np.log10(1.0), num=50),
                alpha=0.4,
                label=args.model_name + " - SEOBNRV5HM",
            )
            rmse = np.sqrt(np.sum(omd_diff ** 2) / len(omd_diff))
            plt.legend(loc="best")
            plt.gca().set_xscale("log")
            plt.xlabel(f"$\Delta d\omega{ell}{m}/dt$")
            plt.title("RMSE = " + f"{np.round(rmse,6)}")
            plt.savefig(
                f"{plt_dir}/hist_{args.model_name}_omegadot{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # Difference across parameter space
            omd_diff_s, q_s, ap_s = map(list, zip(*sorted(zip(omd_diff, q_pl, ap_pl))))
            plt.scatter(
                q_s,
                ap_s,
                c=omd_diff_s,
                linewidths=1,
                # norm=matplotlib.colors.LogNorm()
            )
            plt.ylabel("$\chi_{\mathrm{eff}}$")
            plt.xlabel("$q$")
            cbar = plt.colorbar()
            cbar.set_label(f"$\Delta d\omega{ell}{m}/dt$")
            plt.savefig(
                f"{plt_dir}/scatter_{args.model_name}_omegadot{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()
