import argparse
import importlib
import os
import sys

import numpy as np
from bilby.core.utils import check_directory_exists_and_if_not_mkdir
from pathos.multiprocessing import ProcessingPool as Pool

from pyseobnr.auxiliary.external_models import *
from pyseobnr.auxiliary.sanity_checks.metrics import *
from pyseobnr.auxiliary.sanity_checks.parameters import parameters_random_fast
from pyseobnr.generate_waveform import generate_modes_opt

modes_v5HM = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5)]
modes_v4HM = [(2, 2), (2, 1), (3, 3), (4, 4), (5, 5)]

# Reproducible
seed = 150914

def mismatch_EOB(q, chi1, chi2, model_1_name, model_2_name):

    omega0 = 0.015

    if model_1_name == "SEOBNRv5HM":
        _,_,model_1 = generate_modes_opt(q,chi1,chi2,0.8*omega0,debug=True)

    elif model_1_name == "SEOBNRv4HM":
        model_1 = SEOBNRv4HM_LAL(q, chi1, chi2, 0.8*omega0)
        model_1()


    model_2 = model_2_name

    masses = np.arange(10, 310, 10)

    unf_settings = {"debug": True, "masses": masses}
    unf = UnfaithfulnessModeByModeLAL(settings=unf_settings)

    modes = [(2,2)]
    mms = []
    params = dict(
                q=q,
                chi1=np.array([0.0, 0.0, chi1]),
                chi2=np.array([0.0, 0.0, chi2]),
                omega0= omega0,
                dt=model_1.delta_T,
                df=1 / (len(model_1.t) * model_1.delta_T),
    )

    for mode in modes:
        ell, m = mode
        mm = unf(model_1, model_2, params=params, ell=ell, m=m)
        mms.append(mm)

    return (q, chi1, chi2, np.array(mms))


def process_one_case(input):
    q, chi1, chi2, model_1_name, model_2_name = input
    q, chi1, chi2, mm = mismatch_EOB(q, chi1, chi2, model_1_name, model_2_name)
    return mm


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute mismatch between EOB and Phenom models")
    p.add_argument("--points", type=int, help="Number of points", default="5000")
    p.add_argument("--chi-max", type=float, help="Maximum spin", default="0.995")
    p.add_argument("--q-min", type=float, help="Minimum mass-ratio", default="1.0")
    p.add_argument("--q-max", type=float, help="Maximum mass-ratio", default="100.0")
    p.add_argument(
        "--name",
        type=str,
        help="Name of the output file",
        default="statistics_EOB_comparable",
    )
    p.add_argument(
        "--plots", action="store_true", help="Make diagnostic plots", default=True
    )
    p.add_argument(
        "--model-1-name",
        type=str,
        help="Name of the first EOB model",
        default="SEOBNRv5HM",
    )
    p.add_argument(
        "--model-2-name",
        type=str,
        help="Name of the second Phenom model. Must be one of 'IMRPhenomT', 'IMRPhenomXAS'",
        default="IMRPhenomXAS",
    )
    p.add_argument("--n-cpu", type=int, help="Number of cores to use", default=64)
    p.add_argument(
        "--plots-only",
        action="store_true",
        help="Only generate plots, don't recompute things",
    )
    p.add_argument(
        "--include-all",
        action="store_true",
        help="Include odd m modes close to equal mass",
        default=False,
    )

    args = p.parse_args()

    qarr, chi1arr, chi2arr = parameters_random_fast(
        args.points,
        args.q_min,
        args.q_max,
        -args.chi_max,
        args.chi_max,
        -args.chi_max,
        args.chi_max,
        random_state=seed
    )
    lst = [
        (q, chi1, chi2, args.model_1_name, args.model_2_name)
        for q, chi1, chi2 in zip(qarr, chi1arr, chi2arr)
    ]

    pool = Pool(args.n_cpu)
    all_means = list(pool.map(process_one_case, lst))

    mode_list = [(2,2)]

    aall_means = np.array(all_means)
    # This is an array with shape n cases x p modes x m masses
    mismatches = {}
    for mode in mode_list:
        mismatches[mode] = []
    for i in range(len(all_means)):
        mm_for_case = all_means[i]
        for j, mode in enumerate(mode_list):
            mode_mismatch = mm_for_case[j]
            mismatches[mode].append(mode_mismatch)

    for mode in mode_list:
        ell, m = mode
        np.savetxt(
            f"{args.name}_{args.model_1_name}_{args.model_2_name}{ell}{m}.dat",
            mismatches[mode],
        )

    np.savetxt(
        f"parameters_{args.model_1_name}_{args.model_2_name}_{args.name}.dat",
        np.c_[qarr, chi1arr, chi2arr],
    )
    ### Plots ###

    if args.plots:

        import matplotlib
        import matplotlib.pyplot as plt

        plt_dir = "./plots"
        check_directory_exists_and_if_not_mkdir(plt_dir)

        res_path = args.name
        params = np.genfromtxt(
            f"parameters_{args.model_1_name}_{args.model_2_name}_{args.name}.dat"
        )
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
            mm_M = np.loadtxt(
                f"{args.name}_{args.model_1_name}_{args.model_2_name}{ell}{m}.dat"
            )
            if not args.include_all:
                if m % 2 == 1:
                    idx_notEMES = np.where((q > 1.01) | (np.abs(chi1 - chi2) > 0.01))[0]
                    q_pl = q[idx_notEMES]
                    ap_pl = ap[idx_notEMES]
                    mm_M = mm_M[idx_notEMES]
                else:
                    q_pl = q
                    ap_pl = ap
            else:
                q_pl = q
                ap_pl = ap

            M = np.arange(10, 310, 10)

            # maximum of mm_M across total mass for histogram
            mm = []
            for mp in mm_M:
                mm.append(np.max(mp))
            mm = np.array(mm)

            # Spaghetti plot
            for mp in mm_M:
                plt.plot(M, mp, color="C0", linewidth=0.5)
            plt.axhline(0.01, ls="--", color="red")
            plt.yscale("log")
            plt.xlabel("M")
            plt.ylabel("$\mathcal{M}$")
            plt.xlim(10, 300)
            plt.savefig(
                f"{plt_dir}/mm_spaghetti_{args.model_1_name}_{args.model_2_name}_{args.name}{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

        # Histogram
        plt.hist(
            mm,
            bins=np.logspace(start=np.log10(0.00001), stop=np.log10(1.0), num=50),
            alpha=0.4,
            label=args.model_1_name + " - " + args.model_2_name,
        )
        plt.axvline(np.median(mm), c="C0", ls="--")
        plt.legend(loc="best")
        plt.gca().set_xscale("log")
        plt.xlabel("$\mathcal{M}_{\mathrm{Max}}$")
        plt.title(
            "$\mathcal{M}_{\mathrm{median}} = $" + f"{np.round(np.median(mm),6)}"
        )
        plt.savefig(
            f"{plt_dir}/mm_hist_{args.model_1_name}_{args.model_2_name}_{args.name}{ell}{m}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        # CDF
        plt.hist(
            mm,
            bins=np.logspace(start=np.log10(0.00001), stop=np.log10(1.0), num=100),
            alpha=0.4,
            label=args.model_1_name + " - " + args.model_2_name,
            cumulative=True,
            density=True,
            histtype="step",
            lw=2,
        )
        plt.legend(loc="best")
        plt.gca().set_xscale("log")
        plt.xlabel("$\mathcal{M}_{\mathrm{Max}}$")
        plt.title(
            "$\mathcal{M}_{\mathrm{median}} = $" + f"{np.round(np.median(mm),6)}"
        )
        plt.grid(True, which="both", ls=":")
        plt.savefig(
            f"{plt_dir}/mm_cdf_{args.model_1_name}_{args.model_2_name}_{args.name}{ell}{m}.png",
            bbox_inches="tight",
            dpi=300,
        )

        plt.close()

        # Mismatch across parameter space
        mm_s, q_s, ap_s = map(list, zip(*sorted(zip(mm, q_pl, ap_pl))))
        plt.scatter(
            q_s, ap_s, c=mm_s, linewidths=1, norm=matplotlib.colors.LogNorm()
        )
        plt.ylabel("$\chi_{\mathrm{eff}}$")
        plt.xlabel("$q$")
        cbar = plt.colorbar()
        cbar.set_label("$\mathcal{M}$")
        plt.savefig(
            f"{plt_dir}/mm_scatter_{args.model_1_name}_{args.model_2_name}_{args.name}{ell}{m}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
