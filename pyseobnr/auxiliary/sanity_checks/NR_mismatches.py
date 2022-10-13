import argparse
import csv
import importlib
import os
import sys
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils import check_directory_exists_and_if_not_mkdir
from pathos.multiprocessing import ProcessingPool as Pool

from pyseobnr.auxiliary.sanity_checks.metrics  import UnfaithfulnessModeByModeLAL
from pyseobnr.auxiliary.external_models  import NRModel_SXS,SEOBNRv4HM_LAL
from pyseobnr.generate_waveform import generate_modes_opt

modes_v5HM = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (5, 5)]
modes_v4HM = [(2, 2), (2, 1), (3, 3), (4, 4), (5, 5)]

# modes_v5HM=[(2,2)]


def get_NR_paths(cases_file: str) -> List[str]:
    NR_path = []
    with open(cases_file, "r") as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            NR_path.append(row[5] + "/rhOverM_Asymptotic_GeometricUnits_CoM.h5")

    return NR_path


def single_mismatch(p: Tuple[str, str, bool]) -> Tuple:
    """Compute mode-by-mode mismatches between NR and the given EOB model

    Args:
        p (Tuple[str, str, bool]): The path to NR file, the type of model to use,
        whether to limit the mismatch computation to the merger-ringdown

    Returns:
        Tuple: Parameters and the mismatches, stored as a dict
    """
    path, model = p

    target_model = NRModel_SXS(path)
    q = target_model.q
    if q < 1:
        q = 1 / q
    if model == "SEOBNRv5HM":

        _, _, calib_model = generate_modes_opt(
            q, target_model.chi1[-1], target_model.chi2[-1], 0.7 * target_model.omega0,debug=True
        )
        modes = modes_v5HM
    elif model == "SEOBNRv4HM":
        calib_model = SEOBNRv4HM_LAL(
            q,
            target_model.chi1[-1],
            target_model.chi2[-1],
            0.7 * target_model.omega0,
        )
        calib_model()
        modes = modes_v4HM

    else:
        # print(f"Here, model={model}")
        calib_model = model

    masses = np.arange(10, 310, 10)
    unf_settings = {"debug": True, "masses": masses}
    unf = UnfaithfulnessModeByModeLAL(settings=unf_settings)

    mismatch = {}
    for mode in modes:
        mismatch[mode] = {}

    for mode in modes:
        ell, m = mode

        params = dict(
            q=target_model.q,
            chi1=np.array([0.0, 0.0, target_model.chi1[-1]]),
            chi2=np.array([0.0, 0.0, target_model.chi2[-1]]),
            omega0=0.7 * target_model.omega0,
            dt=target_model.delta_T,
            df=1 / (len(target_model.t) * target_model.delta_T),
        )
        #print(f"ell={ell},m={m}")
        mm = unf(target_model, calib_model, params=params, ell=ell, m=m)

        mismatch[mode] = mm

    q = np.round(target_model.q, 2)
    chi1 = np.round(target_model.chi1[-1], 3)
    chi2 = np.round(target_model.chi2[-1], 3)
    label = f"q_{q}_s{chi1}_s{chi2}"
    return label, q, chi1, chi2, mismatch


def mismatch_NR(
    NR_paths: List[str], n_cpu: int = 80, model: str = "SEOBNRv5HM"
) -> Tuple:
    """Compute mismatches between NR and a given model, in parallel for many
    cases.

    Args:
        NR_paths (List[str]): List of NR waveforms
        n_cpu (int, optional): Number of CPUs to use. Defaults to 80.
        model (str, optional): Model to compare to NR. Defaults to "SEOBNRv5HM".

    Returns:
        Tuple: Labels,parameters,mismatches. The mismatches are a dict, with
                with keys being the modes
    """

    mismatch = {}
    q_arr = []
    chi1_arr = []
    chi2_arr = []
    labels = []

    pool = Pool(n_cpu)
    lst = list(pool.map(single_mismatch, [(x, model) for x in NR_paths]))

    if model == "SEOBNRv5HM":
        modes = modes_v5HM
    elif model == "SEOBNRv4HM":
        modes = modes_v4HM
    else:
        modes = [(2, 2)]
    # Reshuffle the data into a convenient form
    # modes = [(2,2)]
    for mode in modes:
        mismatch[mode] = []
    for row in lst:
        label, q, chi1, chi2, mm = row
        q_arr.append(q)
        chi1_arr.append(chi1)
        chi2_arr.append(chi2)
        labels.append(label)
        for mode in modes:
            mismatch[mode].append(mm[mode])

    for mode in modes:
        mismatch[mode] = np.array(mismatch[mode])
    return (
        np.array(labels),
        np.array(q_arr),
        np.array(chi1_arr),
        np.array(chi2_arr),
        mismatch,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cases-file", type=str, help="File containing the NR case specifications"
    )

    p.add_argument(
        "--mismatch-file", type=str, help="File with mismatches, if precomputed"
    )
    p.add_argument(
        "--model",
        type=str,
        help="Which approximant to compare to NR",
        default="SEOBNRv5HM",
    )
    p.add_argument(
        "--plots", action="store_true", help="Make diagnostic plots", default=True
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

    if not args.plots_only:
        NR_paths = get_NR_paths(args.cases_file)

        # Get the mismatches
        labels, qs, chi1s, chi2s, mms = mismatch_NR(
            NR_paths,
            model=args.model,
            n_cpu=args.n_cpu,
        )

        # Save the parameters in a separate file, for convenience
        params = np.c_[qs, chi1s, chi2s]
        np.savetxt(f"parameters_mismatch_calc_{args.model}.dat", params)

        # Now save the mismatches, one file per mode
        for key in mms.keys():
            ell, m = key
            np.savetxt(f"mismatch_{args.model}{ell}{m}.dat", mms[key])

    if args.plots:
        params = np.genfromtxt(f"parameters_mismatch_calc_{args.model}.dat")
        q = params[:, 0]
        chi1 = params[:, 1]
        chi2 = params[:, 2]

        nu = q / (1 + q) ** 2
        m1 = q / (1 + q)
        m2 = 1 / (1 + q)
        ap = m1 * chi1 + m2 * chi2
        am = m1 * chi1 - m2 * chi2

        plt_dir = "./plots"
        check_directory_exists_and_if_not_mkdir(plt_dir)
        if args.model == "SEOBNRv5HM":
            mode_list = modes_v5HM
        elif args.model == "SEOBNRv4HM":
            mode_list = modes_v4HM
        else:

            mode_list = [(2, 2)]

        for mode in mode_list:
            ell, m = mode
            mm_M = np.loadtxt(f"mismatch_{args.model}{ell}{m}.dat")

            if not args.include_all:
                if m % 2:
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


            if True:
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
                    f"{plt_dir}/mm_spaghetti_{args.model}{ell}{m}.png",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.close()

            # Histogram
            plt.hist(
                mm,
                bins=np.logspace(start=np.log10(0.00001), stop=np.log10(1.0), num=50),
                alpha=0.4,
                label=args.model,
            )
            plt.axvline(np.median(mm), c="C0", ls="--")
            plt.legend(loc="best")
            plt.gca().set_xscale("log")
            plt.xlabel("$\mathcal{M}_{\mathrm{Max}}$")
            plt.title(
                "$\mathcal{M}_{\mathrm{median}} = $" + f"{np.round(np.median(mm),6)}"
            )
            plt.savefig(
                f"{plt_dir}/mm_hist_{args.model}{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # CDF
            plt.hist(
                mm,
                bins=np.logspace(start=np.log10(0.00001), stop=np.log10(1.0), num=100),
                alpha=0.4,
                label=args.model,
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
                f"{plt_dir}/mm_cdf_{args.model}{ell}{m}.png",
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
                f"{plt_dir}/mm_scatter_{args.model}{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()
