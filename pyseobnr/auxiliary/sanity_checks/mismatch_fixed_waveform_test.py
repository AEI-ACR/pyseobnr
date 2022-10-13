import numpy as np
import argparse
import os
import sys
from schwimmbad import JoblibPool
from bilby.core.utils import check_directory_exists_and_if_not_mkdir

sys.path.append(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../../calibration"
    )
)
from models import *
from metrics import *
from parameters import parameters_random

from SEOBNRv5_wrapper import SEOBNRv5_wrapper


def perturbation_mismatch(q_A, chi1_A, chi2_A, q_B, chi1_B, chi2_B):
    
    omega0 = 0.015

    model_1 = SEOBNRv5_wrapper(
        q_A,
        chi1_A,
        chi2_A,
        omega0,
    )

    model_2 = SEOBNRv5_wrapper(
        q_B,
        chi1_B,
        chi2_B,
        omega0,
    )
    
    mm = UnfaithfulnessFlat()(model_1, model_2)

    return (
        q_A,
        chi1_A,
        chi2_A,
        mm
    )


def process_one_case(input):
    q_A, chi1_A, chi2_A, q_B, chi1_B, chi2_B = input
    q_A, chi1_A, chi2_A, mm = perturbation_mismatch(q_A, chi1_A, chi2_A, q_B, chi1_B, chi2_B)
    return np.array([q_A, chi1_A, chi2_A, mm])


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="Perturbation test for a fixed point (q, chi1, chi2)"
    )
    p.add_argument(
        "--points", type=int, help="Number of points", default="5000"
    )
    p.add_argument(
        "--q", type=float, help="Mass-ratio", default="2.0"
    )
    p.add_argument(
        "--chi1", type=float, help="Primary spin", default="0.0"
    )
    p.add_argument(
        "--chi2", type=float, help="Secondary spin", default="0.0"
    )
    p.add_argument(
        "--delta-q", type=float, help="Mass-ratio perturbation", default="1e-9"
    )
    p.add_argument(
        "--delta-chi", type=float, help="Spin perturbation", default="0.01"
    )
    p.add_argument(
        "--name", type=str, help="Name of the output file", default="perturbation_mismatch.dat"
    )
    p.add_argument(
        "--plots", action="store_true", help="Make diagnostic plots", default=True
    )
    args = p.parse_args()

    qarr, chi1arr, chi2arr = parameters_random(args.points, args.q - args.delta_q, args.q + args.delta_q, args.chi1 - args.delta_chi, args.chi1 + args.delta_chi, args.chi2 - args.delta_chi, args.chi2 + args.delta_chi)

    lst = [(a, b, c, args.q, args.chi1, args.chi2) for a, b, c in zip(qarr, chi1arr, chi2arr)]

    with JoblibPool(24) as pool:
        all_means = pool.map(process_one_case, lst)

    all_means = np.array(all_means)

    np.savetxt(args.name, all_means)

    ### Plots ###

    if args.plots:

        import matplotlib
        import matplotlib.pyplot as plt

        plt_dir = './plots'
        check_directory_exists_and_if_not_mkdir(plt_dir)

        res_path = args.name
        res = np.loadtxt(res_path)

        q = res[:,0]
        chi1 = res[:,1]
        chi2 = res[:,2]
        mm = res[:,3]

        nu = q/(1+q)**2
        m1 = q/(1+q)
        m2 = 1/(1 + q)
        ap = m1*chi1+m2*chi2
        am = m1*chi1-m2*chi2

        # Mismatch across parameter space - (chi1, chi2)
        mm_s, chi1_s, chi2_s = map(list, zip(*sorted(zip(mm, chi1, chi2))))
        plt.scatter(chi1_s, chi2_s, c=mm_s,linewidths=1,norm=matplotlib.colors.LogNorm())
        plt.xlabel('$\chi_1$')
        plt.ylabel('$\chi_2$')
        plt.title(f'$q = {args.q},$'+'$\quad \mathcal{M}_{\mathrm{median}} = $' + f'{np.round(np.median(mm), 6)}')
        cbar=plt.colorbar()
        cbar.set_label('$\mathcal{M}$')
        plt.savefig(f'{plt_dir}/mm_pert_chi1chi2.png', bbox_inches = 'tight', dpi = 300)
        plt.close()

        # Mismatch across parameter space - (ap, am)
        mm_s, ap_s, am_s = map(list, zip(*sorted(zip(mm, ap, am))))
        plt.scatter(ap_s, am_s, c=mm_s,linewidths=1,norm=matplotlib.colors.LogNorm())
        plt.xlabel('$a_{+}$')
        plt.ylabel('$a_{-}$')
        plt.title(f'$q = {args.q},$'+'$\quad \mathcal{M}_{\mathrm{median}} = $' + f'{np.round(np.median(mm), 6)}')
        cbar=plt.colorbar()
        cbar.set_label('$\mathcal{M}$')
        plt.savefig(f'{plt_dir}/mm_pert_apam.png', bbox_inches = 'tight', dpi = 300)
        plt.close()