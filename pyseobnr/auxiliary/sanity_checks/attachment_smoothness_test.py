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


def attachment_time(q, chi1, chi2):
    
    omega0 = 0.015

    model = SEOBNRv5_wrapper(
        q,
        chi1,
        chi2,
        omega0,
    )
    
    t_attach = model.t_attach

    return (
        q,
        chi1,
        chi2,
        t_attach
    )


def process_one_case(input):
    q, chi1, chi2 = input
    q, chi1, chi2, t_attach = attachment_time(q, chi1, chi2)
    return np.array([q, chi1, chi2, t_attach])


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="Check attachment time smoothness across parameter space"
    )
    p.add_argument(
        "--points", type=int, help="Number of points", default="5000"
    )
    p.add_argument(
        "--q-max", type=float, help="Upper limit on q", default="20.0"
    )
    p.add_argument(
        "--chi-max", type=float, help="Upper limit on |chi|", default="0.95"
    )
    p.add_argument(
        "--name", type=str, help="Name of the output file", default="attachment_time.dat"
    )
    p.add_argument(
        "--plots", action="store_true", help="Make diagnostic plots", default=True
    )
    args = p.parse_args()

    qarr, chi1arr, chi2arr = parameters_random(args.points, 1.0, args.q_max, -args.chi_max, args.chi_max, -args.chi_max, args.chi_max)

    lst = [(a, b, c) for a, b, c in zip(qarr, chi1arr, chi2arr)]

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
        t_attach = res[:,3]

        nu = q/(1+q)**2
        m1 = q/(1+q)
        m2 = 1/(1 + q)
        ap = m1*chi1+m2*chi2
        am = m1*chi1-m2*chi2

        # Attachment time across parameter space - (q, chieff)
        plt.scatter(q, ap, c=t_attach,linewidths=1,norm=matplotlib.colors.LogNorm())
        plt.ylabel('$\chi_{\mathrm{eff}}$')
        plt.xlabel('$q$')
        cbar=plt.colorbar()
        cbar.set_label('$t_{\mathrm{attach}}/M (\Omega_0 = 0.015)$')
        plt.savefig(f'{plt_dir}/t_attachment.png', bbox_inches = 'tight', dpi = 300)
        plt.close()
