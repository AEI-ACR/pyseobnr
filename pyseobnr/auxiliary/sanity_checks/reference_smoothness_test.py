import numpy as np
import argparse
import os
import sys
from schwimmbad import JoblibPool
from bilby.core.utils import check_directory_exists_and_if_not_mkdir

sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)), "../../calibration" ) )
sys.path.append( os.path.join( os.path.abspath(os.path.dirname(__file__)), "../../toys/hamiltonian_prototype" ) )

from compute_hlms import Kerr_ISCO_mod
from models import *
from metrics import *
from parameters import parameters_random

from SEOBNRv5_wrapper import SEOBNRv5_wrapper


def reference_time(q, chi1, chi2):
    """
    Generate a SEOBNRv5 waveform with parameters (q, chi1, chi2) and compute
    the reference point for the MR attachment (r_ISCO with modified final spin)
    and the last r of the dynamics rend. 

    One should always have r_ISCO > rend
    """
    omega0 = 0.02

    model = SEOBNRv5_wrapper(
        q,
        chi1,
        chi2,
        omega0,
    )
    
    m1 = q/(1+q)
    m2 = 1/(1 + q)
    r_ISCO, L_ISCO = Kerr_ISCO_mod(chi1, chi2, m1, m2)
    rend = model.dynamics[:,1][-1]

    return (
        q,
        chi1,
        chi2,
        rend,
        r_ISCO,
    )


def process_one_case(input):
    q, chi1, chi2 = input
    q, chi1, chi2, rend, r_ISCO = reference_time(q, chi1, chi2)
    return np.array([q, chi1, chi2, rend, r_ISCO])


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="Check reference time smoothness across parameter space"
    )
    p.add_argument(
        "--points", type=int, help="Number of points", default="10000"
    )
    p.add_argument(
        "--q-max", type=float, help="Upper limit on q", default="100.0"
    )
    p.add_argument(
        "--chi-max", type=float, help="Upper limit on |chi|", default="0.995"
    )
    p.add_argument(
        "--name", type=str, help="Name of the output file", default="reference_time.dat"
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
        matplotlib.rcParams['text.usetex'] = True

        plt_dir = './plots'
        check_directory_exists_and_if_not_mkdir(plt_dir)

        res_path = args.name
        res = np.loadtxt(res_path)

        q = res[:,0]
        chi1 = res[:,1]
        chi2 = res[:,2]
        rend = res[:,3]
        r_ISCO = res[:,4]

        nu = q/(1+q)**2
        m1 = q/(1+q)
        m2 = 1/(1 + q)
        ap = m1*chi1+m2*chi2
        am = m1*chi1-m2*chi2

        # End r across parameter space - (q, chieff)
        plt.scatter(q, ap, c=rend,linewidths=1)
        plt.ylabel('$\chi_{\mathrm{eff}}$')
        plt.xlabel('$q$')
        cbar=plt.colorbar()
        cbar.set_label('$r_{\mathrm{end}} (\Omega_0 = 0.015)$')
        plt.savefig(f'{plt_dir}/r_end.png', bbox_inches = 'tight', dpi = 300)
        plt.close()

        # rend - rISCO across parameter space - (q, chieff)
        plt.scatter(q, ap, c=rend - r_ISCO,linewidths=1)
        plt.ylabel('$\chi_{\mathrm{eff}}$')
        plt.xlabel('$q$')
        cbar=plt.colorbar()
        cbar.set_label('$r_{\mathrm{end}} - r_{\mathrm{ISCO}}$')
        plt.savefig(f'{plt_dir}/rend_rISCO.png', bbox_inches = 'tight', dpi = 300)
        plt.close()
