import argparse
import importlib
import os,sys
from mimetypes import MimeTypes
import lal
import numpy as np


import matplotlib.pyplot as plt
from pathlib import Path
from pathos.multiprocessing import ProcessingPool as Pool

from pyseobnr.generate_waveform import generate_modes_opt
from pyseobnr.auxiliary.sanity_checks.parameters import parameters_random_fast
from pyseobnr.auxiliary.sanity_checks.single_waveform_tests import amplitude_hierarchy_test

from pycbc.psd.analytical import aLIGOZeroDetHighPower, aLIGOZeroDetHighPowerGWINC
from pycbc.types import TimeSeries
from pycbc.filter import make_frequency_series
from pycbc.filter.matchedfilter import optimized_match
from pycbc.waveform.utils import taper_timeseries


from typing import Dict, Union, Tuple
def combine_modes(
    iota: float, phi: float, modes_dict: Dict
) -> Tuple[np.array, np.array]:
    """Combine modes to compute the waveform polarizations in the direction
    (iota,np.pi/2-phi)

    Args:
        iota (float): Inclination angle (rad)
        phi (float): Azimuthal angle(rad)
        modes_dict (Dict): Dictionary containing the modes, either time of frequency-domain

    Returns:
        np.array: Waveform in the given direction
    """
    sm = 0.0
    for key in modes_dict.keys():
        #print(key)
        ell, m = [int(x) for x in key.split(",")]
        Ylm0 = lal.SpinWeightedSphericalHarmonic(iota, np.pi / 2 - phi, -2, ell, m)
        sm += Ylm0 * modes_dict[key]

    return np.real(sm), -np.imag(sm)


def timeMtoSec(timeM, mt:float):
    return timeM*mt*lal.MTSUN_SI


def ampNRtoPhysicalTD(ampNR, mt:float,distance:float):
    return ampNR*(lal.C_SI*mt*lal.MTSUN_SI)/distance



def generate_v5PHM_waveform(m1:float, m2:float,
                            s1x:float, s1y:float, s1z:float,
                            s2x:float, s2y:float, s2z:float,
                            f_min:float,
                            iota:float, phi:float, distance:float,
                            delta_t:float, approx:str,
                            ell_max: int = 4):

    mtotal = m1+m2
    q = m1/m2
    omega0 = f_min * (np.pi * (m1 + m2) * lal.MTSUN_SI)

    chi_1 = [s1x, s1y, s1z]
    chi_2 = [s2x, s2y, s2z]

    settings = {'ell_max':ell_max,'beta_approx':None,'M':mtotal,"dt":delta_t,
    "postadiabatic": True,
    "postadiabatic_type": "analytic",
    "initial_conditions":"adiabatic",
    "initial_conditions_postadiabatic_type":"analytic"}
    if approx == 'SEOBNRv5PHM':
        #time, modes, _ = v5P_wrapper.get_EOB_modes(p, ell_max=ell_max)
        time, modes = generate_modes_opt(q,chi_1,chi_2,omega0,approximant=approx,
                           debug=False,settings=settings)

        #amp22 = abs(modes['2,2'])
        #idx_max = np.argmax(amp22)
        #tmax22 = time[idx_max]

        #time -= tmax22

        modes_dict = {}
        for ell in range(2, ell_max + 1):
            for m in range(-ell, ell + 1):
                modes_dict["{},{}".format(ell, m)] = modes["{},{}".format(ell, m)]

    elif approx == 'SEOBNRv5HM':

        time, modes = generate_modes_opt(q,chi_1[-1],chi_2[-1],omega0,approximant=approx,
                                debug=False,settings=settings)

        modes_dict = {}
        for key in modes.keys():
            lm_tag = key.split(',')
            ell,m = lm_tag
            ll = float(ell)
            modes_dict["{},{}".format(ell, m)] = modes[key] # Minus sign to have same convention as v5PHM
            modes_dict["{},-{}".format(ell,m)] = pow(-1.,ll) * np.conjugate(modes[key]) # Minus sign to have same convention as v5PHM


    else:
        raise NotImplementedError



    hp_NR, hc_NR = combine_modes(iota, phi, modes_dict)

    t_s = timeMtoSec(time, mtotal)
    hp = ampNRtoPhysicalTD(hp_NR,mtotal,distance)
    hc = ampNRtoPhysicalTD(hc_NR,mtotal,distance)


    # Taper
    hp_td = TimeSeries(hp, delta_t=delta_t)
    hc_td = TimeSeries(hc, delta_t=delta_t)
    hp_td = taper_timeseries(hp_td, tapermethod="startend")
    hc_td = taper_timeseries(hc_td, tapermethod="startend")

    return time, modes_dict, hp_td, hc_td

# Reproducible
seed = 150914

mode_list = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]#, (5, 5)]

def mismatch_v5P_strain(
    m1: float,
    m2: float,
    chi1: float,
    chi2: float,
    iota_s: float,
):

    Mt = m1+m2
    q=m1/m2

    if q<1:
        q=1

    if m2 < 1:

        m2 = 1.
        m1 = q*m2
        Mt = m1+m2


    phi_s = 0.0

    omega0 = 0.02
    f_min = omega0 / (np.pi * (m1 + m2) * lal.MTSUN_SI)
    distance = 1e6*lal.PC_SI
    delta_t = 1./16384.0

    # Generate TD polatizations of SEOBNRv5HM
    time_v5, modes_v5, hp_td_v5, hc_td_v5 = generate_v5PHM_waveform(m1, m2,
                                0.,0., chi1,
                                0.,0., chi2,
                                f_min,
                                iota_s, phi_s, distance,
                                delta_t, 'SEOBNRv5HM')




    # Generate TD polatizations of SEOBNRv5PHM
    time_v5p, modes_v5p, hp_td_v5p, hc_td_v5p = generate_v5PHM_waveform(m1, m2,
                                0.,0., chi1,
                                0.,0., chi2,
                                f_min,
                                iota_s, phi_s, distance,
                                delta_t, 'SEOBNRv5PHM')

    # Pad zeros
    N = max(len(hp_td_v5), len(hc_td_v5p))
    pad = int(2 ** (np.floor(np.log2(N)) + 2))
    hp_td_v5p.resize(pad)
    hc_td_v5p.resize(pad)
    hp_td_v5.resize(pad)
    hc_td_v5.resize(pad)


    # Perform the Fourier Transform
    hp_v5p = make_frequency_series(hp_td_v5p)
    hc_v5p = make_frequency_series(hc_td_v5p)


    hp_v5 = make_frequency_series(hp_td_v5)
    hc_v5 = make_frequency_series(hc_td_v5)


    amp_strain = abs(hp_v5p.data-1j*hc_v5p.data)
    idx_max = np.argmax(amp_strain)
    fpeak_v5p = hp_v5p.get_sample_frequencies()[idx_max]



    ampv5_strain = abs(hp_v5.data-1j*hc_v5.data)
    idx_max = np.argmax(ampv5_strain)
    fpeak_v5 = hp_v5.get_sample_frequencies()[idx_max]

    fpeak = max([fpeak_v5p,fpeak_v5])
    f_min = 1.35*fpeak

    # Generate PSD
    if f_min<10:
        f_low_phys = 10.
    else:
        f_low_phys = f_min


    f_high_phys = 2048.

    try:

        psd = aLIGOZeroDetHighPower(len(hp_v5), hp_v5.delta_f, f_low_phys)

        # Compute match for hplus
        mm_hp = optimized_match(hp_v5,
              hp_v5p,
              psd,
              low_frequency_cutoff=f_low_phys,
              high_frequency_cutoff=f_high_phys
              )[0]


        # Compute match for hcross
        mm_hc = optimized_match(hc_v5,
              hc_v5p,
              psd,
              low_frequency_cutoff=f_low_phys,
              high_frequency_cutoff=f_high_phys
              )[0]

        # Take the mean
        mm_mean  = 1.-np.mean([mm_hp,mm_hc])
    except:

        print(
            f"Error for the following parameters: q = {q}, chi1 = {chi1}, chi2 = {chi2}, Mt = {Mt}, iota_s = {iota_s}"
        )
        mm_mean = -1
        pass

    #print(m1,m2,chi1,chi2,iota_s,mm_mean)
    return (m1,m2, chi1, chi2, iota_s, mm_mean)


def process_one_case(input):

    m1, m2, chi1, chi2, iota = input

    m1,m2, chi1, chi2, iota, mm = mismatch_v5P_strain(
        m1,m2, chi1, chi2,iota
    )

    return  np.array([m1,m2, chi1, chi2, iota, mm])



if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute mismatch")
    p.add_argument("--points", type=int, help="Number of points", default="5000")
    p.add_argument("--chi-max", type=float, help="Maximum spin", default="0.995")
    p.add_argument("--q-max", type=float, help="Maximum mass-ratio", default="100.0")
    p.add_argument(
        "--name", type=str, help="Name of the output file", default="mismatch"
    )
    p.add_argument(
        "--plots", action="store_true", help="Make diagnostic plots", default=True
    )
    p.add_argument("--n-cpu", type=int, help="Number of cores to use", default=64)
    p.add_argument(
        "--include-all",
        action="store_true",
        help="Include odd m modes close to equal mass",
        default=False,
    )

    p.add_argument(
        "--M-max", type=float, help="Maximum total mass", default="10.0"
    )
    p.add_argument(
        "--M-min", type=float, help="Maximum total mass", default="300.0"
    )

    args = p.parse_args()



    seed=150914
    N = args.points
    a_max = 0.998
    np.random.seed(seed)
    q = np.random.uniform(1., args.q_max, N)
    #m1 = np.random.uniform(args.M_min,args.M_max,N)
    np.random.seed(seed-1)
    mtotal = np.random.uniform(args.M_min, args.M_max, N)

    m1 = q/(1.+q)*mtotal
    m2 = 1./(1.+q)*mtotal

    np.random.seed(seed+999998)
    chi1arr = np.random.uniform(-a_max, a_max, N)
    np.random.seed(seed+999997)
    chi2arr = np.random.uniform(-a_max, a_max, N)
    np.random.seed(seed+999996)
    iota_arr = np.random.uniform(0,np.pi, N)

    lst = [ (m1_msun, m2_msun, s1z, s2z, iota) for m1_msun, m2_msun, s1z, s2z, iota in zip(m1,m2, chi1arr, chi2arr,iota_arr ) ]

    #print(f'Generated {args.points} random parameters')
    #print(lst)

    pool = Pool(args.n_cpu)
    all_means = pool.map(process_one_case, lst)
    all_means = np.array(all_means)

    np.savetxt(f"mismatch_aslimit_strain.dat", all_means)

    #print(f"Finalizing mismatch AS limit")
    ### Plots ###

    if args.plots:

        import matplotlib
        import matplotlib.pyplot as plt

        plt_dir = "./plots"
        Path(plt_dir).mkdir(parents=True, exist_ok=True)

        res_path = args.name
        params = np.genfromtxt(f"mismatch_aslimit_strain.dat")
        m1 = params[:, 0]
        m2 = params[:, 1]
        chi1 = params[:, 2]
        chi2 = params[:, 3]
        iota = params[:, 4]
        mm = params[:, 5]

        q = m1/m2
        mtotal = m1+m2
        nu = q / (1. + q) ** 2
        m1_dim = q / (1. + q)
        m2_dim = 1 / (1. + q)
        ap = (m1 * chi1 + m2 * chi2)/mtotal
        am = (m1 * chi1 - m2 * chi2)/mtotal

        # Histogram

        fig, ax = plt.subplots(figsize=(14,10),dpi=250)
        ax.hist(
            mm,
            bins=np.logspace(start=np.log10(1e-15), stop=np.log10(1.0), num=100),
            alpha=0.4,
            label='SEOBNRv5PHM - SEOBNRv5HM',
            histtype="step",
            fill=False
        )

        ax.axvline(np.median(mm), c="C0", ls="--")
        ax.legend(loc="best")
        ax.set_xlabel("$\mathcal{M}_{\mathrm{Max}}$",fontsize=30)
        ax.set_ylabel("Count",fontsize=30)
        ax.set_title(
            "$\mathcal{M}_{\mathrm{median}} = $" + f"{np.round(np.median(mm),10)}",fontsize=25
        )


        ax.set_xscale('log')

        size=30
        ax.tick_params(axis='x', which='major', pad=10,width=2,length=5,size=7, labelsize=size,direction='in')
        ax.tick_params(axis='x', which='minor', pad=10,width=2,length=5,size=7, labelsize=size,direction='in')
        ax.tick_params(axis='y', which='major', pad=10,width=2,length=5,size=7, labelsize=size,direction='in')
        ax.tick_params(axis='y', which='minor', pad=10,width=2,length=5,size=7, labelsize=size,direction='in')
        plt.tight_layout()
        plt.savefig(f"{plt_dir}/mm_hist_v5PHM_align_strain.png",bbox_inches="tight",dpi=300,)
        plt.close()

        # CDF
        fig, ax = plt.subplots(figsize=(14,10),dpi=250)
        ax.hist(
            mm,
            bins=np.logspace(start=np.log10(1e-15), stop=np.log10(1.0), num=100),
            alpha=0.4,
            label='SEOBNRv5PHM - SEOBNRv5HM',
            cumulative=True,
            density=True,
            histtype="step",
            lw=2,
        )




        ax.legend(loc="best")
        ax.set_xscale("log")
        ax.set_xlabel("$\mathcal{M}_{\mathrm{Max}}$")
        ax.set_title(
            "$\mathcal{M}_{\mathrm{median}} = $" + f"{np.round(np.median(mm),6)}"
        )

        ax.set_xscale('log')

        size=30
        ax.tick_params(axis='x', which='major', pad=10,width=2,length=5,size=7, labelsize=size,direction='in')
        ax.tick_params(axis='x', which='minor', pad=10,width=2,length=5,size=7, labelsize=size,direction='in')
        ax.tick_params(axis='y', which='major', pad=10,width=2,length=5,size=7, labelsize=size,direction='in')
        ax.tick_params(axis='y', which='minor', pad=10,width=2,length=5,size=7, labelsize=size,direction='in')
        plt.grid(True, which="both", ls=":")

        plt.tight_layout()
        plt.savefig(f"{plt_dir}/mm_cdf_v5PHM_align_limit_strain.png",bbox_inches="tight",dpi=300,)
        plt.close()



        # Mismatch across parameter space
        mm_s, q_s, ap_s = map(list, zip(*sorted(zip(mm, q, ap))))
        plt.scatter(
            q_s, ap_s, c=mm_s, linewidths=1, norm=matplotlib.colors.LogNorm()
        )
        plt.ylabel("$\chi_{\mathrm{eff}}$")
        plt.xlabel("$q$")
        cbar = plt.colorbar()
        cbar.set_label("$\mathcal{M}$")
        plt.savefig(
            f"{plt_dir}/mm_scatter_v5PHM_align_limit_strain.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()
