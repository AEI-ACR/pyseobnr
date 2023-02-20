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


def generate_v5PHM_waveform(q:float, mtotal:float,
                            s1x:float, s1y:float, s1z:float,
                            s2x:float, s2y:float, s2z:float,
                            omega_ref:float,
                            omega_start:float,
                            iota:float, phi:float, distance:float,
                            delta_t:float, approx:str,
                            ell_max: int = 5,
                            initial_conditions = "adiabatic",
                            initial_conditions_postadiabatic_type = "analytic",
                            ):



    m1 = q/(1.+q)*mtotal
    m2 = 1./(1.+q)*mtotal

    # Do not test subsolar mass events
    if m2 < 1:
        m2 = 1.
        m1 = q*m2
        mtotal = m1+m2


    #mtotal = m1+m2
    #q = m1/m2
    #omega0 = f_min * (np.pi * mtotal * lal.MTSUN_SI)

    chi_1 = [s1x, s1y, s1z]
    chi_2 = [s2x, s2y, s2z]

    settings = {'ell_max':ell_max,'beta_approx':None,'M':mtotal,"dt":delta_t,
            "initial_conditions" : initial_conditions,
            "initial_conditions_postadiabatic_type" : initial_conditions_postadiabatic_type}

    if approx == 'SEOBNRv5PHM':
        settings["postadiabatic"] = False
        #time, modes, _ = v5P_wrapper.get_EOB_modes(p, ell_max=ell_max)
        time, modes = generate_modes_opt(q,chi_1,chi_2,omega_start,
                           omega_ref=omega_ref, approximant="SEOBNRv5PHM",
                           debug=False,settings=settings)
        modes_dict = {}
        for ell in range(2, ell_max + 1):
            for m in range(-ell, ell + 1):
                modes_dict["{},{}".format(ell, m)] = modes["{},{}".format(ell, m)]

    elif approx == 'SEOBNRv5PHM_PA':
        settings["postadiabatic"] = True
        settings["postadiabatic_type"] = "analytic"
        time, modes = generate_modes_opt(q,chi_1,chi_2, omega_start,
                            omega_ref=omega_ref, approximant="SEOBNRv5PHM",
                           debug=False,settings=settings)
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



def perturbation_mismatch_prec(m1:float, m2:float,
                            s1x:float, s1y:float, s1z:float,
                            s2x:float, s2y:float, s2z:float,
                            iota_s:float,
                            ell_max=5,initial_conditions="adiabatic",omega0 = 0.02):


    Mt = m1+m2
    q=m1/m2

    if q<1:
        q=1

    phi_s = 0.
    pert=1e-15

    f_min = omega0 / (np.pi * (m1 + m2) * lal.MTSUN_SI)
    distance = 1e6*lal.PC_SI
    delta_t = 1./16384.0 
    approx = 'SEOBNRv5PHM_PA'

    # Add small perturbation to m1

    # Generate TD polatizations of SEOBNRv5HM
    time, modes, hp_td, hc_td = generate_v5PHM_waveform(q, Mt,
                                                        s1x, s1y, s1z,
                                                        s2x, s2y, s2z,
                                                        omega0,
                                                        omega0,
                                                        iota_s, phi_s, distance,
                                                        delta_t, approx, ell_max = ell_max,
                                                        initial_conditions = initial_conditions,
                                                        initial_conditions_postadiabatic_type = "analytic",
                                                        )
    # Pad zeros
    N = max(len(hp_td), len(hc_td))
    pad = int(2 ** (np.floor(np.log2(N)) + 2))
    hp_td.resize(pad)
    hc_td.resize(pad)

    # Perform the Fourier Transform
    hp = make_frequency_series(hp_td)
    hc = make_frequency_series(hc_td)


    q_pert = m1*(1.+pert)/m2
    Mt_pert = m1*(1.+pert) + m2
    # Generate TD polarizations of the perturbed system
    time_pert, modes_pert, hp_td, hc_td = generate_v5PHM_waveform(q_pert, Mt_pert,
                                                        s1x, s1y, s1z,
                                                        s2x, s2y, s2z,
                                                        omega0,
                                                        omega0,
                                                        iota_s, phi_s, distance,
                                                        delta_t, approx, ell_max = ell_max,
                                                        initial_conditions = initial_conditions,
                                                        initial_conditions_postadiabatic_type = "analytic",
                                                        )

    # Pad zeros
    N = max(len(hp_td), len(hc_td))
    pad = int(2 ** (np.floor(np.log2(N)) + 2))
    hp_td.resize(pad)
    hc_td.resize(pad)

    # Perform the Fourier Transform
    hp_pert = make_frequency_series(hp_td)
    hc_pert = make_frequency_series(hc_td)


    # Generate PSD
    if f_min<10:
        f_low_phys = 10.
    else:
        f_low_phys = f_min


    f_high_phys = 2048.

    psd = aLIGOZeroDetHighPower(len(hp), hp.delta_f, f_low_phys)

    # Compute match for hplus
    mm_hp = optimized_match(hp,
          hp_pert,
          psd,
          low_frequency_cutoff=f_low_phys,
          high_frequency_cutoff=f_high_phys
          )[0]


    # Compute match for hcross
    mm_hc = optimized_match(hc,
          hc_pert,
          psd,
          low_frequency_cutoff=f_low_phys,
          high_frequency_cutoff=f_high_phys
          )[0]

    # Take the mean
    mm_mean  = 1.-np.mean([mm_hp,mm_hc])
    #print(f"mm = {mm_mean}")

    chi1 = [s1x,s1y,s1z]
    chi2 = [s2x,s2y,s2z]

    if mm_mean>0.001:


        plt.figure(figsize = (10,10/1.618))

        mode_list = ['2,2','2,1','3,3','3,2','4,4','4,3']#,'5,5']

        amp22 = np.abs(modes['2,2'])
        idx_max = np.argmax(amp22)
        tmax22 = time[idx_max]

        amp22_pert = np.abs(modes['2,2'])
        idx_max = np.argmax(amp22_pert)
        tmax22_pert = time_pert[idx_max]

        for mode in mode_list:
            plt.plot(time-tmax22, np.real(modes[mode]),label=str(mode))
            plt.plot(time_pert-tmax22_pert, np.real(modes_pert[mode]),ls = '--')


        plt.xlabel("Time (M)")
        plt.ylabel(r"$\|h_{\ell m}|$")
        plt.xlim(-300,100)
        plt.legend(loc=3)
        plt.title(f'MM = {np.round(mm_mean,7)} - q{np.round(q,3)}_s{np.round(chi1,3)}_s{np.round(chi2,3)} - iota = {np.round(iota_s,3)}')
        Path('./plots').mkdir(parents=True, exist_ok=True)
        plt.savefig(f'./plots/pert_q{np.round(q,3)}_s{np.round(chi1,3)}_s{np.round(chi2,3)}_i{np.round(iota_s,3)}.png', dpi = 200)
        plt.clf()

    res = [m1,m2, *chi1,*chi2, mm_mean,iota_s]

    return (m1,m2, *chi1,*chi2,iota_s, mm_mean)



def process_one_case(input):
    m1,m2,  s1x,s1y,s1z,s2x,s2y,s2z, iota_s, ell_max, initial_conditions, omega0 = input

    m1,m2,  s1x,s1y,s1z,s2x,s2y,s2z, iota_s, mm = perturbation_mismatch_prec(m1,m2, s1x,s1y,s1z,s2x,s2y,s2z, iota_s, ell_max = ell_max,initial_conditions=initial_conditions,omega0 = omega0)
    return np.array([m1,m2,s1x,s1y,s1z,s2x,s2y,s2z, iota_s, mm])


if __name__ == "__main__":

    p = argparse.ArgumentParser(
        description="Perturbation test for the full polarizations"
    )
    p.add_argument(
        "--points", type=int, help="Number of points", default="2000"
    )
    p.add_argument(
        "--plots", action="store_true", help="Make diagnostic plots", default=True
    )
    p.add_argument(
        "--ncores", type=int, help="Number of cores to use", default="64"
    )
    p.add_argument(
        "--q-max", type=float, help="Maximum mass-ratio", default="100.0"
    )
    p.add_argument(
        "--M-max", type=float, help="Maximum total mass", default="10.0"
    )
    p.add_argument(
        "--M-min", type=float, help="Maximum total mass", default="300.0"
    )
    p.add_argument(
        "--ell-max", type=int, help="Maximum l-mode", default="5"
    )
    p.add_argument(
        "--initial-conditions", type=str, help="Type of initial conditions to use for the non-PA evolution", default="adiabatic"
    )
    p.add_argument(
        "--omega0", type=float, help="Starting orbital frequeny in geometric units", default="0.02"
    )
    p.add_argument(
        "--aligned", type=str, help="If set then run aligned-spin configuration", default="False"
    )
    args = p.parse_args()

    seed=150914
    N = args.points
    a_max = 0.99
    ell_max = args.ell_max
    omega0 = args.omega0
    initial_conditions = args.initial_conditions

    np.random.seed(seed)
    q = np.random.uniform(1., args.q_max, N)
    #m1 = np.random.uniform(args.M_min,args.M_max,N)
    np.random.seed(seed-1)
    mtotal = np.random.uniform(args.M_min, args.M_max, N)
    as_case = args.aligned

    m1 = q/(1.+q)*mtotal
    m2 = 1./(1.+q)*mtotal


    if as_case == "True":

        np.random.seed(seed+999998)
        chi1z = np.random.uniform(-a_max, a_max, N)
        np.random.seed(seed+999997)
        chi2z = np.random.uniform(-a_max, a_max, N)
        chi1x = np.zeros(N)
        chi1y = np.zeros(N)
        chi2x = np.zeros(N)
        chi2y = np.zeros(N)

    else:
        np.random.seed(seed+999998)
        a1 = np.random.uniform(0., a_max, N)
        np.random.seed(seed+999997)
        a2 = np.random.uniform(0., a_max, N)
        np.random.seed(seed+999996)
        theta1 = np.random.uniform(0,np.pi,N)
        np.random.seed(seed+999995)
        theta2 = np.random.uniform(0,np.pi,N)
        np.random.seed(seed+999994)
        phi1 = np.random.uniform(0,2*np.pi,N)
        np.random.seed(seed+999993)
        phi2 = np.random.uniform(0,2*np.pi,N)

        chi1x = a1*np.sin(theta1)*np.cos(phi1)
        chi1y = a1*np.sin(theta1)*np.sin(phi1)
        chi1z = a1*np.cos(theta1)

        chi2x = a2*np.sin(theta2)*np.cos(phi2)
        chi2y = a2*np.sin(theta2)*np.sin(phi2)
        chi2z = a2*np.cos(theta2)

    np.random.seed(seed+999992)
    cos_iota = np.random.uniform(-1.0, 1.0, args.points)
    iota_arr = np.arccos(cos_iota)

    ell_max_arr =  np.full(len(iota_arr), ell_max, dtype=int)
    ic_arr =  np.full(len(iota_arr), initial_conditions)
    omega0_arr = np.full(len(iota_arr), omega0)

    lst = [(a, b, c, d,e,f,g,h,i,j,k,l) for a, b, c, d,e,f,g,h,i,j,k,l in zip(m1,m2,chi1x,chi1y,chi1z,chi2x,chi2y,chi2z,iota_arr,ell_max_arr,ic_arr,omega0_arr)]

    with Pool(args.ncores) as pool:
        all_means = pool.map(process_one_case, lst)

    all_means = np.array(all_means)

    np.savetxt(f"perturbation_mismatch.dat", all_means)

    ### Plots ###

    if args.plots:

        import matplotlib
        import matplotlib.pyplot as plt

        plt_dir = './plots'
        Path(plt_dir).mkdir(parents=True, exist_ok=True)

        res_path = f"perturbation_mismatch.dat"
        res = np.loadtxt(res_path)

        m1 = res[:,0]
        m2 = res[:,1]
        q = m1/m2

        chi1x = res[:,2]
        chi1y = res[:,3]
        chi1z = res[:,4]

        chi2x = res[:,5]
        chi2y = res[:,6]
        chi2z = res[:,7]

        iota = res[:,8]
        mm = res[:,9]

        nu = q/(1+q)**2
        m1 = q/(1+q)
        m2 = 1/(1 + q)
        m_total = m1 + m2
        ap = (m1*chi1z+m2*chi2z)/m_total
        am = (m1*chi1z-m2*chi2z)/m_total

        # Mismatch across parameter space
        mm_s, q_s, ap_s, am_s = map(list, zip(*sorted(zip(mm, q, ap, am))))
        plt.scatter(q_s, ap_s, c=mm_s,linewidths=1,norm=matplotlib.colors.LogNorm())
        plt.xlabel('$q$')
        plt.ylabel('$a_{+}$')
        plt.title('$\mathcal{M}_{\mathrm{median}} = $' + f'{np.round(np.median(mm), 7)}')
        cbar=plt.colorbar()
        cbar.set_label('$\mathcal{M}$')
        plt.savefig(f'{plt_dir}/mm_pert_apam.png', bbox_inches = 'tight', dpi = 300)
        plt.close()



        # Histogram
        fig, ax = plt.subplots(figsize=(14,10),dpi=250)
        ax.hist(
            mm,
            bins=np.logspace(start=np.log10(1e-15), stop=np.log10(1.0), num=100),
            alpha=0.4,
            label='SEOBNRv5PHM - SEOBNRv5PHM pert.',
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
        plt.savefig(f"{plt_dir}/mm_hist_v5PHM_pert_test.png",bbox_inches="tight",dpi=300,)
        plt.close()
