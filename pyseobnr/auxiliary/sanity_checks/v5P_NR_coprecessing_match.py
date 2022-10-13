import argparse
import importlib
from mimetypes import MimeTypes
import os
import sys

import lal
import scri
import numpy as np
from bilby.core.utils import check_directory_exists_and_if_not_mkdir
from pathos.multiprocessing import ProcessingPool as Pool

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../calibration")
)
sys.path.append(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../../toys/hamiltonian_prototype"
    )
)
sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../calibration/calibration_parameter_fits/model_0509")
)
from metrics import *
from models import *

from parameters import parameters_random_fast_prec

from SEOBNRv5PHM_wrapper_0509 import *

# Reproducible
seed = 150914

mode_list = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)] # no (5,5) in 7dq4

import gwsurrogate
sur = gwsurrogate.LoadSurrogate('NRSur7dq4')


def package_modes(waveform_modes, ell_min=2, ell_max=5):
    keys = waveform_modes.keys()
    shape = waveform_modes['2,2'].shape
    result = np.zeros((shape[0], 32), dtype=np.complex128)
    i = 0
    for ell in range(ell_min, ell_max + 1):
        for m in range(-ell, ell + 1):

            if str(ell)+','+str(m) not in keys:
                result[:, i] = np.zeros(shape)
            else:
                result[:, i] = waveform_modes[str(ell)+','+str(m)]
            i += 1
    return result


# Set to zero all missing modes in co-precessing frame
def unpack_scri(w):
    result = {}
    for key in w.LM:
        result[f"{key[0]},{key[1]}"] = w.data[
            :, w.index(key[0], key[1])
        ]
    return result



def mismatch_coprecessing(
    q: float,
    chi1x: float,
    chi1y: float,
    chi1z: float,
    chi2x: float,
    chi2y: float,
    chi2z: float,
    approximant_name,
    signal_name,
    NR_file = None,
    ellMax = 5
):

    m1 = q/(1.+q)
    m2 = 1. - m1
    omega0 = 0.022

    Mt = 50.0
    m1 = m1 * Mt
    m2 = m2 * Mt

    f_min = omega0 * (np.pi * (m1 + m2) * lal.MTSUN_SI)

    v5P_wrapper = SEOBNRv5PHMWrapper()

    try:


        if signal_name=="NRSur7dq4":

            model_signal = NRSur7dq4(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, omega0)
            model_signal(sur)

            imr_full_signal = package_modes(model_signal.waveform_modes)

            # Create a inertial frame scri waveform
            w_signal = scri.WaveformModes(
                dataType=scri.h,
                t=model_signal.t,
                data=imr_full_signal,
                ell_min=2,
                ell_max=5,
                frameType=scri.Inertial,
                r_is_scaled_out=True,
                m_is_scaled_out=True,
            )

            w_signal_cop = w_signal.to_coprecessing_frame()
            modes_signal_cop = unpack_scri(w_signal_cop)

        elif signal_name == 'NR_hdf5':

            model_signal, params_NR = generate_NR_modes(NR_file, ellMax)

            chi1x = params_NR["s1x"]
            chi1y = params_NR["s1y"]
            chi1z = params_NR["s1z"]
            chi2x = params_NR["s2x"]
            chi2y = params_NR["s2y"]
            chi2z = params_NR["s2z"]
            f_min = params_NR["f_min"]
            omega0 = params_NR["omega0"]
            m1 = params_NR['m1']
            m2 = params_NR['m2']
            q = params_NR['q']

        if approximant_name == 'SEOBNRv5PHM':
            params = WaveformParams(
                m1=m1, m2=m2, s1x=chi1x, s1y=chi1y, s1z=chi1z, s2x=chi2x,  s2y=chi2y,  s2z=chi2z,
                f_ref=f_min, f_min = f_min, delta_t=1.0 / 16384, alpha = 0, augm_option = 0#, splines = None
            )

            times, modes_all, model = v5P_wrapper.get_EOB_modes(params)
            coprecessing_modes = model.coprecessing_modes

        elif approximant_name == 'SEOBNRv4PHM':
            settings = {'M':m1+m2 }


            model= SEOBNRv4PHM(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, omega0,
                              settings = settings)
            model()
            imr_full = package_modes(model.waveform_modes)

            # Create a inertial frame scri waveform
            w = scri.WaveformModes(
                dataType=scri.h,
                t=model.t,
                data=imr_full,
                ell_min=2,
                ell_max=5,
                frameType=scri.Inertial,
                r_is_scaled_out=True,
                m_is_scaled_out=True,
            )

            w_cop = w.to_coprecessing_frame()
            coprecessing_modes = unpack_scri(w_cop)
            model.coprecessing_modes = coprecessing_modes


        elif approximant_name == 'IMRPhenomTPHM':

            settings = {'M':m1+m2 }
            model= IMRPhenomTPHM(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, omega0,settings= settings)
            model()
            imr_full = package_modes(model.waveform_modes)

            # Create a inertial frame scri waveform
            w = scri.WaveformModes(
                dataType=scri.h,
                t=model.t,
                data=imr_full,
                ell_min=2,
                ell_max=5,
                frameType=scri.Inertial,
                r_is_scaled_out=True,
                m_is_scaled_out=True,
            )

            w_cop = w.to_coprecessing_frame()
            coprecessing_modes = unpack_scri(w_cop)
            model.coprecessing_modes = coprecessing_modes

        for mode in mode_list:
            ell, m = mode
            model.waveform_modes[f'{ell},{m}'] = model.coprecessing_modes[f'{ell},{m}']
            model_signal.waveform_modes[f'{ell},{m}'] = model_signal.coprecessing_modes[f'{ell},{m}']

        masses = np.arange(10, 310, 10)
        unf_settings = {"sigma": 0.001, "debug": True, "masses": masses}
        unf = UnfaithfulnessPhysV4(settings=unf_settings)

        mms = []
        for mode in mode_list:
            ell, m = mode
            mm = unf(model_signal, model, ell=ell, m=m)
            print(f'(ell,m) = {ell},{m}, mm = {np.max(mm)}')
            mms.append(mm)

    except IndexError:
        np.savetxt(f'error_q{np.round(q,5)}.txt', [q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, f_min])
        mms = []
        for mode in mode_list:
            ell, m = mode
            mm = np.ones(30)
            print(f'(ell,m) = {ell},{m}, mm = {np.max(mm)}')
            mms.append(mm)

    return np.array(mms)


def AmpPhysicaltoNRTD(ampphysical, M, dMpc):
    return ampphysical*dMpc*1e6*lal.PC_SI/(lal.C_SI*(M*lal.MTSUN_SI))

def SectotimeM(seconds, M):
    return seconds/(M*lal.MTSUN_SI)

def generate_NR_modes(NR_file:str, ellMax:int):


    try:
        fp = h5py.File(NR_file, "r")
        freq_1M = fp.attrs["f_lower_at_1MSUN"]

        # Add rounding to prevent some errors in the calculation of the symmetric mass ratio (for TEOBResumSe)
        m1_dim = round(fp.attrs["mass1"], 5)
        m2_dim = round(fp.attrs["mass2"], 5)
        e0_signal = round(fp.attrs["eccentricity"], 4)
        mean_anomaly_signal = round(fp.attrs["mean_anomaly"], 4)
        q=m1_dim/m2_dim

        fp.close()
        prefix = os.path.basename(NR_file).split(".")[0]

        LAL_params_signal = lal.CreateDict()
        ma = lalsim.SimInspiralCreateModeArray()

        for ell in range(2, ellMax + 1):
            lalsim.SimInspiralModeArrayActivateAllModesAtL(ma, ell)

        lalsim.SimInspiralWaveformParamsInsertModeArray(LAL_params_signal, ma)
        lalsim.SimInspiralWaveformParamsInsertNumRelData(LAL_params_signal, NR_file)
        (
            s1x,
            s1y,
            s1z,
            s2x,
            s2y,
            s2z,
        ) = lalsim.SimInspiralNRWaveformGetSpinsFromHDF5File(
            0.0, 100, NR_file
        )  # Since we are using f_ref=0, total mass does not matter

        # print(m1_dim, m2_dim, s1x, s1y, s1z, s2x, s2y, s2z)
    except:
        raise print(
            "When using the 'NR_hdf5' approximant, you must provide the NR_file option!"
        )
        exit(-1)

    Mtot = 20.
    distance = 50.0*1e6*lal.PC_SI
    m1 = m1_dim * Mtot
    m2 = m2_dim * Mtot
    fudge_factor = 1.35


    f_min = freq_1M / (m1 + m2)
    omega0 = f_min * (Mtot * lal.MTSUN_SI * np.pi)
    #print(f"f_min = {f_min}, omega0 = {omega0}")
    _, hlm_NR = lalsim.SimInspiralNRWaveformGetHlms(
                1./16384.,
                m1 * lal.MSUN_SI,
                m2 * lal.MSUN_SI,
                distance,
                f_min,
                0.0,
                s1x,
                s1y,
                s1z,
                s2x,
                s2y,
                s2z,
                NR_file,
                ma,
            )


    modes_NR_list =  []

    for ll in range(2,ellMax+1):
        for mm in range(-ll,ll+1):
            modes_NR_list.append([ll,mm])

    modes_NR = {}
    for ll,mm in modes_NR_list:

        key = str(ll)+','+str(mm)
        modes_NR[key] = AmpPhysicaltoNRTD(lalsim.SphHarmTimeSeriesGetMode(hlm_NR, ll,mm).data.data,
                                  m1+ m2, distance/(1e6*lal.PC_SI))

    tmp = lalsim.SphHarmTimeSeriesGetMode(hlm_NR, 2, 2)
    time_NR = tmp.deltaT * np.arange(len(tmp.data.data))
    time_NR = SectotimeM(time_NR, m1 + m2)

    params_NR = {}
    params_NR = {"m1":m1, "m2":m2, 's1x':s1x, 's1y':s1y, 's1z':s1z, 's2x':s2x, 's2y':s2y, 's2z':s2z,
                 "omega0": omega0, 'f_min':f_min, 'q':q }


    alignment_interval_start = 2 # Which peak in Re(h22) is the start
    alignment_interval_end = 12 # Which peak in Re(h22) is the end


    tmin, tmax = get_alignment_interval_from_maxima(
    time_NR,
    np.real(modes_NR["2,2"]),
    tmin=0,
    start=alignment_interval_start,
    end=alignment_interval_end,
    )
    class recast_output:
            def __init__(final, x, fun,tmin,tmax):
                final.t = x
                #final.coprecessing_modes = fun
                final.waveform_modes = fun
                final.tmin = tmin
                final.tmax = tmax


    model_nr = recast_output(time_NR, modes_NR,tmin,tmax)


    modes_nr = model_nr.waveform_modes

    imr_full_nr = package_modes(modes_nr)

    # Create a inertial frame scri waveform
    w_nr = scri.WaveformModes(
        dataType=scri.h,
        t=model_nr.t,
        data=imr_full_nr,
        ell_min=2,
        ell_max=5,
        frameType=scri.Inertial,
        r_is_scaled_out=True,
        m_is_scaled_out=True,
    )

    w_nr_cop = w_nr.to_coprecessing_frame()
    modes_nr_cop = unpack_scri(deepcopy(w_nr_cop))
    model_nr.coprecessing_modes = modes_nr_cop



    return model_nr, params_NR



def process_one_case(input):
    q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, approximant, signal, NR_file, ellMax = input

    mm = mismatch_coprecessing(
        q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, approximant, signal,NR_file,ellMax
    )
    return mm


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute mismatch")
    p.add_argument("--points", type=int, help="Number of points", default="1000")
    p.add_argument("--a-max", type=float, help="Maximum spin", default="0.8")
    p.add_argument("--q-max", type=float, help="Maximum mass-ratio", default="4.0")
    p.add_argument(
        "--name", type=str, help="Name of the output file", default="mismatch"
    )
    p.add_argument(
        "--approximant-name", type=str, help="Name of the approximant", default="SEOBNRv5PHM"
    )

    p.add_argument(
        "--signal-name", type=str, help="Name of the signal", default="SEOBNRv5PHM"
    )
    p.add_argument(
        "--plots", action="store_true", help="Make diagnostic plots", default=True
    )
    p.add_argument("--n-cpu", type=int, help="Number of cores to use", default=64)
    p.add_argument("--ell-max", type=int, help="Maximum number of l-modes to incude", default=5)
    p.add_argument(
        "--include-all",
        action="store_true",
        help="Include odd m modes close to equal mass",
        default=False,
    )

    p.add_argument("--file_list", type=str, help="List of all the NR files to run on")
    p.add_argument("--nr_dir", type=str, help="Directory with the LVCNR format files", default="/work/sossokine/LVCFormatWaveforms",)


    args = p.parse_args()

    nr_dir = args.nr_dir

    if args.signal_name=='NR_hdf5':

        with open(args.file_list, "r") as fp:
            lst = fp.readlines()
        print(len(lst))
        file_list0 = [lst[i].split('\n')[0] for i in range(len(lst))]

        print(len(lst),file_list0)

        lst = [(1., 0., 0., 0., 0., 0., 0., args.approximant_name,args.signal_name,nr_dir+file_list0[i],args.ell_max) for i in range(len(file_list0))]

        print(f'Generated {len(lst)} NR cases')

    else:
        NR_file=None
        qarr, chi1x_arr, chi1y_arr, chi1z_arr, chi2x_arr, chi2y_arr, chi2z_arr = parameters_random_fast_prec(
            args.points,
            1.0,
            args.q_max,
            0.,
            args.a_max,
            0.,
            args.a_max,
        )

        lst = [(a, b, c, d, e, f, g, args.approximant_name,args.signal_name,NR_file,args.ell_max) for a, b, c, d, e, f, g in zip(qarr, chi1x_arr, chi1y_arr, chi1z_arr, chi2x_arr, chi2y_arr, chi2z_arr)]

        print(f'Generated {args.points} random parameters')

    pool = Pool(args.n_cpu)
    all_means = pool.map(process_one_case, lst)

    all_means = np.array(all_means)
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
            f"{args.name}_{args.approximant_name}_{args.signal_name}_copr_{ell}{m}.dat",
            mismatches[mode],
        )

    np.savetxt(f"parameters_{args.approximant_name}_{args.signal_name}_copr.dat", np.c_[qarr, chi1x_arr, chi1y_arr, chi1z_arr, chi2x_arr, chi2y_arr, chi2z_arr])
    ### Plots ###

    if args.plots:

        import matplotlib
        import matplotlib.pyplot as plt

        plt_dir = "./plots"
        check_directory_exists_and_if_not_mkdir(plt_dir)

        res_path = args.name
        params = np.genfromtxt(f"parameters_{args.approximant_name}_{args.signal_name}_copr.dat")
        q = params[:, 0]
        chi1_x = params[:, 1]
        chi1_y = params[:, 2]
        chi1_z = params[:, 3]
        chi2_x = params[:, 4]
        chi2_y = params[:, 5]
        chi2_z = params[:, 6]


        nu = q / (1 + q) ** 2
        m1 = q / (1 + q)
        m2 = 1 / (1 + q)
        ap = m1 * chi1_z + m2 * chi2_z
        am = m1 * chi1_z - m2 * chi2_z

        for mode in mode_list:
            ell, m = mode
            mm_M = np.loadtxt(
                f"{args.name}_{args.approximant_name}_{args.signal_name}_copr_{ell}{m}.dat"
            )
            # maximum of mm_M across total mass for histogram
            mm = []
            for mp in mm_M:
                mm.append(np.max(mp))
            mm = np.array(mm)

            # Do not plot cases with errors
            idx = np.where(mm<1.0)[0]
            q_pl = q[idx]
            ap_pl = ap[idx]
            mm_M = mm_M[idx]
            mm = mm[idx]

            M = np.arange(10, 310, 10)


            # Spaghetti plot
            for mp in mm_M:
                plt.plot(M, mp, color="C0", linewidth=0.5)
            plt.axhline(0.01, ls="--", color="red")
            plt.yscale("log")
            plt.xlabel("M")
            plt.ylabel("$\mathcal{M}$")
            plt.xlim(10, 300)
            plt.savefig(
                f"{plt_dir}/mm_spaghetti_{args.approximant_name}_{args.signal_name}_copr_{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # Histogram
            plt.hist(
                mm,
                bins=np.logspace(start=np.log10(0.00001), stop=np.log10(1.0), num=50),
                alpha=0.4,
                label=f'{args.approximant_name}-{args.signal_name} (coprecessing frame)',
            )
            plt.axvline(np.median(mm), c="C0", ls="--")
            plt.legend(loc="best")
            plt.gca().set_xscale("log")
            plt.xlabel("$\mathcal{M}_{\mathrm{Max}}$")
            plt.title(
                "$\mathcal{M}_{\mathrm{median}} = $" + f"{np.round(np.median(mm),6)}"
            )
            plt.savefig(
                f"{plt_dir}/mm_hist_{args.approximant_name}-{args.signal_name}_copr_{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()

            # CDF
            plt.hist(
                mm,
                bins=np.logspace(start=np.log10(0.00001), stop=np.log10(1.0), num=100),
                alpha=0.4,
                label=f'{args.approximant_name}-{args.signal_name}',
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
                f"{plt_dir}/mm_cdf_{args.approximant_name}-{args.signal_name}_copr_{ell}{m}.png",
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
                f"{plt_dir}/mm_scatter{args.approximant_name}-{args.signal_name}_copr_{ell}{m}.png",
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()
