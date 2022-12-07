"""
Perform several regression tests of the SEOBNRv5HM model.

This file was automatically generated on 2022-10-24 14:10:41.366988.
"""

import pytest
import numpy as np
from pyseobnr.generate_waveform import generate_modes_opt, GenerateWaveform
import lal

# Indices to use for tests of waveforms
indices_waveform = [0, 5000, 10000, 20000, 24000, 23447, 23600]
indices = [0, 100, 500, 1000, 1500, 2000, 2500, 2820]


@pytest.fixture
def model_expert():
    q = 5.0
    chi1 = 0.76
    chi2 = 0.33
    omega0 = 0.0157
    Mt = 60.0
    dt = 1 / 8192.0
    settings = {"M": Mt, "dt": dt}
    t, modes, model = generate_modes_opt(
        q, chi1, chi2, omega0, settings=settings, debug=True
    )
    return model


@pytest.fixture
def modes_SI():
    m1 = 50.0
    m2 = 10.0
    Mt = m1 + m2
    dt = 1 / 8192.0
    distance = 1000.0
    inclination = np.pi / 3.0
    phiRef = 0.0
    approximant = "SEOBNRv5HM"
    s1x = s1y = s2x = s2y = 0.0
    s1z = 0.76
    s2z = 0.33
    f_max = 4096.0
    f_min = 0.0157 / (Mt * np.pi * lal.MTSUN_SI)
    params_dict = {
        "mass1": m1,
        "mass2": m2,
        "spin1x": s1x,
        "spin1y": s1y,
        "spin1z": s1z,
        "spin2x": s2x,
        "spin2y": s2y,
        "spin2z": s2z,
        "deltaT": dt,
        "f22_start": f_min,
        "phi_ref": phiRef,
        "distance": distance,
        "inclination": inclination,
        "f_max": f_max,
        "approximant": approximant,
    }
    wfm_gen = GenerateWaveform(params_dict)  # We call the generator with the parameters
    times, hlm = wfm_gen.generate_td_modes()
    return times, hlm


def test_dynamics(model_expert):
    """
    Check that the dynamics values at certain points in the inspiral
    have not changed.
    """
    rs = np.array(
        [
            15.867019121339561,
            10.531505723413677,
            6.045329454339947,
            5.651540484962651,
            5.1601391822805285,
            4.492859835910911,
            3.4089585745044086,
            1.6368739894468507,
        ]
    )
    prs = np.array(
        [
            -0.0004539973226946,
            -0.0016059691727573,
            -0.0092114454070551,
            -0.0113809478343324,
            -0.0150963007654738,
            -0.0229740183850073,
            -0.050812465887382,
            -0.3658845972876429,
        ]
    )

    dynamics = model_expert.dynamics
    r = dynamics[:, 1]
    pr = dynamics[:, 3]
    np.testing.assert_allclose(rs, r[indices], atol=0, rtol=1e-12)
    np.testing.assert_allclose(prs, pr[indices], atol=0, rtol=1e-12)


def test_modes(model_expert):
    """
    Check that the waveform modes at certain points have not changed.
    """

    values = {}

    values["2,2"] = np.array(
        [
            -4.9670160471169768e-02 + 1.1476140064341465e-02j,
            -2.2437672748292156e-02 - 4.9059805028387263e-02j,
            -7.3428845474157890e-03 - 5.7650393288003809e-02j,
            -7.2184657727662291e-02 + 3.4165897482683651e-02j,
            4.8201320986880289e-09 - 3.5026548265447038e-08j,
            1.2410186768821159e-01 - 1.6871322253603330e-01j,
            3.4623579054240856e-03 - 5.7517888348397406e-03j,
        ]
    )

    values["2,1"] = np.array(
        [
            -3.1583499225760532e-04 - 2.0799956796436481e-03j,
            -1.1488669103894799e-03 + 1.9622303283948193e-03j,
            -1.5727111334759877e-03 + 1.9660569444120957e-03j,
            -1.1786116080562134e-03 - 3.6940743854642789e-03j,
            5.4883634454436089e-10 + 6.5662272188327650e-10j,
            -9.1894934041537304e-04 + 1.3782029911672107e-02j,
            -7.6475210439808024e-05 - 2.8059969966021705e-04j,
        ]
    )

    values["4,3"] = np.array(
        [
            3.8585753171480794e-05 + 9.6989552011944092e-05j,
            1.2032061037868927e-04 + 1.2123218625816370e-05j,
            1.2525627747594521e-04 + 7.6570191973531439e-05j,
            2.2777328907491463e-04 + 2.4925111608339645e-04j,
            1.2480162449720609e-09 + 1.8960186805298799e-10j,
            2.4583870938132677e-03 + 5.2295282999532929e-03j,
            -8.0105813170395053e-05 - 3.2749209067593061e-04j,
        ]
    )

    for mode in values.keys():
        np.testing.assert_allclose(
            values[mode],
            model_expert.waveform_modes[mode][indices_waveform],
            atol=0,
            rtol=1e-14,
        )


def test_SI_modes(modes_SI):
    """
    Check that the SI modes have the right values at certain points in time
    """

    values = {}

    values[(2, 2)] = np.array(
        [
            1.4261542592916378e-22 - 3.2950862001921257e-23j,
            6.4424157794179870e-23 + 1.4086294313839175e-22j,
            2.1083253956592566e-23 + 1.6552866581787464e-22j,
            2.0726016605798606e-22 - 9.8098817791685713e-23j,
            -1.3839802122005062e-29 + 1.0056996096488549e-28j,
            -3.5632743182362025e-22 + 4.8441776438036797e-22j,
            -9.9412935798319619e-24 + 1.6514821106958258e-23j,
        ]
    )

    values[(3, 3)] = np.array(
        [
            -6.5642518776869324e-24 - 1.6796696772315580e-23j,
            -1.9500998115254826e-23 - 2.0875691074407980e-24j,
            -1.8608251631270033e-23 - 1.1537694600509943e-23j,
            -2.3363816008820581e-23 - 2.5788235608474010e-23j,
            -2.9684567653181157e-29 - 6.9049871710376828e-30j,
            -1.0005618098419905e-22 - 1.4332360997808917e-22j,
            4.9473133801433107e-25 + 6.7897357958470015e-24j,
        ]
    )

    values[(4, 4)] = np.array(
        [
            -3.2487782078916292e-24 + 1.7091751952236825e-24j,
            2.5815709927319988e-24 - 3.1949227742120076e-24j,
            4.5606253352664988e-24 - 1.3671513473902351e-24j,
            -5.1753389671241531e-24 + 7.2657132847788700e-24j,
            2.2130297081198305e-30 + 9.5972610653569981e-30j,
            6.1844543616416235e-23 + 2.9278250739332965e-23j,
            -1.0119562653917803e-24 + 2.3636638072691720e-24j,
        ]
    )

    values[(3, 2)] = np.array(
        [
            1.6837802909039010e-24 - 4.3675368049045504e-25j,
            8.6793628477472505e-25 + 1.7610674440853747e-24j,
            3.6424124958857387e-25 + 2.2739823654756563e-24j,
            4.0318039897940146e-24 - 2.1579673178709476e-24j,
            -9.0901269957147127e-31 + 9.4687187544572400e-30j,
            -2.0151480065924556e-23 + 3.9917314008033975e-23j,
            -1.1706862984527676e-24 + 1.6883127010578698e-24j,
        ]
    )

    values[(5, 5)] = np.array(
        [
            4.5352475158856276e-25 + 6.2945747959408049e-25j,
            3.0672295822860062e-25 + 8.3842154574859743e-25j,
            -4.1235830093005229e-25 + 9.9121724528264025e-25j,
            2.2331142448161237e-24 + 7.2639430731712485e-25j,
            3.0376802592951011e-30 + 8.9831287697997125e-31j,
            -1.3117435760531568e-23 + 2.9490861610750671e-23j,
            -7.7214769328720130e-25 + 4.9902542837762871e-25j,
        ]
    )

    t, hlm = modes_SI
    for mode in values.keys():
        np.testing.assert_allclose(
            values[mode], hlm[mode][indices_waveform], atol=0, rtol=1e-14
        )
