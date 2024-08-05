import numpy as np
from pycbc.types import TimeSeries, FrequencySeries
from pyseobnr.generate_waveform import GenerateWaveform

def gen_seobnrv5hm_td(**p):
    '''PyCBC waveform generator for SEOBNRv5HM

    Parameters
    ----------
    dict: p
        A dictionary of parameters used by pycbc.waveform.get_td_waveform

    Returns
    -------
    hp: Array
        The plus polarization of the waveform in time domain
    hc: Array
        The cross polarization of the waveform in time domain
    """
    '''

    # Some waveform input parameters from PyCBC have the same naming 
    # conventions as PySEOBNR, thus they can be directly inherited. 
    # We only update the settings used uniquely by PySEOBNR
    p.update({
        "approximant": "SEOBNRv5HM",  
        "phi_ref": p["coa_phase"], # reference phase needed by SEOBNRv5
        "f22_start": p["f_lower"], # starting frequency
        "deltaT": p["delta_t"],    
        })

    waveform = GenerateWaveform(p)
    hp, hc = waveform.generate_td_polarizations()

    # Build the PyCBC TimeSeries format
    hp = TimeSeries(hp.data.data[:], delta_t=hp.deltaT, epoch=hp.epoch)
    hc = TimeSeries(hc.data.data[:], delta_t=hc.deltaT, epoch=hc.epoch)

    return hp,hc

def gen_seobnrv5hm_fd(**p):
    '''PyCBC waveform generator for SEOBNRv5HM

    Parameters
    ----------
    dict: p
        A dictionary of parameters used by pycbc.waveform.get_fd_waveform

    Returns
    -------
    hp: Array
        The plus polarization of the waveform in frequency domain
    hc: Array
        The cross polarization of the waveform in frequency domain
    """
    '''

    # Some waveform input parameters from PyCBC have the same naming 
    # conventions as PySEOBNR, thus they can be directly inherited. 
    # We only update the settings used uniquely by PySEOBNR
    p.update({
        "approximant": "SEOBNRv5HM",  
        "phi_ref": p["coa_phase"], 
        "f22_start": p["f_lower"],
        "deltaF": p["delta_f"]
        })

    waveform = GenerateWaveform(p)
    hp, hc = waveform.generate_fd_polarizations()

    # Build the PyCBC TimeSeries format
    hp = FrequencySeries(hp.data.data[:], delta_f=hp.deltaF, epoch=hp.epoch)
    hc = FrequencySeries(hc.data.data[:], delta_f=hc.deltaF, epoch=hp.epoch)

    return hp,hc

def gen_seobnrv5phm_td(**p):
    '''PyCBC waveform generator for SEOBNRv5HM

    Parameters
    ----------
    dict: p
        A dictionary of parameters used by pycbc.waveform.get_td_waveform

    Returns
    -------
    hp: Array
        The plus polarization of the waveform in time domain
    hc: Array
        The cross polarization of the waveform in time domain
    """
    '''

    # Some waveform input parameters from PyCBC have the same naming 
    # conventions as PySEOBNR, thus they can be directly inherited. 
    # We only update the settings used uniquely by PySEOBNR
    p.update({
        "approximant": "SEOBNRv5PHM",  
        "phi_ref": p["coa_phase"], # reference phase needed by SEOBNRv5
        "f22_start": p["f_lower"], # starting frequency
        "deltaT": p["delta_t"],    
        })

    waveform = GenerateWaveform(p)
    hp, hc = waveform.generate_td_polarizations()

    # Build the PyCBC TimeSeries format
    hp = TimeSeries(hp.data.data[:], delta_t=hp.deltaT, epoch=hp.epoch)
    hc = TimeSeries(hc.data.data[:], delta_t=hc.deltaT, epoch=hc.epoch)

    return hp,hc

def gen_seobnrv5phm_fd(**p):
    '''PyCBC waveform generator for SEOBNRv5HM

    Parameters
    ----------
    dict: p
        A dictionary of parameters used by pycbc.waveform.get_fd_waveform

    Returns
    -------
    hp: Array
        The plus polarization of the waveform in frequency domain
    hc: Array
        The cross polarization of the waveform in frequency domain
    """
    '''

    # Some waveform input parameters from PyCBC have the same naming 
    # conventions as PySEOBNR, thus they can be directly inherited. 
    # We only update the settings used uniquely by PySEOBNR
    p.update({
        "approximant": "SEOBNRv5PHM",  
        "phi_ref": p["coa_phase"], 
        "f22_start": p["f_lower"],
        "deltaF": p["delta_f"]
        })

    waveform = GenerateWaveform(p)
    hp, hc = waveform.generate_fd_polarizations()

    # Build the PyCBC TimeSeries format
    hp = FrequencySeries(hp.data.data[:], delta_f=hp.deltaF, epoch=hp.epoch)
    hc = FrequencySeries(hc.data.data[:], delta_f=hc.deltaF, epoch=hp.epoch)

    return hp,hc