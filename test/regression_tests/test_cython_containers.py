import numpy as np

from .helpers import create_eob_params


def test_0_spins():
    """In case we have 0 spin, the conditions in cython for nu should be the same as for python"""
    eob_pars, hamiltonian = create_eob_params(
        m_1=0.5000000057439491,
        m_2=0.49999999425605085,
        chi_1=0,
        chi_2=0,
        omega=0.17073493094550068,
        omega_circ=0.012761920876346968,
        omega_avg=0.012761920876346968,
        omega_instant=0.016980735748382013,
        x_avg=0.30767334771596205,
        eccentricity=0.14745898920231795,
        rel_anomaly=0,
    )

    assert np.all(np.array(eob_pars.flux_params.f_coeffs) == 0)

    assert np.abs(eob_pars.p_params.nu - 0.25) < 1e-14
    assert not (np.abs(eob_pars.p_params.m_1 / eob_pars.p_params.m_2 - 1) < 1e-14)
    assert (
        np.abs(eob_pars.p_params.m_1 / eob_pars.p_params.m_2 - 1) < (2 / 5 * 1e-6)
        and np.abs(eob_pars.p_params.chi_A) < 1e-14
    )
