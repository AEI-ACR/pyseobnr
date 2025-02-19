"""Some basic regression tests on the code copied from nrutils"""

from pyseobnr.eob.utils.nr_utils import (
    bbh_final_mass_non_precessing_UIB2016,
    bbh_final_spin_non_precessing_HBR2016,
    bbh_final_spin_precessing_HBR2016,
)


def test_final_mass_non_precessing():
    """Basic smoke test on the non precessing final mass"""
    # note: no test has been found from lalinference that can be backported here

    m1 = 0.625
    m2 = 0.37499999999999994
    chi1 = 0.2814369655223523
    chi2 = -0.24119703862679753

    # hp1, hc1 = gen_test_data("TD")
    final_mass = bbh_final_mass_non_precessing_UIB2016(m1, m2, chi1, chi2)
    assert final_mass == 0.9534724600111834


def test_final_spin_non_precessing():
    """Basic smoke test on the non precessing final spin"""
    # note: no test has been found from lalinference that can be backported here

    m1 = 0.625
    m2 = 0.37499999999999994
    chi1 = 0.2269651688658209
    chi2 = -0.14571951370815447

    final_spin = bbh_final_spin_non_precessing_HBR2016(
        m1, m2, chi1, chi2, version="M3J4"
    )

    assert final_spin == 0.7004451919670823


def test_final_spin_precessing():
    """Basic smoke test on the precessing final spin"""
    # note: no test has been found from lalinference that can be backported here
    m1 = 0.625
    m2 = 0.37499999999999994
    chi1 = 0.6164421777444954
    chi2 = 0.7681144331022736
    tilt1 = 1.0966822131670126
    tilt2 = 1.8902119930187582
    phi12 = 1.1504467138934216

    final_spin = bbh_final_spin_precessing_HBR2016(
        m1, m2, chi1, chi2, tilt1, tilt2, phi12, version="M3J4"
    )
    assert final_spin == 0.7549195105253206
