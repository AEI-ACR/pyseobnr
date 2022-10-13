from .eob.hamiltonian.Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C import (
    Ham_align_a6_apm_AP15_DP23_gaugeL_Tay_C as Ham_aligned_opt,
)
from .eob.waveform.waveform import RR_force, SEOBNRv5RRForce
from .models import SEOBNRv5HM


def generate_modes_opt(
    q, chi1, chi2, omega0, approximant="SEOBNRv5HM", settings=None, debug=False
):
    if approximant == "SEOBNRv5HM":
        RR_f = SEOBNRv5RRForce()
        model = SEOBNRv5HM.SEOBNRv5HM_opt(
            q, chi1, chi2, omega0, Ham_aligned_opt, RR_f, settings=settings
        )
        model()
    if debug:
        return model.t, model.waveform_modes, model
    else:
        return model.t, model.waveform_modes
