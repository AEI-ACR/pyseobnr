import lal
import numpy as np
from jinja2 import Template
from pyseobnr.generate_waveform import GenerateWaveform, generate_modes_opt
from jinja2 import Environment, FileSystemLoader


np.set_printoptions(precision=16)


# Generate expert mode
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


# Generate SI modes
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

# Now generate the data
# These are the time indices to use
indices = [0, 100, 500, 1000, 1500, 2000, 2500, 2820]
indices_waveform = [0, 5000, 10000, 20000, 24000, 23447, 23600]
modes_expert = ["2,2", "2,1", "4,3"]
modes_SI = [(2, 2), (3, 3), (4, 4), (3, 2), (5, 5)]

data = {
    key: np.array2string(model.waveform_modes[key][indices_waveform], separator=",")
    for key in modes_expert
}
data_SI = {
    key: np.array2string(hlm[key][indices_waveform], separator=",") for key in modes_SI
}

# Load the template
file_loader = FileSystemLoader("./")
env = Environment(loader=file_loader)
template = env.get_template("template_v5HM_tests.jinja")
prog = template.render(
    rs=np.array2string(model.dynamics[indices][:, 1], separator=","),
    prs=np.array2string(model.dynamics[indices][:, 3], separator=","),
    modes=data,
    modes_SI=data_SI,
    indices=indices,
    indices_waveform=indices_waveform,
)
with open("test_SEOBNRv5HM.py", "w") as fw:
    fw.write(prog)
