from typing import Dict

pulse = 11089
pulse_to_write = 43000000

diagnostics = ["xrcs", "cxff_tws_c", "ts", "efit"]
opt_quantity = ["xrcs.spectra", "cxff_tws_c.ti", "ts.ne", "ts.te"]

param_names = [
    # "electron_density.y1",
    # "electron_density.y0",
    # "electron_density.peaking",
    # "electron_density.wcenter",
    # "electron_density.wped",
    # "impurity_density:ar.y1",
    "impurity_density:ar.y0",
    # "impurity_density:ar.wcenter",
    # "impurity_density:ar.wped",
    # "impurity_density:ar.peaking",
    # "electron_temperature.y0",
    # "electron_temperature.wped",
    # "electron_temperature.wcenter",
    # "electron_temperature.peaking",
    "ion_temperature.y0",
    # "ion_temperature.wped",
    # "ion_temperature.wcenter",
    # "ion_temperature.peaking",
]

plasma_settings = dict(
    main_ion="h",
    impurities=("ar", "c"),
    impurity_concentration=(0.001, 0.005),
    n_rad=10,
)

filter_coords: Dict = {"cxff_pi":
                       {"ti": ("channel", (3, 5)), "vtor": ("channel", (3, 5))},
                        "cxff_tws_c":
                        {"ti": ("channel", (0, 1)), "vtor": ("channel", (0, 1))}
                       }
model_init: Dict = {"xrcs": {"window_masks": [slice(0.394, 0.396)]}}

phantom = False
mock = True
set_ts = False
profile_params_to_update: Dict = {}
revisions: Dict = {}
tstart = 0.05
tend = 0.06
dt = 0.01
starting_samples = 10
iterations = 5
nwalkers = 5
stopping_criteria_factor = 0.005
sample_method = "random"
stopping_criteria = "mode"
burn_frac = 0.00

mds_write = False
best = False
plot = False
run = "TEST"
run_info = "Test run"
dirname = None
