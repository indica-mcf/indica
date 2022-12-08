import numpy as np
import xarray as xr
from indica.models.plasma import Plasma
from scipy.optimize import least_squares
from xarray import DataArray
from indica.provenance import get_prov_attribute
from indica.readers.manage_data import initialize_bckc_dataarray
from indica.utilities import print_like
from indica.profiles_gauss import Profiles
from indica.models.interferometry import Interferometry
from indica.numpy_typing import LabeledArray


def match_interferometer_los_int(
    models: dict,
    plasma: Plasma,
    data: dict,
    t: float,
    optimise_for: dict = {"smmh1": ["ne"]},
    ne0: float = 5.0e19,
    bounds: tuple = (1.0e17, 1.0e21),
    bckc: dict = {},
):
    """
    Rescale density profiles to match the interferometer measurements
    """

    def residuals(ne0):
        plasma.Ne_prof.y0 = ne0
        plasma.assign_profiles("electron_density", t=t)

        for instrument in optimise_for.keys():
            if instrument in models.keys():
                bckc[instrument] = models[instrument](t=t)

        resid = []
        for instrument in optimise_for.keys():
            if instrument in models.keys():
                for quantity in optimise_for[instrument]:
                    resid.append(
                        data[instrument][quantity].sel(t=t)
                        - bckc[instrument][quantity]
                    )

        return (np.array(resid) ** 2).sum()

    least_squares(residuals, ne0, bounds=bounds, method="dogbox")

    return plasma, bckc
