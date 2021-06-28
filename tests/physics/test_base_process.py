"""
Base script for running InDiCA analysis tests.
Run with specific data source (e.g. JET JPF/PPF data)
"""
import getpass
import json
from pathlib import Path
from typing import Tuple
from typing import Union

from matplotlib import pylab as plt
import numpy as np
import pytest
from xarray.core.dataarray import DataArray

from indica import readers
from indica.equilibrium import Equilibrium
from indica.operators import InvertRadiation
from indica.utilities import coord_array
from .channel_selection import channel_selector


class BaseTestAnalysis:
    """
    Common components of running a benchmark InDiCA analysis for continuous
    testing.
    Subclass and specialise for specific use cases
    """

    def __init__(
        self,
        R: np.ndarray = np.linspace(1.83, 3.9, 25),
        z: np.ndarray = np.linspace(-1.75, 2.0, 25),
        t: np.ndarray = np.linspace(45, 50, 5),
        cameras: list = None,
    ):
        self.cameras = cameras or ["v"]
        self.R = coord_array(R, "R")
        self.z = coord_array(z, "z")
        self.t = coord_array(t, "t")

    def authenticate_reader(self) -> None:
        """
        Method to authenticate chosen reader class
        """
        pass

    def get_diagnostics(self) -> Tuple[dict, Equilibrium]:
        """
        Method to get relevant diagnostics
        """
        pass


class JetTestAnalysis(BaseTestAnalysis):
    """
    Setup and run standard analysis for benchmarking InDiCA against WSX on JET
    data, where JET data readers are available
    """

    def __init__(
        self, pulse: int, time: Tuple[float, float], n_knots: int = 6, **kwargs
    ):
        super().__init__(**kwargs)
        self.pulse = pulse
        self.time = time
        readers.abstractreader.CACHE_DIR = (
            f"{str(Path(__file__).absolute().parent)}/test_cache"
        )
        self.reader = readers.PPFReader(
            pulse=self.pulse,
            tstart=self.time[0],
            tend=self.time[1],
            selector=channel_selector,
        )
        self.authenticate_reader()
        print("Reading diagnostics")
        self.diagnostics, self.equilibrium = self.get_diagnostics()
        print("Inverting radiation")
        self.inverter = InvertRadiation(
            num_cameras=len(self.cameras), datatype="sxr", n_knots=n_knots
        )
        self.emissivity, self.emiss_fit, *self.camera_results = self.invert_sxr()

    def authenticate_reader(self) -> None:
        if self.reader.requires_authentication:
            user = input("JET username: ")
            password = getpass.getpass("JET password: ")
            assert self.reader.authenticate(user, password)

    def get_diagnostics(self) -> Tuple[dict, Equilibrium]:
        """
        Get dictionary of relevant diagnostics and equilibrium object

        Returns
        -------
        """
        diagnostics = {
            "eftp": self.reader.get(uid="jetppf", instrument="eftp", revision=0),
            "hrts": self.reader.get(uid="jetppf", instrument="hrts", revision=0),
            "sxr": self.reader.get(uid="jetppf", instrument="sxr", revision=0),
            # "bolo": self.reader.get(uid="jetppf", instrument="bolo", revision=0),
            # "cxrs": self.reader.get(uid="jetppf", instrument="cxrs", revision=0),
        }
        efit_equilibrium = Equilibrium(equilibrium_data=diagnostics["eftp"])
        for key, diag in diagnostics.items():
            for data in diag.values():
                if hasattr(data.attrs["transform"], "equilibrium"):
                    del data.attrs["transform"].equilibrium
                if "efit" not in key.lower():
                    data.indica.equilibrium = efit_equilibrium
        return diagnostics, efit_equilibrium

    def invert_sxr(self):
        return self.inverter(
            self.R, self.z, self.t, *(self.diagnostics["sxr"][c] for c in self.cameras)
        )

    def plot_sxr_fit(
        self,
        times: Union[int, float, list, np.ndarray, DataArray] = None,
        fig: plt.Figure = None,
        ax: plt.Axes = None,
    ):
        try:
            times = times or self.t
            if isinstance(times, (int, float)):
                times = [times]
            if fig is None or ax is None:
                fig, ax = plt.subplots(
                    nrows=len(times),
                    ncols=len(self.camera_results),
                    sharex="col",
                    sharey="row",
                )
            if not isinstance(ax, np.ndarray):
                ax = np.asarray([ax])
            ax = ax.flatten()
            panel = 0
            assert len(ax) == len(times) * len(self.camera_results)
            for time in times:
                for cresult, cname in zip(self.camera_results, self.cameras):
                    cresult["camera"].sel(t=time).plot.line(
                        "o", x="sxr_v_rho_poloidal", label="From camera", ax=ax[panel]
                    )
                    cresult["back_integral"].sel(t=time).plot(
                        x="sxr_v_rho_poloidal", label="From model", ax=ax[panel]
                    )
                    ax[panel].legend()
                    panel += 1
        finally:
            plt.show()


class TestTungstenBeryllium(JetTestAnalysis):
    """
    Compare analysis to WSX:
        - Be concentration should match `ZEFH` to within a few percent,
          contribution from W should be very low
        - Vrot from W asymmetry should match CXRS
        - VUV recalibrated SXR W density should lead to slightly lower
          radiation than experimental value
    """

    pass


class TestNickelBeryllium(JetTestAnalysis):
    """
    Compare analysis to WSX
    """

    pass


class TestTungstenNickelBeryllium(JetTestAnalysis):
    """
    Compare analysis to WSX
    """

    pass


class TestNeonSeeded(JetTestAnalysis):
    """
    Compare analysis to WSX
    """

    pass


@pytest.mark.skip(
    "Currently unable to run as CI test, "
    "requires JET PPF server connection and credentials"
)
def test_JPN_90279():
    pulse = 90279
    with open(f"{str(Path(__file__).absolute().parent)}/test_cases.json", "r") as f:
        test_cases = json.load(f)
    o = JetTestAnalysis(
        pulse=pulse,
        time=test_cases.get("time", (45.0, 50.0)),
        cameras=test_cases.get("cameras", ["v"]),
        n_knots=test_cases.get("n_knots", 6),
    )
    return o
