"""
Base script for running InDiCA analysis tests.
Run with specific data source (e.g. JET JPF/PPF data)
"""
import getpass
import json
from pathlib import Path
from typing import Hashable
from typing import List
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
from indica.utilities import to_filename
from .channel_selection import channel_selector
from .channel_selection import INSTRUMENT_TYPES


class BaseTestAnalysis:
    """
    Common components of running a benchmark InDiCA analysis for continuous
    testing.
    Subclass and specialise for specific use cases
    """

    def __init__(self, config_key: Hashable):
        """
        Get test parameters from json configuration file, sets defaults for values
        not present in file.

        :param config_key: Key to get parameters for in configuration file
        :type config_key: Hashable
        """
        with open(f"{str(Path(__file__).absolute().parent)}/test_cases.json", "r") as f:
            self.test_cases = json.load(f)
        assert str(config_key) in self.test_cases.keys()
        self.R: DataArray = coord_array(
            np.linspace(*self.test_cases[str(config_key)].get("R", (1.83, 3.9, 25))),
            "R",
        )
        self.z: DataArray = coord_array(
            np.linspace(*self.test_cases[str(config_key)].get("z", (-1.75, 2.0, 25))),
            "z",
        )
        time = self.test_cases[str(config_key)].get("t", (45, 50, 5))
        self.t: DataArray = coord_array(np.linspace(*time), "t")
        self.trange: Tuple[float, float] = time[0], time[1]
        self.cache_dir = Path(__file__).absolute().parent / "test_cache"

    def clean_cache(self):
        """
        Empty test cache directory
        """
        import shutil

        try:
            shutil.rmtree(self.cache_dir)
        except OSError as e:
            print(e)

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

    def __init__(self, pulse: int):
        self.pulse = int(pulse)
        super().__init__(config_key=str(self.pulse))
        self.cameras: List[str] = self.test_cases[str(pulse)].get("cameras", ["v"])
        self.n_knots: int = self.test_cases[str(pulse)].get("n_knots", 6)
        readers.abstractreader.CACHE_DIR = str(self.cache_dir)
        self.reader = readers.PPFReader(
            pulse=self.pulse,
            tstart=self.trange[0],
            tend=self.trange[1],
            selector=channel_selector,
        )
        self.authenticate_reader()
        print("Reading diagnostics")
        self.diagnostics, self.equilibrium = self.get_diagnostics()
        print("Inverting radiation")
        self.inverter = InvertRadiation(
            num_cameras=len(self.cameras), datatype="sxr", n_knots=self.n_knots
        )
        self.emissivity, self.emiss_fit, *self.camera_results = self.invert_sxr()

    def ignore_channels(self, name: str, channels: List[int]):
        """
        Populate local test_cache files with ignored channels.
        Names will be substituted with full cache path name, assume JETPPF for this

        :param name: Name of instrument to set ignored channels for
        :type name: str
        :param channels: List of channels to ignore
        :type channels: List[int]
        """
        split_name = name.split("_")
        instrument_type = INSTRUMENT_TYPES.get(split_name[0])
        if instrument_type is None:
            return
        cache_path = (
            self.cache_dir
            / self.reader.__class__.__name__
            / to_filename(
                self.reader._RECORD_TEMPLATE.format(
                    self.reader._reader_cache_id,
                    instrument_type,
                    split_name[0],
                    "jetppf",
                    split_name[1],
                )
            )
        )
        with cache_path.open("w+") as f:
            f.writelines([f"{val}\n" for val in channels])

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
        ignore_channels = self.test_cases[str(self.pulse)].get("ignore_channels", {})
        for instrument, channels in ignore_channels.items():
            self.ignore_channels(name=instrument, channels=channels)
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


class BaseTestClass:
    """
    Base class containing tests to run on JET data
    """

    def analysis(self, pulse: int):
        o = JetTestAnalysis(pulse=pulse)
        o.clean_cache()
        return o

    @pytest.mark.skip("Not implemented yet")
    def test_JPN_96175(self):
        pulse = 90279
        return self.analysis(pulse=pulse)

    @pytest.mark.skip("Not implemented yet")
    def test_JPN_96375(self):
        pulse = 90279
        return self.analysis(pulse=pulse)

    @pytest.mark.skip("Not implemented yet")
    def test_JPN_94442(self):
        pulse = 90279
        return self.analysis(pulse=pulse)

    @pytest.mark.skip("Currently unable to run as CI test, requires JET PPF server")
    def test_JPN_90279(self):
        pulse = 90279
        return self.analysis(pulse=pulse)

    @pytest.mark.skip("Not implemented yet")
    def test_JPN_97006(self):
        pulse = 90279
        return self.analysis(pulse=pulse)


class TestTungstenBeryllium(BaseTestClass):
    """
    Compare analysis to WSX:
        - Be concentration should match `ZEFH` to within a few percent,
          contribution from W should be very low
        - Vrot from W asymmetry should match CXRS
        - VUV recalibrated SXR W density should lead to slightly lower
          radiation than experimental value
    """

    pass


class TestNickelBeryllium(BaseTestClass):
    """
    Compare analysis to WSX
    """

    pass


class TestTungstenNickelBeryllium(BaseTestClass):
    """
    Compare analysis to WSX
    """

    pass


class TestNeonSeeded(BaseTestClass):
    """
    Compare analysis to WSX
    """

    pass
