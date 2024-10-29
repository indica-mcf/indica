"""
Set of dictionaries providing standard names and units for plasma physics quantities
"""

from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

GeneralDataType = str
SpecificDataType = str

#: Structure for type information for :py:class:`xarray.DataArray` objects.
ArrayType = Tuple[Optional[GeneralDataType], Optional[SpecificDataType]]

#: Structure for type information for :py:class:`xarray.Dataset` objects.
DatasetType = Tuple[Optional[SpecificDataType], Dict[str, GeneralDataType]]

DataType = Union[ArrayType, DatasetType]

#: Units expected by each quantity in Indica (key = string identifier)
UNITS: Dict[str, str] = {
    "none": "",
    "number": "#",
    "length": "m",
    "counts": "Counts",
    "arbitrary": "[a.u]",
    "temperature": "eV",
    "density": "$1/m^3$",
    "particle_source": "$1/m^3 s$",
    "density_integrated": "1/$m^2$",
    "brightness": "$W/m^2$",
    "emissivity": "$W/m^3$",
    "intensity": "$count/s$",
    "radiance": "W/m^2/steradian$",
    "spectral_radiance": "W/m^2/steradian/nm$",
    "velocity": "m/s",
    "magnetic_flux": r"$Wb/2\pi$",
    "f": "Wb m",
    "current": "A",
    "current_density": "$A/m^2$",
    "magnetic_field": "T",
    "energy": "J",
    "power": "W",
    "percent": "%",
    "angular_frequency": "rad/s",
    "frequency": "Hz",
    "time": "s",
    "ionisation_rate": "",
    "recombination_rate": "",
    "emission_rate": "W $m^3$",
    "volume_jacobian": "$m^3$",
    "area_jacobian": "$m^2$",
    "volume": "1/$m^3$",
    "area": "$1/m^2$",
    "voltage": "V",
    "pressure_density": "$Pa/m^3$",
    "pressure": "$Pa$",
    "conductivity": r"1/($\Ohm$ m)",
    "wavelength": "nm",
}

# Dictionary of Indica datatypes
#   {"Variable Name" : ("DataArray LongName", "Unit string identifier")}
DATATYPES: Dict[str, Tuple[str, str]] = {
    "dim_0": ("dim_0", "none"),
    "dim_1": ("dim_1", "none"),
    "dim_2": ("dim_2", "none"),
    "index": ("Index", "none"),
    "label": ("Label", "none"),
    "t": ("Time", "time"),
    "channel": ("Channel", "number"),
    "wavelength": ("Wavelength", "wavelength"),
    "x": ("x", "length"),
    "y": ("y", "length"),
    "R": ("R", "length"),
    "z": ("z", "length"),
    "R_midplane": ("R-midplane", "length"),
    "z_midplane": ("z-midplane", "length"),
    "R_geo": ("$R_{geo}$", "length"),
    "z_geo": ("$z_{geo}$", "length"),
    "R_mag": ("$R_{mag}$", "length"),
    "z_mag": ("$z_{mag}$", "length"),
    "R_fit": ("R", "length"),
    "z_fit": ("z", "length"),
    "R_hfs": ("$R_{HFS}$", "length"),
    "R_lfs": ("$R_{LFS}$", "length"),
    "R_boundary": ("$R_{boundary}$", "length"),
    "z_boundary": ("$z_{boundary}$", "length"),
    "R_shift": ("$R_{shift}$", "length"),
    "r_minor": ("$r_{min}$", "length"),
    "r_minor_boundary": ("a", "length"),
    "rho": (r"$\rho$", "none"),
    "rhop": (r"$\rho_{pol}$", "none"),
    "rhop_fit": (r"$\rho_{pol}$", "none"),
    "rhot": (r"$\rho_{tor}$", "none"),
    "location": ("Location", "length"),  # this is an array (x, y, z)
    "direction": ("Direction", "length"),  # this is an array (dx, dy, dz)
    "element": ("Element", "none"),
    "impurity": ("Impurity", "none"),
    "beamlet": ("Beamlet", "none"),
    "tau": (r"Impurity residence time $\tau$", "time"),
    "electron_density": ("$N_e$", "density"),
    "ion_density": ("$N_i$", "density"),
    "fast_ion_density": ("$N_{fast}$", "density"),
    "impurity_density": ("$N_{imp}$", "density"),
    "thermal_neutral_density": ("$N_{neutrals, th}$", "density"),
    "neutral_density": ("$N_{neutrals}$", "density"),
    "electron_density_integrated": (r"$\int N_e$ dl", "density_integrated"),
    "electron_temperature": ("$T_e$", "temperature"),
    "ion_temperature": ("$T_i$", "temperature"),
    "toroidal_rotation": ("$V_{tor}$", "velocity"),
    "toroidal_angular_frequency": (
        r"$\omega_{tor}$",
        "angular_frequency",
    ),
    "centrifugal_asymmetry": ("Centrifugal asymmetry", "none"),
    "centrifugal_asymmetry_multiplier": (
        "Centrifugal asymmetry multiplier",
        "none",
    ),
    "intensity": ("Intensity", "intensity"),
    "radiance": ("Radiance", "radiance"),
    "spectral_radiance": ("Spectral Radiance", "spectral_radiance"),
    "emissivity": ("Emissivity", "emissivity"),
    "total_radiation": ("$P_{rad, tot}$", "emissivity"),
    "sxr_radiation": ("$P_{rad, sxr}$", "emissivity"),
    "prad_tot": ("$P_{rad, tot}$", "power"),
    "prad_sxr": ("$P_{rad, sxr}$", "power"),
    "brightness": ("Brightness", "brightness"),
    "line_intensity": ("Line intensity", "brightness"),
    "spectra_raw": ("Spectra", "counts"),
    "spectra": ("Spectra", "brightness"),
    "spectra_fit": ("Spectra (fit)", "brightness"),
    "chi_squared": (r"$\chi^2$", "none"),
    "effective_charge": ("$Z_{eff}$", "none"),
    "mean_charge": ("$<q>$", "none"),
    "fractional_abundance": ("$f_q$", "none"),
    "ion_charge": ("Charge state", "none"),
    "atomic_weight": ("Atomic weight", "none"),
    "atomic_number": ("Atomic number", "none"),
    "element_name": ("Element", "none"),
    "element_symbol": ("Element", "none"),
    "equilibrium_f": ("f", "f"),
    "poloidal_flux": ("Poloidal flux", "magnetic_flux"),
    "poloidal_flux_axis": ("Poloidal flux (axis)", "magnetic_flux"),
    "poloidal_flux_boundary": ("Poloidal flux (separatrix)", "magnetic_flux"),
    "psin": ("Normalised poloidal flux", "magnetic_flux"),
    "toroidal_flux": ("Toroidal flux", "magnetic_flux"),
    "volume_jacobian": ("$V_{jac}$", "volume_jacobian"),
    "area_jacobian": ("$A_{jac}$", "area_jacobian"),
    "volume": ("Volume", "volume"),
    "area": ("Cross-sectional area", "area"),
    "plasma_current": ("$I_{P}$", "current"),
    "toroidal_magnetic_field": ("$B_{T}$", "magnetic_field"),
    "poloidal_magnetic_field": ("$B_{P}$", "magnetic_field"),
    "radial_magnetic_field": ("$B_{r}$", "magnetic_field"),
    "vertical_magnetic_field": ("$B_{z}$", "magnetic_field"),
    "concentration": ("Concentration", "percent"),
    "scd": ("SCD rate coefficient", "ionisation_rate"),
    "acd": ("ACD rate coefficient", "recombination_rate"),
    "ccd": ("CCD rate coefficient", "recombination_rate"),
    "pec": ("PEC coefficient", "emission_rate"),
    "plt": ("PLT coefficient", "emission_rate"),
    "prb": ("PRB coefficient", "emission_rate"),
    "prc": ("PRC coefficient", "emission_rate"),
    "pls": ("PLS coefficient", "emission_rate"),
    "prs": ("PRS coefficient", "emission_rate"),
    "total_radiation_loss_parameter": ("$L_{tot}$", "emission_rate"),
    "sxr_radiation_loss_parameter": ("$L_{sxr}$", "emission_rate"),
    "loop_voltage": ("$V_{loop}$", "voltage"),
    "equilibrium_stored_energy": ("$W_{eq}$", "energy"),
    "thermal_stored_energy": ("$W_{th}$", "energy"),
    "fast_ion_stored_energy": ("$W_{fast}$", "energy"),
    "total_stored_energy": ("$W_{tot}$", "energy"),
    "ohmic_current_density": ("$j_{ohm}$", "current_density"),
    "bootstrap_current_density": ("$j_{BS}$", "current_density"),
    "nbi_current_density": ("$j_{NBI}$", "current_density"),
    "total_current_density": ("$j_{tot}$", "current_density"),
    "ohmic_power_density": ("$P_{ohm}$", "power"),
    "nbi_power_density": ("$P_{nbi}$", "power"),
    "nbi_particle_source": ("$S_{nbi}$", "particle_source"),
    "wall_particle_source": ("$S_{wall}$", "particle_source"),
    "total_particle_source": ("$S_{tot}$", "particle_source"),
    "thermal_pressure": ("$P_{th}", "pressure_density"),
    "electron_pressure": ("$P_{th, e}", "pressure_density"),
    "ion_pressure": ("$P_{th, i}", "pressure_density"),
    "total_pressure": ("$P_{tot}", "pressure_density"),
    "fast_ion_pressure": ("$P_{fast}", "pressure_density"),
    "fast_ion_pressure_parallel": ("$P_{fast,//}$", "pressure_density"),
    "fast_ion_pressure_perpendicular": (r"$P_{fast,\perp}$", "pressure_density"),
    "thermal_pressure_integral": ("$P_{th}", "pressure"),
    "total_pressure_integral": ("$P_{tot}", "pressure"),
    "injected_nbi_power": ("$P_{NBI, inj}$", "power"),
    "absorbed_nbi_power": ("$P_{NBI, abs}$", "power"),
    "absorbed_ohmic_power": ("$P_{ohm, abs}$", "power"),
    "safety_factor": ("q", "none"),
    "parallel_conductivity": (r"$\sigma$", "conductivity"),
}
