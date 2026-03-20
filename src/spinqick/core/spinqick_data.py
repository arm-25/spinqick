"""Data handling and saving for spinqick experiments."""

import json
import logging
import os
import time
from typing import Any, Dict, List, Sequence

import importlib_metadata
import netCDF4
import numpy as np
import pydantic
from qick import qick_asm

from spinqick.helper_functions import file_manager, spinqick_enums

logger = logging.getLogger(__name__)


def load_analysis_data(nc_file: netCDF4.Dataset, data_desc: str, attr_name: str | None = None):
    """Load data stored in the 'analyzed_data' folder of the netcdf file."""
    processed = []
    attrs = []
    ana = nc_file["analyzed_data"]
    ana_data_keys = list(ana.variables.keys())
    analysis_type = "None"
    for d in ana_data_keys:
        if data_desc in d:
            data_array = np.asarray(ana[d][:])
            processed.append(data_array)
            if hasattr(ana[d], "units"):
                analysis_type = ana[d].units
            if attr_name is not None:
                if hasattr(ana[d], attr_name):
                    var_attr = getattr(ana[d], attr_name)
                    attrs.append(var_attr)
    return processed, analysis_type, attrs


def _assign_timestamp():
    return int(time.time())


def _get_filename(timestamp, experiment_name):
    data_path = file_manager.get_data_dir()
    data_file = os.path.join(
        data_path,
        str(timestamp) + experiment_name + ".nc",
    )
    return data_file


class SpinqickData:
    """Self describing data object to handle data from a QICK AcquireProgramv2 output."""

    def __init__(
        self,
        raw_data: List[np.ndarray],
        cfg: pydantic.BaseModel,
        triggers: int,
        reps: int,
        experiment_name: str,
        analyzed_data: List[np.ndarray] | None = None,
        timestamp: int | None = None,
        filename: str | None = None,
        prog: qick_asm.AbsQickProgram | str | None = None,
        voltage_state: Dict[str, float] | None = None,
    ):
        self.raw_data = raw_data
        self.analyzed_data = analyzed_data
        self.analysis_type = ""
        self.analysis_averaged: spinqick_enums.AverageLevel | None = None
        self.prog = prog
        self.cfg = cfg
        self.timestamp = _assign_timestamp() if timestamp is None else timestamp
        self.axes: dict = {}
        self.experiment_name = experiment_name
        self.data_file = (
            _get_filename(self.timestamp, self.experiment_name) if filename is None else filename
        )
        self.triggers = triggers
        self.reps = reps
        self.cfg_class = cfg.__class__.__name__
        self.fit_param_dict: dict = {}
        self.best_fit: np.ndarray = np.array([])
        self.fit_axis: str = ""
        self.spinqick_version: str = importlib_metadata.version("spinqick")
        self._cfg = self.cfg.model_dump_json()
        self.voltage_state = voltage_state

    def add_full_average(self, avgs: int):
        """Data contains an outer loop for averaging."""
        self.axes["full_avgs"] = {"size": avgs, "loop_no": 0}

    def add_point_average(self, avgs: int, loop_no: int):
        """Data contains a point average loop.

        User supplies the loop number, where outermost loop is zero
        """
        self.axes["point_avgs"] = {"size": avgs, "loop_no": loop_no}

    def add_axis(
        self,
        data: List[np.ndarray],
        axis_name: str,
        sweep_names: Sequence[str],
        dim_size: int,
        loop_no: int = 0,
        units: List[str] | None = None,
    ):
        """Add information describing a swept variable/ set of variables in your dataset."""
        ax_dict = {}
        for i, sweep in enumerate(sweep_names):
            if units is None:
                unit_val = "NA"
            else:
                unit_val = units[i]
            ax_dict[sweep] = {"data": data[i], "units": unit_val}
        ax_dict["size"] = dim_size
        ax_dict["loop_no"] = loop_no
        self.axes[axis_name] = ax_dict

    def add_fit_params(self, param_dict: dict, best_fit: np.ndarray, fit_axis: str):
        """Add fit parameter attributes to the spinqick data object."""
        self.fit_param_dict = param_dict
        self.best_fit = best_fit
        self.fit_axis = fit_axis

    @staticmethod
    def get_sweep_vars(axis_dict: dict) -> dict:
        """Extract sweep variable entries from an axis dict.

        Sweep variables are entries whose values are dicts containing a ``"data"`` key.
        Metadata keys like ``"size"`` and ``"loop_no"`` are excluded.
        """
        return {k: v for k, v in axis_dict.items() if isinstance(v, dict) and "data" in v}

    def get_config_dict(self) -> dict:
        """Return config parameters as a plain dict, regardless of model type.

        After a save → load round-trip with an unknown config model the parameters
        end up nested under a ``"cfg"`` key. This accessor unwraps that layer so
        callers always get the original flat dict.
        """
        raw: dict = json.loads(self._cfg)
        # If the loader wrapped the original dict in a DynamicFakeConfig / InvalidConfig
        # the structure is {"cfg": {"param_a": ..., ...}}
        if list(raw.keys()) == ["cfg"] and isinstance(raw["cfg"], dict):
            return raw["cfg"]
        return raw

    def to_xarray(self) -> Any:
        """Convert to an :class:`xarray.Dataset`.

        Requires *xarray* to be installed (raises :class:`ImportError` otherwise).
        """
        import xarray as xr  # optional dependency

        coords: dict[str, np.ndarray] = {}
        for _axis_name, axis_dict in self.axes.items():
            for key, val in axis_dict.items():
                if isinstance(val, dict) and "data" in val:
                    coords[key] = np.asarray(val["data"])

        coord_keys = list(coords.keys())
        data_vars: dict[str, tuple] = {}
        for i, raw in enumerate(self.raw_data):
            data_vars[f"raw_data_{i}"] = (coord_keys + ["IQ"], raw)
        if self.analyzed_data:
            for i, ana in enumerate(self.analyzed_data):
                data_vars[f"analyzed_{i}"] = (coord_keys, ana)

        ds = xr.Dataset(data_vars, coords=coords)
        ds.attrs["experiment_name"] = self.experiment_name
        ds.attrs["timestamp"] = self.timestamp
        ds.attrs["cfg"] = json.dumps(self.get_config_dict())
        return ds

    def save_fit_params(self, nc_file: file_manager.SaveData):
        """Save parameters from a fit into a dict."""
        fit_grp = nc_file.createGroup("fits")
        if self.fit_param_dict is not None:
            nc_file.groups["fits"].setncatts(self.fit_param_dict)
        nc_file.add_dataset(
            "best_fit",
            axes=[self.fit_axis],
            data=self.best_fit,
            group_path=fit_grp.path,
        )

    def save_voltage_data(
        self,
        nc_file: file_manager.SaveData,
        nest_in_group: None | str = None,
    ):
        """Saves the all_voltages output from the hardware manager as json string."""
        vstate_json = json.dumps(self.voltage_state)
        if nest_in_group is not None:
            nc_file[nest_in_group].voltage_state = vstate_json
        else:
            nc_file.voltage_state = vstate_json
        return nc_file

    def load_to_fake_config(self, json_cfg):
        DynamicFakeConfig = pydantic.create_model("DynamicFakeConfig", cfg=dict)
        python_dict = json.loads(json_cfg)
        return DynamicFakeConfig(cfg=python_dict)


class PsbData(SpinqickData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.difference_data: List[np.ndarray] | None = None
        self.difference_avged: spinqick_enums.AverageLevel | None = None
        self.thresh_avged: spinqick_enums.AverageLevel | None = None
        self.threshed_data: List[np.ndarray] | None = None
        self.threshold: List[float] | None


class CompositeSpinqickData:
    """Stores a list of SpinqickData objects.

    This is designed for datasets which include sweeps
    """

    def __init__(
        self,
        qdata_array: Sequence[SpinqickData | PsbData],
        dset_labels: List[str],
        experiment_name: str,
        dset_coordinates: np.ndarray | None = None,
        dset_coordinate_units: str | None = None,
        analyzed_data: np.ndarray | None = None,
        timestamp: int | None = None,
        filename: str | None = None,
    ):
        self.qdata_array = qdata_array
        self.experiment_name = experiment_name
        self.spinqick_version: str = importlib_metadata.version("spinqick")
        self.analyzed_data = analyzed_data
        self.dset_coordinates = dset_coordinates
        self.dset_labels = dset_labels
        self.dset_coordinate_units = dset_coordinate_units
        self.timestamp = _assign_timestamp() if timestamp is None else timestamp
        self.data_file = (
            _get_filename(self.timestamp, self.experiment_name) if filename is None else filename
        )
        self.fit_param_dict: dict = {}
        self.best_fit: np.ndarray = np.array([])
        self.fit_axis: str = ""
