"""NetCDF4 backend for SpinQICK data persistence.

This module contains the ``NetCDF4Handler`` which implements all save/load
logic previously embedded in :class:`~spinqick.core.spinqick_data.SpinqickData`,
:class:`~spinqick.core.spinqick_data.PsbData`, and
:class:`~spinqick.core.spinqick_data.CompositeSpinqickData`.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import importlib_metadata
import netCDF4
import numpy as np
import pydantic
from qick import helpers, qick_asm

from spinqick.backends import register_backend
from spinqick.backends.data_protocols import DataHandler
from spinqick.helper_functions import file_manager
from spinqick.models import experiment_models

if TYPE_CHECKING:
    from spinqick.core.spinqick_data import CompositeSpinqickData, PsbData, SpinqickData

logger = logging.getLogger(__name__)


def _load_analysis_data(nc_file: netCDF4.Dataset, data_desc: str, attr_name: str | None = None):
    """Load data stored in the 'analyzed_data' group of the netcdf file.

    This is a 1-to-1 copy of :func:`spinqick.core.spinqick_data.load_analysis_data`.
    """
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


class NetCDF4Handler(DataHandler):
    """DataHandler implementation using netCDF4 files.

    All save/load logic that was previously on SpinqickData, PsbData, and CompositeSpinqickData is
    collected here.
    """

    # -- public save methods --------------------------------------------------

    def save(self, data: SpinqickData) -> file_manager.SaveData:
        """Save a SpinqickData or PsbData object to a netCDF4 file.

        For PsbData, also writes difference and thresholded data when present.
        Returns the open :class:`~spinqick.helper_functions.file_manager.SaveData`
        handle (useful for ``save_last_plot``).
        """
        from spinqick.core.spinqick_data import PsbData

        nc_file = file_manager.SaveData(data.data_file, "a", format="NETCDF4")
        self._basic_save(data, nc_file)
        if isinstance(data, PsbData):
            if data.difference_data is not None:
                self._save_difference_data(data, nc_file)
            if data.threshed_data is not None:
                self._save_threshed_data(data, nc_file)
        logger.info("data saved at %s", data.data_file)
        return nc_file

    def save_composite(self, data: CompositeSpinqickData) -> file_manager.SaveData:
        """Save a CompositeSpinqickData object to a single netCDF4 file.

        Each sub-dataset is nested in its own group labelled by
        ``data.dset_labels``.
        """
        nc_file = file_manager.SaveData(data.data_file, "a", format="NETCDF4")
        for i, d in enumerate(data.qdata_array):
            self._basic_save(d, nc_file, nest_in_group=data.dset_labels[i])
        if data.dset_coordinates is not None:
            nc_file.add_axis("outer_sweep", data.dset_coordinates, units=data.dset_coordinate_units)
        if data.analyzed_data is not None:
            nc_file.add_dataset("analyzed_data", ["outer_sweep"], data.analyzed_data)
        nc_file.experiment_name = data.experiment_name
        nc_file.timestamp = data.timestamp
        logger.info("data saved at %s", data.data_file)
        return nc_file

    # -- public load methods --------------------------------------------------

    def load(self, identifier: str, load_psb: bool = False) -> SpinqickData:
        """Load a SpinqickData or PsbData from a netCDF4 file path.

        :param identifier: Path to the ``.nc`` file.
        :param load_psb: If ``True``, create a
            :class:`~spinqick.core.spinqick_data.PsbData` and load
            difference/thresholded data when present.
        """
        nc_file = netCDF4.Dataset(identifier, "r")
        try:
            data = self._load_spinqick_data(nc_file, load_psb=load_psb)
        finally:
            nc_file.close()
        return data

    def load_composite(
        self,
        identifier: str,
        load_psb: bool = False,
    ) -> CompositeSpinqickData:
        """Load a CompositeSpinqickData from a netCDF4 file path.

        :param identifier: Path to the ``.nc`` file.
        :param load_psb: If ``True``, load sub-datasets as PsbData.
        """
        nc_file = netCDF4.Dataset(identifier, "r")
        try:
            data = self._load_composite_data(nc_file, load_psb=load_psb)
        finally:
            nc_file.close()
        return data

    # -- internal save methods ------------------------------------------------

    def _basic_save(
        self,
        data: SpinqickData,
        nc_file: file_manager.SaveData,
        nest_in_group: str | None = None,
    ) -> file_manager.SaveData:
        """Save data from a SpinqickData object into a netCDF4 file/group.

        1-to-1 extraction of ``SpinqickData.basic_save()``.
        """
        if nest_in_group is None:
            nc_group: netCDF4.Group | file_manager.SaveData = nc_file
        else:
            nc_group = nc_file.createGroup(nest_in_group)
        nc_group.timestamp = data.timestamp
        nc_group.cfg_type = data.cfg_class
        nc_group.experiment_name = data.experiment_name
        data.spinqick_version = importlib_metadata.version("spinqick")
        nc_group.spinqick_version = data.spinqick_version
        ### start by saving raw data
        if data.axes:
            axis_group = "swept_variables"
            sweep_grp = nc_group.createGroup(axis_group)
            for axis, axis_dict in data.axes.items():
                dim_name = axis + "_dim"
                # Bug 2 fix: scope dimension check to current group, not root
                if dim_name in nc_group.dimensions:
                    continue
                else:
                    ax_grp = sweep_grp.createGroup(axis)
                    ax_grp.ax_dim = axis_dict["size"]
                    ax_grp.loop_no = axis_dict["loop_no"]
                    if axis == "full_avgs":
                        nc_group.createDimension("full_avgs_dim", axis_dict["size"])
                    elif axis == "point_avgs":
                        nc_group.createDimension("point_avgs_dim", axis_dict["size"])
                    else:
                        nc_file.add_multivariable_axis(axis, axis_dict, group_path=ax_grp.path)
        nc_group.createDimension("reps_dim", data.reps)
        nc_group.createDimension("triggers_dim", data.triggers)
        nc_group.createDimension("IQ_dim", 2)
        adc_ind = 0
        axes_names: list[str] = []
        for array in data.raw_data:
            axes_names = ["reps", "triggers"]
            sweep_ax_order = []
            sweep_ax_label = []
            # get the axes in the correct order before assigning
            for axis, axis_dict in data.axes.items():
                sweep_ax_order.append(axis_dict["loop_no"])
                sweep_ax_label.append(axis)
            ordered_ax_list = sweep_ax_label.copy()
            for k, i in enumerate(sweep_ax_order):
                ordered_ax_list[i] = sweep_ax_label[k]
            for ax_label in ordered_ax_list:
                axes_names.append(ax_label)
            axes_names.append("IQ")
            if nest_in_group is None:
                gp = None
            else:
                gp = nc_group.path
            nc_file.add_dataset(
                "raw_data_" + str(adc_ind),
                axes_names,
                array,
                units="adc_raw",
                group_path=gp,
            )
            adc_ind += 1
        if data.analyzed_data:
            ana_group = nc_group.createGroup("analyzed_data")
            adc_ind = 0
            if data.analysis_averaged is not None:
                ana_group.analysis_avged = data.analysis_averaged
            for array in data.analyzed_data:
                axes_names_analyzed = []
                for axis in axes_names:
                    # these axes need to be in the correct order
                    if axis == "point_avgs":
                        if data.analysis_averaged not in ["inner", "both"]:
                            axes_names_analyzed.append("point_avgs")
                    elif axis == "full_avgs":
                        if data.analysis_averaged not in ["outer", "both"]:
                            axes_names_analyzed.append("full_avgs")
                    else:
                        if axis != "IQ":
                            axes_names_analyzed.append(axis)
                nc_file.add_dataset(
                    "analyzed_" + str(adc_ind),
                    axes_names_analyzed,
                    array,
                    group_path=ana_group.path,
                    units=data.analysis_type,
                )
                adc_ind += 1
        if data.fit_param_dict:
            self._save_fit_params(data, nc_file)
        nc_group.cfg = data._cfg

        if isinstance(data.prog, str):
            nc_file.prog = data.prog
        elif isinstance(data.prog, qick_asm.AbsQickProgram):
            prog_dict = data.prog.dump_prog()
            prog_json = helpers.progs2json(prog_dict)
            nc_file.prog = prog_json
        if data.voltage_state is not None:
            self._save_voltage_data(data, nc_file, nest_in_group=nest_in_group)
        return nc_file

    def _save_fit_params(self, data: SpinqickData, nc_file: file_manager.SaveData) -> None:
        """Save fit parameters."""
        fit_grp = nc_file.createGroup("fits")
        if data.fit_param_dict is not None:
            nc_file.groups["fits"].setncatts(data.fit_param_dict)
        nc_file.add_dataset(
            "best_fit",
            axes=[data.fit_axis],
            data=data.best_fit,
            group_path=fit_grp.path,
        )

    def _save_voltage_data(
        self,
        data: SpinqickData,
        nc_file: file_manager.SaveData,
        nest_in_group: str | None = None,
    ) -> None:
        """Save voltage state as a JSON string attribute."""
        vstate_json = json.dumps(data.voltage_state)
        if nest_in_group is not None:
            nc_file[nest_in_group].voltage_state = vstate_json
        else:
            nc_file.voltage_state = vstate_json

    def _save_difference_data(self, data: PsbData, ncdf: file_manager.SaveData) -> None:
        """Save difference data.

        :meth:`_save_data_group`.
        """
        ana_group = ncdf["analyzed_data"]
        if data.difference_avged is not None:
            ana_group.difference_avged = data.difference_avged
        source_dims = ana_group.variables["analyzed_0"].dimensions
        assert data.difference_data is not None
        self._save_data_group(
            ncdf,
            ana_group,
            data.difference_data,
            source_dims,
            data.difference_avged,
            prefix="difference",
        )

    def _save_threshed_data(self, data: PsbData, ncdf: file_manager.SaveData) -> None:
        """Save thresholded data.

        :meth:`_save_data_group`.
        """
        ana_group = ncdf["analyzed_data"]
        if data.thresh_avged is not None:
            ana_group.thresh_avged = data.thresh_avged
        if data.difference_data is not None:
            source_dims = ana_group.variables["difference_0"].dimensions
        else:
            source_dims = ana_group.variables["analyzed_0"].dimensions
        assert data.threshed_data is not None
        self._save_data_group(
            ncdf,
            ana_group,
            data.threshed_data,
            source_dims,
            data.thresh_avged,
            prefix="threshed",
            thresholds=data.threshold,
        )

    @staticmethod
    def _filter_dims(
        source_dims: tuple[str, ...],
        avg_level: str | None,
        drop_triggers: bool = True,
    ) -> list[str]:
        """Build a filtered axis-name list from netCDF dimension names.

        Strips ``_dim`` suffixes and drops axes that have been averaged away
        (``point_avgs``/``full_avgs``) or are not relevant (``triggers``).
        """
        filtered: list[str] = []
        for dim in source_dims:
            ax = dim.removesuffix("_dim")
            if ax == "point_avgs" and avg_level in ("inner", "both"):
                continue
            if ax == "full_avgs" and avg_level in ("outer", "both"):
                continue
            if drop_triggers and ax == "triggers":
                continue
            filtered.append(ax)
        return filtered

    @staticmethod
    def _save_data_group(
        ncdf: file_manager.SaveData,
        ana_group: netCDF4.Group,
        data_arrays: list[np.ndarray],
        source_dims: tuple[str, ...],
        avg_level: str | None,
        prefix: str,
        thresholds: list[float] | None = None,
    ) -> None:
        """Write a list of arrays into the analyzed_data group.

        Shared implementation for difference and thresholded data saving.
        """
        axes_names = NetCDF4Handler._filter_dims(source_dims, avg_level, drop_triggers=True)
        for adc_ind, array in enumerate(data_arrays):
            dset_name = f"{prefix}_{adc_ind}"
            ncdf.add_dataset(
                dset_name,
                axes_names,
                array,
                group_path=ana_group.path,
            )
            if thresholds is not None:
                ana_group[dset_name].threshold = thresholds[adc_ind]

    # -- internal load methods ------------------------------------------------

    def _load_spinqick_data(
        self,
        nc_file: netCDF4.Dataset,
        load_psb: bool = False,
    ) -> SpinqickData:
        """Load data from a netCDF4 dataset into a SpinqickData or PsbData.

        1-to-1 extraction of ``SpinqickData.load_spinqick_data()`` and
        ``PsbData.load_spinqick_data()``.
        """
        from spinqick.core.spinqick_data import PsbData, SpinqickData

        cls = PsbData if load_psb else SpinqickData

        cfg_type = nc_file.cfg_type
        try:
            cfg_model = getattr(experiment_models, cfg_type)
            cfg = cfg_model.model_validate_json(nc_file.cfg)
        except AttributeError:
            logger.warning("no experiment model named %s, loading dict as a fake model", cfg_type)
            DynamicFakeConfig = pydantic.create_model("DynamicFakeConfig", cfg=dict)
            python_dict = json.loads(nc_file.cfg)
            cfg = DynamicFakeConfig(cfg=python_dict)
        except pydantic.ValidationError:
            logger.warning("unable to load config into pydantic model")
            InvalidConfig = pydantic.create_model("InvalidConfig", cfg=dict)
            python_dict = json.loads(nc_file.cfg)
            cfg = InvalidConfig(cfg=python_dict)

        data_keys = list(nc_file.variables.keys())
        raw = []
        analysis_type = ""
        axes: dict[str, Any] = {}
        for d in data_keys:
            if "raw_data" in d:
                data_array = np.asarray(nc_file[d][:])
                raw.append(data_array)
        if "analyzed_data" in nc_file.groups.keys():
            processed, analysis_type, ana_avg = _load_analysis_data(
                nc_file, "analyzed", "analysis_avged"
            )
        else:
            processed = None
            ana_avg = None
        if "swept_variables" in nc_file.groups.keys():
            sweep = nc_file["swept_variables"]
            sweep_axis_keys = list(sweep.groups.keys())
            for sw in sweep_axis_keys:
                ax_dict: dict[str, Any] = {}
                ax = sweep[sw]
                sweep_var_keys = list(ax.variables.keys())
                for sv in sweep_var_keys:
                    data_array = np.asarray(ax[sv][:])
                    ax_dict[sv] = {"data": data_array, "units": ax[sv].units}
                ax_dict["size"] = ax.ax_dim
                ax_dict["loop_no"] = ax.loop_no
                axes[sw] = ax_dict

        if hasattr(nc_file, "prog"):
            prog = nc_file.prog
        else:
            prog = None

        data_obj = cls(
            raw_data=raw,
            cfg=cfg,
            experiment_name=nc_file.experiment_name,
            reps=nc_file.dimensions["reps_dim"].size,
            triggers=nc_file.dimensions["triggers_dim"].size,
            analyzed_data=processed,
            filename=nc_file.filepath(),
            timestamp=nc_file.timestamp,
            prog=prog,
        )
        data_obj.analysis_type = analysis_type
        if ana_avg:
            data_obj.analysis_averaged = ana_avg[0]
        data_obj.spinqick_version = nc_file.spinqick_version
        data_obj.axes = axes
        if hasattr(nc_file, "voltage_state"):
            v_json = nc_file.voltage_state
            data_obj.voltage_state = json.loads(v_json)
        if "fits" in nc_file.groups.keys():
            fit_grp = nc_file["fits"]
            best_fit = fit_grp["best_fit"]
            best_fit_dim_label = best_fit.dimensions
            best_fit_dim = best_fit_dim_label[0][:-4]
            fitparams = fit_grp.ncattrs()
            fit_dict = {}
            for p in fitparams:
                fit_dict[p] = getattr(fit_grp, p)
            data_obj.add_fit_params(fit_dict, np.asarray(best_fit), best_fit_dim)

        # Load PsbData-specific fields
        if load_psb and isinstance(data_obj, PsbData) and "analyzed_data" in nc_file.groups.keys():
            processed_diff, _, _ = _load_analysis_data(nc_file, "difference")
            if processed_diff:
                data_obj.difference_data = processed_diff
                data_obj.difference_avged = (
                    nc_file["analyzed_data"].difference_avged
                    if hasattr(nc_file["analyzed_data"], "difference_avged")
                    else None
                )
            threshed, _, thresholds = _load_analysis_data(nc_file, "threshed", "threshold")
            if threshed:
                data_obj.threshed_data = threshed
                data_obj.threshold = thresholds
                data_obj.thresh_avged = (
                    nc_file["analyzed_data"].thresh_avged
                    if hasattr(nc_file["analyzed_data"], "thresh_avged")
                    else None
                )

        return data_obj

    def _load_composite_data(
        self,
        nc_file: netCDF4.Dataset,
        load_psb: bool = False,
    ) -> CompositeSpinqickData:
        """Load a CompositeSpinqickData from a netCDF4 dataset.

        1-to-1 extraction of ``CompositeSpinqickData.load_composite()``.
        """
        from spinqick.core.spinqick_data import CompositeSpinqickData

        sqd_instances = []
        dset_labels = []
        for grp in nc_file.groups:
            dset = nc_file[grp]
            sqd = self._load_spinqick_data(dset, load_psb=load_psb)
            sqd_instances.append(sqd)
            dset_labels.append(grp)
        if "outer_sweep" in nc_file.variables:
            coordinates = nc_file["outer_sweep"][:]
            units = nc_file["outer_sweep"].units
        else:
            coordinates = None
            units = None
        if "analyzed_data" in nc_file.variables:
            analyzed = nc_file["analyzed_data"]
        else:
            analyzed = None
        data_obj = CompositeSpinqickData(
            qdata_array=sqd_instances,
            dset_labels=dset_labels,
            experiment_name=nc_file.experiment_name,
            dset_coordinates=np.asarray(coordinates),
            dset_coordinate_units=units,
            analyzed_data=np.asarray(analyzed),
            filename=nc_file.filepath(),
            timestamp=nc_file.timestamp,
        )
        return data_obj


"Register netcdf4 handler as a useable data backend in SpinQICK."
register_backend("NetCDF4", NetCDF4Handler)
