"""
File for managing filter types/settings and applying them to waveforms.

Unit convention
~~~~~~~~~~~~~~~
``soccfg`` returns sampling frequencies in **MHz**.  ``FilterPath.apply()``
converts ``fs`` from MHz to **Hz** at the boundary so that all filter
internals and config values (e.g. Butterworth ``cutoff``) use base SI
units.  Delays are specified in **nanoseconds** in the config and
converted internally.

To add a filter: 
    1. Define new filter class in bottom section of the file (filters are alphabetized).
    2. In spinqick_enums.py add the new enum to FilterTypes
    3. Use that enum in filter_config to call on your filter
    Note: kwargs get passed through, so you can add what you need for any filter type and
    retroactively support more kwargs for existing filters if needed.
"""

import json
import logging
from abc import ABC
from pathlib import Path

import numpy as np
from scipy import signal

from spinqick.settings import file_settings

logger = logging.getLogger(__name__)

# Module-level cache — populated by load_filter_config() / build_filter_map()
_filter_config: dict = {"filters": {}, "paths": {}, "gate_scheme": {}}
_filter_map: dict[int, "FilterPath"] = {}


class Filter(ABC):
    """
    Class for managing filter types/settings and applying them to waveforms. 
    The settings are specified as keyword arguments during initialization.
    """

    def __init__(self, **kwargs):
        pass

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}({params})"
    
    def apply(self, wf, **kwargs):
        """Apply the filter to the input waveform."""
        pass


class FilterPath():
    """Class that manages the filtering path for waveforms."""

    def __init__(self, filter_conf: list[dict] = None):
        """
        Instantiates a Filter_Path object that applies the sequence of
        filters specified in the filter_config. 
        :param filter_conf: A list of dictionaries specifying the filters to apply and their settings.
        """
        self.filter_path = []
        if filter_conf is not None:
            for filter_dict in filter_conf:
                print("testing filter dict:")
                print(filter_dict)
                filter_obj = self._create_filter(filter_dict)
                self.filter_path.append(filter_obj)

    def __repr__(self) -> str:
        n = len(self.filter_path)
        header = f"Filter_Path({n} filter{'s' if n != 1 else ''}):"
        if n == 0:
            return f"{header} (empty)"
        steps = "\n".join(f"  {i+1}. {f!r}" for i, f in enumerate(self.filter_path))
        return f"{header}\n{steps}"
    
    def _create_filter(self, filter_dict: dict) -> Filter:
        """
        Takes the filter dictionary passed from the filter_config and instantiates a Filter class
        representing that filter.
        """
        from spinqick.helper_functions.spinqick_enums import FilterTypes
        params = filter_dict.copy()
        filter_type = params.pop("type")
        if filter_type in FilterTypes.__members__:
            filter_class = FilterTypes[filter_type].value
            return filter_class(**params)
        else:
            raise ValueError(f"Filter type {filter_type} not recognized.")

    def apply(self, wf, pre=None, post=None, **kwargs):
        """
        Applies the filter path to the waveform. It will optionally prepend and postpend
        repeats of the wf to mitigate edge effects. The number of repeats is determined by the length of the longest filter in the path.
        :param wf: The input waveform to filter.
        :param pre: The number of times to repeat the waveform before the original waveform. If None, no pre-padding will be applied before filtering.
        :param post: The number of times to repeat the waveform after the original waveform. If None, no post-padding will be applied before filtering.
        :param kwargs: Additional keyword arguments passed to each filter's apply method.
            ``fs`` is expected in MHz (as returned by soccfg) and is
            converted to Hz here so all filter internals and config
            values use base SI units.
        """
        # Convert fs from MHz (soccfg convention) to Hz (SI) at the boundary
        if "fs" in kwargs:
            kwargs = {**kwargs, "fs": kwargs["fs"] * 1e6}

        wf_extended = wf
        if pre is not None:
            for i in range(pre):
                wf_extended = np.concatenate((wf, wf_extended))
        if post is not None:
            for i in range(post):
                wf_extended = np.concatenate((wf_extended, wf))
        
        wf_filt = wf_extended
        for filter in self.filter_path:
            wf_filt = filter.apply(wf_filt, **kwargs)

        if pre is not None:
            pre_length = len(wf) * pre
        else:
            pre_length = 0
        if post is not None:
            post_length = -1 * len(wf) * post
            wf_final = wf_filt[pre_length:post_length]
        else:
            wf_final = wf_filt[pre_length:]

        return wf_final
    

def load_filter_config() -> dict:
    """Read the filter config JSON from disk into the module-level cache.

    Call this in DotExperiment.__init__ and update_local() so changes to
    filter_config.json are picked up at each experiment run.
    """
    global _filter_config
    if file_settings.filter_config is not None:
        _filter_config = json.loads(Path(file_settings.filter_config).read_text())
    else:
        _filter_config = {"filters": {}, "paths": {}, "gate_scheme": {}}
    logger.info("Loaded filter config with %d filters, %d paths, %d gate_scheme entries.",
                len(_filter_config.get("filters", {})),
                len(_filter_config.get("paths", {})),
                len(_filter_config.get("gate_scheme", {})))
    return _filter_config


def build_filter_map(hardware_config) -> dict[int, "FilterPath"]:
    """Build a mapping of qick_gen number → FilterPath from the cached filter config.

    Uses three levels of indirection from filter_config.json:
      - gate_scheme: maps gate names or gate types to a path name
      - paths: maps path names to ordered lists of filter names
      - filters: maps filter names to filter type + params

    Resolution priority for each channel: gate name in gate_scheme →
    gate_type in gate_scheme → skip. A gate_scheme value of "none" means
    no filtering for that gate.

    Call this after load_filter_config() and after hardware_config is loaded.
    The resulting map is cached at module level so add_predistorted_envelope
    can access it via get_filter_path(ch) without parameter threading.

    :param hardware_config: HardwareConfig instance with channel definitions.
    :return: dict mapping qick_gen int to FilterPath.
    """
    global _filter_map
    _filter_map = {}
    gate_scheme = _filter_config.get("gate_scheme", {})
    paths = _filter_config.get("paths", {})
    filters = _filter_config.get("filters", {})

    for gate_name, gate_cfg in hardware_config.channels.items():
        if not hasattr(gate_cfg, "qick_gen"):
            continue

        gen = gate_cfg.qick_gen
        gate_type = str(gate_cfg.gate_type) if hasattr(gate_cfg, "gate_type") else None

        # Resolve gate_scheme: gate name → gate_type → skip
        path_name = None
        if str(gate_name) in gate_scheme:
            path_name = gate_scheme[str(gate_name)]
        elif gate_type is not None and gate_type in gate_scheme:
            path_name = gate_scheme[gate_type]

        if path_name is None or path_name == "none":
            logger.debug("No filter path for gen %d (%s).", gen, gate_name)
            continue

        if path_name not in paths:
            logger.warning("gate_scheme references path '%s' for %s, but it is not defined in paths.",
                           path_name, gate_name)
            continue

        filter_list = [filters[name].copy() for name in paths[path_name]]
        _filter_map[gen] = FilterPath(filter_conf=filter_list)
        logger.info("Filter path '%s' assigned to gen %d (%s).", path_name, gen, gate_name)

    return _filter_map


def get_filter_path(ch: int) -> "FilterPath | None":
    """Look up the cached FilterPath for a given qick generator channel.

    :param ch: qick generator number.
    :return: FilterPath if one is configured, None otherwise.
    """
    return _filter_map.get(ch)

### ------------- Filters defined below ------------- ###

class FilterButterworth(Filter):
    """
    Class for applying a Butterworth filter to a waveform.
    """

    def __init__(self, order=5, cutoff=0.5, zero_phase=True):
        self.order = order
        self.cutoff = cutoff
        self.zero_phase = zero_phase

    def apply(self, wf, **kwargs):
        """Apply the Butterworth filter to the input waveform.

        :param wf: The input waveform.
        :param fs: Sampling frequency in Hz (required).  Converted from
            MHz by :meth:`FilterPath.apply` before reaching here.
        """
        fs = kwargs.get("fs")
        if fs is None:
            raise ValueError("FilterButterworth.apply requires fs (sampling frequency in Hz).")
        b, a = signal.butter(self.order, self.cutoff, fs=fs)
        if self.zero_phase:
            return signal.filtfilt(b, a, wf)
        else:
            return signal.lfilter(b, a, wf)


class FilterFIR(Filter):
    """Class for applying an FIR filter to a waveform."""

    def __init__(self, taps):
        self.taps = taps

    def apply(self, wf, **kwargs):
        """Apply the FIR filter to the input waveform."""
        return signal.convolve(wf, self.taps, mode="same")


class FilterFractionalDelay(Filter):
    """Shift a waveform by a sub-sample amount using a windowed-sinc FIR.

    This implements the sub-sample time shifting technique used in qubit
    shuttling experiments (van Riggelen-Doelman et al., Nat. Commun. 15,
    5716 (2024)), generalized from edge-only interpolation to arbitrary
    waveform shapes via a fractional-delay FIR filter.

    The delay is specified in nanoseconds in the filter config.  At apply
    time the DAC sampling frequency ``fs`` (in MHz) is used to convert to
    fractional samples.

    :param delay_ns: Desired time shift in nanoseconds.  Positive values
        delay the waveform (shift right); negative values advance it
        (shift left).
    :param num_taps: Number of FIR taps (forced odd).  More taps give a
        more accurate shift at the cost of needing more edge padding.
        Default 21.
    :param window: Window function name accepted by
        ``scipy.signal.get_window``.  Default ``"hamming"``.
    """

    def __init__(self, delay_ns: float = 0.0, num_taps: int = 21, window: str = "hamming"):
        self.delay_ns = delay_ns
        # Ensure odd tap count so the kernel is symmetric about its centre
        self.num_taps = num_taps if num_taps % 2 == 1 else num_taps + 1
        self.window = window

    def apply(self, wf, **kwargs):
        """Apply the fractional delay to the input waveform.

        :param wf: The input waveform array.
        :param fs: Sampling frequency in Hz (required).  Converted from
            MHz by :meth:`FilterPath.apply` before reaching here.
        """
        fs = kwargs.get("fs")
        if fs is None:
            raise ValueError(
                "FilterFractionalDelay.apply requires fs (sampling frequency in Hz)."
            )

        # Convert nanosecond delay to fractional samples.
        # delay_samples = delay_ns × 1e-9 s/ns × fs Hz = delay_ns × fs × 1e-9
        delay_samples = self.delay_ns * 1e-9 * fs

        # Build windowed-sinc kernel centred on the fractional delay
        M = self.num_taps // 2
        n = np.arange(self.num_taps) - M  # -M … 0 … +M
        kernel = np.sinc(n - delay_samples) * signal.get_window(self.window, self.num_taps)
        kernel /= kernel.sum()  # normalise to preserve DC gain

        return signal.convolve(wf, kernel, mode="same")


class FilterIIR(Filter):
    """Class for applying an IIR filter to a waveform."""

    def __init__(self, b=None, a=None):
        self.b = b
        self.a = a

    def apply(self, wf, **kwargs):
        """Apply the IIR filter to the input waveform."""
        return signal.lfilter(self.b, self.a, wf)


class FilterTukey(Filter):
    """Class for applying a Tukey window to a waveform."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def apply(self, wf, **kwargs):
        """Apply the Tukey window to the input waveform."""
        return wf * signal.windows.tukey(len(wf), self.alpha)