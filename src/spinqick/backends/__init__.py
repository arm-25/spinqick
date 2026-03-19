"""Pluggable data backend registry for SpinQICK."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinqick.backends.data_protocols import DataBackend

_BACKENDS: dict[str, type[DataBackend]] = {}


def register_backend(name: str, backend_cls: type[DataBackend]) -> None:
    """Register a backend class by name."""
    _BACKENDS[name] = backend_cls


def get_backend(name: str | None = None) -> DataBackend:
    """Get configured backend.

    Falls back to netcdf if not specified.
    """
    from spinqick.settings import spinqick_settings

    name = name or spinqick_settings.data_backend or "netcdf"
    if name not in _BACKENDS:
        raise KeyError(f"Unknown data backend {name!r}. Available: {list(_BACKENDS.keys())}")
    return _BACKENDS[name]()
