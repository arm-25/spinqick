"""Pluggable data backend registry for SpinQICK."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spinqick.backends.data_protocols import DataBackend

_BACKENDS: dict[str, type[DataBackend]] = {}


def register_backend(name: str, backend_cls: type[DataBackend]) -> None:
    """Register a backend class by name (case-insensitive).

    Anyone can call this function to add a new backend following the pattern in netcdf4_backend.py.
    Backends must implement the DataBackend interface. Backends do not need to be located in
    spinqick repository.
    """
    _BACKENDS[name.lower()] = backend_cls


def get_backend(name: str | None = None) -> DataBackend:
    """Get configured backend.

    Falls back to netcdf if not specified. Lookup is case-insensitive.
    """
    from spinqick.settings import file_settings

    name = name or file_settings.data_backend
    key = name.lower()
    if key not in _BACKENDS:
        raise KeyError(f"Unknown data backend {name!r}. Available: {list(_BACKENDS.keys())}")
    return _BACKENDS[key]()
