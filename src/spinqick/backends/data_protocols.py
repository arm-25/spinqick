from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from spinqick.core.spinqick_data import CompositeSpinqickData, SpinqickData


class DataHandler(ABC):
    """Interface for persisting SpinqickData to a storage backend.

    Handles saving and loading of SpinqickData objects.
    """

    @abstractmethod
    def save(self, data: SpinqickData) -> Any:
        """Save a SpinqickData object.

        Returns a backend-specific handle.
        """
        ...

    @abstractmethod
    def save_composite(self, data: CompositeSpinqickData) -> Any:
        """Save a CompositeSpinqickData object."""
        ...

    @abstractmethod
    def load(self, identifier: str) -> SpinqickData:
        """Load a SpinqickData object by identifier (path, run_id, etc)."""
        ...

    # Shared helpers available to all backends:
    @staticmethod
    def get_sweep_vars(axis_dict: dict) -> dict:
        """Extract sweep variables from a flat axis dict."""
        return {k: v for k, v in axis_dict.items() if isinstance(v, dict) and "data" in v}


@runtime_checkable
class DataPlotter(Protocol):
    """Interface for visualizing SpinqickData."""

    def plot_1d(self, data: SpinqickData, x_key: str, **kwargs) -> Any:
        """Create a 1D plot from SpinqickData."""
        ...

    def plot_2d(self, data: SpinqickData, x_key: str, y_key: str, **kwargs) -> Any:
        """Create a 2D plot from SpinqickData."""
        ...

    def save_plot(self, handle: Any, plot: Any) -> None:
        """Attach a plot to a saved dataset."""
        ...


@dataclass
class DataBackend:
    """Combined handler + plotter."""

    name: str
    handler: DataHandler
    plotter: DataPlotter
