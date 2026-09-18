"""Backend capabilities exposed to the experiment runner."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    name: str
    supports_indexing: bool
    implemented: bool = True
