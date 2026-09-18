"""Backend registry with explicit availability checks."""
from rq1.backends.base import Backend


_BACKENDS = {
    "ms_graphrag": Backend("ms_graphrag", supports_indexing=True),
    "mock": Backend("mock", supports_indexing=True),
    "hipporag": Backend("hipporag", supports_indexing=True, implemented=False),
}


def get_backend(name: str) -> Backend:
    try:
        backend = _BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown backend: {name}") from None
    if not backend.implemented:
        raise NotImplementedError(f"Backend {name} is scaffolded but not implemented")
    return backend
