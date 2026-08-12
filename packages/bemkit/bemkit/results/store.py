"""Where result arrays live.

Three cases, one interface. `.npz` is the solver-neutral format: NumCalc's
``be.out/`` is its private output and gets ingested into one of these, while an
in-process backend writes `.npz` directly and never has a ``be.out/``.
"""

from pathlib import Path

import numpy as np

#: Quantity name -> is it a 3-vector per point?
QUANTITIES = {"pBoundary": False, "vBoundary": False,
              "pEvalGrid": False, "vEvalGrid": True}


class Store:
    """Base class. get(name) -> ndarray; available() -> list[str]."""

    def get(self, quantity):
        raise NotImplementedError

    def available(self):
        raise NotImplementedError

    def save(self, path):
        """Write every available quantity to a .npz."""
        np.savez_compressed(path, **{q: self.get(q) for q in self.available()})


class MemoryStore(Store):
    """Arrays already in RAM — an in-process solve with no workdir."""

    def __init__(self, arrays):
        self._arrays = dict(arrays)

    def get(self, quantity):
        if quantity not in self._arrays:
            raise KeyError(
                f"no {quantity!r} in this result "
                f"(have: {', '.join(sorted(self._arrays)) or 'nothing'})")
        return self._arrays[quantity]

    def available(self):
        return sorted(self._arrays)


class NpzStore(Store):
    """Lazily read from a .npz, cached after first access."""

    def __init__(self, path):
        self.path = Path(path)
        self._cache = {}

    def get(self, quantity):
        if quantity not in self._cache:
            with np.load(self.path) as z:
                if quantity not in z:
                    raise KeyError(
                        f"no {quantity!r} in {self.path} "
                        f"(have: {', '.join(sorted(z.files))})")
                self._cache[quantity] = z[quantity]
        return self._cache[quantity]

    def available(self):
        with np.load(self.path) as z:
            return sorted(z.files)
