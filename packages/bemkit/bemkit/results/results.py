"""Results: solver-neutral access to a solved Model.

A thin layer over a :class:`~bemkit.results.store.Store`: it knows the shape
of the data (which quantity is which field-point set, which axis is
frequency) but nothing about how the numbers were produced — a NumCalc bundle
and an in-process backend both end up here.

``scaled_by`` exists because the BEM is linear: a frequency-dependent
excitation is a post-hoc scale on a unit-amplitude solve, not a re-solve
(see ``bemkit.model.conditions`` for why). It returns a *view* — the scale
factor is applied lazily on each read, so scaling never copies the
underlying arrays and stacks losslessly (``scaled_by(a).scaled_by(b)`` is the
same as ``scaled_by(a * b)``).
"""

import json
from pathlib import Path

import numpy as np

from bemkit.results.store import QUANTITIES, MemoryStore, NpzStore

#: npz keys that are metadata rather than a quantity array (see save/load).
_FREQ_KEY = "__frequencies__"
_OFFSET_PREFIX = "__offset__"


class Results:
    """Eval-point and boundary results for one solved Model.

    Args:
        store: a Store (``MemoryStore`` or ``NpzStore``) holding the raw
            quantity arrays (``pEvalGrid``, ``vEvalGrid``, ``pBoundary``,
            ``vBoundary`` — see ``bemkit.results.store.QUANTITIES``).
        frequencies: (nf,) Hz, ascending — must match the store's arrays'
            frequency axis.
        offsets: ``{set_name: (k,) int array}`` — row indices into the
            eval-grid quantities for each field-point set (see
            ``bemkit.solvers.fieldpoints.combine``).
        scale: optional (nf,) complex array. When given, every quantity read
            through this Results is multiplied by it along the frequency
            axis — the mechanism behind :meth:`scaled_by`. None means no
            scaling (equivalent to an all-ones spectrum, but skips the
            multiply).
    """

    def __init__(self, store, frequencies, offsets, scale=None):
        self.store = store
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.offsets = {name: np.asarray(idx, dtype=int)
                        for name, idx in offsets.items()}
        if scale is None:
            self.scale = None
        else:
            scale = np.asarray(scale, dtype=complex)
            if scale.shape != self.frequencies.shape:
                raise ValueError(
                    f"scale must have {len(self.frequencies)} values (one "
                    f"per frequency), got shape {scale.shape}")
            self.scale = scale

    # -- set bookkeeping ---------------------------------------------------

    @property
    def set_names(self):
        return sorted(self.offsets)

    def _rows(self, arr, name):
        if name is None:
            return arr
        if name not in self.offsets:
            available = ", ".join(self.set_names) or "none"
            raise KeyError(
                f"no field-point set {name!r} (available: {available})")
        return arr[self.offsets[name]]

    def _scaled(self, quantity):
        arr = self.store.get(quantity)
        if self.scale is None:
            return arr
        # arr is (N, nf) for pressure or (N, nf, 3) for velocity; the scale
        # always broadcasts along axis 1 regardless of how many axes follow.
        shape = [1] * arr.ndim
        shape[1] = len(self.scale)
        return arr * self.scale.reshape(shape)

    # -- eval-grid quantities, split by field-point set --------------------

    def pressure(self, name=None):
        """(N, nf) complex pressure. ``name=None`` returns every eval point."""
        return self._rows(self._scaled("pEvalGrid"), name)

    def velocity(self, name=None):
        """(N, nf, 3) complex velocity. ``name=None`` returns every eval point."""
        return self._rows(self._scaled("vEvalGrid"), name)

    # -- boundary (surface mesh) quantities ---------------------------------

    def boundary_pressure(self):
        """(M, nf) complex surface pressure."""
        return self._scaled("pBoundary")

    def boundary_velocity(self):
        """(M, nf) complex surface normal velocity."""
        return self._scaled("vBoundary")

    # -- linearity -----------------------------------------------------------

    def scaled_by(self, spectrum):
        """A view of this Results multiplied by ``spectrum`` per frequency.

        ``spectrum`` is (nf,), one complex value per frequency in
        ``self.frequencies``. Composes: ``r.scaled_by(a).scaled_by(b)`` reads
        the same as ``r.scaled_by(a * b)``.
        """
        spectrum = np.asarray(spectrum, dtype=complex)
        if spectrum.shape != self.frequencies.shape:
            raise ValueError(
                f"spectrum must have {len(self.frequencies)} values (one "
                f"per frequency), got shape {spectrum.shape}")
        combined = spectrum if self.scale is None else self.scale * spectrum
        return Results(self.store, self.frequencies, self.offsets, scale=combined)

    # -- materialisation / persistence ---------------------------------------

    def materialise(self):
        """Force every available quantity into memory (in place).

        Applies the current scale eagerly and replaces ``self.store`` with a
        ``MemoryStore`` of the result, then drops the now-redundant scale.
        Right after this call, reading is pure dict lookup — no file I/O and
        no per-read multiply.
        """
        arrays = {q: self._scaled(q) for q in self.store.available()}
        self.store = MemoryStore(arrays)
        self.scale = None
        return self

    def save(self, path):
        """Write quantities + frequencies + set offsets to a ``.npz``.

        ``path`` may be the file itself or a directory (writes
        ``results.npz`` inside it). The *effective* (scaled) quantities are
        written — reading the saved file back gives the same numbers
        :meth:`pressure`/:meth:`velocity`/etc. give now.
        """
        path = Path(path)
        if path.is_dir():
            path = path / "results.npz"
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {q: self._scaled(q) for q in self.store.available()}
        arrays[_FREQ_KEY] = self.frequencies
        for name, idx in self.offsets.items():
            arrays[f"{_OFFSET_PREFIX}{name}"] = idx
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path):
        """Load a Results from a ``.npz`` file, or a directory holding one
        (``results.npz``, as written by :meth:`save`/:meth:`from_beout`)."""
        path = Path(path)
        if path.is_dir():
            path = path / "results.npz"
        if not path.exists():
            raise FileNotFoundError(path)

        with np.load(path) as z:
            frequencies = z[_FREQ_KEY]
            offsets = {}
            quantities = {}
            for key in z.files:
                if key == _FREQ_KEY:
                    continue
                if key.startswith(_OFFSET_PREFIX):
                    offsets[key[len(_OFFSET_PREFIX):]] = z[key]
                else:
                    quantities[key] = z[key]
        return cls(MemoryStore(quantities), frequencies, offsets)

    @classmethod
    def from_beout(cls, workdir, frequencies, offsets):
        """Ingest a NumCalc ``be.out/`` (or packed ``be.npz``) into a Results.

        Reads every quantity in ``bemkit.results.store.QUANTITIES`` that is
        present (a model with no field points has no ``*EvalGrid`` files;
        that is not an error here), then saves the result as
        ``workdir/results.npz`` — so the ingest only has to happen once, even
        for a raw ASCII ``be.out/`` that would otherwise be re-parsed on
        every load.

        Missing bins (a partial run) come back as NaN, distinct from a bin
        that solved to an actual zero — see ``numcalckit.load_bins``.
        """
        from numcalckit import load_bins

        workdir = Path(workdir)
        frequencies = np.asarray(frequencies, dtype=float)
        nf = len(frequencies)

        arrays = {}
        for quantity in QUANTITIES:
            try:
                arrays[quantity] = load_bins(workdir, quantity, nf)
            except FileNotFoundError:
                continue

        results = cls(MemoryStore(arrays), frequencies, offsets)
        results.save(workdir / "results.npz")
        return results
