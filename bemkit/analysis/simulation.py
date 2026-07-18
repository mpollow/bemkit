"""Lazy accessor for NumCalc simulation results.

Provides numpy-like access to all data in a numcalc/{variant}/ directory
with lazy loading — files are read on first access and cached.

Usage::

    sim = SimulationData.from_model("models/sphcap_10")

    sim.p_onaxis              # (N, nf) complex — normalized pressure (default)
    sim.p_raw_onaxis          # (N, nf) complex — raw pressure (velocity BC)
    sim.v_onaxis              # (N, nf, 3) complex — normalized velocity (default)
    sim.v_raw_onaxis          # (N, nf, 3) complex — raw velocity (velocity BC)
    sim.nodes_onaxis          # (N, 3) coordinates
    sim.boundary_pressure()          # (M, nf) complex (normalized, default)
    sim.boundary_pressure(norm="raw")# (M, nf) complex (raw)
    sim.boundary_velocity()          # (M, nf) complex (normalized, default)
    sim.boundary_velocity(norm="raw")# (M, nf) complex (normal velocity, raw)
    sim.boundary_pressure_acc        # (M, nf) alias for boundary_pressure(norm="acc")
    sim.boundary_velocity_acc        # (M, nf) alias for boundary_velocity(norm="acc")
    sim.set_names                    # ['onaxis', 'plane_hor', ...]

    result = sim.result("onaxis")  # BEMResult for existing viz functions
"""

from functools import cached_property
from pathlib import Path

import numpy as np


class SimulationData:
    """Lazy accessor for all data in a numcalc/{variant}/ directory.

    Reads files on first access and caches them. Supports dynamic attribute
    access for fieldpoint sets: ``sim.p_<set>``, ``sim.v_<set>``,
    ``sim.nodes_<set>``.
    """

    def __init__(self, nc_dir, config):
        """
        Parameters
        ----------
        nc_dir : Path or str
            Path to ``numcalc/{variant}/`` directory.
        config : dict
            Model config with keys ``sr``, ``nbins``, and optionally
            ``air_density`` and ``speed_of_sound``.
        """
        self._nc_dir = Path(nc_dir)

        # Prefer the bundle's config snapshot (what was exported) over the live
        # config (current config.toml); warn if they drift. This keeps result
        # interpretation (frequencies, air properties) tied to the actual run.
        from bemkit.sim.config_snapshot import read_snapshot, warn_on_drift
        snapshot = read_snapshot(self._nc_dir)
        if snapshot is not None:
            warn_on_drift(snapshot, config, nc_dir=self._nc_dir)
            eff = snapshot
        else:
            eff = config
        self._rho0 = eff.get("air_density", 1.2)
        self._c0 = eff.get("speed_of_sound", 340.0)
        sr = eff["sr"]
        nbins = eff["nbins"]
        self.frequencies = np.linspace(0, sr / 2, nbins)[1:]

        # Prefer the packed be.npz when present; otherwise scan be.out/.
        # Quantities are cached decompressed on first access (an NpzFile
        # re-reads and decompresses the whole member on every __getitem__,
        # and keeping one open would hold a stale handle across re-runs).
        from bemkit.sim.status import available_bins
        npz_file = self._nc_dir / "be.npz"
        self._npz_path = npz_file if npz_file.exists() else None
        self._npz_cache = {}
        self.available_bins = available_bins(self._nc_dir)

        # Load offsets (tiny file, eager)
        from bemkit.sim.fieldpoints import load_offsets
        offsets_file = self._nc_dir / "eval_offsets.json"
        if offsets_file.exists():
            self._offsets = load_offsets(offsets_file)
            self.set_names = list(self._offsets.get("sets", {}).keys())
        else:
            self._offsets = None
            self.set_names = []

        # Per-set split caches
        self._pressure_sets = {}
        self._velocity_sets = {}
        self._nodes_sets = {}

    # -- Convenience factory --

    @classmethod
    def from_model(cls, model_dir, variant=None):
        """Create from a model directory, auto-discovering variant and config.

        Parameters
        ----------
        model_dir : Path or str
        variant : str or None
            If None, uses the first variant with ``be.out/``.
        """
        from bemkit.config import ModelConfig
        from bemkit.sim.discovery import discover_variants, numcalc_dir

        model_dir = Path(model_dir)
        config = ModelConfig.from_dir(model_dir)
        if variant is None:
            variants = discover_variants(model_dir, need_be_out=True)
            if not variants:
                raise FileNotFoundError(
                    f"No variants with be.out/ found in {model_dir / 'numcalc'}")
            variant = variants[0]
        nc_dir = numcalc_dir(model_dir, variant)
        obj = cls(nc_dir, config)
        obj._variant = variant
        return obj

    # -- Boundary (surface mesh) data --

    @cached_property
    def boundary_nodes(self):
        """(M, 3) float — surface mesh node coordinates in meters."""
        from bemkit.sim.readers import read_object_mesh_nodes
        return read_object_mesh_nodes(self._nc_dir)

    @cached_property
    def boundary_elements(self):
        """(T, 3) int32 — surface mesh triangle connectivity (0-indexed)."""
        from bemkit.sim.readers import read_object_mesh_elements
        return read_object_mesh_elements(self._nc_dir)

    @cached_property
    def _boundary_pressure_raw(self):
        """(M, nf) complex — surface pressure across all frequency bins (raw)."""
        return self._load_all_bins("pBoundary")

    @cached_property
    def _boundary_velocity_raw(self):
        """(M, nf) complex — surface normal velocity across all bins (raw).

        Note: vBoundary stores scalar normal velocity (not 3D vector),
        unlike vEvalGrid which stores full 3D velocity vectors.
        """
        return self._load_all_bins("vBoundary")

    def boundary_pressure(self, norm="acc"):
        """(M, nf) complex — surface pressure for all frequency bins.

        Parameters
        ----------
        norm : "acc" or "raw"
            "acc" (default): normalized ÷ (j·ρ₀·ω·A_vib), cached via boundary_pressure_acc.
            "raw": raw velocity BC output.
        """
        if norm == "raw":
            return self._boundary_pressure_raw
        return self.boundary_pressure_acc  # delegates to cached_property

    def boundary_velocity(self, norm="acc"):
        """(M, nf) complex — surface normal velocity for all frequency bins.

        Note: vBoundary stores scalar normal velocity (not 3D vector),
        unlike vEvalGrid which stores full 3D velocity vectors.

        Parameters
        ----------
        norm : "acc" or "raw"
            "acc" (default): same ÷ (j·ρ₀·ω·A_vib) normalization as pressure,
                giving v_acc = 1/(j·ρ₀·ω·A_vib) on the vibrating patch.
                Primarily useful for API symmetry; Z = p_acc/v_acc = p_raw/v_raw.
            "raw": raw BC output (≈1 on vibrating elements, ≈0 on rigid).
        """
        if norm == "raw":
            return self._boundary_velocity_raw
        return self.boundary_velocity_acc  # delegates to cached_property

    @cached_property
    def boundary_pressure_acc(self):
        """(M, nf) complex — surface pressure normalized to acceleration BC.

        Cached. Equivalent to ``boundary_pressure(norm="acc")``.
        """
        from bemkit.analysis.normalization import normalize_pressure
        return normalize_pressure(self._boundary_pressure_raw, self.frequencies,
                                  self._rho0, self.A_vib)

    @cached_property
    def boundary_velocity_acc(self):
        """(M, nf) complex — surface normal velocity normalized to acceleration BC.

        Applies the same ÷ (j·ρ₀·ω·A_vib) formula as pressure normalization.
        Cached. Equivalent to ``boundary_velocity(norm="acc")``.
        """
        from bemkit.analysis.normalization import normalize_pressure
        return normalize_pressure(self._boundary_velocity_raw, self.frequencies,
                                  self._rho0, self.A_vib)

    # -- Eval grid (combined, private) --

    @cached_property
    def _eval_nodes(self):
        """(N_total, 3) float — all eval points concatenated."""
        from bemkit.sim.readers import read_eval_grid_nodes
        return read_eval_grid_nodes(self._nc_dir)

    @cached_property
    def _eval_pressure(self):
        """(N_total, nf) complex — eval grid pressure across all bins."""
        return self._load_all_bins("pEvalGrid")

    @cached_property
    def _eval_velocity(self):
        """(N_total, nf, 3) complex — eval grid velocity across all bins."""
        return self._load_all_bins("vEvalGrid")

    # -- Vibrating area (for normalization) --

    @cached_property
    def A_vib(self):
        """Vibrating surface area in m^2, or None if no vibrating elements."""
        from bemkit.meshing.io import element_areas
        from bemkit.sim.readers import read_nc_inp_boundary_conditions
        bc = read_nc_inp_boundary_conditions(self._nc_dir / "NC.inp")
        vib_idx = bc.get("vibrating", np.array([], dtype=int))
        if len(vib_idx) == 0:
            return None
        return element_areas(self.boundary_nodes,
                             self.boundary_elements[vib_idx]).sum()

    # -- Per-set accessors --

    def pressure(self, set_name, norm="acc"):
        """(N, nf) complex — eval pressure for a fieldpoint set.

        Parameters
        ----------
        set_name : str or None
            Fieldpoint set name (e.g. 'onaxis'). None returns all eval points.
        norm : "acc" or "raw"
            "acc" (default): normalized to acceleration boundary condition.
            "raw": raw velocity boundary condition output.
        """
        if set_name is None:
            p_raw = self._eval_pressure
        else:
            if set_name not in self._pressure_sets:
                self._pressure_sets[set_name] = self._split_set(
                    self._eval_pressure, set_name)
            p_raw = self._pressure_sets[set_name]

        if norm == "raw":
            return p_raw

        from bemkit.analysis.normalization import normalize_pressure
        return normalize_pressure(p_raw, self.frequencies, self._rho0, self.A_vib)

    def velocity(self, set_name, norm="acc"):
        """(N, nf, 3) complex — eval velocity vectors for a fieldpoint set.

        Parameters
        ----------
        set_name : str or None
            Fieldpoint set name (e.g. 'onaxis'). None returns all eval points.
        norm : "acc" or "raw"
            "acc" (default): normalized to acceleration boundary condition.
            "raw": raw velocity boundary condition output.
        """
        if set_name is None:
            v_raw = self._eval_velocity
        else:
            if set_name not in self._velocity_sets:
                self._velocity_sets[set_name] = self._split_set(
                    self._eval_velocity, set_name)
            v_raw = self._velocity_sets[set_name]

        if norm == "raw":
            return v_raw

        from bemkit.analysis.normalization import normalize_pressure
        return normalize_pressure(v_raw, self.frequencies, self._rho0, self.A_vib)

    def nodes(self, set_name):
        """(N, 3) float — eval point coordinates for a fieldpoint set.

        Parameters
        ----------
        set_name : str or None
            Fieldpoint set name. None returns all eval points.
        """
        if set_name is None:
            return self._eval_nodes
        if set_name not in self._nodes_sets:
            self._nodes_sets[set_name] = self._split_set(
                self._eval_nodes, set_name)
        return self._nodes_sets[set_name]

    def result(self, set_name=None):
        """Build a BEMResult for a fieldpoint set (compatible with viz functions).

        Pressure normalization: ``pressure_acc = pressure_raw / (1j * rho0 * omega * A_vib)``

        Velocity is stored as scalar magnitude in BEMResult for shape
        compatibility with existing consumers. Use ``self.velocity(set_name)``
        for the full (N, nf, 3) vector.

        Parameters
        ----------
        set_name : str or None
            Fieldpoint set name. None returns all eval points unsplit.
        """
        from bemkit.analysis.result import BEMResult
        from bemkit.analysis.normalization import normalize_pressure

        if set_name is None:
            p_raw = self._eval_pressure
            pts = self._eval_nodes
            v_raw_vec = self._eval_velocity
        else:
            p_raw = self.pressure(set_name, norm="raw")
            pts = self.nodes(set_name)
            v_raw_vec = self.velocity(set_name, norm="raw")

        p_acc = normalize_pressure(p_raw, self.frequencies, self._rho0, self.A_vib)
        v_mag = np.linalg.norm(v_raw_vec, axis=-1)           # (N, nf) real magnitude
        v_acc = np.linalg.norm(                               # magnitude of normalized 3D velocity
            normalize_pressure(v_raw_vec, self.frequencies, self._rho0, self.A_vib),
            axis=-1)

        return BEMResult(
            name=set_name or "evalpoints",
            frequencies=self.frequencies,
            eval_points=pts,
            pressure_raw=p_raw,
            pressure_acc=p_acc,
            velocity_raw=v_mag,
            velocity_acc=v_acc,
            rho0=self._rho0,
            speed_of_sound=self._c0,
            source_path=self._nc_dir,
        )

    # -- Single-bin access (for interactive viewer) --

    def _normalize_bin(self, arr, bin_idx):
        """Acc-normalize a single-bin array via the shared normalize_pressure formula.

        ``arr`` is (N,) or (N, 3) for one frequency bin. normalize_pressure expects
        an nf axis, so insert a length-1 one and drop it again afterwards.
        """
        from bemkit.analysis.normalization import normalize_pressure
        freqs = self.frequencies[bin_idx - 1:bin_idx]  # length-1
        out = normalize_pressure(arr[:, None, ...], freqs, self._rho0, self.A_vib)
        return out[:, 0, ...]

    def pressure_at(self, bin_idx, domain="eval", set_name=None, norm="acc"):
        """Read pressure for a single frequency bin without loading all bins.

        Parameters
        ----------
        bin_idx : int
            1-based be.{i} index.
        domain : "eval" or "boundary"
        set_name : str or None
            Split eval result to this fieldpoint set. Ignored for boundary.
        norm : "acc" or "raw"
            "acc" (default): normalize to acceleration BC.
            "raw": return raw velocity-BC output.

        Returns
        -------
        (N,) complex array
        """
        filename = "pEvalGrid" if domain == "eval" else "pBoundary"
        if self._npz_path is not None:
            p = self._packed(filename)[:, bin_idx - 1]
        else:
            from bemkit.sim.readers import _read_numcalc_data
            p = _read_numcalc_data(
                self._nc_dir / "be.out" / f"be.{bin_idx}" / filename)
        if domain == "eval" and set_name is not None:
            p = self._split_set(p, set_name)
        if norm == "acc" and self.A_vib is not None:
            p = self._normalize_bin(p, bin_idx)
        return p

    def velocity_at(self, bin_idx, domain="eval", set_name=None, norm="acc"):
        """Read velocity for a single frequency bin without loading all bins.

        Parameters
        ----------
        bin_idx : int
            1-based be.{i} index.
        domain : "eval" or "boundary"
        set_name : str or None
            Split eval result to this fieldpoint set. Ignored for boundary.
        norm : "acc" or "raw"
            "acc" (default): normalize to acceleration BC.
            "raw": return raw velocity-BC output.

        Returns
        -------
        (N, 3) complex for eval (3D vector), (N,) complex for boundary (scalar normal)
        """
        filename = "vEvalGrid" if domain == "eval" else "vBoundary"
        if self._npz_path is not None:
            arr = self._packed(filename)
            v = arr[:, bin_idx - 1, :] if domain == "eval" else arr[:, bin_idx - 1]
        else:
            from bemkit.sim.readers import _read_numcalc_velocity_data, _read_numcalc_data
            if domain == "eval":
                v = _read_numcalc_velocity_data(
                    self._nc_dir / "be.out" / f"be.{bin_idx}" / "vEvalGrid")
            else:
                # vBoundary is scalar normal velocity, same format as pBoundary
                v = _read_numcalc_data(
                    self._nc_dir / "be.out" / f"be.{bin_idx}" / "vBoundary")
        if domain == "eval" and set_name is not None:
            v = self._split_set(v, set_name)
        if norm == "acc" and self.A_vib is not None:
            v = self._normalize_bin(v, bin_idx)
        return v

    # -- Dynamic attribute access --

    def __getattr__(self, name):
        """Support sim.p_<set>, sim.p_raw_<set>, sim.v_<set>, sim.v_raw_<set>, sim.nodes_<set> syntax."""
        prefixes = [
            ("p_raw_", lambda s: self.pressure(s, norm="raw")),
            ("p_", lambda s: self.pressure(s, norm="acc")),
            ("v_raw_", lambda s: self.velocity(s, norm="raw")),
            ("v_", lambda s: self.velocity(s, norm="acc")),
            ("nodes_", self.nodes),
        ]
        for prefix, method in prefixes:
            if name.startswith(prefix):
                set_name = name[len(prefix):]
                if set_name in self.set_names:
                    return method(set_name)
                available = ", ".join(self.set_names) or "(none)"
                raise AttributeError(
                    f"No fieldpoint set '{set_name}'. "
                    f"Available: {available}")
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'")

    def __dir__(self):
        attrs = list(super().__dir__())
        for s in self.set_names:
            attrs.extend([f"p_{s}", f"p_raw_{s}", f"v_{s}", f"v_raw_{s}", f"nodes_{s}"])
        return attrs

    def __repr__(self):
        variant = getattr(self, "_variant", self._nc_dir.name)
        sets = ", ".join(self.set_names) if self.set_names else "no offsets"
        n_bins = len(self.available_bins)
        n_freq = len(self.frequencies)
        loaded = []
        if "_eval_pressure" in self.__dict__:
            loaded.append("eval_p")
        if "_eval_velocity" in self.__dict__:
            loaded.append("eval_v")
        if "_boundary_pressure_raw" in self.__dict__:
            loaded.append("bnd_p")
        if "_boundary_velocity_raw" in self.__dict__:
            loaded.append("bnd_v")
        status = f", loaded=[{', '.join(loaded)}]" if loaded else ""
        return (f"SimulationData({variant!r}, {n_bins}/{n_freq} bins, "
                f"sets=[{sets}]{status})")

    # -- Internal helpers --

    def _packed(self, quantity):
        """Cached complex128 array for one quantity from be.npz.

        Opens the file per miss instead of holding an NpzFile for the object's
        lifetime, and caches the decompressed array so single-bin accessors
        (frequency-slider dragging) don't re-decompress the whole quantity
        on every call.
        """
        if quantity not in self._npz_cache:
            with np.load(self._npz_path, allow_pickle=False) as nz:
                if quantity not in nz.files:
                    raise FileNotFoundError(f"{quantity} not in {self._npz_path}")
                self._npz_cache[quantity] = nz[quantity].astype(
                    np.complex128, copy=False)
        return self._npz_cache[quantity]

    def _load_all_bins(self, filename):
        """Load a quantity across all frequency bins (be.npz preferred)."""
        if self._npz_path is not None:
            return self._packed(filename)
        from bemkit.sim.readers import load_bins
        return load_bins(self._nc_dir, filename, len(self.frequencies))

    def _split_set(self, combined, set_name):
        """Extract a fieldpoint set from combined eval array using offsets."""
        from bemkit.sim.fieldpoints import split_by_offsets

        if self._offsets is None:
            if set_name is not None:
                raise ValueError(
                    f"No eval_offsets.json — cannot split by set '{set_name}'")
            return combined

        if set_name not in self._offsets.get("sets", {}):
            available = ", ".join(self.set_names)
            raise KeyError(
                f"Set '{set_name}' not found. Available: {available}")

        splits = split_by_offsets(combined, self._offsets)
        return splits[set_name]
