"""BEMResult data class for BEM simulation results."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bemkit.analysis._grid import detect_grid, reshape_to_grid


@dataclass(frozen=True)
class BEMResult:
    """Immutable container for BEM simulation results.

    Attributes
    ----------
    name : str
        Human-readable name (fieldpoint set name).
    frequencies : (nf,) float array
        Frequency bins in Hz.
    eval_points : (N, 3) float array
        Evaluation point coordinates in meters. NaN rows for dropped points.
    pressure_raw : (N, nf) complex array
        Raw pressure (velocity BC).
    pressure_acc : (N, nf) complex array
        Pressure normalized to acceleration BC.
    velocity_raw : (N, nf) complex array
        Raw velocity (velocity BC).
    velocity_acc : (N, nf) complex array
        Velocity normalized to acceleration BC.
    rho0 : float
        Air density (kg/m^3).
    speed_of_sound : float
        Speed of sound (m/s).
    source_path : Path or None
        Path to the numcalc/{variant}/ directory this was loaded from.
    """

    name: str
    frequencies: np.ndarray
    eval_points: np.ndarray
    pressure_raw: np.ndarray
    pressure_acc: np.ndarray
    velocity_raw: np.ndarray
    velocity_acc: np.ndarray
    rho0: float
    speed_of_sound: float
    source_path: Path | None = None

    # -- Factories --

    @classmethod
    def from_numcalc(cls, nc_dir, config, set_name=None):
        """Build a BEMResult directly from a numcalc/{variant}/ directory.

        Parameters
        ----------
        nc_dir : str or Path
            Path to numcalc/{variant}/.
        config : dict
            Model config with keys ``sr``, ``nbins``, and optionally
            ``air_density`` and ``speed_of_sound``.
        set_name : str or None
            Which fieldpoint set to extract.  If None, all points are returned.

        Returns
        -------
        BEMResult
        """
        from bemkit.analysis.simulation import SimulationData
        return SimulationData(nc_dir, config).result(set_name)

    @classmethod
    def load_variant(cls, model_dir, variant, config):
        """Load all fieldpoint sets for a variant directly from be.out.

        Parameters
        ----------
        model_dir : str or Path
            Model root directory.
        variant : str
            Variant name (= subdirectory of numcalc/).
        config : dict
            Model config (from config.toml).

        Returns
        -------
        dict[str, BEMResult]
            Keyed by set name.  Falls back to ``{"evalpoints": result}`` when
            no offsets file exists.
        """
        from bemkit.analysis.simulation import SimulationData
        nc_dir = Path(model_dir) / "numcalc" / variant
        sim = SimulationData(nc_dir, config)
        if not sim.set_names:
            return {"evalpoints": sim.result(None)}
        return {sn: sim.result(sn) for sn in sim.set_names}

    # -- Point masks and counts --

    @property
    def valid_mask(self):
        """(N,) bool array — True where eval_points are not NaN."""
        return ~np.any(np.isnan(self.eval_points), axis=1)

    @property
    def n_points(self):
        """Total number of evaluation points (including dropped)."""
        return len(self.eval_points)

    @property
    def n_valid(self):
        """Number of valid (non-NaN) evaluation points."""
        return int(self.valid_mask.sum())

    @property
    def n_freq(self):
        """Number of frequency bins."""
        return len(self.frequencies)

    # -- Data accessors --

    def data(self, quantity="pressure", norm="acc"):
        """Generic accessor for complex data arrays.

        Parameters
        ----------
        quantity : "pressure" or "velocity"
        norm : "acc" or "raw"

        Returns
        -------
        (N, nf) complex array
        """
        key = f"{quantity}_{norm}"
        mapping = {
            "pressure_acc": self.pressure_acc,
            "pressure_raw": self.pressure_raw,
            "velocity_acc": self.velocity_acc,
            "velocity_raw": self.velocity_raw,
        }
        if key not in mapping:
            raise ValueError(f"Unknown quantity/norm: {key!r}. "
                             f"Use {list(mapping.keys())}")
        return mapping[key]

    def magnitude(self, quantity="pressure", norm="acc"):
        """Absolute value of data. Returns (N, nf) float."""
        return np.abs(self.data(quantity, norm))

    def magnitude_db(self, quantity="pressure", norm="acc", ref=1.0):
        """Magnitude in dB. Returns (N, nf) float.

        Parameters
        ----------
        ref : float
            Reference value. ``20 * log10(|data| / ref)``.
        """
        mag = self.magnitude(quantity, norm)
        with np.errstate(divide="ignore"):
            return 20 * np.log10(mag / ref)

    def phase(self, quantity="pressure", norm="acc", unwrap=False):
        """Phase angle in radians. Returns (N, nf) float."""
        p = np.angle(self.data(quantity, norm))
        if unwrap:
            p = np.unwrap(p, axis=1)
        return p

    # -- Frequency helpers --

    def at_freq(self, freq_hz):
        """Index of the nearest frequency bin to ``freq_hz``."""
        return int(np.argmin(np.abs(self.frequencies - freq_hz)))

    def freq_slice(self, f_min=None, f_max=None):
        """Slice object for a frequency range.

        Parameters
        ----------
        f_min, f_max : float or None
            Bounds in Hz. None means no bound.
        """
        i0 = 0 if f_min is None else self.at_freq(f_min)
        i1 = self.n_freq if f_max is None else self.at_freq(f_max) + 1
        return slice(i0, i1)

    # -- Time domain --

    def impulse_response(self, quantity="pressure", norm="acc"):
        """Compute impulse response via IRFFT.

        Returns
        -------
        time_s : (M,) float array
            Time in seconds.
        ir : (N, M) float array
            Impulse response waveforms.
        """
        d = self.data(quantity, norm)
        ir = np.fft.irfft(d, axis=1)
        dt = 1.0 / (2 * self.frequencies[-1])  # sr = 2 * f_max
        time_s = np.arange(ir.shape[1]) * dt
        return time_s, ir

    # -- Cartesian coordinates --

    @property
    def x(self):
        """(N,) x coordinates."""
        return self.eval_points[:, 0]

    @property
    def y(self):
        """(N,) y coordinates."""
        return self.eval_points[:, 1]

    @property
    def z(self):
        """(N,) z coordinates."""
        return self.eval_points[:, 2]

    # -- Spherical coordinates --

    @property
    def r(self):
        """(N,) radial distance."""
        return np.sqrt(np.sum(self.eval_points**2, axis=1))

    @property
    def theta(self):
        """(N,) polar angle in radians (0=+z)."""
        r = self.r
        return np.arccos(np.clip(self.z / np.where(r > 0, r, 1), -1, 1))

    @property
    def phi(self):
        """(N,) azimuthal angle in radians."""
        return np.arctan2(self.y, self.x)

    @property
    def theta_deg(self):
        """(N,) polar angle in degrees."""
        return np.degrees(self.theta)

    @property
    def phi_deg(self):
        """(N,) azimuthal angle in degrees."""
        return np.degrees(self.phi)

    # -- Grid support --

    @property
    def grid_shape(self):
        """(nr, nc) if points form a grid, else None."""
        info = self._grid_info
        return info["shape"] if info is not None else None

    @property
    def is_grid(self):
        """True if evaluation points form a regular grid."""
        return self._grid_info is not None

    @property
    def _grid_info(self):
        """Cached grid detection result."""
        # Use object.__getattribute__ to bypass frozen check
        try:
            return object.__getattribute__(self, "_cached_grid_info")
        except AttributeError:
            info = detect_grid(self.eval_points)
            object.__setattr__(self, "_cached_grid_info", info)
            return info

    def as_grid(self, values):
        """Reshape a flat (N,) or (N,...) array into a 2D grid.

        Parameters
        ----------
        values : (N,) or (N, ...) array
            Flat values aligned with eval_points.

        Returns
        -------
        row2d, col2d, val2d : 2D arrays
            Grid coordinates and reshaped values.

        Raises
        ------
        ValueError
            If points don't form a grid.
        """
        info = self._grid_info
        if info is None:
            raise ValueError("Points do not form a regular grid")
        return reshape_to_grid(self.eval_points, values, info)

    # -- Point slicing --

    def __getitem__(self, idx):
        """Slice by point index, returning a new BEMResult.

        Parameters
        ----------
        idx : int, slice, array, or bool mask
            Point selector.
        """
        if isinstance(idx, (int, np.integer)):
            idx = [idx]

        return BEMResult(
            name=self.name,
            frequencies=self.frequencies,
            eval_points=self.eval_points[idx],
            pressure_raw=self.pressure_raw[idx],
            pressure_acc=self.pressure_acc[idx],
            velocity_raw=self.velocity_raw[idx],
            velocity_acc=self.velocity_acc[idx],
            rho0=self.rho0,
            speed_of_sound=self.speed_of_sound,
            source_path=self.source_path,
        )

    def __repr__(self):
        return (f"BEMResult({self.name!r}, {self.n_valid}/{self.n_points} points, "
                f"{self.n_freq} freqs)")


def load_result(model_dir, set_name=None):
    """Load a BEMResult from the first variant with results.

    Returns:
        (BEMResult, variant_name)

    Raises:
        FileNotFoundError: no variant with result data.
    """
    from bemkit.config import ModelConfig
    from bemkit.sim.discovery import discover_variants
    from bemkit.sim.packing import has_results

    model_dir = Path(model_dir)
    config = ModelConfig.from_dir(model_dir)
    for v in discover_variants(model_dir):
        nc_dir = model_dir / "numcalc" / v
        if has_results(nc_dir):
            try:
                return BEMResult.from_numcalc(nc_dir, config, set_name=set_name), v
            except (FileNotFoundError, ValueError):
                continue
    raise FileNotFoundError(f"No result data found in {model_dir}")
