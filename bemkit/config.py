"""Configuration for NumCalc BEM simulations.

ModelConfig is the single configuration object of the toolkit. It also
supports dict-style access (``config["sr"]``, ``config.get(...)``) so code
paths that historically took the raw config.toml dict keep working.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelConfig:
    """Typed wrapper around a model directory's config.toml."""

    model_dir: Path
    modelname: str
    sr: int
    nbins: int
    air_density: float = 1.2
    speed_of_sound: float = 340.0
    bem_type: int = 4
    time_factor: float = -1.0
    is_internal: int = 0
    mesh_cfg: dict = field(default_factory=dict)
    fieldpoints_cfg: dict | str = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dir(cls, model_dir):
        """Load config.toml; modelname defaults to the directory name."""
        model_dir = Path(model_dir).resolve()
        with open(model_dir / "config.toml", "rb") as f:
            raw = tomllib.load(f)
        missing = [k for k in ("sr", "nbins") if k not in raw]
        if missing:
            raise ValueError(
                f"{model_dir / 'config.toml'} is missing required "
                f"key(s): {', '.join(missing)}")
        raw.setdefault("modelname", model_dir.name)
        return cls(
            model_dir=model_dir,
            modelname=raw["modelname"],
            sr=raw["sr"],
            nbins=raw["nbins"],
            air_density=raw.get("air_density", 1.2),
            speed_of_sound=raw.get("speed_of_sound", 340.0),
            bem_type=raw.get("bem_type", 4),
            time_factor=raw.get("time_factor", -1.0),
            is_internal=raw.get("is_internal", 0),
            mesh_cfg=raw.get("mesh", {}),
            fieldpoints_cfg=raw.get("fieldpoints", {}),
            raw=raw,
        )

    # -- dict-style access (backward compatible with raw-dict callers) --
    #
    # Lookups consult config.toml first, then fall back to the typed dataclass
    # fields, so config.get("air_density") returns the default 1.2 even when
    # config.toml omits it — dict access never diverges from attribute access.

    def __getitem__(self, key):
        if key in self.raw:
            return self.raw[key]
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key):
        return key in self.raw or key in self.__dataclass_fields__

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    # -- helpers --

    def freq_array(self):
        """Frequency array excluding DC, shape (nbins-1,)."""
        return np.linspace(0, self.sr / 2, self.nbins)[1:]

    def variants(self, prefixes=None):
        """Discover variant names from [variants] table or mesh/ directory."""
        from bemkit.sim.discovery import discover_variants
        return discover_variants(self.model_dir, prefixes)

    def mesh_map(self):
        """Return {variant: mesh_stem} from [variants] table, or {}."""
        tbl = self.raw.get("variants") or {}
        return {n: v.get("mesh", n) for n, v in tbl.items()}


def freq_bin(freq_hz, sr, nbins):
    """Return (0-based index, 1-based be.out index, actual_freq_hz) of nearest bin."""
    freqs = np.linspace(0, sr / 2, nbins)[1:]
    idx = int(np.argmin(np.abs(freqs - freq_hz)))
    return idx, idx + 1, freqs[idx]
