"""Tests for bemkit.config — ModelConfig, freq_bin."""

import numpy as np
import pytest

from bemkit.config import freq_bin, ModelConfig


# ── ModelConfig.freq_array ──────────────────────────────────────────────────

class TestFreqArray:
    def _mc(self, tmp_path, sr, nbins):
        (tmp_path / "config.toml").write_text(f"[bem]\nsr = {sr}\nnbins = {nbins}\n")
        return ModelConfig.from_dir(tmp_path)

    def test_basic(self, tmp_path):
        freqs = self._mc(tmp_path, 48000, 65).freq_array()
        assert len(freqs) == 64
        assert freqs[0] == pytest.approx(48000 / 2 / 64)  # first bin after DC
        assert freqs[-1] == pytest.approx(24000)

    def test_excludes_dc(self, tmp_path):
        freqs = self._mc(tmp_path, 16000, 9).freq_array()
        # linspace(0, 8000, 9) = [0, 1000, ..., 8000], drop DC → 8 bins
        assert len(freqs) == 8
        assert freqs[0] == pytest.approx(1000)

    def test_single_bin(self, tmp_path):
        freqs = self._mc(tmp_path, 1000, 2).freq_array()
        assert len(freqs) == 1
        assert freqs[0] == pytest.approx(500)

    def test_spacing_is_uniform(self, tmp_path):
        freqs = self._mc(tmp_path, 48000, 65).freq_array()
        diffs = np.diff(freqs)
        np.testing.assert_allclose(diffs, diffs[0])


# ── ModelConfig validation ──────────────────────────────────────────────────

class TestModelConfigValidation:
    def test_missing_required_keys_raises_with_filename(self, tmp_path):
        (tmp_path / "config.toml").write_text("modelname = 'x'\n")
        with pytest.raises(ValueError, match=r"config\.toml.*sr.*nbins"):
            ModelConfig.from_dir(tmp_path)

    def test_missing_nbins_only(self, tmp_path):
        (tmp_path / "config.toml").write_text("[bem]\nsr = 48000\n")
        with pytest.raises(ValueError, match="nbins"):
            ModelConfig.from_dir(tmp_path)

    def test_old_flat_keys_raise_migration_error(self, tmp_path):
        (tmp_path / "config.toml").write_text("sr = 48000\nnbins = 65\n")
        with pytest.raises(ValueError, match=r"moved into the.*\[bem\]"):
            ModelConfig.from_dir(tmp_path)

    def test_unknown_bem_key_raises(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            "[bem]\nsr = 48000\nnbins = 65\nsamplerate = 1\n")
        with pytest.raises(ValueError, match="samplerate"):
            ModelConfig.from_dir(tmp_path)

    def test_old_fieldpoints_string_raises(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            'fieldpoints = "fieldpoints"\n'
            '[bem]\nsr = 48000\nnbins = 65\n')
        with pytest.raises(ValueError, match="old directory form"):
            ModelConfig.from_dir(tmp_path)

    def test_old_fieldpoints_path_value_raises(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            '[bem]\nsr = 48000\nnbins = 65\n'
            '[fieldpoints]\nonaxis = "fieldpoints/onaxis.npy"\n')
        with pytest.raises(ValueError, match=r'file = "fieldpoints/onaxis\.npy"'):
            ModelConfig.from_dir(tmp_path)

    def test_fieldpoints_set_needs_type_or_file(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            '[bem]\nsr = 48000\nnbins = 65\n'
            '[fieldpoints.onaxis]\ndim = "z"\n')
        with pytest.raises(ValueError, match="exactly one of type/file"):
            ModelConfig.from_dir(tmp_path)

    def test_fieldpoints_unknown_type_raises(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            '[bem]\nsr = 48000\nnbins = 65\n'
            '[fieldpoints.onaxis]\ntype = "spiral"\n')
        with pytest.raises(ValueError, match="spiral"):
            ModelConfig.from_dir(tmp_path)

    def test_unknown_mesh_key_raises(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            "[bem]\nsr = 48000\nnbins = 65\n[mesh]\nlc_mni = 5.0\n")
        with pytest.raises(ValueError, match="lc_mni"):
            ModelConfig.from_dir(tmp_path)

    def test_unknown_distance_mode_raises(self, tmp_path):
        # A key-only whitelist would let a typo through and silently fall back
        # to euclidean sizing.
        (tmp_path / "config.toml").write_text(
            '[bem]\nsr = 48000\nnbins = 65\n[mesh]\ndistance = "geodeisc"\n')
        with pytest.raises(ValueError, match="geodeisc"):
            ModelConfig.from_dir(tmp_path)

    def test_known_distance_modes_accepted(self, tmp_path):
        for mode in ("euclidean", "geodesic"):
            (tmp_path / "config.toml").write_text(
                f'[bem]\nsr = 48000\nnbins = 65\n[mesh]\ndistance = "{mode}"\n')
            assert ModelConfig.from_dir(tmp_path).mesh_cfg["distance"] == mode

    def test_variant_unknown_key_raises(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            '[bem]\nsr = 48000\nnbins = 65\n'
            '[variants.fine]\nmeshh = "x"\n')
        with pytest.raises(ValueError, match="meshh"):
            ModelConfig.from_dir(tmp_path)

    def test_unknown_user_table_is_allowed(self, tmp_path):
        # Free-form tables (e.g. [cad] read by the model's own scripts) pass.
        (tmp_path / "config.toml").write_text(
            "[bem]\nsr = 48000\nnbins = 65\n[cad]\nradius = 50\n")
        assert ModelConfig.from_dir(tmp_path)["cad"] == {"radius": 50}


# ── freq_bin ─────────────────────────────────────────────────────────────────

class TestFreqBin:
    def test_exact_match(self):
        # sr=16000, nbins=9 → freqs = [1000, 2000, ..., 8000]
        idx, be_idx, actual = freq_bin(3000, 16000, 9)
        assert idx == 2
        assert be_idx == 3
        assert actual == pytest.approx(3000)

    def test_snaps_to_nearest(self):
        idx, be_idx, actual = freq_bin(3100, 16000, 9)
        assert actual == pytest.approx(3000)
        idx2, _, actual2 = freq_bin(3600, 16000, 9)
        assert actual2 == pytest.approx(4000)

    def test_first_bin(self):
        idx, be_idx, _ = freq_bin(0, 16000, 9)
        assert idx == 0
        assert be_idx == 1

    def test_last_bin(self):
        idx, be_idx, actual = freq_bin(99999, 16000, 9)
        assert idx == 7
        assert be_idx == 8
        assert actual == pytest.approx(8000)


# ── ModelConfig.from_dir / dict-style access ────────────────────────────────

class TestModelConfigFromDir:
    def test_loads_vars_toml(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            '[bem]\nsr = 48000\nnbins = 65\n'
        )
        cfg = ModelConfig.from_dir(tmp_path)
        assert cfg["sr"] == 48000
        assert cfg["nbins"] == 65

    def test_modelname_defaults_to_dirname(self, tmp_path):
        d = tmp_path / "my_model"
        d.mkdir()
        (d / "config.toml").write_text('[bem]\nsr = 48000\nnbins = 65\n')
        cfg = ModelConfig.from_dir(d)
        assert cfg["modelname"] == "my_model"
        assert cfg.modelname == "my_model"

    def test_explicit_modelname(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            'modelname = "custom"\n[bem]\nsr = 48000\nnbins = 65\n'
        )
        cfg = ModelConfig.from_dir(tmp_path)
        assert cfg["modelname"] == "custom"
        assert cfg.modelname == "custom"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ModelConfig.from_dir(tmp_path / "nonexistent")

    def test_dict_style_get_and_contains(self, tmp_path):
        (tmp_path / "config.toml").write_text('[bem]\nsr = 48000\nnbins = 65\n')
        cfg = ModelConfig.from_dir(tmp_path)
        assert cfg.get("missing", 42) == 42
        assert cfg.get("sr") == 48000
        assert "sr" in cfg
        assert "missing" not in cfg


# ── ModelConfig ──────────────────────────────────────────────────────────────

class TestModelConfig:
    @pytest.fixture()
    def model_dir(self, tmp_path):
        d = tmp_path / "test_model"
        d.mkdir()
        (d / "config.toml").write_text(
            '[bem]\nsr = 32000\nnbins = 33\nair_density = 1.25\n'
            'speed_of_sound = 343.0\nbem_type = 1\n'
        )
        return d

    def test_from_dir(self, model_dir):
        mc = ModelConfig.from_dir(model_dir)
        assert mc.sr == 32000
        assert mc.nbins == 33
        assert mc.air_density == 1.25
        assert mc.speed_of_sound == 343.0
        assert mc.bem_type == 1
        assert mc.modelname == "test_model"

    def test_defaults(self, tmp_path):
        (tmp_path / "config.toml").write_text('[bem]\nsr = 48000\nnbins = 65\n')
        mc = ModelConfig.from_dir(tmp_path)
        assert mc.air_density == 1.2
        assert mc.speed_of_sound == 340.0
        assert mc.bem_type == 4
        assert mc.time_factor == -1.0
        assert mc.is_internal == 0

    def test_freq_array_method(self, model_dir):
        mc = ModelConfig.from_dir(model_dir)
        freqs = mc.freq_array()
        assert len(freqs) == 32
        assert freqs[-1] == pytest.approx(16000)

    def test_mesh_map_empty(self, tmp_path):
        (tmp_path / "config.toml").write_text('[bem]\nsr = 48000\nnbins = 65\n')
        mc = ModelConfig.from_dir(tmp_path)
        assert mc.mesh_map() == {}

    def test_mesh_map_with_variants(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            '[bem]\nsr = 48000\nnbins = 65\n'
            '[variants.left]\nmesh = "shared"\n'
            '[variants.right]\n'
        )
        mc = ModelConfig.from_dir(tmp_path)
        m = mc.mesh_map()
        assert m == {"left": "shared", "right": "right"}

    def test_variant_mesh_overrides(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            '[bem]\nsr = 48000\nnbins = 65\n'
            '[mesh]\nlc_min = 1.5\nlc_max = 3.5\ndistance = "euclidean"\n'
            '[variants.fine]\nsource = "box"\n'
            'mesh = { lc_min = 1.0, distance = "geodesic" }\n'
            '[variants.coarse]\nsource = "box"\n'
            '[variants.left]\nmesh = "shared"\n'
        )
        mc = ModelConfig.from_dir(tmp_path)
        # generated variants mesh to their own name; trusted keeps the stem
        assert mc.mesh_map() == {"fine": "fine", "coarse": "coarse",
                                 "left": "shared"}
        assert mc.variant_mesh_params("fine") == {
            "lc_min": 1.0, "lc_max": 3.5, "distance": "geodesic"}
        assert mc.variant_mesh_params("coarse") == {
            "lc_min": 1.5, "lc_max": 3.5, "distance": "euclidean"}
        assert mc.variant_mesh_params("left") is None  # trusted mesh file
        assert mc.variant_source("fine") == "box"
        assert mc.variant_source("left") is None


class TestDictAccessFallback:
    """Dict-style access falls back to typed defaults (no raw/attribute divergence)."""

    def test_get_falls_back_to_dataclass_default(self, tmp_path):
        (tmp_path / "config.toml").write_text("[bem]\nsr = 48000\nnbins = 65\n")
        cfg = ModelConfig.from_dir(tmp_path)
        assert cfg.get("air_density") == cfg.air_density == 1.2
        assert cfg["speed_of_sound"] == 340.0
        assert "bem_type" in cfg

    def test_unknown_key_still_raises(self, tmp_path):
        (tmp_path / "config.toml").write_text("[bem]\nsr = 48000\nnbins = 65\n")
        cfg = ModelConfig.from_dir(tmp_path)
        assert cfg.get("no_such_key", 7) == 7
        with pytest.raises(KeyError):
            cfg["no_such_key"]
