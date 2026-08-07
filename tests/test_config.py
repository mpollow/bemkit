"""Tests for bemkit.config — ModelConfig, freq_bin, freq_grid, resolve_bins."""

import numpy as np
import pytest

from bemkit.config import freq_bin, freq_grid, resolve_bins, ModelConfig


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


# ── freq_grid ────────────────────────────────────────────────────────────────

def test_freq_grid_matches_freq_array(tmp_path):
    """The grid formula has one home; ModelConfig must not drift from it."""
    (tmp_path / "config.toml").write_text("[bem]\nsr = 32000\nnbins = 65\n")
    cfg = ModelConfig.from_dir(tmp_path)
    assert np.array_equal(cfg.freq_array(), freq_grid(32000, 65))


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


# ── resolve_bins ────────────────────────────────────────────────────────────

class TestResolveBins:
    """sphcap's grid: sr=32000, nbins=65 → 64 bins, 250 Hz apart."""

    SR, NBINS = 32000, 65

    def _r(self, **kw):
        return resolve_bins(self.SR, self.NBINS, **kw)

    def test_literal_bins_sorted_and_deduped(self):
        assert self._r(bins=[16, 4, 16]) == [4, 16]

    def test_bin_index_above_the_band_raises(self):
        with pytest.raises(ValueError, match="outside the band"):
            self._r(bins=[65])

    def test_bin_index_zero_raises(self):
        """Bins are 1-based because NumCalc's -istart is."""
        with pytest.raises(ValueError, match="1-based"):
            self._r(bins=[0])

    def test_every_out_of_range_bin_is_named(self):
        """_validate joins all offenders; fixing them one re-run at a time is
        the thing that convention exists to prevent."""
        with pytest.raises(ValueError, match="0.*99|99.*0"):
            self._r(bins=[0, 4, 99])

    def test_non_integral_bin_raises(self):
        with pytest.raises(ValueError, match="whole numbers"):
            self._r(bins=[3.9])

    def test_bins_accepts_a_numpy_array(self):
        """The freq selector accepts arrays; the two must not disagree about
        what an input may be."""
        assert self._r(bins=np.array([16, 4])) == [4, 16]

    def test_empty_numpy_bins_raises_cleanly(self):
        with pytest.raises(ValueError, match="empty"):
            self._r(bins=np.array([]))

    def test_frequency_snaps_to_nearest_bin(self):
        assert self._r(freq=[1000]) == [4]      # exactly on bin 4
        assert self._r(freq=[1100]) == [4]      # 1100 → 1000 is nearest

    def test_tie_resolves_to_the_lower_bin(self):
        """875 Hz is equidistant from 750 (bin 3) and 1000 (bin 4)."""
        assert self._r(freq=[875]) == [3]

    def test_frequencies_landing_on_one_bin_collapse(self):
        assert self._r(freq=[500, 600]) == [2]

    def test_frequency_above_the_band_raises(self):
        """Silently snapping 30 kHz to the 16 kHz top bin would be a lie."""
        with pytest.raises(ValueError, match="above the simulated band"):
            self._r(freq=[30000])

    def test_frequency_below_the_first_bin_snaps_up(self):
        assert self._r(freq=[10]) == [1]

    def test_nan_frequency_raises(self):
        """NaN passes a > comparison silently and would snap to bin 1."""
        with pytest.raises(ValueError, match="not valid"):
            self._r(freq=[float("nan")])

    def test_negative_frequency_raises(self):
        with pytest.raises(ValueError, match="not valid"):
            self._r(freq=[-500])

    def test_infinite_frequency_raises(self):
        """Before the finiteness guard, inf already raised via the
        above-band check — so this must pin the guard's own message, not
        just any ValueError, or it can't tell the two apart."""
        with pytest.raises(ValueError, match="not valid"):
            self._r(freq=[float("inf")])

    def test_preview_is_log_spaced(self):
        bins = self._r(preview=8)
        assert bins == [1, 2, 3, 6, 11, 20, 35, 64]
        # log spacing on a linear grid ⇒ non-decreasing gaps between bins
        assert all(np.diff(np.diff(bins)) >= 0)

    def test_preview_deduplicates_at_the_low_end(self):
        """Log spacing crowds the bottom of a linear grid, so a big N yields
        fewer bins than asked."""
        bins = self._r(preview=20)
        assert len(bins) < 20
        assert bins == sorted(set(bins))

    def test_preview_covers_both_ends_of_the_band(self):
        assert self._r(preview=8)[0] == 1
        assert self._r(preview=8)[-1] == 64

    def test_preview_of_one_is_the_lowest_bin(self):
        assert self._r(preview=1) == [1]

    def test_preview_below_one_raises(self):
        with pytest.raises(ValueError, match="at least 1"):
            self._r(preview=0)

    def test_two_selectors_raise(self):
        with pytest.raises(ValueError, match="exactly one"):
            self._r(freq=[1000], preview=4)

    def test_no_selector_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            self._r()

    def test_empty_selector_raises(self):
        with pytest.raises(ValueError, match="empty"):
            self._r(freq=[])

    def test_zero_bin_model_raises_clearly(self):
        """nbins=1 is DC only; numpy's IndexError names nothing useful."""
        with pytest.raises(ValueError, match="nbins"):
            resolve_bins(32000, 1, preview=4)

    def test_every_selector_returns_bins_inside_the_band(self):
        """The merge contract: a partial run's bins must be a subset of the
        full run's, or the later run could not fill in around them."""
        nf = self.NBINS - 1
        for kwargs in ({"preview": 8}, {"freq": [500, 4000]}, {"bins": [1, 64]}):
            assert set(self._r(**kwargs)) <= set(range(1, nf + 1))


def test_infinite_bin_index_is_rejected_as_non_integral():
    """floor(inf) == inf, so inf passes an integrality test and then casts to
    INT_MIN — the range error would name -9223372036854775808."""
    from bemkit.config import resolve_bins
    with pytest.raises(ValueError, match="whole numbers"):
        resolve_bins(32000, 65, bins=[float("inf")])


# ── Physics-section validation (BOUNDARY / SYMMETRY / sources) ──────────────

class TestPhysicsSectionValidation:
    """The sections whose silent breakage yields plausible-but-wrong physics
    must be rejected at load, same as the solver/mesh sections already are."""

    def _write(self, tmp_path, body):
        # body first: top-level keys (velocity = ...) must precede any table.
        (tmp_path / "config.toml").write_text(
            body + "\n[bem]\nsr = 32000\nnbins = 65\n")
        return tmp_path

    def test_boundary_missing_velocity_rejected(self, tmp_path):
        d = self._write(tmp_path, "[BOUNDARY]\ncap = [100, 100]\n")
        with pytest.raises(ValueError, match=r"BOUNDARY.*cap"):
            ModelConfig.from_dir(d)

    def test_boundary_string_entry_rejected(self, tmp_path):
        d = self._write(tmp_path, '[BOUNDARY]\ncap = "oops"\n')
        with pytest.raises(ValueError, match=r"BOUNDARY.*cap"):
            ModelConfig.from_dir(d)

    def test_boundary_radius_typo_rejected(self, tmp_path):
        d = self._write(
            tmp_path,
            "[BOUNDARY.v1]\nspot = { near = [0.0, 0.0, 0.05], radios = 0.01 }\n")
        with pytest.raises(ValueError, match=r"radios"):
            ModelConfig.from_dir(d)

    def test_boundary_bad_near_rejected(self, tmp_path):
        d = self._write(tmp_path,
                        "[BOUNDARY.v1]\nspot = { near = [0.0, 0.0] }\n")
        with pytest.raises(ValueError, match=r"near"):
            ModelConfig.from_dir(d)

    def test_boundary_negative_radius_rejected(self, tmp_path):
        d = self._write(
            tmp_path,
            "[BOUNDARY.v1]\nspot = { near = [0.0, 0.0, 0.05], radius = -0.01 }\n")
        with pytest.raises(ValueError, match=r"radius"):
            ModelConfig.from_dir(d)

    def test_boundary_nested_subtable_rejected(self, tmp_path):
        d = self._write(
            tmp_path,
            "[BOUNDARY.v1.deeper]\nspot = { near = [0.0, 0.0, 0.05] }\n")
        with pytest.raises(ValueError, match=r"subtable"):
            ModelConfig.from_dir(d)

    def test_boundary_valid_forms_pass(self, tmp_path):
        d = self._write(tmp_path, (
            "[BOUNDARY]\n"
            "a = [100, 100, 1.0]\n"
            "b = [[1, 2, 1.0], [5, 9, 0.5]]\n"
            "[BOUNDARY.v1]\n"
            "spot = { near = [0.0, 0.0, 0.05], radius = 0.01, velocity = 0.5 }\n"
            "multi = { near = [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]] }\n"))
        ModelConfig.from_dir(d)  # must not raise

    def test_symmetry_unknown_key_rejected(self, tmp_path):
        d = self._write(tmp_path, "[SYMMETRY]\nplanez = [1, 0, 0]\n")
        with pytest.raises(ValueError, match=r"SYMMETRY"):
            ModelConfig.from_dir(d)

    def test_symmetry_bad_planes_rejected(self, tmp_path):
        d = self._write(tmp_path, "[SYMMETRY]\nplanes = [1, 0]\n")
        with pytest.raises(ValueError, match=r"planes"):
            ModelConfig.from_dir(d)

    def test_symmetry_valid_passes(self, tmp_path):
        d = self._write(tmp_path,
                        "[SYMMETRY]\nplanes = [1, 0, 0]\n"
                        "coordinates = [0.0, 0.0, 0.0]\n")
        ModelConfig.from_dir(d)

    def test_plane_waves_short_row_rejected(self, tmp_path):
        d = self._write(tmp_path, '["PLANE WAVES"]\nw = [1, 0, 0, -1]\n')
        with pytest.raises(ValueError, match=r"PLANE WAVES"):
            ModelConfig.from_dir(d)

    def test_point_sources_valid_passes(self, tmp_path):
        d = self._write(
            tmp_path,
            '["POINT SOURCES"]\ns = [1, 0.0, 0.0, 0.5, 1.0, -1, 0.0, -1]\n')
        ModelConfig.from_dir(d)

    def test_sources_multi_row_valid_passes(self, tmp_path):
        d = self._write(
            tmp_path,
            '["PLANE WAVES"]\n'
            'w = [[1, 0.0, 0.0, 1.0, 1.0, -1, 0.0, -1],\n'
            '     [2, 1.0, 0.0, 0.0, 1.0, -1, 0.0, -1]]\n')
        ModelConfig.from_dir(d)

    def test_top_level_velocity_must_be_number(self, tmp_path):
        d = self._write(tmp_path, 'velocity = "fast"\n')
        with pytest.raises(ValueError, match=r"velocity"):
            ModelConfig.from_dir(d)

    def test_top_level_velocity_number_passes(self, tmp_path):
        d = self._write(tmp_path, "velocity = 0.5\n")
        assert ModelConfig.from_dir(d).get("velocity") == 0.5

    def test_user_extension_tables_still_pass_through(self, tmp_path):
        d = self._write(tmp_path, "[cad]\nwidth = 120\n")
        assert ModelConfig.from_dir(d)["cad"] == {"width": 120}


# ── from_dict seam + typed physics accessors ────────────────────────────────

class TestFromDictAndTypedAccessors:
    def test_from_dict_runs_validation(self, tmp_path):
        raw = {"bem": {"sr": 32000, "nbins": 65},
               "BOUNDARY": {"cap": [100, 100]}}      # malformed → must raise
        with pytest.raises(ValueError, match="BOUNDARY"):
            ModelConfig.from_dict(raw, tmp_path)

    def test_from_dict_valid(self, tmp_path):
        cfg = ModelConfig.from_dict({"bem": {"sr": 32000, "nbins": 65}}, tmp_path)
        assert cfg.sr == 32000
        assert cfg.model_dir == tmp_path.resolve()
        assert cfg.modelname == tmp_path.name

    def test_from_dir_delegates_to_from_dict(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            "[bem]\nsr = 48000\nnbins = 33\n")
        cfg = ModelConfig.from_dir(tmp_path)
        assert cfg.sr == 48000 and cfg.nbins == 33

    def test_typed_physics_accessors(self, tmp_path):
        (tmp_path / "config.toml").write_text(
            'velocity = 0.5\n'
            '[bem]\nsr = 32000\nnbins = 65\n'
            '[BOUNDARY]\ncap = [100, 100, 1.0]\n'
            '[SYMMETRY]\nplanes = [1, 0, 0]\n'
            '["PLANE WAVES"]\nw = [1, 0.0, 0.0, 1.0, 1.0, -1, 0.0, -1]\n')
        cfg = ModelConfig.from_dir(tmp_path)
        assert cfg.default_velocity == 0.5
        assert cfg.boundary_cfg == {"cap": [100, 100, 1.0]}
        assert cfg.symmetry_cfg == {"planes": [1, 0, 0]}
        assert cfg.sources_cfg["PLANE WAVES"] == {
            "w": [1, 0.0, 0.0, 1.0, 1.0, -1, 0.0, -1]}
        assert cfg.sources_cfg["POINT SOURCES"] == {}

    def test_typed_accessors_default_empty(self, tmp_path):
        cfg = ModelConfig.from_dict({"bem": {"sr": 32000, "nbins": 65}}, tmp_path)
        assert cfg.boundary_cfg == {}
        assert cfg.symmetry_cfg == {}
        assert cfg.default_velocity == 1.0
