"""Tests for bemkit.sim.builder — ModelConfig -> Deck.

These are the behaviour tests that used to live in tests/sim/test_writers.py
against write_nc_inp. They now assert on the Deck, which is what they were
really testing: which elements get a boundary condition, and which config
mistakes must fail loudly.
"""

import pytest

from bemkit.sim.builder import deck_from_config


def _counts(tmp_path, n_elem=30):
    """Minimal ObjectMesh/EvalGrid count files — the builder reads these back."""
    for sub, n in [("ObjectMesh", n_elem), ("EvalGrid", 3)]:
        d = tmp_path / sub
        d.mkdir(exist_ok=True)
        (d / "Nodes.txt").write_text(f"{n}\n")
        (d / "Elements.txt").write_text(f"{n}\n")


def _config(**kw):
    base = {"modelname": "tiny", "sr": 8000, "nbins": 5, "bem_type": 4,
            "air_density": 1.2, "speed_of_sound": 340.0}
    base.update(kw)
    return base


def _ranges(deck):
    """(start, end, velocity) of every emitted BC, for compact assertions."""
    return [(bc.start, bc.end, bc.real) for bc in deck.boundary]


class TestBoundaryAssembly:
    def test_physical_group_and_boundary_table_combine(self, tmp_path):
        _counts(tmp_path)
        deck = deck_from_config(
            _config(BOUNDARY={"cap": [0, 0, 1.0]}), "tiny", tmp_path,
            vibrating_indices=[1, 2], verbose=False)
        assert _ranges(deck) == [(1, 2, 1.0), (0, 0, 1.0)]
        assert all(bc.kind == "VELO" for bc in deck.boundary)

    def test_variant_boundary_section_used_exclusively(self, tmp_path):
        _counts(tmp_path)
        cfg = _config(BOUNDARY={"left": {"ear": [10, 10, 1.0]},
                                "right": {"ear": [20, 20, 1.0]}})
        deck = deck_from_config(cfg, "left", tmp_path, verbose=False)
        assert _ranges(deck) == [(10, 10, 1.0)]  # other variant's BC excluded

    def test_recipe_suffixed_variant_matches_target_section(self, tmp_path):
        # The config-fallback path must use the same variant matching as
        # export_variant (resolve_boundary): 'left__r05' falls back to
        # [BOUNDARY.left] after stripping the __recipe suffix.
        _counts(tmp_path)
        cfg = _config(BOUNDARY={"left": {"ear": [10, 10, 1.0]}})
        deck = deck_from_config(cfg, "left__r05", tmp_path, verbose=False)
        assert _ranges(deck) == [(10, 10, 1.0)]

    def test_nested_range_boundary_entries(self, tmp_path):
        _counts(tmp_path)
        cfg = _config(BOUNDARY={"cap": [[1, 2, 1.0], [5, 6, 0.5]]})
        deck = deck_from_config(cfg, "tiny", tmp_path, verbose=False)
        assert _ranges(deck) == [(1, 2, 1.0), (5, 6, 0.5)]

    def test_geometric_anchor_without_mesh_raises(self, tmp_path):
        # near= anchors need a mesh to resolve against; this path has none, so
        # it must raise instead of silently emitting a fully rigid surface.
        _counts(tmp_path)
        cfg = _config(BOUNDARY={"spot": {"near": [0.0, 0.0, 0.05]}})
        with pytest.raises(ValueError, match="near"):
            deck_from_config(cfg, "tiny", tmp_path, verbose=False)

    def test_out_of_range_boundary_raises(self, tmp_path):
        _counts(tmp_path, n_elem=5)  # valid indices 0..4
        cfg = _config(BOUNDARY={"cap": [10, 10, 1.0]})
        with pytest.raises(ValueError, match="out of bounds"):
            deck_from_config(cfg, "tiny", tmp_path, verbose=False)

    def test_warns_when_boundary_table_yields_no_elements(self, tmp_path, capsys):
        _counts(tmp_path)
        # Per-variant table exists but nothing matches variant 'tiny'
        cfg = _config(BOUNDARY={"left": {"ear": [1, 1, 1.0]}})
        deck = deck_from_config(cfg, "tiny", tmp_path, verbose=True)
        assert "fully rigid" in capsys.readouterr().out
        assert deck.boundary == []


class TestSources:
    def test_source_sections_carry_into_the_deck(self, tmp_path):
        _counts(tmp_path)
        cfg = _config(**{"PLANE WAVES": {"1": [1, 0, -1, 0, 1, -1, 0, -1]},
                         "POINT SOURCES": {"1": [1, 0, 2, 0, 1, -1, 0, -1]}})
        deck = deck_from_config(cfg, "tiny", tmp_path, verbose=False)
        assert [s.vec for s in deck.plane_waves] == [(0, -1, 0)]
        assert [s.vec for s in deck.point_sources] == [(0, 2, 0)]

        # The header counts NumCalc reads must match the emitted rows.
        deck.write(tmp_path / "NC.inp")
        text = (tmp_path / "NC.inp").read_text()
        assert "1 1 0 0.0000e+00 0 0 0" in text
        assert "PLANE WAVES\n1 0.0 -1.0 0.0 1.0 -1 0.0 -1\n##" in text
        assert "POINT SOURCES\n1 0.0 2.0 0.0 1.0 -1 0.0 -1\n##" in text

    def test_malformed_source_entry_raises(self, tmp_path):
        # A silently dropped entry would leave the header count claiming a
        # source line that never appears — NumCalc would misparse the file.
        _counts(tmp_path)
        cfg = _config(**{"PLANE WAVES": {"1": [1, 0, -1]}})  # not 8 values
        with pytest.raises(ValueError, match="PLANE WAVES"):
            deck_from_config(cfg, "tiny", tmp_path, verbose=False)

    @pytest.mark.parametrize("section, wanted, unwanted", [
        ("PLANE WAVES", "direction", "position in meters"),
        ("POINT SOURCES", "position in meters", "direction"),
    ])
    def test_source_error_names_the_right_triple(self, tmp_path, section,
                                                 wanted, unwanted):
        """The two sections share a row layout but not a meaning for fields
        1-3: a plane wave travels along a direction, a point source sits at a
        position. A row copied between them is accepted and silently wrong, so
        the message must name the triple the section actually wants."""
        _counts(tmp_path)
        cfg = _config(**{section: {"1": [1, 0, -1]}})  # not 8 values
        with pytest.raises(ValueError) as exc:
            deck_from_config(cfg, "tiny", tmp_path, verbose=False)
        assert wanted in str(exc.value)
        assert unwanted not in str(exc.value)


class TestConfigMapping:
    def test_modelconfig_typed_defaults_fill_omitted_keys(self, tmp_path):
        from bemkit.config import ModelConfig
        _counts(tmp_path)
        model = tmp_path / "model"
        model.mkdir()
        (model / "config.toml").write_text("[bem]\nsr = 8000\nnbins = 5\n")
        deck = deck_from_config(ModelConfig.from_dir(model), "tiny", tmp_path,
                                verbose=False)
        assert deck.time_factor == -1.0     # the typed default, not 1.0
        assert deck.bem_type == 4
        assert deck.air_density == 1.2

    def test_frequencies_are_the_fft_grid(self, tmp_path):
        import numpy as np
        _counts(tmp_path)
        deck = deck_from_config(_config(sr=8000, nbins=5), "tiny", tmp_path,
                                verbose=False)
        # nbins-1 frequencies, DC excluded, up to Nyquist.
        np.testing.assert_allclose(deck.frequencies, [1000, 2000, 3000, 4000])

    def test_symmetry_carries_through(self, tmp_path):
        _counts(tmp_path)
        cfg = _config(SYMMETRY={"planes": [0, 0, 1],
                                "coordinates": [0.0, 0.0, 0.25]})
        deck = deck_from_config(cfg, "tiny", tmp_path, verbose=False)
        assert deck.symmetry_planes == [0, 0, 1]
        assert deck.n_symmetric_planes == 1

    def test_inactive_symmetry_is_dropped(self, tmp_path):
        _counts(tmp_path)
        cfg = _config(SYMMETRY={"planes": [0, 0, 0]})
        deck = deck_from_config(cfg, "tiny", tmp_path, verbose=False)
        assert deck.symmetry_planes is None

    def test_totals_are_object_plus_eval(self, tmp_path):
        _counts(tmp_path, n_elem=30)
        deck = deck_from_config(_config(), "tiny", tmp_path, verbose=False)
        assert deck.n_total_nodes == 33 and deck.n_total_elements == 33


def test_written_deck_is_a_complete_nc_inp(tmp_path):
    _counts(tmp_path)
    cfg = _config(BOUNDARY={"cap": [0, 0, 1.0]})
    deck = deck_from_config(cfg, "tiny", tmp_path, vibrating_indices=[1, 2],
                            verbose=False)
    deck.write(tmp_path / "NC.inp")
    text = (tmp_path / "NC.inp").read_text()
    assert text.startswith("##")            # header block
    assert "POST PROCESS" in text and text.endswith("END\n")
    assert "{" not in text                  # no unresolved placeholders
