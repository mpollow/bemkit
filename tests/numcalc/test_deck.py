"""Tests for bemkit.numcalc.deck — the NC.inp document model.

The load-bearing test here is the round trip: ``Deck.read(deck.write(p)) == deck``
over a corpus covering every section. That single property is what keeps
"support every NumCalc setting" testable as fields are added later.
"""

import numpy as np
import pytest

from bemkit.numcalc.deck import BC, Curve, Deck, Source, _contiguous_ranges


class TestContiguousRanges:
    def test_empty(self):
        assert _contiguous_ranges([]) == []

    def test_single_run(self):
        assert _contiguous_ranges([3, 4, 5]) == [(3, 5)]

    def test_gaps_and_order(self):
        assert _contiguous_ranges([7, 1, 2, 9]) == [(1, 2), (7, 7), (9, 9)]


def _minimal(**kw):
    base = dict(title="tiny", frequencies=np.array([250.0, 500.0, 750.0]),
                n_total_nodes=33, n_total_elements=33,
                boundary=[BC(0, 4, "VELO", 1.0)])
    base.update(kw)
    return Deck(**base)


#: One deck per section, so a regression names the section it broke.
CORPUS = {
    "minimal": _minimal(),
    "symmetry": _minimal(symmetry_planes=[0, 0, 1],
                         symmetry_coordinates=[0.0, 0.0, 0.5]),
    "multi_bc": _minimal(boundary=[
        BC(0, 4, "VELO", 1.0), BC(5, 5, "VELO", 0.5),
        BC(10, 20, "PRES", 0.0), BC(21, 30, "ADMI", 0.4, imag=-0.2),
    ]),
    "sources": _minimal(
        plane_waves=[Source(1, (0.0, 1.0, 0.0), 1.0)],
        point_sources=[Source(1, (0.0, -1.5, 0.0), 1.0, imag=0.25)]),
    "curves": _minimal(
        boundary=[BC(0, 4, "ADMI", 1.0, imag=1.0, curve_real=1, curve_imag=2)],
        curves=[Curve(1, [(100.0, 0.1), (1000.0, 0.4)]),
                Curve(2, [(100.0, -0.05), (1000.0, -0.2)])]),
    "nonuniform_freqs": _minimal(
        # NumCalc's frequency grid is a curve, not a range — a log sweep is
        # representable even though bemkit only builds FFT grids today.
        frequencies=np.array([100.0, 200.0, 400.0, 800.0, 1600.0])),
    "solver_knobs": _minimal(bem_type=0, isolver=4, min_expansion_terms=7,
                             cluster_edge_length=0.05, preconditioner=-2,
                             is_internal=1),
}


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_roundtrip(name, tmp_path):
    deck = CORPUS[name]
    path = tmp_path / "NC.inp"
    deck.write(path)
    assert Deck.read(path) == deck


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_roundtrip_is_stable_on_rewrite(name, tmp_path):
    """Writing a deck that was read back must reproduce the same bytes."""
    first, second = tmp_path / "a.inp", tmp_path / "b.inp"
    CORPUS[name].write(first)
    Deck.read(first).write(second)
    assert first.read_text() == second.read_text()


class TestFrequencyEncoding:
    def test_frequencies_survive_exactly(self, tmp_path):
        freqs = np.array([250.0, 500.0, 750.0, 1000.0])
        d = _minimal(frequencies=freqs)
        d.write(tmp_path / "NC.inp")
        np.testing.assert_array_equal(
            Deck.read(tmp_path / "NC.inp").frequencies, freqs)

    def test_curve_is_interpolated_like_numcalc(self, tmp_path):
        """A two-point curve must expand the same way NC_ComputeFrequencies does:
        linear between nodes, clamped flat outside."""
        d = _minimal()
        d.write(tmp_path / "NC.inp")
        text = (tmp_path / "NC.inp").read_text()
        # Replace our per-step curve with the 2-point shorthand bemkit used to
        # write, over the same span: steps 1..3 -> 250, 500, 750.
        old = "0 3\n1 250.0 0.0\n2 500.0 0.0\n3 750.0 0.0\n"
        new = "0 2\n1 250.0 0.0\n3 750.0 0.0\n"
        assert old in text, "curve block not found — this test would pass vacuously"
        (tmp_path / "NC.inp").write_text(text.replace(old, new))
        np.testing.assert_allclose(
            Deck.read(tmp_path / "NC.inp").frequencies, [250.0, 500.0, 750.0])


class TestValidation:
    def test_rejects_unsupported_bc_kind(self):
        with pytest.raises(ValueError, match="TRAD/TRIM"):
            _minimal(boundary=[BC(0, 1, "TRAD", 1.0)]).validate()

    def test_rejects_half_specified_curve_refs(self):
        # NumCalc: ALL curve numbers in one line must be -1, or none.
        with pytest.raises(ValueError, match="both -1 or both set"):
            _minimal(boundary=[BC(0, 1, "ADMI", 1.0, curve_real=1)]).validate()

    def test_rejects_impe_with_curve(self):
        # NumCalc exits(-1) on this combination rather than reporting an error.
        with pytest.raises(ValueError, match="ADMI instead"):
            _minimal(boundary=[
                BC(0, 1, "IMPE", 1.0, imag=1.0, curve_real=1, curve_imag=2)
            ]).validate()

    def test_rejects_zero_plane_wave_direction(self):
        with pytest.raises(ValueError, match="all zeros"):
            _minimal(plane_waves=[Source(1, (0.0, 0.0, 0.0), 1.0)]).validate()

    def test_rejects_bad_isolver(self):
        with pytest.raises(ValueError, match="ISOLVER"):
            _minimal(isolver=1).validate()

    def test_rejects_out_of_band_expansion_terms(self):
        with pytest.raises(ValueError, match=r"\[5, 10\]"):
            _minimal(min_expansion_terms=3).validate()

    def test_rejects_too_many_symmetry_planes(self):
        with pytest.raises(ValueError, match="at most 3"):
            _minimal(symmetry_planes=[1, 1, 1, 1]).validate()


class TestLexerCompatibility:
    """NumCalc's NC_GetLine splits on the space character only and treats a
    first term containing '#' as a comment. Violating either silently
    misparses the deck."""

    def test_no_tabs_anywhere(self, tmp_path):
        CORPUS["curves"].write(tmp_path / "NC.inp")
        assert "\t" not in (tmp_path / "NC.inp").read_text()

    def test_comments_are_whole_line_only(self, tmp_path):
        CORPUS["curves"].write(tmp_path / "NC.inp")
        for line in (tmp_path / "NC.inp").read_text().splitlines():
            if "#" in line:
                assert line.lstrip().startswith("#"), (
                    f"inline comment would not be stripped by NumCalc: {line!r}")

    def test_file_ends_with_newline(self, tmp_path):
        # NC_GetLine drops the last character of every line, assuming '\n'.
        CORPUS["minimal"].write(tmp_path / "NC.inp")
        assert (tmp_path / "NC.inp").read_text().endswith("END\n")

    def test_header_comments_are_preserved(self, tmp_path):
        d = _minimal(header=["## Created using bemkit", "## Date: 2026-01-01"])
        d.write(tmp_path / "NC.inp")
        assert Deck.read(tmp_path / "NC.inp").header == d.header


class TestReadErrors:
    def test_truncated_file_raises(self, tmp_path):
        CORPUS["minimal"].write(tmp_path / "NC.inp")
        lines = (tmp_path / "NC.inp").read_text().splitlines()
        (tmp_path / "NC.inp").write_text("\n".join(lines[:6]) + "\n")
        with pytest.raises(ValueError, match="unexpected end of file"):
            Deck.read(tmp_path / "NC.inp")

    def test_malformed_boundary_line_raises(self, tmp_path):
        CORPUS["minimal"].write(tmp_path / "NC.inp")
        text = (tmp_path / "NC.inp").read_text()
        old = "ELEM 0 TO 4 VELO 1.0 -1 0.0 -1"
        assert old in text, "BC line not found — this test would pass vacuously"
        (tmp_path / "NC.inp").write_text(text.replace(old, "ELEM 0 TO 4 VELO 1.0"))
        with pytest.raises(ValueError, match="at least 9 terms"):
            Deck.read(tmp_path / "NC.inp")

    def test_missing_version_line_raises(self, tmp_path):
        (tmp_path / "NC.inp").write_text("NotMesh2HRTF 1.0\n")
        with pytest.raises(ValueError, match="Mesh2HRTF"):
            Deck.read(tmp_path / "NC.inp")


class TestSolverKnobValidation:
    """NumCalc accepts these combinations and then quietly does something else,
    which is worse than rejecting them."""

    def test_direct_solver_requires_tbem(self):
        with pytest.raises(ValueError, match="silently falls back"):
            _minimal(isolver=4, bem_type=4).validate()

    def test_direct_solver_with_tbem_is_fine(self):
        _minimal(isolver=4, bem_type=0).validate()

    def test_knobs_round_trip(self, tmp_path):
        d = _minimal(bem_type=0, isolver=4, min_expansion_terms=9,
                     cluster_edge_length=0.05, preconditioner=0)
        d.write(tmp_path / "NC.inp")
        assert Deck.read(tmp_path / "NC.inp") == d
