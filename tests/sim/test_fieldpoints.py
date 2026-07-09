"""Tests for pynumcalc.sim.fieldpoints — inside/distance checks, combine/split roundtrip."""

import numpy as np
import pytest

from pynumcalc.sim.fieldpoints import (
    points_inside_mesh,
    min_distance_to_mesh,
    combine_and_track,
    load_fieldpoint_sets,
    split_by_offsets,
    save_offsets,
    load_offsets,
)


@pytest.fixture()
def unit_cube():
    """Closed unit cube mesh [0,1]^3 — 8 vertices, 12 triangles.

    Outward-facing normals (CCW winding from outside).
    """
    v = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    t = np.array([
        # bottom (z=0, normal -z)
        [0, 1, 2], [0, 2, 3],
        # top (z=1, normal +z)
        [4, 6, 5], [4, 7, 6],
        # front (y=0, normal -y)
        [0, 5, 1], [0, 4, 5],
        # back (y=1, normal +y)
        [2, 7, 3], [2, 6, 7],
        # left (x=0, normal -x)
        [0, 7, 4], [0, 3, 7],
        # right (x=1, normal +x)
        [1, 6, 2], [1, 5, 6],
    ], dtype=np.int32)
    return v, t


class TestPointsInsideMesh:
    def test_inside_point(self, unit_cube):
        v, t = unit_cube
        # Offset from face diagonals to avoid ray-edge ambiguity
        pts = np.array([[0.3, 0.4, 0.6]])
        inside = points_inside_mesh(pts, v, t)
        assert inside[0]

    def test_outside_point(self, unit_cube):
        v, t = unit_cube
        pts = np.array([[2.0, 0.5, 0.5]])
        inside = points_inside_mesh(pts, v, t)
        assert not inside[0]

    def test_multiple_points(self, unit_cube):
        v, t = unit_cube
        pts = np.array([
            [0.3, 0.4, 0.6],  # inside (offset from diagonals)
            [2.0, 2.0, 2.0],  # outside
            [0.1, 0.2, 0.3],  # inside (corner region)
        ])
        inside = points_inside_mesh(pts, v, t)
        assert inside[0]
        assert not inside[1]
        assert inside[2]


class TestMinDistanceToMesh:
    def test_on_surface(self, unit_cube):
        v, t = unit_cube
        # Point on the z=0 face
        pts = np.array([[0.5, 0.5, 0.0]])
        dist = min_distance_to_mesh(pts, v, t)
        assert dist[0] == pytest.approx(0.0, abs=1e-10)

    def test_outside_distance(self, unit_cube):
        v, t = unit_cube
        # Point 1 unit above center of top face
        pts = np.array([[0.5, 0.5, 2.0]])
        dist = min_distance_to_mesh(pts, v, t)
        assert dist[0] == pytest.approx(1.0, abs=1e-10)

    def test_inside_distance(self, unit_cube):
        v, t = unit_cube
        # Point at center — distance to nearest face is 0.5
        pts = np.array([[0.5, 0.5, 0.5]])
        dist = min_distance_to_mesh(pts, v, t)
        assert dist[0] == pytest.approx(0.5, abs=1e-10)


# ── Combine / Split roundtrip ───────────────────────────────────────────────

class TestCombineAndTrack:
    def test_basic_combine(self):
        sets = {
            "a": np.array([[1.0, 0, 0], [2.0, 0, 0]]),
            "b": np.array([[3.0, 0, 0]]),
        }
        combined, offsets = combine_and_track(sets)
        assert combined.shape == (3, 3)
        assert offsets["sets"]["a"]["start"] == 0
        assert offsets["sets"]["a"]["end"] == 2
        assert offsets["sets"]["b"]["start"] == 2
        assert offsets["sets"]["b"]["end"] == 3

    def test_with_valid_masks(self):
        sets = {
            "a": np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]),
        }
        masks = {"a": np.array([True, False, True])}
        combined, offsets = combine_and_track(sets, valid_masks=masks)
        assert combined.shape == (2, 3)
        assert offsets["sets"]["a"]["n_original"] == 3
        assert offsets["sets"]["a"]["n_valid"] == 2
        assert offsets["dropped"]["a"] == [1]

    def test_empty_set(self):
        sets = {"a": np.array([[1, 0, 0]])}
        masks = {"a": np.array([False])}
        combined, offsets = combine_and_track(sets, valid_masks=masks)
        assert combined.shape == (0, 3)


class TestSplitByOffsets:
    def test_roundtrip_no_drops(self):
        sets = {
            "a": np.array([[1.0, 0, 0], [2.0, 0, 0]]),
            "b": np.array([[3.0, 0, 0]]),
        }
        combined, offsets = combine_and_track(sets)
        result = split_by_offsets(combined, offsets)
        np.testing.assert_array_equal(result["a"], sets["a"])
        np.testing.assert_array_equal(result["b"], sets["b"])

    def test_roundtrip_with_drops(self):
        original = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
        sets = {"a": original}
        masks = {"a": np.array([True, False, True])}
        combined, offsets = combine_and_track(sets, valid_masks=masks)
        result = split_by_offsets(combined, offsets)
        assert result["a"].shape == (3, 3)
        np.testing.assert_allclose(result["a"][0], [1, 0, 0])
        assert np.all(np.isnan(result["a"][1]))
        np.testing.assert_allclose(result["a"][2], [3, 0, 0])

    def test_works_with_pressure_data(self):
        """Roundtrip with (N, nf) complex arrays like pressure data."""
        sets = {"pts": np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)}
        combined, offsets = combine_and_track(sets)
        pressure = np.array([[1+2j, 3+4j], [5+6j, 7+8j], [9+10j, 11+12j]])
        result = split_by_offsets(pressure, offsets)
        np.testing.assert_array_equal(result["pts"], pressure)


class TestSaveLoadOffsets:
    def test_roundtrip(self, tmp_path):
        offsets = {
            "sets": {"a": {"start": 0, "end": 5, "n_original": 5, "n_valid": 5}},
            "dropped": {},
        }
        path = tmp_path / "offsets.json"
        save_offsets(offsets, path)
        loaded = load_offsets(path)
        assert loaded == offsets


class TestLoadFieldpointSets:
    def test_missing_directory_raises(self, tmp_path):
        # A configured fieldpoints dir that does not exist must raise, not
        # silently return {} (which would corrupt the EvalGrid downstream).
        with pytest.raises(FileNotFoundError):
            load_fieldpoint_sets("does_not_exist", tmp_path)

    def test_empty_directory_raises(self, tmp_path):
        (tmp_path / "fieldpoints").mkdir()
        with pytest.raises(FileNotFoundError):
            load_fieldpoint_sets("fieldpoints", tmp_path)

    def test_empty_dict_raises(self, tmp_path):
        with pytest.raises(ValueError):
            load_fieldpoint_sets({}, tmp_path)

    def test_loads_directory(self, tmp_path):
        fp = tmp_path / "fieldpoints"
        fp.mkdir()
        np.save(fp / "onaxis.npy", np.zeros((5, 3)))
        sets = load_fieldpoint_sets("fieldpoints", tmp_path)
        assert list(sets) == ["onaxis"]
        assert sets["onaxis"].shape == (5, 3)
