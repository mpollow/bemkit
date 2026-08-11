"""Tests for bemkit.sim.fieldpoints — inside/distance checks, combine/split roundtrip."""

import numpy as np
import pytest

from bemkit.sim.fieldpoints import (
    points_inside_mesh,
    min_distance_to_mesh,
    combine_and_track,
    load_fieldpoint_sets,
    split_by_offsets,
    save_offsets,
    load_offsets,
    validate_points,
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
    def test_empty_directory_raises(self, tmp_path):
        # No named sets and no .npy files must raise, not silently return
        # {} (which would corrupt the EvalGrid downstream).
        (tmp_path / "fieldpoints").mkdir()
        with pytest.raises(ValueError, match="No fieldpoint sets"):
            load_fieldpoint_sets({}, tmp_path)

    def test_empty_config_no_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No fieldpoint sets"):
            load_fieldpoint_sets({}, tmp_path)

    def test_dir_fallback_loads_all_npy(self, tmp_path):
        fp = tmp_path / "fieldpoints"
        fp.mkdir()
        np.save(fp / "onaxis.npy", np.zeros((5, 3)))
        np.save(fp / "polar.npy", np.zeros((7, 3)))
        sets = load_fieldpoint_sets({}, tmp_path)
        assert list(sets) == ["onaxis", "polar"]
        assert sets["onaxis"].shape == (5, 3)

    def test_exec_key_is_ignored(self, tmp_path):
        fp = tmp_path / "fieldpoints"
        fp.mkdir()
        np.save(fp / "onaxis.npy", np.zeros((5, 3)))
        sets = load_fieldpoint_sets({"exec": "./make.py"}, tmp_path)
        assert list(sets) == ["onaxis"]

    def test_file_set(self, tmp_path):
        np.save(tmp_path / "pts.npy", np.zeros((4, 3)))
        sets = load_fieldpoint_sets({"custom": {"file": "pts.npy"}}, tmp_path)
        assert list(sets) == ["custom"]
        assert sets["custom"].shape == (4, 3)

    def test_file_set_bad_shape_raises(self, tmp_path):
        np.save(tmp_path / "pts.npy", np.zeros((4, 2)))
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            load_fieldpoint_sets({"custom": {"file": "pts.npy"}}, tmp_path)

    def test_generator_set_onaxis(self, tmp_path):
        sets = load_fieldpoint_sets(
            {"onaxis": {"type": "onaxis", "dim": "z", "radii": [0.5, 1.0]}},
            tmp_path)
        np.testing.assert_allclose(
            sets["onaxis"], [[0, 0, 0.5], [0, 0, 1.0]])

    def test_generator_set_polar(self, tmp_path):
        sets = load_fieldpoint_sets(
            {"polar": {"type": "polar", "dims": "zx", "res": 90}}, tmp_path)
        assert sets["polar"].shape == (4, 3)

    def test_generator_bad_kwarg_names_the_set(self, tmp_path):
        with pytest.raises(ValueError, match=r"\[fieldpoints\.onaxis\]"):
            load_fieldpoint_sets(
                {"onaxis": {"type": "onaxis", "radius": 1.0}}, tmp_path)

    def test_named_sets_win_over_directory(self, tmp_path):
        fp = tmp_path / "fieldpoints"
        fp.mkdir()
        np.save(fp / "ignored.npy", np.zeros((5, 3)))
        sets = load_fieldpoint_sets(
            {"onaxis": {"type": "onaxis", "radii": [1.0]}}, tmp_path)
        assert list(sets) == ["onaxis"]


class TestValidatePointsWithSymmetry:
    """A symmetry-reduced mesh only covers part of the object; eval points
    must be checked against the geometry NumCalc actually solves.

    Regression: with [SYMMETRY] planes = [1, 1, 0] and full-span field
    planes, a point landing on the mirror image of the source patch passed
    validation and NumCalc then died in post-processing subdividing the
    element under it ("nsbe must be < MSBE").
    """

    @pytest.fixture()
    def quarter_box_file(self, tmp_path):
        """Quarter of the box [-1,1] x [-1,1] x [0,1], kept in x,y >= 0,
        open at both cut planes. Written as .stl so it is read in metres."""
        import meshio
        v = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], float)
        t = np.array([
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [1, 2, 6], [1, 6, 5],
            [3, 7, 6], [3, 6, 2],
        ])
        path = tmp_path / "quarter.stl"
        meshio.write_points_cells(path, v, [("triangle", t)])
        return path

    QUARTER_SYMMETRY = {"planes": [1, 1, 0], "coordinates": [0.0, 0.0, 0.0]}

    def test_point_inside_the_mirrored_geometry_is_dropped(self, quarter_box_file):
        inside_the_mirror = np.array([[-0.6, -0.3, 0.5]])
        assert validate_points(inside_the_mirror, quarter_box_file).all()
        assert not validate_points(inside_the_mirror, quarter_box_file,
                                   symmetry=self.QUARTER_SYMMETRY).any()

    def test_point_on_a_mirrored_surface_is_dropped(self, quarter_box_file):
        on_the_mirrored_face = np.array([[-1.0, 0.5, 0.5]])
        assert validate_points(on_the_mirrored_face, quarter_box_file).all()
        assert not validate_points(on_the_mirrored_face, quarter_box_file,
                                   symmetry=self.QUARTER_SYMMETRY).any()

    def test_point_clear_of_the_whole_object_stays_valid(self, quarter_box_file):
        outside = np.array([[-3.0, -3.0, 0.5]])
        assert validate_points(outside, quarter_box_file,
                               symmetry=self.QUARTER_SYMMETRY).all()

    def test_without_symmetry_validation_is_unchanged(self, quarter_box_file):
        inside_the_mirror = np.array([[-0.6, -0.3, 0.5]])
        assert validate_points(inside_the_mirror, quarter_box_file,
                               symmetry=None).all()

    def test_inactive_planes_are_a_no_op(self, quarter_box_file):
        inside_the_mirror = np.array([[-0.6, -0.3, 0.5]])
        assert validate_points(inside_the_mirror, quarter_box_file,
                               symmetry={"planes": [0, 0, 0]}).all()

    def test_coordinates_default_to_the_origin(self, quarter_box_file):
        """[SYMMETRY] may omit coordinates; NC.inp defaults them to 0."""
        inside_the_mirror = np.array([[-0.6, -0.3, 0.5]])
        assert not validate_points(inside_the_mirror, quarter_box_file,
                                   symmetry={"planes": [1, 1, 0]}).any()
