"""Tests for bemviz.viz._angles — set geometry detection and angle selection.

These helpers are what make the dataset-tailored plots refuse the wrong
dataset, so the negative cases matter as much as the positive ones.
"""

import numpy as np
import pytest

from bemkit.analysis.result import BEMResult
from bemviz.viz._angles import (
    DEFAULT_ANGLES, NotCircular, NotOnAxis, angle_indices, axis_positions,
    circle_angles, is_circular, is_onaxis,
)


def make_result(points, *, name="test", nf=3):
    """A BEMResult carrying the given points and dummy unit pressure."""
    points = np.asarray(points, dtype=float)
    ones = np.ones((len(points), nf), dtype=complex)
    return BEMResult(
        name=name,
        frequencies=np.linspace(100.0, 300.0, nf),
        eval_points=points,
        pressure_raw=ones,
        pressure_acc=ones,
        velocity_raw=ones,
        velocity_acc=ones,
        rho0=1.2,
        speed_of_sound=340.0,
    )


def circle_points(dims="zx", res=5.0, radius=1.0):
    """The same circle ``make_fields_polar`` generates."""
    from bemkit.sim.fields import make_fields_polar
    return make_fields_polar(dims=dims, res=res, radius=radius)


class TestCircleAngles:
    def test_zx_circle_puts_zero_on_plus_z(self):
        result = make_result(circle_points("zx"))
        angles, radius, normal = circle_angles(result)

        on_axis = result.eval_points[np.argmin(np.abs(angles))]
        assert on_axis == pytest.approx([0.0, 0.0, 1.0])
        assert radius == pytest.approx(1.0)
        assert normal == 1  # y is the plane normal

    def test_quarter_turn_is_plus_ninety(self):
        result = make_result(circle_points("zx"))
        angles, _, _ = circle_angles(result)

        at90 = result.eval_points[np.argmin(np.abs(angles - 90.0))]
        assert at90 == pytest.approx([1.0, 0.0, 0.0])

    def test_xy_circle_puts_zero_on_plus_x(self):
        result = make_result(circle_points("xy"))
        angles, _, normal = circle_angles(result)

        on_axis = result.eval_points[np.argmin(np.abs(angles))]
        assert on_axis == pytest.approx([1.0, 0.0, 0.0])
        assert normal == 2

    def test_range_is_half_open_at_minus_180(self):
        result = make_result(circle_points("zx"))
        angles, _, _ = circle_angles(result)

        assert angles.min() >= -180.0
        assert angles.max() < 180.0
        # 72 points at 5° spacing: exactly one per direction, none duplicated.
        assert len(np.unique(np.round(angles, 6))) == 72

    def test_dropped_points_stay_aligned_as_nan(self):
        pts = circle_points("zx", res=90.0)
        pts[2] = np.nan
        angles, _, _ = circle_angles(make_result(pts))

        assert len(angles) == len(pts)
        assert np.isnan(angles[2])
        assert not np.any(np.isnan(angles[[0, 1, 3]]))

    def test_arbitrary_resolution(self):
        for res in (1.0, 5.0, 15.0, 45.0):
            angles, _, _ = circle_angles(make_result(circle_points("zx", res)))
            assert len(angles) == int(360 / res)


class TestCircleRejection:
    def test_plane_grid_is_not_circular(self):
        from bemkit.sim.fields import make_fields_plane
        result = make_result(make_fields_plane(dims="xz", res=100))
        with pytest.raises(NotCircular):
            circle_angles(result)
        assert not is_circular(result)

    def test_line_is_not_circular(self):
        from bemkit.sim.fields import make_fields_onaxis
        result = make_result(make_fields_onaxis("z", [0.1, 0.5, 1.0, 2.0]))
        assert not is_circular(result)

    def test_offset_plane_is_not_circular(self):
        pts = circle_points("zx", res=45.0)
        pts[:, 1] += 0.5  # circle in y = 0.5: angles about the origin are moot
        assert not is_circular(make_result(pts))

    def test_varying_radius_is_not_circular(self):
        pts = circle_points("zx", res=45.0)
        pts[3] *= 1.5
        with pytest.raises(NotCircular, match="not on"):
            circle_angles(make_result(pts))

    def test_spiral_within_tolerance_is_still_circular(self):
        pts = circle_points("zx", res=45.0) * (1.0 + 1e-4)
        assert is_circular(make_result(pts))


class TestAngleIndices:
    def test_default_fan_on_a_five_degree_set(self):
        result = make_result(circle_points("zx", res=5.0))
        indices, matched = angle_indices(result)

        assert list(matched) == pytest.approx(list(DEFAULT_ANGLES))
        assert len(set(indices.tolist())) == len(indices)

    def test_requests_between_samples_snap(self):
        result = make_result(circle_points("zx", res=45.0))
        _, matched = angle_indices(result, [30.0])
        assert matched[0] == pytest.approx(45.0)

    def test_uncovered_directions_are_dropped(self):
        # Half circle, 0°…180°: nothing at -30/-60/-90.
        pts = circle_points("zx", res=10.0)
        angles_all, _, _ = circle_angles(make_result(pts))
        result = make_result(pts[angles_all >= 0])

        _, matched = angle_indices(result)
        assert list(matched) == pytest.approx([0.0, 30.0, 60.0, 90.0])

    def test_wraparound_matches_across_the_seam(self):
        result = make_result(circle_points("zx", res=10.0))
        _, matched = angle_indices(result, [180.0])
        assert abs(matched[0]) == pytest.approx(180.0)

    def test_duplicate_requests_collapse(self):
        result = make_result(circle_points("zx", res=90.0))
        indices, _ = angle_indices(result, [0.0, 10.0, 20.0])
        assert len(indices) == 1


class TestAxisPositions:
    def test_signed_positions_along_z(self):
        from bemkit.sim.fields import make_fields_onaxis
        radii = [-1.0, 0.1, 0.5, 2.0]
        result = make_result(make_fields_onaxis("z", radii))

        positions, axis = axis_positions(result)
        assert axis == 2
        assert list(positions) == pytest.approx(radii)
        assert is_onaxis(result)

    def test_circle_is_not_on_axis(self):
        with pytest.raises(NotOnAxis, match="not collinear"):
            axis_positions(make_result(circle_points("zx", res=45.0)))

    def test_offset_line_is_rejected(self):
        from bemkit.sim.fields import make_fields_onaxis
        pts = make_fields_onaxis("z", [0.1, 0.5, 1.0])
        pts[:, 0] += 0.2  # parallel to z but not through the origin
        with pytest.raises(NotOnAxis, match="offset from the origin"):
            axis_positions(make_result(pts))

    def test_dropped_points_stay_aligned_as_nan(self):
        from bemkit.sim.fields import make_fields_onaxis
        pts = make_fields_onaxis("z", [0.1, 0.5, 1.0])
        pts[1] = np.nan
        positions, axis = axis_positions(make_result(pts))
        assert np.isnan(positions[1])
        assert positions[[0, 2]] == pytest.approx([0.1, 1.0])
