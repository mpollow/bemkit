"""Sources, field generators, and the Model that ties the spine together."""

import numpy as np
import pytest

from bemkit.model import (
    BC, Medium, Mesh, Model, Near, PlaneWave, PointSource, Velocity,
    circle, fft_grid, on_axis, plane,
)

VERT = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
TRI = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
MESH = Mesh(VERT, TRI, groups={"vibrating": np.array([0])})


def test_fft_grid_excludes_dc():
    f = fft_grid(8000, 5)
    np.testing.assert_allclose(f, [1000, 2000, 3000, 4000])


def test_on_axis_points_lie_on_one_axis():
    p = on_axis("z", [0.1, 0.5])
    assert p.shape == (2, 3)
    np.testing.assert_allclose(p[:, 2], [0.1, 0.5])
    np.testing.assert_allclose(p[:, :2], 0)


def test_on_axis_empty_distances():
    p = on_axis("z", [])
    assert p.shape == (0, 3)


def test_circle_is_closed_and_in_plane():
    p = circle("zx", res=90, radius=2.0)
    assert p.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(p, axis=1), 2.0)
    np.testing.assert_allclose(p[:, 1], 0)


def test_circle_res_not_dividing_360_still_ends_near_start():
    # res=100 does not divide 360 evenly; the last point should still land
    # within one step of closing the loop, not silently truncate it.
    p = circle("xy", res=100, radius=1.0)
    ang = np.arctan2(p[:, 1], p[:, 0])
    # Points should be within a step of the full circle, not stop far short.
    assert len(p) >= 3
    np.testing.assert_allclose(np.linalg.norm(p, axis=1), 1.0)


def test_plane_grid_shape():
    p = plane("xz", res=0.5, limits=(-1, 1, -1, 1))
    assert p.shape == (25, 3)


def test_plane_wave_direction_must_not_be_zero():
    with pytest.raises(ValueError, match="all zeros"):
        PlaneWave((0, 0, 0))


def test_point_source_carries_a_position():
    s = PointSource((0, 0, -1.5), amplitude=2.0)
    np.testing.assert_allclose(s.position, [0, 0, -1.5])


def test_model_resolves_boundary_conditions_to_elements():
    m = Model(MESH, fft_grid(8000, 5), Medium(),
              bcs=[BC(Near(MESH.centroids[0]), Velocity(1.0))])
    resolved = m.resolved_bcs()
    assert len(resolved) == 1
    np.testing.assert_array_equal(resolved[0].elements, [0])
    assert resolved[0].condition.kind == "VELO"


def test_model_rejects_an_empty_frequency_list():
    with pytest.raises(ValueError, match="at least one frequency"):
        Model(MESH, np.array([]), Medium())


def test_model_rejects_unsorted_frequencies():
    with pytest.raises(ValueError, match="ascending"):
        Model(MESH, np.array([100.0, 50.0]), Medium())


def test_fields_are_a_plain_dict_of_arrays():
    m = Model(MESH, fft_grid(8000, 5), Medium(),
              fields={"onaxis": on_axis("z", [1.0, 2.0])})
    assert m.fields["onaxis"].shape == (2, 3)
    assert m.n_field_points == 2


def test_fields_must_be_n_by_3():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        Model(MESH, fft_grid(8000, 5), Medium(), fields={"bad": np.zeros((2, 2))})


def test_uniform_band_detection():
    assert Model(MESH, fft_grid(8000, 5), Medium()).is_uniform_band
    assert not Model(MESH, np.array([100.0, 200.0, 400.0]),
                     Medium()).is_uniform_band


def test_uniform_band_rejects_evenly_spaced_band_not_starting_at_its_step():
    # [150, 250, 350] is evenly spaced (step 100) but does not start at the
    # step -- it is not the DC-excluded FFT grid for any (sr, nbins), so it
    # must not report as uniform.
    m = Model(MESH, np.array([150.0, 250.0, 350.0]), Medium())
    assert not m.is_uniform_band


def test_medium_default_is_not_replaced_by_falsy_check():
    # Medium has no __bool__/__len__, so every instance is truthy and `medium
    # or Medium()` cannot silently replace a caller-supplied Medium.
    custom = Medium(c=340.0, rho=1.0)
    m = Model(MESH, fft_grid(8000, 5), custom)
    assert m.medium is custom


def test_empty_field_set_is_accepted():
    # A field set that has been filtered down to nothing (e.g. by upstream
    # ray-cast validation) is a legitimate degenerate case, not an error.
    m = Model(MESH, fft_grid(8000, 5), Medium(), fields={"empty": np.zeros((0, 3))})
    assert m.fields["empty"].shape == (0, 3)
    assert m.n_field_points == 0


def test_field_points_reject_non_finite_values():
    bad = np.array([[0.0, 0.0, np.nan]])
    with pytest.raises(ValueError, match="finite"):
        Model(MESH, fft_grid(8000, 5), Medium(), fields={"bad": bad})


def test_plane_grid_does_not_overshoot_limits():
    # res=0.3 does not evenly divide the 2 m span; a naive
    # arange(a0, a1 + res/2, res) overshoots to 1.1, outside the requested
    # limits entirely.
    p = plane("xz", res=0.3, limits=(-1, 1, -1, 1))
    assert p[:, 0].max() <= 1.0 + 1e-9
    assert p[:, 2].max() <= 1.0 + 1e-9


def test_circle_rejects_non_positive_resolution():
    with pytest.raises(ValueError, match="res"):
        circle("xy", res=0)


def test_plane_rejects_non_positive_resolution():
    with pytest.raises(ValueError, match="res"):
        plane("xy", res=0)
