"""Tests for bemkit.geometry.generate.mesh_sphere_cap.

This mesher exists so that `models/sphcap_meshing` can be scored against a
closed form, which only means anything if the mesh really is the sphere and
the cap that closed form assumes. Everything checked here has an exact
answer, and every failure mode it covers is one that still meshes, still
solves, and quietly answers a different question.

gmsh is only needed to *generate*; nothing in bemkit needs it to read a mesh,
so the whole module skips without it.
"""

import math

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from bemkit.geometry import load_mesh                          # noqa: E402
from bemkit.geometry.generate import mesh_sphere_cap           # noqa: E402
from bemkit.geometry.quality import (                          # noqa: E402
    check_orientation, check_watertight)
from bemkit.geometry.sizing import edge_field, uniform_field   # noqa: E402

RADIUS_MM = 50.0
ANGLE_DEG = 30.0


def _cap(tmp_path, name="cap", *, sizing=None, angle_deg=ANGLE_DEG, lc=8.0):
    path = mesh_sphere_cap(tmp_path / f"{name}.msh",
                           sizing if sizing is not None else uniform_field(lc),
                           radius_mm=RADIUS_MM, angle_deg=angle_deg)
    return load_mesh(path)


def _areas(mesh):
    v = mesh.vertices[mesh.triangles]
    return 0.5 * np.linalg.norm(
        np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1) * 1e6


def exact_cap_area(angle_deg=ANGLE_DEG):
    return (2 * math.pi * RADIUS_MM ** 2
            * (1 - math.cos(math.radians(angle_deg))))


class TestTheMeshIsTheSphere:
    def test_every_vertex_is_on_the_sphere(self, tmp_path):
        mesh = _cap(tmp_path)
        radii = np.linalg.norm(mesh.vertices, axis=1) * 1e3
        np.testing.assert_allclose(radii, RADIUS_MM, rtol=1e-9)

    def test_the_surface_is_closed(self, tmp_path):
        mesh = _cap(tmp_path)
        assert check_watertight(mesh.vertices, mesh.triangles).level == "ok"

    def test_normals_point_outward(self, tmp_path):
        """Not merely consistent — outward. A uniformly inverted sphere is
        perfectly consistent, passes NumCalc's -check_normals, and solves a
        different problem."""
        mesh = _cap(tmp_path)
        assert check_orientation(mesh.vertices, mesh.triangles).level == "ok"

    def test_the_two_revolutions_share_their_rim(self, tmp_path):
        """removeAllDuplicates is load-bearing: without it each revolution
        keeps its own copy of the cap edge circle and the surface has a slit
        along the rim, which check_watertight above would see as holes."""
        mesh = _cap(tmp_path)
        assert len(np.unique(mesh.triangles)) == len(mesh.vertices)


class TestTheGroupsAreTheCap:
    def test_groups_are_named_the_way_selectors_expect(self, tmp_path):
        assert set(_cap(tmp_path).groups) == {"vibrating", "rigid"}

    def test_the_driven_group_subtends_the_cap_angle(self, tmp_path):
        """The one way a study could pass every comparison while measuring the
        wrong physics: a driven group that is not the cap the closed form
        assumes."""
        mesh = _cap(tmp_path)
        colatitude = np.degrees(np.arccos(np.clip(
            mesh.vertices[:, 2] / np.linalg.norm(mesh.vertices, axis=1),
            -1, 1)))
        driven = np.unique(mesh.triangles[mesh.groups["vibrating"]])
        rigid = np.unique(mesh.triangles[mesh.groups["rigid"]])
        assert colatitude[driven].max() == pytest.approx(ANGLE_DEG, abs=1e-6)
        assert colatitude[rigid].min() == pytest.approx(ANGLE_DEG, abs=1e-6)

    @pytest.mark.parametrize("angle_deg", [10.0, 30.0, 90.0])
    def test_the_cap_angle_is_a_parameter(self, tmp_path, angle_deg):
        """5 % rather than 1 %: at a fixed element size a small cap has a
        proportionally coarser rim polygon, and the 10 degree case really is
        3 % short. That deficit is the study's subject, not a defect — what is
        pinned here is only that `angle_deg` selects the cap it names."""
        mesh = _cap(tmp_path, f"a{angle_deg}", angle_deg=angle_deg, lc=4.0)
        cap = _areas(mesh)[mesh.groups["vibrating"]].sum()
        assert cap == pytest.approx(exact_cap_area(angle_deg), rel=0.05)

    @pytest.mark.parametrize("angle_deg", [0.0, 180.0, 200.0, -5.0])
    def test_a_cap_angle_outside_the_open_range_is_rejected(
            self, tmp_path, angle_deg):
        with pytest.raises(ValueError, match="half-angle"):
            _cap(tmp_path, "bad", angle_deg=angle_deg)


class TestRefinementShrinksTheDeficit:
    def test_the_faceted_cap_converges_on_the_spherical_one(self, tmp_path):
        """Flat triangles inscribed in a sphere under-state its area, and for
        the cap that deficit is the dominant error in a low-frequency BEM
        result. It has to fall with h — a mesher that refined without closing
        the gap would leave a floor no solver could get under."""
        exact = exact_cap_area()
        deficits = [1 - _areas(m)[m.groups["vibrating"]].sum() / exact
                    for m in (_cap(tmp_path, "c", lc=10.0),
                              _cap(tmp_path, "f", lc=2.5))]
        assert deficits[0] > deficits[1] > 0
        assert deficits[1] < 0.005


class TestEdgeFieldRefinesTheRim:
    def test_the_band_is_around_the_rim_and_not_the_whole_surface(
            self, tmp_path):
        """What edge_field is for, and the reason it switches gmsh's boundary
        size extension off: with that left on, lc_min spreads from the rim
        across the entire sphere and the "band" is the whole mesh."""
        mesh = _cap(tmp_path, "rim", sizing=edge_field(1.5, 6.0, 2.0, 8.0))
        area = _areas(mesh)
        h = np.sqrt(4 * area / np.sqrt(3))
        centroids = mesh.vertices[mesh.triangles].mean(axis=1) * 1e3
        theta = math.radians(ANGLE_DEG)
        to_rim = np.hypot(
            np.hypot(centroids[:, 0], centroids[:, 1])
            - RADIUS_MM * math.sin(theta),
            centroids[:, 2] - RADIUS_MM * math.cos(theta))
        assert h[to_rim < 3.0].mean() == pytest.approx(1.5, rel=0.25)
        assert h[to_rim > 30.0].mean() == pytest.approx(6.0, rel=0.25)
