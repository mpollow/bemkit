"""Fusing overlapping bodies with manifold3d."""

import numpy as np
import pytest

from bemkit.geometry.combine import (changed_elements, dilate_elements,
                                     min_clearance, shells, submesh,
                                     unchanged_elements)
from bemkit.model import Mesh

from .meshes import enclosed_volume, sphere

manifold3d = pytest.importorskip("manifold3d",
                                 reason="boolean ops need bemkit[csg]")

from bemkit.geometry.boolean import union            # noqa: E402


def _ball(center=(0.0, 0.0, 0.0), radius=1.0, subdiv=3):
    return Mesh(*sphere(center, radius, subdiv))


# ----------------------------------------------------------------------- union

def test_union_of_overlapping_balls_is_one_closed_body():
    out = union(a=_ball(radius=0.1, subdiv=4),
                b=_ball(center=(0.19, 0, 0), radius=0.1, subdiv=3))

    assert len(shells(out)) == 1
    assert enclosed_volume(out.vertices, out.triangles) > 0
    #: two 0.1 m balls overlapping by 10 mm — less than twice one ball's volume
    one = 4.0 / 3.0 * np.pi * 0.1 ** 3
    assert one < enclosed_volume(out.vertices, out.triangles) < 2 * one


def test_union_preserves_the_untouched_part_of_the_larger_body_exactly():
    """The property the frozen-head pipeline depends on. manifold3d's default
    Mesh type is float32 and would perturb every vertex; Mesh64 does not."""
    head = _ball(radius=0.1, subdiv=4)
    device = _ball(center=(0.19, 0, 0), radius=0.1, subdiv=3)

    out = union(head=head, device=device)

    kept = {tuple(v) for v in out.vertices}
    survived = [v for v in head.vertices if tuple(v) in kept]
    assert len(survived) > 0.9 * head.n_nodes


def test_the_head_vertices_a_union_drops_are_the_ones_inside_the_device():
    head = _ball(radius=0.1, subdiv=4)
    device = _ball(center=(0.19, 0, 0), radius=0.1, subdiv=3)

    kept = {tuple(v) for v in union(head=head, device=device).vertices}
    lost = np.array([v for v in head.vertices if tuple(v) not in kept])

    #: every lost vertex is within the device's radius of its centre
    assert np.all(np.linalg.norm(lost - np.array([0.19, 0, 0]), axis=1)
                  <= 0.1 + 1e-9)


def test_union_names_each_body_and_covers_every_element():
    head = _ball(radius=0.1, subdiv=4)
    device = _ball(center=(0.19, 0, 0), radius=0.1, subdiv=3)

    out = union(head=head, device=device)

    assert set(out.groups) == {"head", "device"}
    total = sum(len(g) for g in out.groups.values())
    assert total == out.n_elements
    #: and the partition is disjoint
    assert len(np.unique(np.concatenate(list(out.groups.values())))) == total


def test_union_attribution_puts_each_group_on_the_right_side():
    head = _ball(radius=0.1, subdiv=4)
    device = _ball(center=(0.19, 0, 0), radius=0.1, subdiv=3)

    out = union(head=head, device=device)
    head_x = out.centroids[out.groups["head"]][:, 0].mean()
    device_x = out.centroids[out.groups["device"]][:, 0].mean()
    assert head_x < device_x


def test_union_of_disjoint_bodies_leaves_two_shells():
    """Nothing to fuse is not an error — the result is simply both bodies."""
    out = union(a=_ball(radius=1.0, subdiv=2),
                b=_ball(center=(5, 0, 0), radius=1.0, subdiv=2))
    assert len(shells(out)) == 2


def test_union_needs_at_least_two_bodies():
    with pytest.raises(ValueError, match="at least two bodies"):
        union(only=_ball())


def test_union_rejects_a_surface_with_holes():
    """An open surface is not an oriented 2-manifold, and manifold3d silently
    returns empty rather than raising — so we turn that into a real error
    naming the checks that diagnose it.
    models/hutubs21_glassesearbuds's openbuds_parts.stl
    is exactly this case."""
    v, t = sphere(subdiv=2)
    holed = Mesh(v, t[:-4])                        # remove four faces
    with pytest.raises(ValueError, match="not an oriented 2-manifold"):
        union(a=holed, b=_ball(center=(0.5, 0, 0), radius=1.0, subdiv=2))


# --------------------------------------------------- dirty-zone identification

def test_changed_elements_is_empty_when_nothing_changed():
    mesh = _ball(subdiv=2)
    assert len(changed_elements(mesh, mesh)) == 0
    assert len(unchanged_elements(mesh, mesh)) == mesh.n_elements


def test_changed_elements_finds_only_what_the_union_reworked():
    head = _ball(radius=0.1, subdiv=4)
    device = _ball(center=(0.19, 0, 0), radius=0.1, subdiv=3)
    out = union(head=head, device=device)

    dirty = changed_elements(head, out)
    clean = unchanged_elements(head, out)
    assert len(dirty) + len(clean) == out.n_elements

    #: everything left verbatim from the head is on the far side from the seam,
    #: which sits between the two centres at x ≈ 0.095
    assert out.centroids[clean][:, 0].max() < 0.19


def test_changed_elements_survives_a_renumbering():
    """Identity has to be geometric: mmgs and manifold3d both renumber."""
    mesh = _ball(subdiv=2)
    order = np.random.default_rng(0).permutation(mesh.n_elements)
    shuffled = Mesh(mesh.vertices, mesh.triangles[order])
    assert len(changed_elements(mesh, shuffled)) == 0


def test_dilate_grows_a_seed_and_contains_it():
    mesh = _ball(radius=1.0, subdiv=3)
    seed = np.array([0])
    grown = dilate_elements(mesh, seed, radius=0.5)

    assert set(seed.tolist()) <= set(grown.tolist())
    assert len(grown) > len(seed)
    assert len(grown) < mesh.n_elements


def test_dilate_with_a_bigger_radius_grows_further():
    mesh = _ball(radius=1.0, subdiv=3)
    small = dilate_elements(mesh, [0], radius=0.2)
    big = dilate_elements(mesh, [0], radius=0.8)
    assert len(big) > len(small)
    assert set(small.tolist()) <= set(big.tolist())


def test_dilate_of_nothing_is_nothing():
    assert len(dilate_elements(_ball(subdiv=2), [], radius=1.0)) == 0


def test_geodesic_dilation_does_not_jump_an_air_gap():
    """The reason dilation is geodesic: a halo around a seam on the temple must
    not swallow the pinna a few millimetres away across the gap."""
    near = _ball(radius=1.0, subdiv=3)
    far = _ball(center=(2.01, 0, 0), radius=1.0, subdiv=3)
    from bemkit.geometry.combine import concat_meshes
    both = concat_meshes(a=near, b=far)
    assert min_clearance(near, far) < 0.02

    seed = both.groups["a"][:1]
    grown = dilate_elements(both, seed, radius=5.0, geodesic=True)
    #: 5 m of surface path cannot reach the other body — there is no path
    assert set(grown.tolist()).isdisjoint(set(both.groups["b"].tolist()))

    #: through space, 5 m reaches everything
    through = dilate_elements(both, seed, radius=5.0, geodesic=False)
    assert not set(through.tolist()).isdisjoint(set(both.groups["b"].tolist()))


# ----------------------------------------------------- the two policies agree

def test_union_and_concat_produce_the_same_group_names():
    """Stage B swaps one for the other, so a build script must not care which
    it called."""
    from bemkit.geometry.combine import concat_meshes

    head = _ball(radius=0.1, subdiv=3)
    device = _ball(center=(0.19, 0, 0), radius=0.1, subdiv=2)

    fused = union(head=head, glasses=device)
    apart = concat_meshes(head=head, glasses=device)
    assert set(fused.groups) <= set(apart.groups)
    assert {"head", "glasses"} <= set(fused.groups)
