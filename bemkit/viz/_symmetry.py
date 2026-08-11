"""Ghost geometry for models with [SYMMETRY].

A symmetry-reduced mesh stores one half/quarter/eighth of the object and lets
NumCalc reflect the elements internally. The mesh file therefore shows less
surface than the solver ever saw, while the evaluation points — authored in
full space and never mirrored — span the whole domain. Both 3D viewers use
this module to add the missing copies as a second "ghost" actor, so the picture
matches the problem that was actually solved.

The reflection itself lives in ``bemkit.meshing.io.mirror_mesh``, which
evaluation-point validation also uses; deriving the ghosts from the same call
keeps the drawn geometry and the validated geometry from drifting apart.
"""

from typing import NamedTuple

import numpy as np

# Edge styling for the *simulated* elements. Ghosts are drawn opaque and
# unstyled, so the wireframe is what distinguishes the elements NumCalc
# actually solved from the copies it reflected. Opacity cannot do that job:
# a see-through surface superimposes the pressure on its far side onto the
# near side, and on a horn those two are nothing alike.
EDGE_STYLE = {"edge_color": 'k', "edge_opacity": 0.1, "line_width": 3}

# NumCalc remaps the NC.inp plane code to an internal value (NC_Input.cpp:474)
# and only ever tests it for == 0, != 0 and < 0. So the magnitude (1/2 vs 3/4)
# never reaches the field computation, and only these two codes flip the sign
# of the reflected field (NC_3dFunctions.cpp:239).
_SIGN_FLIPPING_CODES = (2, 4)

# 1/2 mean the mesh is a fraction of the object and the ghosts are its real
# other half; 3/4 mean the mesh is the whole object and the ghosts are its
# image in an infinite baffle. Invisible to the solver, but not to a viewer.
_REFLECTING_CODES = (3, 4)


class GhostGeometry(NamedTuple):
    """The mirrored copies of a mesh, without the original.

    Attributes:
        nodes: (N, 3) ghost vertex coordinates
        elements: (M, 3) ghost triangles, indexing into ``nodes``
        source: (M,) index of the real element each ghost reflects, for
            looking up its boundary condition or surface scalar
        sign: (M,) +1.0 or -1.0, the factor the reflected field picks up from
            the antisymmetric planes that copy crossed
    """

    nodes: np.ndarray
    elements: np.ndarray
    source: np.ndarray
    sign: np.ndarray


def _active(symmetry):
    """(planes, coordinates) if any plane is active, else None."""
    if not symmetry:
        return None
    planes = list(symmetry.get("planes", [0, 0, 0]))
    if not any(planes):
        return None
    return planes, list(symmetry.get("coordinates", [0.0, 0.0, 0.0]))


def ghost_geometry(nodes, elements, symmetry):
    """Build the mirrored copies of a mesh that the solver adds implicitly.

    Args:
        nodes: (N, 3) vertex coordinates of the meshed surface
        elements: (M, 3) triangle connectivity
        symmetry: ``{"planes": [...], "coordinates": [...]}`` as returned by
            ``bemkit.numcalc.readers.read_nc_inp_symmetry``, or None

    Returns:
        A ``GhostGeometry``, or None when there is nothing to mirror — the
        single branch each viewer has to check.
    """
    from bemkit.meshing.io import mirror_mesh

    active = _active(symmetry)
    if active is None:
        return None
    planes, coordinates = active

    elements = np.asarray(elements)
    all_nodes, all_elements, copies = mirror_mesh(
        nodes, elements, planes, coordinates, return_copies=True)

    n_nodes, n_elem = len(nodes), len(elements)
    signs = [
        np.prod([-1.0 if planes[a] in _SIGN_FLIPPING_CODES else 1.0
                 for a in flipped] or [1.0])
        for flipped in copies[1:]
    ]

    return GhostGeometry(
        nodes=all_nodes[n_nodes:],
        elements=all_elements[n_elem:] - n_nodes,
        source=np.tile(np.arange(n_elem), len(copies) - 1),
        sign=np.repeat(signs, n_elem),
    )


def describe_symmetry(symmetry):
    """One-line description of the active planes and what the ghosts mean.

    The rendering is identical for a symmetric and a reflecting plane, so
    without a label a viewer can mistake a baffle image for real geometry.

    Returns:
        A printable string, or None when nothing is mirrored.
    """
    active = _active(symmetry)
    if active is None:
        return None
    planes, coordinates = active

    parts = []
    for axis, (code, position) in enumerate(zip(planes, coordinates)):
        if not code:
            continue
        kind = "virtual image" if code in _REFLECTING_CODES else "mirrored half"
        if code in _SIGN_FLIPPING_CODES:
            kind += ", antisymmetric"
        parts.append(f"{'xyz'[axis]} = {position:g} ({kind})")

    n_ghosts = 2 ** len(parts) - 1
    return f"{'; '.join(parts)} -> {n_ghosts} ghost cop{'y' if n_ghosts == 1 else 'ies'}"
