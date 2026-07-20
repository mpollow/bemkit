"""Parametric mesh generation from CAD (STEP) and scan surfaces using GMSH.

GMSH is imported lazily -- only needed when generating meshes.
"""

from pathlib import Path

from bemkit.meshing.io import RIGID, VIBRATING

# STEP surface colors for boundary condition detection
_RIGID_COLOR = (0, 0, 255, 255)
_VIBRATING_COLOR = (255, 0, 0, 255)


def mesh_from_step(step_file, output_file,
                   lc_min=5.0, lc_max=25.0,
                   dist_min=50.0, dist_max=100.0,
                   distance="euclidean"):
    """Generate GMSH mesh from a STEP file.

    Reads color-coded surfaces: blue = rigid, red = vibrating.
    Applies distance-based mesh refinement near vibrating surfaces.

    Args:
        step_file: Path to input STEP file
        output_file: Path for output .msh file
        lc_min: Minimum element size (mm)
        lc_max: Maximum element size (mm)
        dist_min: Distance field minimum range (mm)
        dist_max: Distance field maximum range (mm)
        distance: "euclidean" (straight-line through space, GMSH's own Distance
            field) or "geodesic" (measured over the surface -- see
            bemkit.meshing.geodesic). Geodesic costs an extra coarse meshing
            pass but stops geometry that folds back on itself, such as the
            outer wall of a horn, from being refined as if it were adjacent to
            the vibrating surface.

    Returns:
        Path to the generated .msh file
    """
    import gmsh

    step_file = Path(step_file)
    output_file = Path(output_file)

    gmsh.initialize()
    try:
        return _mesh_from_step(gmsh, step_file, output_file,
                               lc_min, lc_max, dist_min, dist_max, distance)
    finally:
        # Always finalize — a leftover initialized gmsh makes every later
        # call in this process fail on initialize().
        gmsh.finalize()


def _mesh_from_step(gmsh, step_file, output_file,
                    lc_min, lc_max, dist_min, dist_max,
                    distance="euclidean"):
    gmsh.option.setNumber("General.Terminal", 1)

    gmsh.merge(str(step_file))

    rigid, vibrating = [], []

    for dim, tag in gmsh.model.getEntities(2):
        color = gmsh.model.getColor(dim, tag)
        if color == _RIGID_COLOR:
            rigid.append(tag)
        elif color == _VIBRATING_COLOR:
            vibrating.append(tag)

    if not rigid and not vibrating:
        # No color tags -- treat everything as rigid so GMSH writes triangles
        all_surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]
        gmsh.model.addPhysicalGroup(2, all_surfaces, RIGID)
        gmsh.model.setPhysicalName(2, RIGID, "rigid")
        print(f"  No color tags found -- {len(all_surfaces)} surface(s) marked rigid")
        print("  Set boundary conditions via [BOUNDARY] in config.toml")
    else:
        if rigid:
            gmsh.model.addPhysicalGroup(2, rigid, RIGID)
            gmsh.model.setPhysicalName(2, RIGID, "rigid")
            print(f"  Found {len(rigid)} rigid surface(s)")

        if vibrating:
            gmsh.model.addPhysicalGroup(2, vibrating, VIBRATING)
            gmsh.model.setPhysicalName(2, VIBRATING, "vibrating")
            print(f"  Found {len(vibrating)} vibrating surface(s)")

    if vibrating:
        if distance == "geodesic":
            _apply_geodesic_field(gmsh, lc_min, lc_max, dist_min, dist_max)
        else:
            d = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(d, "FacesList", vibrating)

            t = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(t, "IField", d)
            gmsh.model.mesh.field.setNumber(t, "LcMin", lc_min)
            gmsh.model.mesh.field.setNumber(t, "LcMax", lc_max)
            gmsh.model.mesh.field.setNumber(t, "DistMin", dist_min)
            gmsh.model.mesh.field.setNumber(t, "DistMax", dist_max)

            gmsh.model.mesh.field.setAsBackgroundMesh(t)
    elif distance == "geodesic":
        print("  No vibrating surface -- geodesic sizing has no source; "
              "mesh will be ungraded")

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    print("Generating 2D mesh...")
    gmsh.model.mesh.generate(2)

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    n_triangles = len(gmsh.model.mesh.getElementsByType(2)[0])

    print(f"Mesh generated: {n_nodes} nodes, {n_triangles} triangles")
    gmsh.write(str(output_file))
    print(f"Mesh saved: {output_file}")

    return output_file


def _apply_geodesic_field(gmsh, lc_min, lc_max, dist_min, dist_max):
    """Build a size field from distance measured over the surface, not through it.

    GMSH has no geodesic field, so the metric is computed here and handed back
    as a background mesh view. That takes two passes: a coarse uniform pilot
    mesh to carry the distance computation, then the real graded mesh.
    """
    import numpy as np

    from bemkit.meshing.geodesic import (
        geodesic_distance, size_from_distance, pack_scalar_triangles)

    # The pilot is the graph the distances travel on, so it must resolve the
    # ramp -- not just be cheap. At lc_max the default ramp spans only ~4 hops,
    # which both inflates Dijkstra's metrication error well past its asymptotic
    # few percent and coarsely quantizes a small vibrating face. ~20 hops
    # across the ramp restores both. Never finer than the final mesh's finest
    # region, so the pilot stays the cheaper of the two passes.
    pilot_lc = max(lc_min, min(lc_max, dist_max / 20.0))

    print(f"Generating pilot mesh for geodesic sizing (lc={pilot_lc:.3g} mm)...")
    gmsh.option.setNumber("Mesh.MeshSizeMin", pilot_lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", pilot_lc)
    gmsh.model.mesh.generate(2)

    tags, coords, _ = gmsh.model.mesh.getNodes()
    tags = tags.astype(np.int64)
    nodes = coords.reshape(-1, 3)

    # GMSH node tags are 1-based but neither dense nor sorted, so remap through
    # a scatter table rather than searchsorted.
    lut = np.zeros(tags.max() + 1, dtype=np.int64)
    lut[tags] = np.arange(len(tags))

    # Deliberately every triangle, not just the vibrating group: a geodesic path
    # reaches the far side of the model by travelling across the rigid faces.
    _, elem_nodes = gmsh.model.mesh.getElementsByType(2)
    triangles = lut[elem_nodes.astype(np.int64)].reshape(-1, 3)

    vib_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(2, VIBRATING)
    source = lut[np.asarray(vib_tags, dtype=np.int64)]

    if source.size == 0:
        raise RuntimeError(
            f"geodesic sizing: the vibrating surface has no nodes in the "
            f"pilot mesh (pilot lc={pilot_lc:.3g} mm). It is probably smaller "
            f"than the pilot element size -- lower lc_min or dist_max.")

    print(f"  Pilot: {len(nodes)} nodes, {len(triangles)} triangles, "
          f"{len(source)} source nodes")

    d = geodesic_distance(nodes, triangles, source)

    unreachable = int(np.isinf(d).sum())
    if unreachable:
        pct = 100.0 * unreachable / len(d)
        print(f"  WARNING: {unreachable}/{len(d)} pilot nodes ({pct:.1f}%) are "
              f"not reachable over the surface from the vibrating faces.")
        print(f"           They were sized at lc_max={lc_max} mm. This usually "
              f"means the STEP model is not watertight, or consists of "
              f"disconnected shells.")

    sizes = size_from_distance(d, lc_min, lc_max, dist_min, dist_max)
    data = pack_scalar_triangles(nodes, triangles, sizes)

    # List-based view data carries its own coordinates, so unlike node-tagged
    # model data it survives the mesh.clear() below.
    view = gmsh.view.add("geodesic_size")
    gmsh.view.addListData(view, "ST", len(triangles), data.tolist())

    gmsh.model.mesh.clear()

    # These are hard clamps applied on top of the background field -- left at
    # pilot_lc they would pin the entire final mesh to the pilot size.
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", 1e22)

    # Note: the other Mesh.MeshSizeFrom*/ExtendFromBoundary options are left at
    # their defaults, exactly as the euclidean path leaves them. Suppressing
    # them so that lc_max is the only size source is tempting -- it removes a
    # ~3% far-field undershoot -- but measured on examples/sphcap it costs far
    # more than it buys: boundary-extended sizes are what hold element growth
    # in check across a curved face, and without them the far field overshoots
    # lc_max by ~50% (25 mm cap, 38 mm edges). Overshooting lc_max breaks the
    # elements-per-wavelength criterion it exists to enforce; undershooting it
    # only costs a few elements.

    f = gmsh.model.mesh.field.add("PostView")
    # ViewTag, not ViewIndex: the index is a position in the loaded-view list
    # and shifts if views are removed.
    gmsh.model.mesh.field.setNumber(f, "ViewTag", view)
    gmsh.model.mesh.field.setAsBackgroundMesh(f)


# Tessellated surface formats accepted as remesh sources in geometry/
# (anything gmsh.merge can read; treated as meters, scaled to mm)
SURFACE_EXTENSIONS = ['.ply', '.stl', '.obj']


def mesh_from_surface(surface_file, output_file,
                      lc_min=5.0, lc_max=25.0,
                      dist_min=50.0, dist_max=100.0):
    """Generate a GMSH surface mesh from a tessellated scan (.ply/.stl/.obj).

    Intended for acoustic BEM on scanned geometries (e.g. head + pinna).

    The input is treated as a discrete, already-clean surface:
    - must be watertight
    - must have consistent normals
    - no topology repair is performed here

    A distance-based size field is applied globally.

    Args:
        surface_file: Path to input surface mesh (any format gmsh can merge)
        output_file: Path for output .msh file
        lc_min: Minimum element size (mm)
        lc_max: Maximum element size (mm)
        dist_min: Distance field minimum range (mm)
        dist_max: Distance field maximum range (mm)

    Returns:
        Path to the generated .msh file
    """
    import gmsh

    surface_file = Path(surface_file)
    output_file = Path(output_file)

    if not surface_file.exists():
        raise FileNotFoundError(f"Input mesh not found: {surface_file}")

    gmsh.initialize()
    try:
        return _mesh_from_surface(gmsh, surface_file, output_file,
                                  lc_min, lc_max, dist_min, dist_max)
    finally:
        gmsh.finalize()


def _mesh_from_surface(gmsh, surface_file, output_file,
                       lc_min, lc_max, dist_min, dist_max):
    gmsh.option.setNumber("General.Terminal", 1)

    print(f"Loading surface: {surface_file}")
    gmsh.merge(str(surface_file))

    # Convert from meters (surface scan) to millimeters (GMSH/BEM convention).
    # One bulk affine transform instead of a per-node setNode loop, which is
    # one API round-trip per node and crawls on large scans.
    print("Converting units from meters to millimeters...")
    gmsh.model.mesh.affineTransform(
        [1000, 0, 0, 0,
         0, 1000, 0, 0,
         0, 0, 1000, 0])

    # Classify discrete surface and create geometry
    # Angle threshold is deliberately high to avoid artificial feature splitting
    gmsh.model.mesh.classifySurfaces(
        angle=40 * 3.141592653589793 / 180.0,
        boundary=True,
        forReparametrization=True
    )
    gmsh.model.mesh.createGeometry()

    surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]

    if not surfaces:
        raise RuntimeError(f"No surface entities found in {surface_file.name}")

    SURFACE = 1
    gmsh.model.addPhysicalGroup(2, surfaces, SURFACE)
    gmsh.model.setPhysicalName(2, SURFACE, "surface")

    print(f"  Found {len(surfaces)} surface patch(es)")

    # Distance-based mesh size field (global)
    d = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(d, "FacesList", surfaces)

    t = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(t, "IField", d)
    gmsh.model.mesh.field.setNumber(t, "LcMin", lc_min)
    gmsh.model.mesh.field.setNumber(t, "LcMax", lc_max)
    gmsh.model.mesh.field.setNumber(t, "DistMin", dist_min)
    gmsh.model.mesh.field.setNumber(t, "DistMax", dist_max)

    gmsh.model.mesh.field.setAsBackgroundMesh(t)

    # Meshing options suitable for BEM
    gmsh.option.setNumber("Mesh.Algorithm", 1)       # MeshAdapt
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)

    gmsh.option.setNumber("Mesh.ElementOrder", 1)    # Linear triangles
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

    print("Generating 2D surface mesh...")
    gmsh.model.mesh.generate(2)

    nodes = gmsh.model.mesh.getNodes()[0]
    tris = gmsh.model.mesh.getElementsByType(2)[0]

    print(f"Mesh generated: {len(nodes)} nodes, {len(tris)} triangles")

    gmsh.write(str(output_file))
    print(f"Mesh saved: {output_file}")

    return output_file


# Backward-compatible name from when PLY was the only supported scan format
mesh_from_ply = mesh_from_surface
