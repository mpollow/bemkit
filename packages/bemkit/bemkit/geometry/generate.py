"""Parametric mesh generation from CAD (STEP) and scan surfaces using GMSH.

GMSH is imported lazily -- only needed when generating meshes.
"""

import math
from pathlib import Path

from bemkit.geometry.io import RIGID, VIBRATING

# STEP surface colors for boundary condition detection
_RIGID_COLOR = (0, 0, 255, 255)
_VIBRATING_COLOR = (255, 0, 0, 255)


def mesh_from_step(step_file, output_file, sizing, *, configure=None):
    """Generate a GMSH mesh from a STEP file using a `sizing` builder.

    Reads color-coded surfaces (blue=rigid, red=vibrating) and hands the
    vibrating face tags to `sizing(gmsh, sources)`. `configure(gmsh)` runs
    just before generate() for advanced gmsh control.
    """
    import gmsh
    step_file, output_file = Path(step_file), Path(output_file)
    gmsh.initialize()
    try:
        return _mesh_from_step(gmsh, step_file, output_file, sizing, configure)
    finally:
        gmsh.finalize()


def _mesh_from_step(gmsh, step_file, output_file, sizing, configure):
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

    # Only install a graded field when there is a source to measure from.
    if vibrating:
        sizing(gmsh, vibrating)
    elif getattr(sizing, "kind", None) == "uniform":
        sizing(gmsh, [])
    elif getattr(sizing, "kind", None) == "geodesic":
        print("  No vibrating surface -- geodesic sizing has no source; "
              "mesh will be ungraded")

    return _run_mesh(gmsh, output_file, default_algorithm=6, configure=configure)


def _run_mesh(gmsh, output_file, *, default_algorithm, configure=None,
              extra_defaults=None):
    """Set overridable defaults, run configure(), generate, and write."""
    gmsh.option.setNumber("Mesh.Algorithm", default_algorithm)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    if extra_defaults:
        extra_defaults(gmsh)
    if configure is not None:
        configure(gmsh)   # advanced override hook (runs last, wins)

    print("Generating 2D mesh...")
    gmsh.model.mesh.generate(2)
    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    n_triangles = len(gmsh.model.mesh.getElementsByType(2)[0])
    print(f"Mesh generated: {n_nodes} nodes, {n_triangles} triangles")
    gmsh.write(str(output_file))
    print(f"Mesh saved: {output_file}")
    return output_file


def mesh_sphere_cap(output_file, sizing, *, radius_mm=50.0, angle_deg=30.0,
                    configure=None):
    """A rigid sphere with a vibrating polar cap, meshed with `sizing`.

    The verification geometry: a spherical cap has a closed-form exterior
    solution (:func:`bemkit.model.analytic.sphcap_pressure`), so a mesh of one
    can be scored against the exact answer rather than against another mesh.
    Written parametrically with the OCC kernel instead of shipping a STEP file
    per cap angle, because the whole point is to sweep the geometry.

    `angle_deg` is the cap's **half**-angle from +z, matching
    ``sphcap_pressure(angle_deg=...)``. Groups are named ``"vibrating"``
    (the cap) and ``"rigid"`` (the remainder), as everywhere else.

    Returns `output_file`.
    """
    import gmsh
    output_file = Path(output_file)
    gmsh.initialize()
    try:
        return _mesh_sphere_cap(gmsh, output_file, sizing, radius_mm,
                                angle_deg, configure)
    finally:
        gmsh.finalize()


def _mesh_sphere_cap(gmsh, output_file, sizing, radius_mm, angle_deg,
                     configure):
    if not 0.0 < angle_deg < 180.0:
        raise ValueError(
            f"angle_deg is the cap half-angle in (0, 180), got {angle_deg}")
    gmsh.option.setNumber("General.Terminal", 1)

    radius, theta = float(radius_mm), math.radians(angle_deg)
    occ = gmsh.model.occ

    # Two arcs revolved about +z, rather than a sphere cut by a boolean: the
    # cap edge is then a *shared* curve of the two faces by construction (what
    # sizing.edge_field measures from), and neither face is a trimmed fragment
    # whose area depends on the kernel's tolerance. Both arcs stay under 180
    # degrees for any half-angle in (0, 180), which is what addCircleArc needs.
    centre = occ.addPoint(0, 0, 0)
    north = occ.addPoint(0, 0, radius)
    edge = occ.addPoint(radius * math.sin(theta), 0, radius * math.cos(theta))
    south = occ.addPoint(0, 0, -radius)

    cap_arc = occ.addCircleArc(north, centre, edge)
    rest_arc = occ.addCircleArc(edge, centre, south)
    cap = _revolved_surface(occ, cap_arc)
    rest = _revolved_surface(occ, rest_arc)
    # The revolution sweeps clockwise about +z (see _revolved_surface): with
    # these arcs running north-to-south, that is the sense that puts the
    # normals *outward*. _check_mesh_is_a_closed_outward_sphere holds it there.

    # The two revolutions each produce their own copy of the cap edge circle
    # and of the pole point; without this the mesh would have a slit along the
    # rim and NumCalc would see an open surface.
    occ.removeAllDuplicates()
    occ.synchronize()

    _verify_sphere_split(gmsh, cap, rest, radius, theta)

    gmsh.model.addPhysicalGroup(2, [cap], VIBRATING)
    gmsh.model.setPhysicalName(2, VIBRATING, "vibrating")
    gmsh.model.addPhysicalGroup(2, [rest], RIGID)
    gmsh.model.setPhysicalName(2, RIGID, "rigid")

    sizing(gmsh, [cap])
    _run_mesh(gmsh, output_file, default_algorithm=6, configure=configure)
    _check_mesh_is_a_closed_outward_sphere(output_file, radius)
    return output_file


def _revolved_surface(occ, arc):
    """Revolve `arc` a full turn about +z and return the one surface tag.

    Negative angle: the sweep direction is what decides which side of the
    resulting surface gmsh calls the front, and so which way the mesh normals
    end up pointing. Inward normals are the error the paper calls the most
    common one that still produces plausible output, and NumCalc's own
    ``-check_normals`` cannot see it (a uniformly inverted mesh is perfectly
    consistent), so it is fixed here rather than left to the caller.
    """
    made = occ.revolve([(1, arc)], 0, 0, 0, 0, 0, 1, -2 * math.pi)
    surfaces = [tag for dim, tag in made if dim == 2]
    if len(surfaces) != 1:
        raise RuntimeError(
            f"revolving arc {arc} produced {len(surfaces)} surfaces, expected "
            "1 — gmsh's OCC kernel did not build a surface of revolution")
    return surfaces[0]


def _verify_sphere_split(gmsh, cap, rest, radius, theta):
    """Fail loudly if the two faces are not the sphere and the cap.

    A boolean or a revolution that went subtly wrong still meshes, still
    solves, and produces a study of a shape nobody chose. These are cheap
    closed forms, so there is no reason to find out later from the results.
    """
    exact_cap = 2 * math.pi * radius ** 2 * (1 - math.cos(theta))
    exact_total = 4 * math.pi * radius ** 2
    got_cap = gmsh.model.occ.getMass(2, cap)
    got_total = got_cap + gmsh.model.occ.getMass(2, rest)
    for what, got, want in (("cap", got_cap, exact_cap),
                            ("total", got_total, exact_total)):
        if abs(got / want - 1.0) > 1e-6:
            raise RuntimeError(
                f"sphere/cap split is wrong: {what} area is {got:.4f} mm^2, "
                f"expected {want:.4f} mm^2")


def _check_mesh_is_a_closed_outward_sphere(output_file, radius_mm):
    """Read the written mesh back and refuse to hand over a broken one.

    Everything checked here is cheap, has a closed form, and produces results
    that look entirely plausible when it is wrong — an inverted normal or a
    slit along the cap rim does not crash a solve, it just answers a different
    question. Duplicates part of ``bem STUDY check`` on purpose: this is the
    generator's own postcondition, and a study should not be the first thing
    to notice its mesher is broken.
    """
    import numpy as np

    from bemkit.geometry.load import load_mesh
    from bemkit.geometry.quality import check_orientation, check_watertight

    mesh = load_mesh(output_file)
    for finding in (check_watertight(mesh.vertices, mesh.triangles),
                    check_orientation(mesh.vertices, mesh.triangles)):
        if finding.level != "ok":
            raise RuntimeError(
                f"mesh_sphere_cap wrote a bad mesh to {output_file}: "
                f"{finding.check} — {finding.message}")

    radii = np.linalg.norm(mesh.vertices, axis=1) * 1e3
    if not np.allclose(radii, radius_mm, rtol=1e-9):
        raise RuntimeError(
            f"mesh_sphere_cap wrote vertices off the sphere: radii span "
            f"{radii.min():.6f}-{radii.max():.6f} mm, expected {radius_mm}")


# Tessellated surface formats accepted as remesh sources in geometry/
# (anything gmsh.merge can read; treated as meters, scaled to mm)
SURFACE_EXTENSIONS = ['.ply', '.stl', '.obj']


def mesh_from_surface(surface_file, output_file, sizing, *, configure=None):
    """Generate a GMSH surface mesh from a tessellated scan (.ply/.stl/.obj)."""
    import gmsh
    surface_file, output_file = Path(surface_file), Path(output_file)
    if not surface_file.exists():
        raise FileNotFoundError(f"Input mesh not found: {surface_file}")
    gmsh.initialize()
    try:
        return _mesh_from_surface(gmsh, surface_file, output_file, sizing, configure)
    finally:
        gmsh.finalize()


def _mesh_from_surface(gmsh, surface_file, output_file, sizing, configure):
    gmsh.option.setNumber("General.Terminal", 1)
    print(f"Loading surface: {surface_file}")
    gmsh.merge(str(surface_file))
    print("Converting units from meters to millimeters...")
    gmsh.model.mesh.affineTransform([1000, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1000, 0])
    gmsh.model.mesh.classifySurfaces(
        angle=40 * 3.141592653589793 / 180.0, boundary=True, forReparametrization=True)
    gmsh.model.mesh.createGeometry()
    surfaces = [tag for dim, tag in gmsh.model.getEntities(2)]
    if not surfaces:
        raise RuntimeError(f"No surface entities found in {surface_file.name}")
    SURFACE = 1
    gmsh.model.addPhysicalGroup(2, surfaces, SURFACE)
    gmsh.model.setPhysicalName(2, SURFACE, "surface")
    print(f"  Found {len(surfaces)} surface patch(es)")

    # Scan has no vibrating group; distance/geodesic measure from all surfaces.
    sizing(gmsh, surfaces)

    def scan_defaults(g):
        g.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        g.option.setNumber("Mesh.MeshSizeMin", 0.1)
        g.option.setNumber("Mesh.Optimize", 1)
        g.option.setNumber("Mesh.OptimizeNetgen", 1)

    return _run_mesh(gmsh, output_file, default_algorithm=1,
                     extra_defaults=scan_defaults, configure=configure)


# Backward-compatible name from when PLY was the only supported scan format
mesh_from_ply = mesh_from_surface
