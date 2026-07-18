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
                   dist_min=50.0, dist_max=100.0):
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

    Returns:
        Path to the generated .msh file
    """
    import gmsh

    step_file = Path(step_file)
    output_file = Path(output_file)

    gmsh.initialize()
    try:
        return _mesh_from_step(gmsh, step_file, output_file,
                               lc_min, lc_max, dist_min, dist_max)
    finally:
        # Always finalize — a leftover initialized gmsh makes every later
        # call in this process fail on initialize().
        gmsh.finalize()


def _mesh_from_step(gmsh, step_file, output_file,
                    lc_min, lc_max, dist_min, dist_max):
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
        d = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(d, "FacesList", vibrating)

        t = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(t, "IField", d)
        gmsh.model.mesh.field.setNumber(t, "LcMin", lc_min)
        gmsh.model.mesh.field.setNumber(t, "LcMax", lc_max)
        gmsh.model.mesh.field.setNumber(t, "DistMin", dist_min)
        gmsh.model.mesh.field.setNumber(t, "DistMax", dist_max)

        gmsh.model.mesh.field.setAsBackgroundMesh(t)

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
