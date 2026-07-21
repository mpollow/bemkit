"""Parametric mesh generation from CAD (STEP) and scan surfaces using GMSH.

GMSH is imported lazily -- only needed when generating meshes.
"""

from pathlib import Path

from bemkit.meshing.io import RIGID, VIBRATING

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
