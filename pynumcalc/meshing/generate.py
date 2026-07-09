"""Parametric mesh generation from STEP/PLY files using GMSH.

GMSH is imported lazily -- only needed when generating meshes.
"""

from pathlib import Path

from pynumcalc.meshing.io import RIGID, VIBRATING

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
    import numpy as np

    step_file = Path(step_file)
    output_file = Path(output_file)

    gmsh.initialize()
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
        print("  Set boundary conditions via [BOUNDARY] in vars.toml")
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
    gmsh.finalize()

    return output_file


def mesh_from_ply(ply_file, output_file,
                  lc_min=5.0, lc_max=25.0,
                  dist_min=50.0, dist_max=100.0):
    """Generate a GMSH surface mesh from a PLY scan.

    Intended for acoustic BEM on scanned geometries (e.g. head + pinna).

    The PLY is treated as a discrete, already-clean surface:
    - must be watertight
    - must have consistent normals
    - no topology repair is performed here

    A distance-based size field is applied globally.

    Args:
        ply_file: Path to input PLY surface mesh
        output_file: Path for output .msh file
        lc_min: Minimum element size (mm)
        lc_max: Maximum element size (mm)
        dist_min: Distance field minimum range (mm)
        dist_max: Distance field maximum range (mm)

    Returns:
        Path to the generated .msh file
    """
    import gmsh
    import numpy as np

    ply_file = Path(ply_file)
    output_file = Path(output_file)

    if not ply_file.exists():
        raise FileNotFoundError(f"Input mesh not found: {ply_file}")

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    print(f"Loading PLY surface: {ply_file}")
    gmsh.merge(str(ply_file))

    # Convert from meters (PLY) to millimeters (GMSH/BEM convention)
    print("Converting units from meters to millimeters...")
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    for i, tag in enumerate(node_tags):
        xyz = [node_coords[3*i] * 1000,
               node_coords[3*i+1] * 1000,
               node_coords[3*i+2] * 1000]
        gmsh.model.mesh.setNode(tag, xyz, [])

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
        gmsh.finalize()
        raise RuntimeError("No surface entities found in PLY file")

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

    gmsh.finalize()
    return output_file
