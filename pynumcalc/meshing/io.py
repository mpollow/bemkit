"""Mesh file I/O: reading, metadata, and geometry helpers.

For mesh generation (GMSH), see pynumcalc.meshing.generate.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Material IDs
RIGID = 1
VIBRATING = 2


@dataclass(frozen=True)
class MeshInfo:
    """Metadata about a mesh file produced or discovered by the pipeline."""

    mesh_file: Path
    mesh_format: str   # ".msh", ".stl", etc.
    scale: float       # 0.001 for .msh, 1.0 otherwise

    @classmethod
    def from_file(cls, mesh_file):
        """Create MeshInfo from an existing mesh file path."""
        mesh_file = Path(mesh_file)
        return cls(
            mesh_file=mesh_file,
            mesh_format=mesh_file.suffix,
            scale=get_mesh_scale(mesh_file),
        )

# Supported mesh file extensions (in priority order)
MESH_EXTENSIONS = ['.msh', '.stl', '.ply', '.mesh']


def find_mesh_file(model_dir, modelname):
    """Find mesh file by name or stem.

    If modelname includes a known extension (e.g. "LR.mesh"), the file is
    looked up directly — no priority search. If modelname is a bare stem
    (e.g. "LR"), extensions are tried in priority order: .msh, .stl, .ply, .mesh.

    Args:
        model_dir: Path to model directory
        modelname: Mesh filename (with or without extension)

    Returns:
        Path to the found mesh file

    Raises:
        FileNotFoundError: If no mesh file is found
    """
    model_dir = Path(model_dir)
    if Path(modelname).suffix in MESH_EXTENSIONS:
        path = model_dir / modelname
        if path.exists():
            return path
        raise FileNotFoundError(f"Mesh file not found: {path}")
    for ext in MESH_EXTENSIONS:
        path = model_dir / f"{modelname}{ext}"
        if path.exists():
            return path
    tried = [f"{modelname}{ext}" for ext in MESH_EXTENSIONS]
    raise FileNotFoundError(
        f"No mesh file found in {model_dir}. Tried: {', '.join(tried)}"
    )


def get_mesh_scale(mesh_file):
    """Scale factor to convert mesh coordinates to meters.

    .msh files use mm (GMSH convention) -> 0.001
    Other formats (.stl, .ply, .mesh) assumed meters -> 1.0

    Args:
        mesh_file: Path to mesh file

    Returns:
        Float scale factor (0.001 for .msh, 1.0 otherwise)
    """
    return 0.001 if Path(mesh_file).suffix == '.msh' else 1.0


def read_mesh_triangles(mesh_file):
    """Read mesh from file and return vertices in meters and triangle indices.

    Args:
        mesh_file: Path to mesh file (.msh, .stl, .ply, .mesh)

    Returns:
        Tuple of (vertices, triangles) where:
            vertices: (N, 3) node coordinates in meters
            triangles: (T, 3) triangle connectivity (0-indexed)
    """
    import meshio
    mesh = meshio.read(str(mesh_file))
    scale = get_mesh_scale(mesh_file)
    vertices = mesh.points * scale
    tri_list = [c.data for c in mesh.cells if c.type == 'triangle']
    if not tri_list:
        raise ValueError(f"No triangles in {mesh_file}")
    return vertices, np.vstack(tri_list)


def read_mesh_physical_groups(mesh_file: str | Path) -> dict:
    """Read physical group assignments from GMSH .msh file.

    Args:
        mesh_file: Path to .msh file

    Returns:
        Dictionary mapping physical group names to element indices.
        E.g., {"rigid": array([0,1,2,...]), "vibrating": array([100,101,...])}
    """
    import meshio

    mesh = meshio.read(str(mesh_file))
    groups = {}
    element_offset = 0  # Track global element index across blocks

    # Meshio splits elements into separate blocks by physical group
    for block_idx, block in enumerate(mesh.cells):
        if block.type == "triangle":
            n_elements = len(block.data)

            # Get physical group tags for this block
            if "gmsh:physical" in mesh.cell_data:
                tags = mesh.cell_data["gmsh:physical"][block_idx]

                # Map tag IDs to names
                for tag_id in np.unique(tags):
                    # Find physical name for this tag
                    name = f"group_{tag_id}"  # fallback
                    if hasattr(mesh, 'field_data') and mesh.field_data:
                        for pname, (pid, _) in mesh.field_data.items():
                            if pid == tag_id:
                                name = pname
                                break

                    # Get element indices for this group (with global offset)
                    local_indices = np.where(tags == tag_id)[0]
                    global_indices = local_indices + element_offset

                    # Append to existing group or create new
                    if name in groups:
                        groups[name] = np.concatenate([groups[name], global_indices])
                    else:
                        groups[name] = global_indices

            element_offset += n_elements

    return groups


def element_areas(nodes, elements):
    """Triangle areas of a surface mesh.

    Args:
        nodes: (M, 3) node coordinates
        elements: (T, 3) triangle node indices (0-indexed)

    Returns:
        (T,) areas in squared input units
    """
    v0 = nodes[elements[:, 0]]
    v1 = nodes[elements[:, 1]]
    v2 = nodes[elements[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
