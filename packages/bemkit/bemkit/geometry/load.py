"""Read a mesh file into a Mesh, in metres.

Unit conversion happens here and nowhere else. Everything downstream of a Mesh
is metres, which retires the ".msh is millimetres, .stl is metres" trap that
used to live in every consumer.
"""

from pathlib import Path

import numpy as np

from bemkit.model import Mesh

#: File extension -> metres per file unit. GMSH's .msh is millimetres by
#: convention here; tessellated formats are written in metres.
SCALE_BY_SUFFIX = {".msh": 0.001, ".stl": 1.0, ".ply": 1.0,
                   ".obj": 1.0, ".mesh": 1.0}


def load_mesh(path, scale=None):
    """Load ``path`` as a Mesh in metres.

    Args:
        path: mesh file.
        scale: metres per file unit. Defaults to SCALE_BY_SUFFIX for the
            extension; pass it explicitly for anything unusual.
    """
    import meshio

    path = Path(path)
    if scale is None:
        scale = SCALE_BY_SUFFIX.get(path.suffix.lower())
        if scale is None:
            raise ValueError(
                f"unknown units for {path.suffix!r} — pass scale= (metres per "
                f"file unit). Known: "
                f"{', '.join(f'{k}={v}' for k, v in SCALE_BY_SUFFIX.items())}")

    m = meshio.read(path)
    tri_blocks = [b.data for b in m.cells if b.type == "triangle"]
    if not tri_blocks:
        raise ValueError(f"no triangle elements in {path}")
    tri = np.vstack(tri_blocks)
    groups = _physical_groups(m, tri)
    return Mesh(m.points * scale, tri, groups)


def _physical_groups(m, triangles):
    """{name: element indices} from a mesh file's physical groups, if any."""
    tags = m.cell_data_dict.get("gmsh:physical", {}).get("triangle")
    if tags is None or not getattr(m, "field_data", None):
        return {}
    by_tag = {int(v[0]): name for name, v in m.field_data.items()}
    return {name: np.nonzero(tags == tag)[0]
            for tag, name in by_tag.items()
            if np.any(tags == tag)}
