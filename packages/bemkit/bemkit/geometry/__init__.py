"""Model meshing: mesh file I/O, GMSH generation."""

from bemkit.geometry.io import (
    RIGID, VIBRATING, MESH_EXTENSIONS, MeshInfo,
    find_mesh_file, get_mesh_scale, read_mesh_triangles,
    read_mesh_physical_groups, element_areas, mirror_mesh,
)
from bemkit.geometry.load import SCALE_BY_SUFFIX, load_mesh
