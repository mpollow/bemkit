"""Model meshing: mesh file I/O, GMSH generation."""

from bemkit.geometry.io import (
    RIGID, VIBRATING, MESH_EXTENSIONS, MeshInfo,
    find_mesh_file, get_mesh_scale, read_mesh_triangles,
    read_mesh_physical_groups, element_areas, mirror_mesh,
)
from bemkit.geometry.load import SCALE_BY_SUFFIX, load_mesh
from bemkit.geometry.generate import (
    mesh_from_step, mesh_from_surface, mesh_sphere_cap,
)
from bemkit.geometry.sizing import (
    distance_field, edge_field, geodesic_field, patch_edge_curves,
    uniform_field,
)
