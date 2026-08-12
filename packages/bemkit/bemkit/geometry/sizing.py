"""Size-field builders for gmsh meshing.

Each builder returns a `sizing` callable ``apply(gmsh, sources) -> None`` that
installs a background size field on the live gmsh model. `sources` is the list
of surface tags to measure distance from (the vibrating faces); `uniform_field`
ignores it. Callables carry `.kind` and `.params` for config mapping and tests.

The geodesic builder delegates the distance math to `bemkit.geometry.geodesic`
(pure numpy/scipy) and owns only the gmsh two-pass orchestration.
"""


def _tag(fn, kind, params):
    fn.kind = kind
    fn.params = params
    return fn


def uniform_field(h):
    """Constant element size `h` everywhere (a flat Threshold on the sources)."""
    def apply(gmsh, sources):
        f = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(f, "F", str(float(h)))
        gmsh.model.mesh.field.setAsBackgroundMesh(f)
    return _tag(apply, "uniform", {"h": h})


def distance_field(lc_min, lc_max, dist_min, dist_max):
    """Euclidean straight-line distance from `sources` → gmsh Distance+Threshold."""
    def apply(gmsh, sources):
        d = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(d, "FacesList", sources)
        t = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(t, "IField", d)
        gmsh.model.mesh.field.setNumber(t, "LcMin", lc_min)
        gmsh.model.mesh.field.setNumber(t, "LcMax", lc_max)
        gmsh.model.mesh.field.setNumber(t, "DistMin", dist_min)
        gmsh.model.mesh.field.setNumber(t, "DistMax", dist_max)
        gmsh.model.mesh.field.setAsBackgroundMesh(t)
    return _tag(apply, "euclidean",
                {"lc_min": lc_min, "lc_max": lc_max,
                 "dist_min": dist_min, "dist_max": dist_max})


def geodesic_field(lc_min, lc_max, dist_min, dist_max):
    """Over-the-surface distance from `sources` via a two-pass pilot mesh.

    Pilot mesh → multi-source Dijkstra (bemkit.geometry.geodesic) → size ramp →
    PostView background view (survives mesh.clear()) → the real graded mesh.
    """
    def apply(gmsh, sources):
        import numpy as np
        from bemkit.geometry.geodesic import (
            geodesic_distance, size_from_distance, pack_scalar_triangles)

        pilot_lc = max(lc_min, min(lc_max, dist_max / 20.0))
        print(f"Generating pilot mesh for geodesic sizing (lc={pilot_lc:.3g} mm)...")
        gmsh.option.setNumber("Mesh.MeshSizeMin", pilot_lc)
        gmsh.option.setNumber("Mesh.MeshSizeMax", pilot_lc)
        gmsh.model.mesh.generate(2)

        tags, coords, _ = gmsh.model.mesh.getNodes()
        tags = tags.astype(np.int64)
        nodes = coords.reshape(-1, 3)
        lut = np.zeros(tags.max() + 1, dtype=np.int64)
        lut[tags] = np.arange(len(tags))
        _, elem_nodes = gmsh.model.mesh.getElementsByType(2)   # ALL entities: paths cross rigid faces
        triangles = lut[elem_nodes.astype(np.int64)].reshape(-1, 3)

        from bemkit.geometry.io import VIBRATING
        vib_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(2, VIBRATING)
        source = lut[np.asarray(vib_tags, dtype=np.int64)]
        if source.size == 0:
            raise RuntimeError(
                f"geodesic sizing: the vibrating surface has no nodes in the "
                f"pilot mesh (pilot lc={pilot_lc:.3g} mm). Lower lc_min or dist_max.")
        print(f"  Pilot: {len(nodes)} nodes, {len(triangles)} triangles, "
              f"{len(source)} source nodes")

        d = geodesic_distance(nodes, triangles, source)
        unreachable = int(np.isinf(d).sum())
        if unreachable:
            pct = 100.0 * unreachable / len(d)
            print(f"  WARNING: {unreachable}/{len(d)} pilot nodes ({pct:.1f}%) are "
                  f"not reachable over the surface from the vibrating faces.")
            print(f"           They were sized at lc_max={lc_max} mm. Model may be "
                  f"non-watertight or made of disconnected shells.")

        sizes = size_from_distance(d, lc_min, lc_max, dist_min, dist_max)
        data = pack_scalar_triangles(nodes, triangles, sizes)

        view = gmsh.view.add("geodesic_size")
        gmsh.view.addListData(view, "ST", len(triangles), data.tolist())
        gmsh.model.mesh.clear()
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 1e22)
        # Do NOT suppress MeshSizeExtendFromBoundary/FromPoints: measured on
        # examples/sphcap, suppressing them overshoots lc_max by ~50%.
        f = gmsh.model.mesh.field.add("PostView")
        gmsh.model.mesh.field.setNumber(f, "ViewTag", view)   # ViewTag, not ViewIndex
        gmsh.model.mesh.field.setAsBackgroundMesh(f)
    return _tag(apply, "geodesic",
                {"lc_min": lc_min, "lc_max": lc_max,
                 "dist_min": dist_min, "dist_max": dist_max})


def sizing_from_config(mesh_cfg):
    """Map an unchanged [mesh] config dict to a sizing builder.

    Keys: lc_min, lc_max, dist_min, dist_max, distance ("euclidean"|"geodesic").
    Preserves current behaviour: lc_min == lc_max still routes through the
    distance field (equal bounds → constant), so meshes stay byte-identical.
    """
    lc_min, lc_max = mesh_cfg["lc_min"], mesh_cfg["lc_max"]
    dist_min, dist_max = mesh_cfg["dist_min"], mesh_cfg["dist_max"]
    if mesh_cfg.get("distance", "euclidean") == "geodesic":
        return geodesic_field(lc_min, lc_max, dist_min, dist_max)
    return distance_field(lc_min, lc_max, dist_min, dist_max)
