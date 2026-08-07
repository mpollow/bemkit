"""Per-variant NumCalc export: fieldpoints + mesh + NC.inp in one call."""

from dataclasses import dataclass, field
from pathlib import Path

from bemkit.meshing.io import (
    DEFAULT_VIBRATING_GROUP, MeshInfo, find_mesh_file,
    read_mesh_physical_groups, resolve_vibrating_group,
)
from bemkit.sim.fieldpoints import (
    combine_and_track, load_fieldpoint_sets, save_offsets, validate_points,
)
from bemkit.sim.writers import (
    write_eval_grid_elements, write_eval_grid_nodes, write_nc_inp,
    write_object_mesh_elements, write_object_mesh_nodes,
)


@dataclass(frozen=True)
class ExportInfo:
    """Per-variant output of the export stage."""

    variant: str
    nc_dir: Path
    mesh_info: MeshInfo
    n_nodes: int
    n_elements: int
    n_eval_points: int
    fieldpoint_sets: list[str] = field(default_factory=list)


def export_variant(config, variant, mesh_file=None, verbose=True):
    """Export fieldpoints, ObjectMesh, and NC.inp for one variant.

    Args:
        config: ModelConfig
        variant: variant name (numcalc/{variant}/ is created)
        mesh_file: mesh path override; default resolves via config.mesh_map()

    Returns:
        ExportInfo

    Raises:
        ValueError: no fieldpoint sets configured or found
        FileNotFoundError: mesh file missing
    """
    model_dir = config.model_dir
    nc_dir = model_dir / "numcalc" / variant
    if mesh_file is None:
        mesh_stem = config.mesh_map().get(variant, variant)
        try:
            mesh_file = find_mesh_file(model_dir / "mesh", mesh_stem)
        except FileNotFoundError as e:
            if config.variant_mesh_params(variant) is not None:
                raise FileNotFoundError(
                    f"{e} — variant '{variant}' is meshed from geometry/; "
                    f"run 'bem mesh' first") from e
            raise
    nc_dir.mkdir(parents=True, exist_ok=True)

    # -- fieldpoints --
    sets = load_fieldpoint_sets(config.fieldpoints_cfg, model_dir)
    valid_masks = {}
    for name, points in sets.items():
        mask = validate_points(points, mesh_file, is_internal=config.is_internal,
                               edge_length_factor=0.1,
                               symmetry=config.symmetry_cfg or None)
        if verbose:
            n = f"{mask.sum()}/{len(mask)}" if not mask.all() else f"all {len(points)}"
            print(f"  '{name}': {n} points valid")
        if not mask.all():
            valid_masks[name] = mask
    combined, offsets = combine_and_track(sets, valid_masks or None)
    if len(combined) == 0:
        raise ValueError(
            f"No valid evaluation points for variant '{variant}': all "
            f"{sum(len(p) for p in sets.values())} fieldpoint(s) were dropped by "
            f"validation (inside the mesh or too close to the surface). "
            f"Check the fieldpoint sets against the mesh.")
    write_eval_grid_nodes(combined, nc_dir)
    write_eval_grid_elements(len(combined), nc_dir)
    save_offsets(offsets, nc_dir / "eval_offsets.json")
    if verbose:
        print(f"EvalGrid: {len(combined)} nodes → {nc_dir}")

    # -- mesh --
    n_nodes = write_object_mesh_nodes(mesh_file, nc_dir)
    n_elements = write_object_mesh_elements(mesh_file, nc_dir)
    if verbose:
        print(f"ObjectMesh: {n_nodes} nodes, {n_elements} elements → {nc_dir}")

    # -- NC.inp --
    # Resolve BOUNDARY entries (including geometric near= anchors) against the
    # mesh, then write the concrete element ranges.
    from bemkit.sim.boundary import flatten, resolve_boundary
    groups = read_mesh_physical_groups(mesh_file)
    vib_name, vib_indices = resolve_vibrating_group(
        groups, config.vibrating_group())
    resolved = resolve_boundary(config.boundary_cfg, variant, mesh_file,
                                default_velocity=config.default_velocity)
    ranges = flatten(resolved)
    if verbose and vib_name:
        print(f"  Driven group '{vib_name}': {len(vib_indices)} elements")
    if vib_indices is None and not ranges and groups:
        # The mesh has groups but none of them is the driven one, and no
        # [BOUNDARY] makes up for it — NumCalc would solve a fully rigid
        # surface with no source and return silence without complaining.
        print(f"  WARNING: variant '{variant}' has no driven elements. The mesh "
              f"defines physical group(s) {', '.join(groups)} but none is named "
              f"'{DEFAULT_VIBRATING_GROUP}'. Name the driven one in config.toml:\n"
              f"    [groups]\n"
              f"    vibrating = \"<one of: {', '.join(groups)}>\"\n"
              f"  Otherwise the whole surface is rigid and the result is silence.")
    write_nc_inp(config, variant, nc_dir,
                 vibrating_indices=vib_indices,
                 boundary_ranges=ranges, verbose=verbose)

    # -- config snapshot (records the effective config + resolved BC) --
    from bemkit.sim.config_snapshot import write_snapshot
    write_snapshot(nc_dir, config, mesh_file=mesh_file, resolved_boundary=resolved)

    return ExportInfo(
        variant=variant,
        nc_dir=nc_dir,
        mesh_info=MeshInfo.from_file(mesh_file),
        n_nodes=n_nodes,
        n_elements=n_elements,
        n_eval_points=len(combined),
        fieldpoint_sets=list(sets.keys()),
    )
