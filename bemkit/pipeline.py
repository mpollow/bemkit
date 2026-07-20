"""Pipeline stage interface and stage implementations.

Stages cover mesh → export → run → pack. Plotting is CLI-level (it needs
the viz layer, which core must not import).
"""

import os
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path

from bemkit.config import ModelConfig
from bemkit.meshing.io import MESH_EXTENSIONS, MeshInfo, find_mesh_file
from bemkit.sim.export import export_variant
from bemkit.sim.packing import clear_pack, has_results, is_packed, pack_variant
from bemkit.sim.runner import plan_jobs, run_jobs
from bemkit.sim.status import solver_warnings


class PipelineStage(ABC):
    """Base class for BEM pipeline stages."""

    @abstractmethod
    def run(self, config: ModelConfig, **kwargs):
        """Execute the stage. Returns stage output data object(s)."""

    @abstractmethod
    def check(self, config: ModelConfig, variant: str) -> bool:
        """Check if stage output already exists for a variant."""


def _run_exec_scripts(model_dir, exec_spec):
    """Run a section's exec script(s) with cwd=model_dir."""
    if not exec_spec:
        return
    scripts = [exec_spec] if isinstance(exec_spec, str) else exec_spec
    # Scripts run with cwd=model_dir, so a repo-checkout bemkit (not pip-
    # installed) would not be importable — put its root on PYTHONPATH.
    env = os.environ.copy()
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = (pkg_root + os.pathsep + env["PYTHONPATH"]
                         if env.get("PYTHONPATH") else pkg_root)
    for s in scripts:
        script = Path(model_dir) / s
        if not script.exists():
            raise FileNotFoundError(f"exec script not found: {script}")
        cmd = ([sys.executable, str(script)] if script.suffix == ".py"
               else ["bash", str(script)])
        print(f"\n{'=' * 60}\n  {' '.join(cmd)}\n{'=' * 60}")
        r = subprocess.run(cmd, cwd=str(model_dir), env=env)
        if r.returncode:
            raise RuntimeError(f"exec script failed (exit {r.returncode}): {s}")


def _generate_mesh(input_file, output_file, mesh_cfg):
    """Generate output_file from one geometry/ source (CAD or scan)."""
    from bemkit.meshing.generate import mesh_from_step, mesh_from_surface  # lazy gmsh

    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True)
    is_step = input_file.suffix == ".step"
    defaults = (5.0, 25.0, 50.0, 100.0) if is_step else (2.0, 4.0, 20.0, 80.0)
    params = {k: mesh_cfg.get(k, d) for k, d in
              zip(("lc_min", "lc_max", "dist_min", "dist_max"), defaults)}
    fn = mesh_from_step if is_step else mesh_from_surface

    # Kept out of the zip above: unlike the four numeric knobs, `distance` is
    # not shared with mesh_from_surface. Only passed through when it is not the
    # default, so the euclidean path stays byte-identical.
    distance = mesh_cfg.get("distance", "euclidean")
    if distance == "geodesic":
        if not is_step:
            raise ValueError(
                f'distance = "geodesic" is only supported for STEP/CAD '
                f'sources; {input_file.name} is a scan surface. Remove the '
                f'key or set distance = "euclidean".')
        params["distance"] = distance

    print(f"Generating surface mesh from {input_file} → {output_file.name}")
    print(f"  Mesh size: lc_min={params['lc_min']} mm, lc_max={params['lc_max']} mm")
    print(f"  Distance field: {distance}, dist_min={params['dist_min']} mm, "
          f"dist_max={params['dist_max']} mm")
    fn(input_file, output_file, **params)
    return output_file


def _resolve_source(sources, variant, source_stem):
    """Pick the geometry/ source a generated variant is meshed from."""
    available = ", ".join(f.stem for f in sources) or "none"
    if source_stem:
        match = [f for f in sources if f.stem == source_stem]
        if not match:
            raise FileNotFoundError(
                f"variant '{variant}': geometry source '{source_stem}' not "
                f"found in geometry/ (available: {available})")
        return match[0]
    match = [f for f in sources if f.stem == variant]
    if match:
        return match[0]
    if len(sources) == 1:
        return sources[0]
    raise ValueError(
        f"variant '{variant}': cannot determine geometry source — set "
        f"source = \"<stem>\" in [variants.{variant}] (available: {available})")


class MeshStage(PipelineStage):
    def check(self, config, variant):
        stem = config.mesh_map().get(variant, variant)
        try:
            find_mesh_file(config.model_dir / "mesh", stem)
            return True
        except FileNotFoundError:
            return False

    def run(self, config, **kwargs):
        from bemkit.meshing.generate import SURFACE_EXTENSIONS

        model_dir = config.model_dir
        leftovers = sorted(s.name for s in model_dir.glob("pre_*.[ps][yh]")
                           if not s.name.startswith("__"))
        if leftovers:
            print(f"  Note: pre_* scripts are no longer run automatically "
                  f"({', '.join(leftovers)}) — reference them via "
                  f"[geometry].exec or [fieldpoints].exec in config.toml.")
        _run_exec_scripts(model_dir, config.geometry_cfg.get("exec"))

        # geometry/ holds remesh sources (CAD .step + tessellated scans);
        # everything in mesh/ is a trusted, ready-to-use mesh, never remeshed.
        geom_dir = model_dir / "geometry"
        mesh_dir = model_dir / "mesh"
        sources = sorted(
            (f for f in (geom_dir.iterdir() if geom_dir.is_dir() else [])
             if f.suffix in ['.step'] + SURFACE_EXTENSIONS), key=lambda f: f.name)

        variants_tbl = config.raw.get("variants") or {}
        if variants_tbl:
            return self._run_variants(config, mesh_dir, sources, variants_tbl)

        results = []
        if sources:
            mesh_files = sorted(f for f in (mesh_dir.iterdir() if mesh_dir.is_dir()
                                            else []) if f.suffix in MESH_EXTENSIONS)
            for src in sources:
                derived = [f for f in mesh_files if f.stem == src.stem]
                if derived:
                    print(f"  Mesh exists, skipping {src.name}: "
                          f"{', '.join(f.name for f in derived)}")
                    results += [MeshInfo.from_file(f) for f in derived]
                else:
                    results.append(MeshInfo.from_file(_generate_mesh(
                        src, mesh_dir / f"{src.stem}.msh", config.mesh_cfg)))
        else:
            existing = sorted(
                (f for f in (mesh_dir.iterdir() if mesh_dir.is_dir() else [])
                 if f.suffix in MESH_EXTENSIONS), key=lambda f: f.name)
            if existing:
                print(f"  Mesh files already exist: {', '.join(f.name for f in existing)}")
                results = [MeshInfo.from_file(f) for f in existing]
            else:
                print("  No geometry sources found "
                      "(nothing in geometry/, no mesh files in mesh/)")
        return results

    def _run_variants(self, config, mesh_dir, sources, variants_tbl):
        """Mesh [variants] entries: generated ones get mesh/<variant>.msh."""
        results = []
        for name in variants_tbl:
            params = config.variant_mesh_params(name)
            if params is None:
                # Trusted mesh/ file (e.g. BC-defined variant); never remeshed.
                stem = config.mesh_map()[name]
                try:
                    f = find_mesh_file(mesh_dir, stem)
                except FileNotFoundError:
                    print(f"  {name}: trusted mesh mesh/{stem}.* not found "
                          f"(export will fail until it exists)")
                    continue
                print(f"  {name}: trusted mesh {f.name}")
                results.append(MeshInfo.from_file(f))
                continue
            output = mesh_dir / f"{name}.msh"
            if output.exists():
                print(f"  Mesh exists, skipping {name}: {output.name}")
                results.append(MeshInfo.from_file(output))
                continue
            src = _resolve_source(sources, name, config.variant_source(name))
            results.append(MeshInfo.from_file(_generate_mesh(src, output, params)))
        return results


class ExportStage(PipelineStage):
    def check(self, config, variant):
        return (config.model_dir / "numcalc" / variant / "NC.inp").exists()

    def run(self, config, *, prefixes=None, **kwargs):
        _run_exec_scripts(config.model_dir, config.fieldpoints_cfg.get("exec"))
        variants = config.variants(prefixes)
        if not variants:
            all_v = config.variants(None)
            if not all_v:
                raise FileNotFoundError(
                    f"No mesh files in {config.model_dir / 'mesh'}. Run 'mesh' first.")
            raise ValueError(f"No variants matching: {', '.join(prefixes)}\n"
                             f"  Available: {', '.join(all_v)}")
        results = []
        for v in variants:
            print(f"\n--- Export: {v} ---")
            results.append(export_variant(config, v))
        return results


class RunStage(PipelineStage):
    def check(self, config, variant):
        return has_results(config.model_dir / "numcalc" / variant)

    def run(self, config, *, prefixes=None, max_jobs=None, split=None,
            force=False, **kwargs):
        # Default is resume: only bins absent from be.out//be.npz are
        # computed, so interrupted runs continue and complete runs are
        # no-ops. --force recomputes everything (discarding old results) —
        # needed after a re-export that changed the mesh or config.
        missing_only = not force
        variants = config.variants(prefixes)
        if not variants:
            raise FileNotFoundError(
                f"No mesh files in {config.model_dir / 'mesh'}. Run 'export' first.")
        from bemkit.sim.status import numcalc_pids
        pairs = []
        for v in variants:
            nc_dir = config.model_dir / "numcalc" / v
            if not nc_dir.is_dir():
                print(f"  {v}: skipping (no numcalc/{v}/ — run 'export' first)")
                continue
            pids = numcalc_pids(nc_dir)
            if pids:
                # A second NumCalc in the same dir would race the first one's
                # be.out/ writes and corrupt bins.
                print(f"  {v}: skipping (NumCalc already running, "
                      f"pid {', '.join(map(str, pids))})")
                continue
            pairs.append((v, nc_dir))
        nf = config.nbins - 1
        jobs = plan_jobs(pairs, nf, split=split, missing_only=missing_only)
        if not jobs:
            print("  Nothing to run (all bins present — use --force to recompute).")
            return [], []
        if shutil.which("NumCalc") is None:
            raise RuntimeError(
                "NumCalc not found on PATH — install Mesh2HRTF's NumCalc "
                "or add its directory to PATH.")
        # A leftover be.npz would shadow the fresh be.out/ results; for a
        # resume, restore its bins into be.out/ first so the runs merge.
        job_dirs = {j.nc_dir for j in jobs}
        for nc_dir in sorted(job_dirs):
            clear_pack(nc_dir, restore=missing_only)
            if not missing_only:
                # Full recompute: NumCalc wipes be.out/ but not old logs,
                # whose warnings would otherwise stay in `bem status` forever.
                for log in nc_dir.glob("NC*.out"):
                    log.unlink()
        n_at_once = max_jobs or len(jobs)
        print(f"\n{'=' * 60}\n  Running NumCalc ({len(jobs)} job(s), "
              f"{n_at_once} at a time)\n{'=' * 60}")
        completed, failed = run_jobs(jobs, max_jobs=max_jobs)
        for v, nc_dir in pairs:
            logs = [j.log_name for j in jobs if j.nc_dir == nc_dir]
            issues = solver_warnings(nc_dir, logs=logs)
            if issues:
                bins = sorted({b for _, b, _ in issues if b is not None})
                where = f" (bins {', '.join(map(str, bins))})" if bins else ""
                print(f"  {v}: {len(issues)} solver warning(s) in logs{where}"
                      f" — non-converged bins usually mean a defective mesh")
        return completed, failed


class PackStage(PipelineStage):
    def check(self, config, variant):
        return is_packed(config.model_dir / "numcalc" / variant)

    def run(self, config, *, prefixes=None, delete_original=False, **kwargs):
        from bemkit.sim.discovery import discover_variants
        from bemkit.sim.readers import list_bins
        # dedup: a variant can surface twice (e.g. cone.msh + cone.ply) — pack once.
        variants = list(dict.fromkeys(
            discover_variants(config.model_dir, prefixes, need_be_out=True)))
        if not variants:
            print(f"  No variants with results found in {config.model_dir / 'numcalc'}")
            return []
        packed = []
        for v in variants:
            nc_dir = config.model_dir / "numcalc" / v
            be_out = nc_dir / "be.out"
            if not be_out.is_dir():
                print(f"  {v}: already packed" if is_packed(nc_dir)
                      else f"  {v}: no be.out/ to pack")
                continue
            if not list_bins(be_out):
                # be.out/ exists but has no bin results (exported, never run) —
                # skip it rather than aborting the whole model's pack.
                print(f"  {v}: be.out/ has no bin results — skipping")
                continue
            print(f"\n--- Pack: {v} ---")
            pack_variant(nc_dir, nf=config.nbins - 1,
                         delete_original=delete_original)
            packed.append(v)
        return packed


STAGES = {
    "mesh": MeshStage(),
    "export": ExportStage(),
    "run": RunStage(),
    "pack": PackStage(),
}
