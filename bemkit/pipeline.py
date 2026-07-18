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
from bemkit.meshing.io import MESH_EXTENSIONS, MeshInfo
from bemkit.sim.export import export_variant
from bemkit.sim.packing import clear_pack, has_results, is_packed, pack_variant
from bemkit.sim.runner import plan_jobs, run_jobs
from bemkit.sim.status import is_complete, solver_warnings


class PipelineStage(ABC):
    """Base class for BEM pipeline stages."""

    @abstractmethod
    def run(self, config: ModelConfig, **kwargs):
        """Execute the stage. Returns stage output data object(s)."""

    @abstractmethod
    def check(self, config: ModelConfig, variant: str) -> bool:
        """Check if stage output already exists for a variant."""


def _generate_mesh(config, input_file):
    """Generate mesh/{stem}.msh from one geometry/ source (CAD or scan)."""
    from bemkit.meshing.generate import mesh_from_step, mesh_from_surface  # lazy gmsh

    input_file = Path(input_file)
    mesh_dir = config.model_dir / "mesh"
    mesh_dir.mkdir(exist_ok=True)
    output_file = mesh_dir / f"{input_file.stem}.msh"
    is_step = input_file.suffix == ".step"
    defaults = (5.0, 25.0, 50.0, 100.0) if is_step else (2.0, 4.0, 20.0, 80.0)
    lc_min = config.mesh_cfg.get("lc_min", defaults[0])
    lc_max = config.mesh_cfg.get("lc_max", defaults[1])
    dist_min = config.mesh_cfg.get("dist_min", defaults[2])
    dist_max = config.mesh_cfg.get("dist_max", defaults[3])
    fn = mesh_from_step if is_step else mesh_from_surface

    print(f"Generating surface mesh from {input_file}")
    print(f"  Mesh size: lc_min={lc_min} mm, lc_max={lc_max} mm")
    print(f"  Distance field: dist_min={dist_min} mm, dist_max={dist_max} mm")
    fn(input_file, output_file, lc_min=lc_min, lc_max=lc_max,
       dist_min=dist_min, dist_max=dist_max)
    return output_file


class MeshStage(PipelineStage):
    def check(self, config, variant):
        mesh_dir = config.model_dir / "mesh"
        return mesh_dir.is_dir() and any(
            f.suffix in MESH_EXTENSIONS for f in mesh_dir.iterdir())

    def run(self, config, **kwargs):
        model_dir = config.model_dir
        # Scripts run with cwd=model_dir, so a repo-checkout bemkit (not pip-
        # installed) would not be importable — put its root on PYTHONPATH.
        env = os.environ.copy()
        pkg_root = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = (pkg_root + os.pathsep + env["PYTHONPATH"]
                             if env.get("PYTHONPATH") else pkg_root)
        for s in sorted(model_dir.glob("pre_*.[ps][yh]")):
            if s.name.startswith("__"):
                continue
            cmd = [sys.executable if s.suffix == ".py" else "bash", str(s)]
            print(f"\n{'=' * 60}\n  {' '.join(cmd)}\n{'=' * 60}")
            r = subprocess.run(cmd, cwd=str(model_dir), env=env)
            if r.returncode:
                raise RuntimeError(f"pre-script failed (exit {r.returncode}): {s.name}")

        # geometry/ holds remesh sources (CAD .step + tessellated scans);
        # everything in mesh/ is a trusted, ready-to-use mesh, never remeshed.
        from bemkit.meshing.generate import SURFACE_EXTENSIONS
        geom_dir = model_dir / "geometry"
        mesh_dir = model_dir / "mesh"
        sources = sorted(
            (f for f in (geom_dir.iterdir() if geom_dir.is_dir() else [])
             if f.suffix in ['.step'] + SURFACE_EXTENSIONS), key=lambda f: f.name)

        results = []
        if sources:
            mesh_files = sorted(f for f in (mesh_dir.iterdir() if mesh_dir.is_dir()
                                            else []) if f.suffix in MESH_EXTENSIONS)
            for src in sources:
                # A pre_* script may have meshed this source under
                # recipe-suffixed names (stem.recipe.msh) — those count too.
                # Match on the full source stem so a dotted source name
                # (head.v2.step) is not confused with head.step recipes.
                derived = [f for f in mesh_files
                           if f.stem == src.stem
                           or f.stem.startswith(src.stem + '.')]
                if derived:
                    print(f"  Mesh exists, skipping {src.name}: "
                          f"{', '.join(f.name for f in derived)}")
                    results += [MeshInfo.from_file(f) for f in derived]
                else:
                    results.append(MeshInfo.from_file(_generate_mesh(config, src)))
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


class ExportStage(PipelineStage):
    def check(self, config, variant):
        return (config.model_dir / "numcalc" / variant / "NC.inp").exists()

    def run(self, config, *, prefixes=None, **kwargs):
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
            missing_only=False, force=False, **kwargs):
        variants = config.variants(prefixes)
        if not variants:
            raise FileNotFoundError(
                f"No mesh files in {config.model_dir / 'mesh'}. Run 'export' first.")
        pairs = []
        for v in variants:
            nc_dir = config.model_dir / "numcalc" / v
            if not nc_dir.is_dir():
                print(f"  {v}: skipping (no numcalc/{v}/ — run 'export' first)")
                continue
            pairs.append((v, nc_dir))
        nf = config.nbins - 1
        # A full (non-resume) re-run wipes be.out/ before recomputing. Refuse to
        # discard already-complete results unless explicitly forced.
        if not missing_only and not force:
            complete = [v for v, nc_dir in pairs if is_complete(nc_dir, nf)]
            if complete:
                raise RuntimeError(
                    f"{len(complete)} variant(s) already complete "
                    f"({', '.join(complete)}). A full re-run would discard those "
                    f"results. Use --missing-only to resume, or --force to recompute.")
        jobs = plan_jobs(pairs, nf, split=split, missing_only=missing_only)
        if not jobs:
            print("  Nothing to run (all bins present).")
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
            pack_variant(nc_dir, delete_original=delete_original)
            packed.append(v)
        return packed


STAGES = {
    "mesh": MeshStage(),
    "export": ExportStage(),
    "run": RunStage(),
    "pack": PackStage(),
}
