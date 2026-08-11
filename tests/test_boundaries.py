"""Enforce the layer dependency rules from the design spec.

config/meshing/sim/numcalc never import analysis or viz — not even lazily — and
never touch the viz libraries (pyvista/matplotlib) directly either.
__init__, pipeline, and cli may use the upper layers only inside functions
(lazily), never at module top level.

numcalc is stricter still: it is the NumCalc format and process layer and
imports *nothing* from bemkit at all, on stdlib and numpy only. That is what
keeps it extractable as a standalone package.

(Geometry authoring moved to the external `bemgeo` package, so `cad/` is no
longer part of bemkit — see CLAUDE.md.)
"""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "bemkit"
UPPER = ("bemkit.analysis", "bemkit.viz")
VIZ_LIBS = ("pyvista", "matplotlib")
CORE = ["config.py", "meshing", "sim", "numcalc"]
GLUE = ["__init__.py", "pipeline.py", "web.py", "cli"]


def _py_files(entry):
    p = PKG / entry
    return [p] if p.is_file() else sorted(p.rglob("*.py"))


def _all_imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module, node.lineno


def _top_level_imports(tree):
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, node.lineno
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module, node.lineno


def _violations(entries, getter, prefixes=UPPER):
    bad = []
    for entry in entries:
        for f in _py_files(entry):
            for mod, lineno in getter(ast.parse(f.read_text())):
                if mod.startswith(prefixes):
                    bad.append(f"{f.relative_to(PKG.parent)}:{lineno} imports {mod}")
    return bad


def test_layout_exists():
    for entry in CORE + GLUE:
        assert (PKG / entry).exists(), f"missing {entry} — layout changed?"


def test_core_never_imports_upper_layers():
    assert _violations(CORE, _all_imports) == []


def test_core_never_uses_viz_libraries():
    assert _violations(CORE, _all_imports, VIZ_LIBS) == []


def test_glue_imports_upper_layers_only_lazily():
    assert _violations(GLUE, _top_level_imports) == []


def test_numcalc_never_imports_bemkit():
    """bemkit/numcalc/ is the NumCalc format and process layer: it knows the
    NC.inp grammar, be.out and the solver binary, and nothing about config.toml,
    meshes or variants. Keeping it free of *every* bemkit import is what makes
    it extractable as a standalone package (docs/numcalc-feature-coverage.md
    §4) — and the direction rots within a month if nothing checks it.

    bemkit.sim may import bemkit.numcalc. Never the reverse.
    """
    bad = []
    for f in _py_files("numcalc"):
        for mod, lineno in _all_imports(ast.parse(f.read_text())):
            if mod == "bemkit" or mod.startswith("bemkit."):
                # Self-imports within the layer are fine.
                if mod.startswith("bemkit.numcalc"):
                    continue
                bad.append(f"{f.relative_to(PKG.parent)}:{lineno} imports {mod}")
    assert bad == [], (
        "bemkit/numcalc must not import from bemkit:\n  " + "\n  ".join(bad))


def test_numcalc_uses_only_stdlib_and_numpy():
    """The layer's dependency budget is numpy — no meshio, no gmsh, no scipy.
    A mesh reader creeping in here is how it would stop being extractable."""
    banned = ("meshio", "gmsh", "scipy", "matplotlib", "pyvista")
    bad = []
    for f in _py_files("numcalc"):
        for mod, lineno in _all_imports(ast.parse(f.read_text())):
            if mod.split(".")[0] in banned:
                bad.append(f"{f.relative_to(PKG.parent)}:{lineno} imports {mod}")
    assert bad == [], (
        "bemkit/numcalc may only use stdlib and numpy:\n  " + "\n  ".join(bad))
