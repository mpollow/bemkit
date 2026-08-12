"""Enforce the layer dependency rules from the design spec.

config/meshing/sim never import analysis — not even lazily — and never touch
matplotlib or pyvista. __init__, pipeline and cli may use analysis only inside
functions (lazily), never at module top level.

Two boundaries are now enforced by packaging as well as by these tests:
`pynumcalc` (packages/pynumcalc) imports nothing from bemkit and needs only
numpy, and `bemviz` (packages/bemviz) owns everything that needs a display
stack. bemkit depends on pynumcalc and never on bemviz.

(Geometry authoring moved to the external `bemgeo` package, so `cad/` is no
longer part of bemkit — see CLAUDE.md.)
"""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "bemkit"
UPPER = ("bemkit.analysis", "bemviz.viz")
VIZ_LIBS = ("pyvista", "matplotlib")
CORE = ["config.py", "meshing", "sim"]
PYNUMCALC = Path(__file__).resolve().parents[1] / "packages" / "pynumcalc" / "pynumcalc"
GLUE = ["__init__.py", "pipeline.py", "cli"]


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


def test_analysis_never_imports_viz():
    """analysis/ is the results API — numpy only, no matplotlib, no pyvista.
    It is what `SimulationData` lives in, so it must stay usable without a
    display; viz depends on it, never the other way round."""
    assert _violations(["analysis"], _all_imports, ("bemviz.viz",)) == []
    assert _violations(["analysis"], _all_imports, VIZ_LIBS) == []


def test_core_never_uses_viz_libraries():
    assert _violations(CORE, _all_imports, VIZ_LIBS) == []


def test_glue_imports_upper_layers_only_lazily():
    assert _violations(GLUE, _top_level_imports) == []


def test_numcalc_never_imports_bemkit():
    """pynumcalc is the NumCalc format and process layer: it knows the NC.inp
    grammar, be.out and the solver binary, and nothing about config.toml,
    meshes or variants. It is now a separate distribution, and this keeps it
    one — bemkit may import pynumcalc, never the reverse.
    """
    bad = []
    for f in sorted(PYNUMCALC.rglob("*.py")):
        for mod, lineno in _all_imports(ast.parse(f.read_text())):
            if mod == "bemkit" or mod.startswith("bemkit."):
                # Self-imports within the layer are fine.
                if mod.startswith("pynumcalc"):
                    continue
                bad.append(f"{f.name}:{lineno} imports {mod}")
    assert bad == [], (
        "pynumcalc must not import from bemkit:\n  " + "\n  ".join(bad))


def test_numcalc_uses_only_stdlib_and_numpy():
    """The layer's dependency budget is numpy — no meshio, no gmsh, no scipy.
    A mesh reader creeping in here is how it would stop being extractable."""
    banned = ("meshio", "gmsh", "scipy", "matplotlib", "pyvista")
    bad = []
    for f in sorted(PYNUMCALC.rglob("*.py")):
        for mod, lineno in _all_imports(ast.parse(f.read_text())):
            if mod.split(".")[0] in banned:
                bad.append(f"{f.name}:{lineno} imports {mod}")
    assert bad == [], (
        "pynumcalc may only use stdlib and numpy:\n  " + "\n  ".join(bad))


def test_model_layer_imports_nothing_of_ours():
    """model/ is pure data — the bottom of the spine.

    If it ever imports geometry, solvers, results or view, the layering has
    inverted and the Model stops being independently testable, which is the
    whole reason two solver backends can be compared on one problem.
    """
    bad = []
    for f in sorted((PKG / "model").rglob("*.py")):
        for mod, lineno in _all_imports(ast.parse(f.read_text())):
            if mod.startswith("bemkit.") and not mod.startswith("bemkit.model"):
                bad.append(f"{f.name}:{lineno} imports {mod}")
    assert bad == [], ("bemkit/model must import nothing else from bemkit:\n  "
                       + "\n  ".join(bad))


def test_model_layer_needs_only_numpy():
    """No meshio, no gmsh, no scipy, no solver — a Model must be constructible
    with nothing installed but numpy."""
    banned = ("meshio", "gmsh", "scipy", "matplotlib", "pyvista", "pynumcalc")
    bad = []
    for f in sorted((PKG / "model").rglob("*.py")):
        for mod, lineno in _all_imports(ast.parse(f.read_text())):
            if mod.split(".")[0] in banned:
                bad.append(f"{f.name}:{lineno} imports {mod}")
    assert bad == [], ("bemkit/model may only use stdlib and numpy:\n  "
                       + "\n  ".join(bad))
