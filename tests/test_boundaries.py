"""Enforce the layer dependency rules from the design spec.

config/meshing/sim never import analysis or viz — not even lazily — and
never touch the viz libraries (pyvista/matplotlib) directly either.
__init__, pipeline, and cli may use the upper layers only inside functions
(lazily), never at module top level.

(Geometry authoring moved to the external `bemgeo` package, so `cad/` is no
longer part of pynumcalc — see CLAUDE.md.)
"""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "pynumcalc"
UPPER = ("pynumcalc.analysis", "pynumcalc.viz")
VIZ_LIBS = ("pyvista", "matplotlib")
CORE = ["config.py", "meshing", "sim"]
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


def test_core_never_uses_viz_libraries():
    assert _violations(CORE, _all_imports, VIZ_LIBS) == []


def test_glue_imports_upper_layers_only_lazily():
    assert _violations(GLUE, _top_level_imports) == []
