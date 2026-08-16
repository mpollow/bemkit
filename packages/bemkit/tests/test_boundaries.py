"""Enforce the layer dependency rules of the spine.

The spine runs bottom-up and imports one way only::

    model/  ->  geometry/  ->  solvers/  ->  results/  ->  view/  ->  cli/

* ``model/`` is pure data: numpy, and nothing of ours. That is what lets a
  Model be built, compared and unit-tested with no solver, mesher or display
  stack installed. Its one scipy user, ``analytic.py``, is reached through a
  lazy ``__getattr__`` and is exempted by name below.
* ``geometry/`` and ``solvers/`` and ``results/`` never touch pyvista or
  matplotlib — producing and reading results must never drag in a display.
* ``cli/`` may use heavy layers, but only inside functions, so ``bem --help``
  does not pay for pyvista.

One boundary is enforced by packaging as well as by these tests: ``numcalckit``
(packages/numcalckit) imports nothing from bemkit and needs only numpy.
``bemkit/view/`` used to be a third distribution (``bemviz``) and is now a
layer inside this one, so its isolation is enforced only here — the display
stack is a ``bemkit[view]`` extra, and the rules below are what keep it
optional in fact rather than only on paper.
"""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "bemkit"
#: packages/bemkit/tests -> packages/bemkit -> packages -> repo root
REPO = Path(__file__).resolve().parents[3]
PYNUMCALC = REPO / "packages" / "numcalckit" / "numcalckit"

VIZ_LIBS = ("pyvista", "matplotlib")
#: bemkit layers that must stay free of a display stack. Everything that
#: produces or reads results lives here.
CORE = ["model", "geometry", "solvers", "results"]
#: May reach into any layer, but only lazily (inside a function body).
GLUE = ["__init__.py", "cli"]
#: view/ is the display layer; the core must never import it.
UPPER = ("bemkit.view",)


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
    """A renamed or deleted layer must fail loudly here rather than silently
    turning every rule below into a scan over zero files."""
    for entry in CORE + GLUE + ["view"]:
        assert (PKG / entry).exists(), f"missing {entry} — layout changed?"
    assert PYNUMCALC.is_dir(), f"missing {PYNUMCALC} — layout changed?"
    assert list(PYNUMCALC.rglob("*.py")), f"no modules under {PYNUMCALC}"


def test_the_upper_layer_rule_is_not_vacuous():
    """UPPER names a prefix that something actually imports.

    When view/ was the separate ``bemviz`` distribution, folding it in
    renamed every import of it — and the rules below, still spelled
    ``bemviz``, would have gone on passing over zero matches forever. This
    fails the moment UPPER stops naming the real display layer.
    """
    hits = [f for f in _py_files("view") + _py_files("cli")
            for mod, _ in _all_imports(ast.parse(f.read_text()))
            if mod.startswith(UPPER)]
    assert hits, f"nothing imports {UPPER} — is the prefix still right?"


def test_core_never_imports_the_view_layer():
    assert _violations(CORE, _all_imports) == []


def test_core_never_uses_viz_libraries():
    assert _violations(CORE, _all_imports, VIZ_LIBS) == []


def test_glue_imports_heavy_layers_only_lazily():
    """__init__ and cli may use view/pyvista, but never at module level."""
    assert _violations(GLUE, _top_level_imports, UPPER + VIZ_LIBS) == []


def test_numcalc_never_imports_bemkit():
    """numcalckit is the NumCalc format and process layer: it knows the NC.inp
    grammar, be.out and the solver binary, and nothing about models, meshes or
    variants. It is a separate distribution, and this keeps it one — bemkit
    may import numcalckit, never the reverse.
    """
    bad = []
    for f in sorted(PYNUMCALC.rglob("*.py")):
        for mod, lineno in _all_imports(ast.parse(f.read_text())):
            if mod == "bemkit" or mod.startswith("bemkit."):
                bad.append(f"{f.name}:{lineno} imports {mod}")
    assert bad == [], (
        "numcalckit must not import from bemkit:\n  " + "\n  ".join(bad))


def test_numcalc_tests_never_import_bemkit():
    """The rule covers the suite too. A numcalckit test that reaches for a
    bemkit reader is testing the wrong layer, and it silently makes the
    distribution un-installable on its own."""
    tests = PYNUMCALC.parent / "tests"
    bad = []
    for f in sorted(tests.rglob("*.py")):
        for mod, lineno in _all_imports(ast.parse(f.read_text())):
            if mod == "bemkit" or mod.startswith("bemkit."):
                bad.append(f"{f.name}:{lineno} imports {mod}")
    assert bad == [], (
        "the numcalckit suite must not import from bemkit:\n  "
        + "\n  ".join(bad))


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
        "numcalckit may only use stdlib and numpy:\n  " + "\n  ".join(bad))


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
    """No meshio, no gmsh, no solver — a Model must be constructible with
    nothing installed but numpy.

    ``analytic.py`` is the one exemption: the exact spherical-cap solution
    genuinely needs scipy's spherical harmonics. It is kept off
    ``bemkit/model/__init__.py``'s eager imports (see the lazy ``_LAZY``
    table there) so importing the layer still costs nothing, which
    ``tests/test_public_api.py`` pins from the other side by asserting
    ``import bemkit.model`` leaves scipy out of ``sys.modules``.
    """
    banned = ("meshio", "gmsh", "scipy", "matplotlib", "pyvista", "numcalckit")
    bad = []
    for f in sorted((PKG / "model").rglob("*.py")):
        allowed = ("scipy",) if f.name == "analytic.py" else ()
        for mod, lineno in _all_imports(ast.parse(f.read_text())):
            head = mod.split(".")[0]
            if head in banned and head not in allowed:
                bad.append(f"{f.name}:{lineno} imports {mod}")
    assert bad == [], ("bemkit/model may only use stdlib and numpy:\n  "
                       + "\n  ".join(bad))


def test_solver_layer_never_imports_the_cli():
    """A solver is a library object. If it reaches for the CLI, `bem` has
    stopped being a thin driver over the spine and become the spine."""
    bad = []
    for entry in ("model", "geometry", "solvers", "results"):
        for f in _py_files(entry):
            for mod, lineno in _all_imports(ast.parse(f.read_text())):
                if mod.startswith("bemkit.cli"):
                    bad.append(f"{f.relative_to(PKG.parent)}:{lineno} imports {mod}")
    assert bad == [], ("the spine must not import bemkit.cli:\n  "
                       + "\n  ".join(bad))
