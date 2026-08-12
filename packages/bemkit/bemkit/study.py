"""A study directory: model.py, optionally params.toml, and its runs/ layout.

A study is Python, so variants and sweeps are a loop rather than a config
table.

    build() -> Model                    required
    build(params: dict) -> Model        also accepted; params from params.toml
    variants() -> {name: Model}         optional; wins over build() when present

``params.toml`` is read with ``tomllib`` and handed to ``build`` only when its
signature accepts a positional argument. It is deliberately **not**
validated against anything -- a typo in it surfaces as a ``KeyError`` (or
whatever else ``build`` raises) at build time, which is early enough for
research code.

The rest of the module is the *layout* half: where a solve puts its bundle,
and how a ``--variant`` prefix filter resolves. Both used to live in
``bemkit.cli._commands``, which meant the view layer reached into the CLI's
privates to find a bundle it wanted to draw. They are neither CLI-specific
nor solver-specific -- they are the convention a study directory follows.
"""

import importlib.util
import inspect
import tomllib
from pathlib import Path

#: Everything a run writes lives under ``STUDY/runs/``, one level per solver.
RUNS_DIR = "runs"


def workdir(study_dir, variant, solver="numcalc"):
    """``STUDY/runs/<solver>/<variant>/`` -- the bundle for one variant.

    Self-contained and relocatable: NumCalc runs with it as CWD using only
    relative paths, and everything read back out of a solve (results.npz,
    be.out/, the exported ObjectMesh and EvalGrid) lives inside it.
    """
    return Path(study_dir) / RUNS_DIR / solver / variant


def match_variants(names, prefixes):
    """Variant names matching ``prefixes`` -- all of them when it is falsy.

    Raises:
        KeyError: no name matched. The message lists what the study does
            have, since a prefix typo is the usual cause.
    """
    names = list(names)
    if not prefixes:
        return names
    chosen = [n for n in names if any(n.startswith(p) for p in prefixes)]
    if not chosen:
        raise KeyError(f"No variant matches {', '.join(prefixes)} — this "
                       f"study has: {', '.join(names)}")
    return chosen


def load_study(path):
    """Return ``{variant name: Model}`` for the study at ``path``."""
    path = Path(path)
    model_py = path / "model.py"
    if not model_py.exists():
        raise FileNotFoundError(f"no model.py in {path}")

    module = _load_module(model_py)

    if hasattr(module, "variants"):
        variants = module.variants
        if not callable(variants):
            raise TypeError(
                f"{model_py}: variants must be a function returning "
                f"{{name: Model}}, got {type(variants).__name__} -- did "
                f"you mean 'def variants():'?")
        return dict(variants())

    if not hasattr(module, "build"):
        raise AttributeError(
            f"{model_py} defines neither build() nor variants()")

    params_file = path / "params.toml"
    params = (tomllib.loads(params_file.read_text())
              if params_file.exists() else {})

    if _build_takes_params(module.build):
        return {"default": module.build(params)}
    return {"default": module.build()}


def _load_module(model_py):
    """Import ``model_py`` as an isolated module, not registered in
    sys.modules -- two studies loaded in the same process (e.g. a directory
    named "study" under two different parents) must not collide, so the
    spec name is derived from the full resolved path rather than the bare
    directory name."""
    resolved = model_py.resolve()
    spec = importlib.util.spec_from_file_location(
        f"study_{abs(hash(str(resolved)))}", model_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_takes_params(build):
    """Whether ``build`` follows the ``build(params)`` shape.

    Only the two documented shapes are accepted: no arguments at all, or
    exactly one plain positional argument. Anything else (keyword-only,
    ``**kwargs``, more than one argument) is a study author's mistake and
    gets a clear error here rather than a confusing TypeError from actually
    calling it wrong.
    """
    params = list(inspect.signature(build).parameters.values())
    if not params:
        return False
    if len(params) == 1 and params[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD):
        return True
    raise TypeError(
        f"build{inspect.signature(build)} is not a supported signature -- "
        f"build must take no arguments, or exactly one positional "
        f"'params' argument")
