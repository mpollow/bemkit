"""Load a study directory: model.py, and optionally params.toml.

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
"""

import importlib.util
import inspect
import tomllib
from pathlib import Path


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
