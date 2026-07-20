"""Unified `bem` command-line interface.

Heavy layers (analysis, viz, matplotlib, pyvista) are imported lazily
inside command functions only — `bem MODEL export` must work without them.
"""

import argparse
import importlib
import sys
import tomllib
from pathlib import Path


def require(module, extra):
    """Import an optional dependency or exit with an install hint."""
    try:
        return importlib.import_module(module)
    except ImportError:
        sys.exit(f"This command requires '{module}'. "
                 f"Install it with: pip install bemkit[{extra}]")


class VariantAction(argparse.Action):
    """--variant filter that accumulates across both forms.

    ``--variant a b`` and ``--variant a --variant b`` both yield ``['a', 'b']``.
    A plain ``nargs='+'`` lets the last flag win (``--variant a --variant b`` →
    ``['b']``); a plain ``action='append'`` rejects the space-separated form.
    Use with ``nargs='+'`` so each occurrence contributes a list, which is then
    flattened into one accumulator.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        items = list(getattr(namespace, self.dest, None) or [])
        items.extend(values)
        setattr(namespace, self.dest, items)


def load_model_config(model_dir):
    """ModelConfig.from_dir with CLI-friendly errors."""
    from bemkit.config import ModelConfig
    try:
        return ModelConfig.from_dir(model_dir)
    except FileNotFoundError:
        sys.exit(f"No config.toml in {model_dir}")
    except (ValueError, tomllib.TOMLDecodeError) as e:
        sys.exit(str(e))


def _iter_cli_entry_points():
    from importlib.metadata import entry_points
    return entry_points(group="bemkit.cli")


def _register_plugins(sub):
    """Let installed packages add `bem` subcommands (group: bemkit.cli)."""
    for ep in _iter_cli_entry_points():
        try:
            ep.load()(sub)
        except Exception as exc:  # a broken plugin must not take down the CLI
            print(f"warning: skipping CLI plugin {ep.name!r}: {exc}", file=sys.stderr)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # model_dir is effectively optional (defaults to the current directory,
    # see below) but must stay a required positional for argparse — so the
    # usage string is written by hand to show the truth.
    parser = argparse.ArgumentParser(
        prog="bem", usage="bem [MODEL_DIR] [COMMAND] [options]",
        description="BEM pipeline for NumCalc models. "
        "Run `bem` (or `bem COMMAND`) from inside a model directory, "
        "or pass MODEL_DIR explicitly.")
    parser.add_argument(
        "model_dir", metavar="MODEL_DIR",
        help="Path to model directory (default: current directory)")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND",
                                prog="bem [MODEL_DIR]")

    from bemkit.cli import _pipeline, _plot, _viz3d, _web
    _pipeline.register(sub)
    _plot.register(sub)
    _viz3d.register(sub)
    _web.register(sub)
    _register_plugins(sub)

    # argparse can't reliably combine an *optional* model_dir positional with
    # the subparsers action - e.g. `nargs="?"` on model_dir makes plain
    # `bem MODEL_DIR` (no COMMAND) ambiguous and fails to parse. So model_dir
    # stays required from argparse's point of view, and we resolve the
    # "default to cwd" behavior ourselves before argparse ever sees the
    # tokens: if the first token names a known subcommand, or there are no
    # tokens at all, there's no model_dir token to consume, so insert "."
    # explicitly. A model directory that happens to share a name with a
    # subcommand (e.g. a dir literally called "mesh") must be passed as
    # "./mesh" to disambiguate.
    if not argv or argv[0] in sub.choices:
        argv = ["."] + argv

    args = parser.parse_args(argv)
    model_dir = Path(args.model_dir).resolve()
    if args.command is None:
        from bemkit.cli._pipeline import show_status
        return show_status(model_dir)
    return args.func(model_dir, args)
