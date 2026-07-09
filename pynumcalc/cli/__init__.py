"""Unified `bem` command-line interface.

Heavy layers (analysis, viz, matplotlib, pyvista) are imported lazily
inside command functions only — `bem MODEL export` must work without them.
"""

import argparse
import importlib
import sys
from pathlib import Path


def require(module, extra):
    """Import an optional dependency or exit with an install hint."""
    try:
        return importlib.import_module(module)
    except ImportError:
        sys.exit(f"This command requires '{module}'. "
                 f"Install it with: pip install pynumcalc[{extra}]")


def load_model_config(model_dir):
    """ModelConfig.from_dir with CLI-friendly errors."""
    from pynumcalc.config import ModelConfig
    try:
        return ModelConfig.from_dir(model_dir)
    except FileNotFoundError:
        sys.exit(f"No vars.toml in {model_dir}")


def _iter_cli_entry_points():
    from importlib.metadata import entry_points
    return entry_points(group="pynumcalc.cli")


def _register_plugins(sub):
    """Let installed packages add `bem` subcommands (group: pynumcalc.cli)."""
    for ep in _iter_cli_entry_points():
        try:
            ep.load()(sub)
        except Exception as exc:  # a broken plugin must not take down the CLI
            print(f"warning: skipping CLI plugin {ep.name!r}: {exc}", file=sys.stderr)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # `bem metric ...` is the one command without a model directory.
    if argv[:1] == ["metric"]:
        from pynumcalc.cli._metric import main as metric_main
        return metric_main(argv[1:])

    parser = argparse.ArgumentParser(
        prog="bem", description="BEM pipeline for NumCalc models. "
        "Run `bem MODEL_DIR` for status, or `bem metric --help` for the "
        "standalone mmgs metric tool.")
    parser.add_argument("model_dir", help="Path to model directory")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    from pynumcalc.cli import _pipeline, _plot, _viz3d, _web
    _pipeline.register(sub)
    _plot.register(sub)
    _viz3d.register(sub)
    _web.register(sub)
    _register_plugins(sub)

    args = parser.parse_args(argv)
    model_dir = Path(args.model_dir).resolve()
    if args.command is None:
        from pynumcalc.cli._pipeline import show_status
        return show_status(model_dir)
    return args.func(model_dir, args)
