"""Unified `bem` command-line interface.

A study is a directory holding a ``model.py``; the CLI builds its Models and
drives them through a solver. Heavy layers (viz, matplotlib, pyvista) are
imported lazily inside command functions only.
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


def _iter_cli_entry_points():
    from importlib.metadata import entry_points
    return entry_points(group="bemkit.cli")


def _register_plugins(sub):
    """Let installed packages add `bem` subcommands (group: bemkit.cli).

    The seam the private `bemlab` package uses for its own verbs. bemkit's own
    view commands no longer come through here — they live in `bemkit.view` and
    are registered directly by :func:`main`.
    """
    for ep in _iter_cli_entry_points():
        try:
            ep.load()(sub)
        except Exception as exc:  # a broken plugin must not take down the CLI
            print(f"warning: skipping CLI plugin {ep.name!r}: {exc}", file=sys.stderr)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # study_dir is effectively optional (defaults to the current directory,
    # see below) but must stay a required positional for argparse — so the
    # usage string is written by hand to show the truth.
    parser = argparse.ArgumentParser(
        prog="bem", usage="bem [STUDY_DIR] [COMMAND] [options]",
        description="BEM studies driven from model.py. "
        "Run `bem` (or `bem COMMAND`) from inside a study directory, "
        "or pass STUDY_DIR explicitly.")
    parser.add_argument(
        "study_dir", metavar="STUDY_DIR",
        help="Path to study directory (default: current directory)")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND",
                                prog="bem [STUDY_DIR]")

    from bemkit.cli import _check, _commands
    _check.register(sub)
    _commands.register(sub)
    # geometry / results / export-web live in bemkit.view. Imported here and
    # not at module level, so `bem --help` never pays for pyvista; an
    # ImportError means the [view] extra is not installed, and the rest of the
    # CLI must still work.
    try:
        from bemkit.view import register_cli as _register_view
    except ImportError:
        pass          # pyvista/matplotlib not installed — view verbs absent
    else:
        _register_view(sub)
    _register_plugins(sub)

    # argparse can't reliably combine an *optional* study_dir positional with
    # the subparsers action - e.g. `nargs="?"` on study_dir makes plain
    # `bem STUDY_DIR` (no COMMAND) ambiguous and fails to parse. So study_dir
    # stays required from argparse's point of view, and we resolve the
    # "default to cwd" behavior ourselves before argparse ever sees the
    # tokens: if the first token names a known subcommand, or there are no
    # tokens at all, there's no study_dir token to consume, so insert "."
    # explicitly. A study directory that happens to share a name with a
    # subcommand (e.g. a dir literally called "check") must be passed as
    # "./check" to disambiguate.
    if not argv or argv[0] in sub.choices:
        argv = ["."] + argv

    args = parser.parse_args(argv)
    study_dir = Path(args.study_dir).resolve()
    if args.command is None:
        return _commands.show_status(study_dir)
    return args.func(study_dir, args)
