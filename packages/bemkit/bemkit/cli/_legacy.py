"""The config.toml-era `bem` entry point, kept alive only for its own tests.

``bemkit.cli.main`` now drives the model.py spine. This is the previous
``main()`` verbatim — same argparse layout, same subcommand set, same
default-to-cwd resolution — so ``test_cli.py`` and ``test_cli_init.py`` keep
exercising ``_pipeline``/``_init``/``_check_legacy`` unchanged while the old
stack is still on disk. It is not wired to any console script and goes when
``pipeline.py``/``config.py`` do.
"""

import argparse
from pathlib import Path

from bemkit.cli import _register_plugins


def main(argv=None):
    import sys

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

    from bemkit.cli import _check_legacy, _init, _pipeline
    _check_legacy.register(sub)
    _init.register(sub)
    _pipeline.register(sub)
    # plot / geometry / results / export-web come from bemviz, through the
    # same entry-point group any third-party plugin uses.
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
