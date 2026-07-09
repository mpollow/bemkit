"""Third-party packages extend `bem` via the pynumcalc.cli entry-point group."""

import argparse

from pynumcalc import cli


class FakeEntryPoint:
    def load(self):
        def register(sub):
            p = sub.add_parser("fancycheck")
            p.set_defaults(func=lambda model_dir, args: 0)
        return register


def test_register_plugins_adds_subcommand(monkeypatch):
    monkeypatch.setattr(cli, "_iter_cli_entry_points", lambda: [FakeEntryPoint()])
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    cli._register_plugins(sub)
    args = parser.parse_args(["fancycheck"])
    assert args.command == "fancycheck"
    assert args.func("model", args) == 0


def test_plugin_subcommand_dispatches_through_main(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_iter_cli_entry_points", lambda: [FakeEntryPoint()])
    assert cli.main([str(tmp_path), "fancycheck"]) == 0


def test_no_plugins_is_fine(monkeypatch):
    monkeypatch.setattr(cli, "_iter_cli_entry_points", lambda: [])
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    cli._register_plugins(sub)  # must not raise
