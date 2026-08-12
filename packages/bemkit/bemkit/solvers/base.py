"""The solver seam.

``workdir`` reconciles two lifecycles. NumCalc needs one — it is a binary that
reads ./NC.inp from its cwd, and the directory *is* the resumable, inspectable,
relocatable bundle. An in-process backend ignores it. ``bins``/``jobs`` are
meaningful to batch backends and no-ops for direct ones, so the CLI drives both
without asking which kind it has.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Status:
    """What a backend can say about a run in progress."""

    bins_present: tuple = ()
    bins_missing: tuple = ()
    running_pids: tuple = ()
    warnings: tuple = ()

    @property
    def complete(self):
        return not self.bins_missing


class Solver(Protocol):
    """A BEM backend."""

    name: str

    def solve(self, model, *, workdir=None, bins=None, jobs=1):
        """Solve ``model`` and return Results.

        Args:
            workdir: bundle directory. Required in spirit by file-based
                backends; when None they use a temporary directory and
                materialise results into memory before it is cleaned up.
            bins: 1-based frequency indices to compute (None = all).
            jobs: max concurrent processes. Ignored by in-process backends.
        """
        ...

    def status(self, workdir):
        """Status of a run in ``workdir``."""
        ...
