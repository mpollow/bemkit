"""NumCalc format and process layer — everything the solver itself understands.

This subpackage is the boundary between *what NumCalc reads* and *what bemkit
decides*. It knows the ``NC.inp`` deck grammar, the ``be.out/`` result format, and
how to invoke and track the solver binary. It knows nothing about ``config.toml``,
mesh files, variants, or any other bemkit concept.

**It must not import anything from ``bemkit``** — only stdlib and numpy. The rule
is enforced by ``tests/test_boundaries.py``; ``bemkit.sim`` may import from here,
never the reverse. Keeping that direction true is what leaves the door open to
publishing this layer as a standalone package (see
``docs/numcalc-feature-coverage.md`` §4).
"""
