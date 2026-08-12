"""Rigid sphere with a vibrating cap -- the reference case.

Its exact solution is in bemkit.model.analytic, which makes it the model every
solver gets checked against.
"""

from pathlib import Path

from bemkit.geometry import load_mesh
from bemkit.model import (
    BC, Group, Medium, Model, Velocity, circle, fft_grid, on_axis,
)

HERE = Path(__file__).parent


def build(params):
    mesh = load_mesh(HERE / "mesh" / f"{params['mesh']}.msh")
    return Model(
        mesh,
        fft_grid(params["sr"], params["nbins"]),
        Medium(c=params["c"], rho=params["rho"]),
        bcs=[BC(Group("vibrating"), Velocity(1.0))],
        fields={
            "onaxis": on_axis("z", [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]),
            "polar": circle("zx", res=5.0, radius=1.0),
        },
    )


def variants():
    """One model per mesh resolution -- a loop, not a [variants] table."""
    base = dict(sr=32000, nbins=65, c=340.0, rho=1.2)
    return {name: build({**base, "mesh": name})
            for name in ("coarse", "standard", "fine")}
