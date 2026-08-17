"""Which meshing strategy buys accuracy, measured against the exact answer.

A spherical cap on a rigid sphere is one of the few radiation problems with a
fast closed form (bemkit.model.analytic.sphcap_pressure), so a mesh of one can
be scored against the truth rather than against another mesh. Nine variants of
the same 100 mm sphere with a +/-30 degree driven cap, solved over the same
250 Hz - 16 kHz band, each a controlled comparison against another -- see
README.md for the pairings and mesh_sphcap_meshing.py for the sizing.

The caveat the study cannot answer: this shape is everywhere smooth. Its only
discontinuity is the cap rim, where the velocity BC jumps -- which is what the
`rim` variant is about, and as close to an edge as this geometry gets. What a
real geometric edge needs is a separate study.
"""

from pathlib import Path

from bemkit.geometry import load_mesh
from bemkit.model import (
    BC, Group, Medium, Model, Velocity, circle, fft_grid, on_axis,
)

HERE = Path(__file__).parent

#: The geometry the closed form is evaluated with. Kept here rather than in
#: mesh_sphcap_meshing.py because this is what compare_analytic.py reads: the
#: mesh script runs once, this is the study's standing description of it.
RADIUS_MM = 50.0
ANGLE_DEG = 30.0

#: Ordered coarse-to-fine within each family, which is the order the report
#: and the plot legends use.
VARIANTS = ("u2", "u4", "u6", "u8", "u10",
            "cap_far", "cap_mid", "geodesic", "rim")


def build(params):
    mesh = load_mesh(HERE / "mesh" / f"{params['mesh']}.msh")
    return Model(
        mesh,
        fft_grid(params["sr"], params["nbins"]),
        Medium(c=params["c"], rho=params["rho"]),
        bcs=[BC(Group("vibrating"), Velocity(1.0))],
        fields={
            # The primary error metric: full directivity in the plane of
            # symmetry, far enough out to be a clean radiation pattern.
            "polar": circle("zx", res=5.0, radius=1.0),
            # Near to far on the driven axis, for the same reason models/plate
            # keeps one: a single number per frequency that is easy to eyeball.
            "onaxis": on_axis("z", [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]),
            # 25 mm off the surface, where mesh error shows up first and where
            # the far-field pattern has not yet formed. Clears even u2's
            # 10.6 mm elements by more than two element lengths.
            "near": circle("zx", res=5.0, radius=0.075),
        },
    )


def variants():
    """One model per mesh variant -- a loop, not a [variants] table."""
    base = dict(sr=32000, nbins=65, c=340.0, rho=1.2)
    return {name: build({**base, "mesh": name}) for name in VARIANTS}
