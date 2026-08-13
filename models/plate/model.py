"""A finite plate with a vibrating membrane -- the simplest demonstration of
geodesic mesh sizing.

The back-face center is 5 mm from the membrane straight through the plate,
but ~100 mm over the surface (out to an edge, around, back). Euclidean
sizing refines a shadow disc on the back face; geodesic leaves the whole back
face at lc_max. See mesh_plate.py for how mesh/*.msh was generated and
compare_meshes.py to see the effect measured.
"""

from pathlib import Path

from bemkit.geometry import load_mesh
from bemkit.model import BC, Group, Medium, Model, Velocity, fft_grid, on_axis

HERE = Path(__file__).parent

#: Two sizing metrics (euclidean/geodesic, same lc_min/lc_max/dist_min/
#: dist_max ramp) plus three uniform resolutions for a convergence check
#: independent of the ramp -- see mesh_plate.py.
VARIANTS = ("euclidean", "geodesic", "uniform3", "uniform5", "uniform10")


def build(params):
    mesh = load_mesh(HERE / "mesh" / f"{params['mesh']}.msh")
    return Model(
        mesh,
        fft_grid(params["sr"], params["nbins"]),
        Medium(c=params["c"], rho=params["rho"]),
        bcs=[BC(Group("vibrating"), Velocity(1.0))],
        fields={"onaxis": on_axis("z", params["onaxis_radii_m"])},
    )


def variants():
    """One model per mesh variant -- a loop, not a [variants] table."""
    base = dict(sr=32000, nbins=65, c=340.0, rho=1.2,
                onaxis_radii_m=[0.1, 0.25, 0.5])
    return {name: build({**base, "mesh": name}) for name in VARIANTS}
