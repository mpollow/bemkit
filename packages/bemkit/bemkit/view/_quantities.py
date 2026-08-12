"""Complex field -> displayable numbers.

The arithmetic behind every colour the 3D viewers paint, kept apart from both
the rendering (``scene.py``) and the plumbing (``bundle.py``) because it is
the only part of the view layer that can be checked without a display.

Everything here takes plain arrays. Nothing knows about Results, bundles or
pyvista.
"""

import numpy as np

#: The five real-valued views of a complex field a viewer can show.
QUANTITIES = ("magnitude_db", "magnitude", "phase", "real", "imag")

#: Dynamic range of the phase plot's magnitude fade: the peak is opaque, and
#: this many dB below it fades to white.
PHASE_DB_RANGE = 80.0

#: Quantities whose colour limits are a window a user can shift/expand.
CLIM_QUANTITIES = frozenset({"magnitude_db", "real", "imag"})

#: ...of those, the ones whose window is symmetric about zero.
SYMMETRIC_QUANTITIES = frozenset({"real", "imag"})

#: Scalar-bar labels. The unit depends on the normalization, so the caller
#: appends it rather than these pretending to know.
LABELS = {"magnitude_db": "dB", "magnitude": "|p|", "phase": "Phase (°)",
          "real": "Re(p)", "imag": "Im(p)"}


def magnitude_db(values, floor=1e-12):
    """20*log10|values|, floored so a hard zero is a number and not -inf."""
    return 20.0 * np.log10(np.maximum(np.abs(values), floor))


def quantity_values(values, quantity):
    """Real-valued view of a complex field.

    Args:
        values: complex array of any shape.
        quantity: one of :data:`QUANTITIES`.

    Returns:
        A float array of the same shape. Phase is in degrees, not radians —
        the viewers label their colour bars in degrees, and a silent radian
        would look like a plausible ±3 dynamic range.
    """
    if quantity == "magnitude_db":
        return magnitude_db(values)
    if quantity == "magnitude":
        return np.abs(values)
    if quantity == "phase":
        return np.degrees(np.angle(values))
    if quantity == "real":
        return np.asarray(values).real
    if quantity == "imag":
        return np.asarray(values).imag
    raise ValueError(f"unknown quantity {quantity!r} (have: "
                     f"{', '.join(QUANTITIES)})")


def phase_rgba(values, db_ref=None, db_range=PHASE_DB_RANGE):
    """(N, 4) uint8 RGBA: hue is phase, low amplitude fades to white.

    A flat hsv phase map colours the noise floor as confidently as the main
    lobe. Fading by magnitude is what makes a phase picture readable: at
    ``db_ref`` the hue is full, ``db_range`` below it the point is white.

    Args:
        values: (N,) complex.
        db_ref: dB level treated as the top of the fade. None uses the
            maximum in ``values``; pass one shared across actors so a surface
            and its eval points fade on the same scale.
        db_range: the fade's dynamic range in dB.
    """
    import matplotlib.pyplot as plt

    values = np.asarray(values)
    mag_db = magnitude_db(values)
    if db_ref is None:
        db_ref = np.nanmax(mag_db)
    weight = np.clip((mag_db - (db_ref - db_range)) / db_range, 0.0, 1.0)
    phase_norm = (np.degrees(np.angle(values)) + 180.0) / 360.0
    phase_rgb = plt.cm.hsv(phase_norm)[:, :3]
    rgb = weight[:, None] * phase_rgb + (1.0 - weight[:, None])
    alpha = np.ones(len(values))
    return (np.column_stack([rgb, alpha]) * 255).astype(np.uint8)


def impedance_scalars(points, pressure, velocity, rho0, c0, quantity):
    """Specific acoustic impedance Z = p / v_r, normalised by rho0*c0.

    ``v_r`` is the *radial* velocity component — the projection onto the unit
    vector from the origin — so this assumes the source sits near the origin.
    That is the case it is useful for: in the far field of such a source
    Re(Z0) tends to 1, which is the reading that tells you the grid is far
    enough out.

    Args:
        points: (N, 3) coordinates in metres.
        pressure: (N,) complex.
        velocity: (N, 3) complex.
        rho0, c0: medium density and sound speed.
        quantity: one of :data:`QUANTITIES`.

    Returns:
        (N,) float. Points with a vanishing radial velocity come back NaN
        rather than as a huge finite number that would blow the colour scale.
    """
    points = np.asarray(points, dtype=float)
    r_norm = np.linalg.norm(points, axis=1, keepdims=True)
    r_hat = points / np.maximum(r_norm, 1e-30)

    v_r = np.sum(np.asarray(velocity) * r_hat, axis=1)
    # The NaN is deliberate (see the docstring), so silence the warning it
    # raises rather than letting every such point print one.
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.asarray(pressure) / np.where(np.abs(v_r) > 1e-30, v_r,
                                            np.nan + 0j)
    z0 = z / (rho0 * c0)
    # magnitude_db of a ratio that is already normalised: report the level of
    # the normalised impedance, so 0 dB is the matched case.
    return quantity_values(z0, quantity)


def intensity(pressure, velocity):
    """Active acoustic intensity I = 1/2 Re(p v*), in W/m². (N, 3) float."""
    return 0.5 * np.real(np.asarray(pressure)[:, None]
                         * np.conj(np.asarray(velocity)))
