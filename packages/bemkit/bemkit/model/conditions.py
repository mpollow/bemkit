"""Conditions: *what* a boundary condition imposes.

Frequency dependence is allowed for Admittance and Impedance only, and this is
physics rather than a limitation. The BEM is linear, so an excitation's
frequency response is a scale factor on the answer:

    p(f) for velocity v(f)  =  v(f) * p_unit(f)

Solve once at unit amplitude and multiply afterwards (``Results.scaled_by``) —
which also lets you try ten driver responses without re-solving. Admittance and
impedance are different in kind: they enter the boundary operator rather than
the right-hand side, so the system matrix genuinely changes per frequency.

That is exactly why NumCalc supports CURVES for ADMI, ignores them for
VELO/PRES, and exit(-1)s on IMPE with a curve.
"""

import numpy as np

#: Above this, an admittance or impedance is almost certainly in SI units
#: rather than normalised by rho0*c. Normalised admittance sits around 0-2 and
#: normalised impedance around 0.5-20; an SI impedance for air starts near 400.
_IMPLAUSIBLE = 100.0

_NORMALISED_HELP = (
    "NumCalc wants a **normalised** quantity — admittance divided by rho0*c "
    "(so 1.0 is a matched, anechoic termination), impedance multiplied by the "
    "same. Divide an SI admittance by ~1/408, or an SI impedance by ~408.")


class Condition:
    """Base class. ``kind`` is the NumCalc keyword this maps to."""

    kind = None
    frequency_dependent_allowed = False

    def __init__(self, value):
        if callable(value):
            self._value = value
        else:
            self._value = np.asarray(value)
        if self.is_frequency_dependent and not self.frequency_dependent_allowed:
            raise ValueError(
                f"{type(self).__name__} takes a single value, not a spectrum. "
                f"The BEM is linear, so a frequency-dependent excitation is a "
                f"scale factor on the result: solve at unit amplitude and use "
                f"Results.scaled_by(spectrum) — which also means you can try "
                f"another response without re-solving.")

    @property
    def is_frequency_dependent(self):
        return callable(self._value) or np.ndim(self._value) > 0

    def at(self, frequencies):
        """(nf,) complex values, one per frequency."""
        frequencies = np.asarray(frequencies, dtype=float)
        if callable(self._value):
            out = np.asarray(self._value(frequencies), dtype=complex)
        elif np.ndim(self._value) == 0:
            out = np.full(len(frequencies), complex(self._value))
        else:
            out = np.asarray(self._value, dtype=complex)
        if out.shape != frequencies.shape:
            raise ValueError(
                f"{type(self).__name__} has {out.size} values but the model "
                f"has {len(frequencies)} frequencies")
        return out

    def __repr__(self):
        v = "<callable>" if callable(self._value) else self._value
        return f"{type(self).__name__}({v})"


class Velocity(Condition):
    """Normal particle velocity in m/s. ``0`` is sound-hard (the default for
    any element with no condition)."""

    kind = "VELO"

    def __init__(self, value=1.0):
        super().__init__(value)


class Pressure(Condition):
    """Sound pressure in Pa. ``0`` is sound-soft."""

    kind = "PRES"

    def __init__(self, value=0.0):
        super().__init__(value)


class _Normalised(Condition):
    """Base for Admittance/Impedance: may vary with frequency, and is checked
    for the SI-vs-normalised units trap.

    A non-callable value's magnitude is known immediately, so it is checked in
    __init__ — the earliest possible point, and what the constructor-level
    tests below rely on. A callable's magnitude is *not* known until it has
    been evaluated against real frequencies, so checking it in __init__ would
    mean either calling it speculatively (with made-up frequencies, which
    could itself raise or mislead) or skipping the guard for callables
    entirely. The latter would be a silent hole — `Admittance(lambda f: 408.0
    * np.ones_like(f))` sailing through unchecked — so instead the guard is
    re-applied to whatever value at() actually produces, callable or not.
    """

    frequency_dependent_allowed = True

    def __init__(self, value=1.0):
        super().__init__(value)
        if not callable(self._value):
            self._check_magnitude(np.asarray(self._value, dtype=complex))

    def _check_magnitude(self, values):
        values = np.asarray(values)
        if values.size and np.max(np.abs(values)) > _IMPLAUSIBLE:
            raise ValueError(
                f"{type(self).__name__} value looks like SI units. "
                f"{_NORMALISED_HELP}")

    def at(self, frequencies):
        out = super().at(frequencies)
        self._check_magnitude(out)
        return out


class Admittance(_Normalised):
    """Normalised surface admittance Y/(rho0*c). ``1.0`` is anechoic."""

    kind = "ADMI"


class Impedance(_Normalised):
    """Normalised surface impedance Z/(rho0*c)."""

    kind = "IMPE"
