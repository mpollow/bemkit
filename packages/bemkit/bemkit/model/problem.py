"""The Model: a complete, solver-neutral BEM problem."""

from dataclasses import dataclass

import numpy as np

from bemkit.model.mesh import Medium


@dataclass(frozen=True)
class BC:
    """A condition applied where a selector says."""

    select: object
    condition: object


@dataclass(frozen=True)
class ResolvedBC:
    """A BC after its selector has been resolved against the mesh."""

    elements: np.ndarray
    condition: object


class Model:
    """Mesh + frequencies + medium + boundary conditions + sources + field points.

    Pure data. Nothing here knows about NumCalc, files, or bempp — which is
    what lets two backends be compared on the same problem.

    Args:
        mesh: a Mesh, in metres.
        frequencies: (nf,) ascending Hz.
        medium: a Medium. Defaults to ``Medium()`` when omitted (``None``) —
            checked with an explicit ``is None``, not truthiness: ``Medium``
            defines no ``__bool__``/``__len__``, so every instance (including
            one a caller deliberately built) is truthy, but the explicit
            check keeps that an invariant of this code rather than of
            Medium's current field set.
        bcs: sequence of BC.
        sources: sequence of PlaneWave/PointSource.
        fields: {name: (N, 3) array} of evaluation points, in metres. An empty
            (0, 3) set is accepted — e.g. the result of upstream ray-cast
            filtering dropping every point — but non-finite coordinates are
            rejected, since a NaN/Inf field point never resolves to anything
            meaningful and would otherwise propagate silently into results.
        exterior: True for a radiation/scattering problem outside the surface.
    """

    def __init__(self, mesh, frequencies, medium=None, *, bcs=(), sources=(),
                 fields=None, exterior=True, time_factor=-1.0):
        self.mesh = mesh
        self.frequencies = np.asarray(frequencies, dtype=float)
        self.medium = medium if medium is not None else Medium()
        self.bcs = tuple(bcs)
        self.sources = tuple(sources)
        self.exterior = bool(exterior)
        # Only -1 and +1 exist, and the check is strict on purpose: NumCalc
        # coerces anything that is not exactly -1.0 to +1.0
        # (NC_Input.cpp:334), silently flipping the sign of every imaginary
        # part. A typo or an omission must not be able to choose the
        # convention for you.
        self.time_factor = float(time_factor)
        if self.time_factor not in (-1.0, 1.0):
            raise ValueError(
                f"time_factor must be exactly -1.0 (e^-iwt) or +1.0 (e^+iwt), "
                f"got {time_factor!r}. NumCalc silently reads anything else "
                f"as +1.0, so there is no safe middle value.")

        if self.frequencies.ndim != 1 or len(self.frequencies) == 0:
            raise ValueError("a model needs at least one frequency")
        if np.any(np.diff(self.frequencies) <= 0):
            raise ValueError("frequencies must be strictly ascending")

        self.fields = {}
        for name, pts in (fields or {}).items():
            pts = np.asarray(pts, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 3:
                raise ValueError(
                    f"field set {name!r} must be (N, 3) metres, got {pts.shape}")
            if not np.all(np.isfinite(pts)):
                raise ValueError(
                    f"field set {name!r} has non-finite coordinates "
                    f"(NaN/Inf) — these never resolve to a usable field "
                    f"point and would otherwise propagate silently")
            self.fields[name] = pts

    @property
    def n_frequencies(self):
        return len(self.frequencies)

    @property
    def n_field_points(self):
        return sum(len(p) for p in self.fields.values())

    @property
    def is_uniform_band(self):
        """True when the frequencies are the evenly spaced FFT grid.

        Time-domain results are an inverse FFT of the band, so they only exist
        for this case. An evenly spaced band that does not *start* at its own
        step (e.g. [150, 250, 350], step 100 but starting at 150) is not the
        DC-excluded grid `fft_grid` produces for any (sr, nbins) and must not
        report as uniform.
        """
        f = self.frequencies
        if len(f) < 2:
            return True
        step = np.diff(f)
        return bool(np.allclose(step, step[0], rtol=1e-6)
                    and np.isclose(f[0], step[0], rtol=1e-6))

    def resolved_bcs(self):
        """Resolve every selector against the mesh. Raises on a bad selector."""
        return [ResolvedBC(bc.select.resolve(self.mesh), bc.condition)
                for bc in self.bcs]

    @property
    def vibrating_area(self):
        """Driven surface area in m², or ``None`` when nothing drives it.

        This is the ``A_vib`` the acceleration normalization divides by:
        ``Results.scaled_by(1 / (1j * rho * omega * A_vib))`` turns NumCalc's
        raw velocity-BC output into acceleration-normalized pressure.

        Two rules, both of which silently distort every normalized number if
        broken rather than failing:

        * **Each element counts once.** Selectors may legitimately overlap —
          a patch on top of a vibrating physical group — and summing per-BC
          would inflate the area.
        * **``Velocity(0)`` is sound-hard, not a source.** ``VELO 0.0`` is
          exactly how NumCalc spells a rigid element, so a zero-amplitude
          range contributes nothing and must not enter the area.

        Pressure/Admittance/Impedance elements are boundary *conditions*, not
        velocity sources, and have no place in a velocity-derived
        normalization either.

        ``None`` rather than ``0.0`` when there is no drive at all: a
        scattering model (plane wave or point source against a rigid or
        absorbing surface) has no vibrating area, and the acc normalization is
        undefined for it rather than infinite. Callers must say so instead of
        handing back raw numbers under an acc label.
        """
        driven = set()
        for rbc in self.resolved_bcs():
            cond = rbc.condition
            if cond.kind != "VELO":
                continue
            if not np.any(cond.at(self.frequencies) != 0):
                continue
            driven.update(int(e) for e in rbc.elements)
        if not driven:
            return None
        return float(self.mesh.areas[sorted(driven)].sum())

    def __repr__(self):
        return (f"Model({self.mesh!r}, {self.n_frequencies} freqs "
                f"{self.frequencies[0]:.0f}-{self.frequencies[-1]:.0f} Hz, "
                f"{len(self.bcs)} bcs, {len(self.sources)} sources, "
                f"fields={list(self.fields)})")
