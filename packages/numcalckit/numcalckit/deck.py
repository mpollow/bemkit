"""The ``NC.inp`` input deck as a Python object.

A *deck* is the complete NumCalc input file: one field per thing the solver
actually reads, in the solver's own terms. It is deliberately literal — it holds
an explicit array of frequencies in Hz, not ``sr``/``nbins``; it holds
``ELEM 100 TO 100 VELO 1.0``, not the ``near=`` anchor that produced it. Those
are bemkit's opinions and live in :mod:`bemkit.solvers.numcalc`.

Two properties are the point of the abstraction:

* **Literal** — anything NumCalc can express is representable here, whether or
  not bemkit currently has a way to ask for it.
* **Round-trippable** — ``Deck.read(path)`` after ``deck.write(path)`` returns an
  equal deck. That single property is what keeps "support every NumCalc setting"
  testable as fields are added.

Lexer constraints this writer must respect (``NC_GetLine``, ``NC_Input.cpp:1596``):

* Terms are split on the **space character only** — a tab is not a separator, so
  every emitted line uses plain spaces.
* A line is a comment when its **first** term contains ``#``. Only whole-line
  comments are safe; a trailing ``250.0 # freq`` would parse as three terms.
* Blank lines are skipped, and the final line must end with a newline (the lexer
  drops the last character of a line, assuming it is ``\\n``).

Format reference: ``mesh2hrtf/NumCalc/src/NC_Input.cpp``; see also
``docs/numcalc-feature-coverage.md`` §1 for what each field means and which
values are hard errors.
"""

from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Boundary condition on an inclusive, 0-based element range.
#: ``kind`` is VELO / PRES / ADMI / IMPE (§1.1). ``curve_*`` are CURVES ids, or
#: -1 for a constant — NumCalc requires both to be -1 or both to be set.
BC = namedtuple("BC", "start end kind real imag curve_real curve_imag")
BC.__new__.__defaults__ = (0.0, -1, -1)  # imag, curve_real, curve_imag

#: Incident source. ``vec`` is a travel *direction* for a plane wave and a
#: *position* in metres for a point source — same layout, different meaning.
Source = namedtuple("Source", "nr vec real imag curve_real curve_imag")
Source.__new__.__defaults__ = (0.0, -1, -1)

#: Frequency curve: ``points`` is [(hz, value), ...], interpolated linearly and
#: clamped flat outside its range.
Curve = namedtuple("Curve", "id points")

BC_KINDS = ("VELO", "PRES", "ADMI", "IMPE")

#: Read and discarded by NumCalc (analysis type / unit system), but the line
#: must be present. Kept as a constant rather than a field for that reason.
_CONTROL_PARAMETER_I = "0 0 0 0 7 0"
#: Likewise: "Main Parameters III" is consumed with no assignment at all
#: (``NC_Input.cpp:325``).
_MAIN_PARAMETERS_III = "0 0 0 0"

_VERSION_LINE = "Mesh2HRTF 1.0.0"

#: Section keywords that terminate a run of data rows when scanning.
_KEYWORDS = ("NODES", "ELEMENTS", "SYMMETRY", "BOUNDARY", "RETU", "PLANE",
             "POINT", "CURVES", "POST", "END")


def _g(x):
    """Format a float losslessly, always keeping a visible decimal point.

    ``.15g`` alone round-trips but renders 1.0 as ``1``, which turns a boundary
    line into ``VELO 1 -1 0 -1`` — four bare integers where two are values and
    two are curve references. Keeping the point makes the values readable
    without costing precision.
    """
    s = f"{float(x):.15g}"
    return s if ("." in s or "e" in s or "n" in s) else s + ".0"


def _contiguous_ranges(indices):
    """Collapse integer indices into inclusive (start, end) runs."""
    ranges = []
    for idx in sorted(int(i) for i in indices):
        if ranges and idx == ranges[-1][1] + 1:
            ranges[-1][1] = idx
        else:
            ranges.append([idx, idx])
    return [tuple(r) for r in ranges]


@dataclass
class Deck:
    """A complete NumCalc ``NC.inp`` input deck.

    Counts are the *totals* NumCalc reads from "Main Parameters I" — the
    object/eval split is not in the file at all (the solver derives it from the
    mesh files), so the deck does not carry it either.
    """

    title: str
    frequencies: np.ndarray                 # (nf,) Hz, explicit

    # -- Main Parameters IV: medium --
    speed_of_sound: float = 340.0
    air_density: float = 1.2
    time_factor: float = -1.0               # coerced to +-1 by NumCalc

    # -- Main Parameters I/II: solver --
    bem_type: int = 4                       # 0 TBEM, 1 SLFMBEM, 4 DMLFMBEM
    is_internal: int = 0
    n_element_groups: int = 2
    min_expansion_terms: int = 0            # 0 => NumCalc default of 8; else [5,10]
    isolver: int = 0                        # 0 iterative CGS, 4 direct Gauss
    cluster_edge_length: float = 0.0        # ClusEdgL0, metres; 0 => default
    preconditioner: int = 0                 # IPRECOND in [-3, 2]

    # -- mesh --
    n_total_nodes: int = 0
    n_total_elements: int = 0
    #: Mesh files named by the NODES/ELEMENTS sections, in read order.
    #:
    #: Two by default, one per file when there is no EvalGrid. The count is
    #: load-bearing: NumCalc reads node files in a ``while(i < numNodes_)``
    #: loop (``NC_Input.cpp:371``) and stops as soon as it has
    #: ``n_total_nodes`` of them, so a listed-but-uncounted EvalGrid line is
    #: never consumed and is then read where the ELEMENTS keyword is
    #: expected -- "Key word ELEMENTS expected!", and the run dies. A
    #: boundary-only bundle must therefore name the object mesh alone.
    node_files: tuple = ("ObjectMesh/Nodes.txt", "EvalGrid/Nodes.txt")
    element_files: tuple = ("ObjectMesh/Elements.txt", "EvalGrid/Elements.txt")

    # -- optional sections --
    symmetry_planes: list | None = None
    symmetry_coordinates: list | None = None
    boundary: list = field(default_factory=list)        # list[BC]
    plane_waves: list = field(default_factory=list)     # list[Source]
    point_sources: list = field(default_factory=list)   # list[Source]
    curves: list = field(default_factory=list)          # list[Curve]

    #: Leading ``##`` comment block, stored verbatim and emitted unchanged.
    header: list = field(default_factory=list)

    # ------------------------------------------------------------------ #

    def __post_init__(self):
        self._normalize()

    def _normalize(self):
        """Put equivalent decks into one canonical form.

        ``planes = [0, 0, 0]`` and ``planes = None`` mean the same thing and emit
        the same file (no SYMMETRY section), so they must compare equal — else
        a write/read round trip would report a spurious difference.
        """
        if self.n_symmetric_planes == 0:
            self.symmetry_planes = None
            self.symmetry_coordinates = None
        self.node_files = tuple(self.node_files)
        self.element_files = tuple(self.element_files)

    @property
    def n_symmetric_planes(self):
        """How many mirror planes are active — what NumCalc reads, not len()."""
        return sum(1 for p in (self.symmetry_planes or []) if p != 0)

    def validate(self):
        """Raise on anything NumCalc would reject or silently misread."""
        nf = len(self.frequencies)
        if nf < 1:
            raise ValueError("deck has no frequencies")
        if self.bem_type not in (0, 1, 4):
            raise ValueError(f"bem_type must be 0, 1 or 4, got {self.bem_type}")
        if self.isolver not in (0, 4):
            raise ValueError(f"isolver (ISOLVER) must be 0 or 4, got {self.isolver}")
        if self.isolver == 4 and self.bem_type != 0:
            # NC_Input.cpp:290 downgrades this to the iterative solver without
            # a word, so the deck would not do what it says.
            raise ValueError(
                f"isolver = 4 (direct) requires bem_type = 0 (TBEM); with "
                f"bem_type = {self.bem_type} NumCalc silently falls back to the "
                f"iterative solver")
        if self.min_expansion_terms != 0 and not (5 <= self.min_expansion_terms <= 10):
            raise ValueError(
                f"min_expansion_terms must be 0 (NumCalc default 8) or in [5, 10], "
                f"got {self.min_expansion_terms}")
        if not (-3 <= self.preconditioner <= 2):
            raise ValueError(
                f"preconditioner (IPRECOND) must be in [-3, 2], "
                f"got {self.preconditioner}")
        if self.is_internal not in (0, 1):
            raise ValueError(f"is_internal must be 0 or 1, got {self.is_internal}")
        if not 0 <= self.n_symmetric_planes <= 3:
            raise ValueError(
                f"at most 3 symmetry planes, got {self.n_symmetric_planes}")
        for bc in self.boundary:
            if bc.kind not in BC_KINDS:
                raise ValueError(
                    f"boundary kind must be one of {', '.join(BC_KINDS)}, "
                    f"got {bc.kind!r} (TRAD/TRIM are parsed by NumCalc and then "
                    f"hard-error, so they are not supported)")
            if bc.start > bc.end:
                raise ValueError(f"boundary range {bc.start}..{bc.end} is inverted")
            # NumCalc rejects a line mixing curve refs with constants.
            if (bc.curve_real == -1) != (bc.curve_imag == -1):
                raise ValueError(
                    f"boundary {bc.kind} {bc.start}..{bc.end}: curve refs must be "
                    f"both -1 or both set (NC_Input.cpp:844)")
            if bc.kind == "IMPE" and bc.curve_real != -1:
                raise ValueError(
                    "IMPE with a frequency curve makes NumCalc exit(-1) — "
                    "express a frequency-dependent surface as ADMI instead")
        for name, srcs in (("plane wave", self.plane_waves),
                           ("point source", self.point_sources)):
            for s in srcs:
                if len(s.vec) != 3:
                    raise ValueError(f"{name} {s.nr}: vec must be 3 numbers")
                if (s.curve_real == -1) != (s.curve_imag == -1):
                    raise ValueError(
                        f"{name} {s.nr}: curve refs must be both -1 or both set")
        for s in self.plane_waves:
            if not any(float(c) for c in s.vec):
                raise ValueError(
                    f"plane wave {s.nr}: direction is all zeros — NumCalc "
                    f"normalises it and would divide by zero")
        ids = [c.id for c in self.curves]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate CURVES ids: {ids}")

    # ------------------------------------------------------------------ #

    def write(self, path):
        """Write this deck to ``path`` as an ``NC.inp`` file."""
        self.validate()
        freqs = np.asarray(self.frequencies, dtype=float)
        nf = len(freqs)
        out = []

        out += list(self.header)
        out += [_VERSION_LINE, "##", self.title, "##"]
        out += ["## Controlparameter I", _CONTROL_PARAMETER_I, "##"]

        # Controlparameter II + Load Frequency Curve together define which
        # frequencies are solved: step i is evaluated at t = first + (i+1)*incr
        # and the curve is interpolated there (NC_ComputeFrequencies). Using
        # incr=1 and one curve point per step makes that mapping the identity,
        # so any frequency list -- uniform or not -- is expressed the same way.
        #
        # It is also exact. The two-point shorthand bemkit wrote until now
        # (span 0..nf*1e-6 mapped onto 0..sr/2) reaches each step through a
        # division, so it asked for 250.00000000000003 Hz where it meant 250.
        # That is harmless physically but not free: a 1e-16 perturbation moves
        # the CGS iteration to a different point inside its own 1e-6 tolerance,
        # which showed up as ~1e-5 relative differences in the radiated field.
        out += ["## Controlparameter II", f"1 {nf} 1 0 1 0 0", "##"]
        out += ["## Load Frequency Curve", f"0 {nf}"]
        out += [f"{i + 1} {_g(f)} 0.0" for i, f in enumerate(freqs)]
        out += ["##"]

        out += ["## 1. Main Parameters I",
                f"{self.n_element_groups} {self.n_total_elements} "
                f"{self.n_total_nodes} {self.min_expansion_terms} "
                f"{self.n_symmetric_planes} 2 2 {self.bem_type} {self.isolver}",
                "##"]
        out += ["## 2. Main Parameters II",
                f"{len(self.plane_waves)} {len(self.point_sources)} "
                f"{self.is_internal} {self.cluster_edge_length:.4e} "
                f"{self.preconditioner} 0 0",
                "# 4th = ClusEdgL0, the FMM initial bounding-box edge length in m:",
                "# 0 means NumCalc's default of (sqrt(N)*A0)^0.5. Shrinking it gives",
                "# smaller cluster radii, a shorter multipole expansion L, and lower",
                "# peak memory -- the documented remedy when a large mesh exhausts",
                "# RAM or L exceeds ~30 (arXiv:2312.00005 Sec. 3.3.2, 4.2).",
                "# 5th = IPRECOND in [-3,2]: 0 none, negative selects incomplete-LU depth.",
                "##"]
        out += ["## 3. Main Parameters III", _MAIN_PARAMETERS_III, "##"]
        out += ["## 4. Main Parameters IV",
                f"{_g(self.speed_of_sound)} {self.air_density:.4e} "
                f"{_g(self.time_factor)} 0.0e+00 0.0e+00 0.0e+00 0.0e+00",
                "##"]

        out += ["NODES"] + list(self.node_files) + ["##"]
        out += ["ELEMENTS"] + list(self.element_files) + ["##"]

        if self.n_symmetric_planes > 0:
            p = self.symmetry_planes
            c = self.symmetry_coordinates or [0.0, 0.0, 0.0]
            out += ["SYMMETRY",
                    " ".join(str(int(v)) for v in p),
                    " ".join(_g(v) for v in c),
                    "##"]

        out += ["BOUNDARY"]
        for bc in self.boundary:
            out += [f"ELEM {int(bc.start)} TO {int(bc.end)} {bc.kind} "
                    f"{_g(bc.real)} {int(bc.curve_real)} "
                    f"{_g(bc.imag)} {int(bc.curve_imag)}"]
        out += ["RETU", "##"]

        for keyword, srcs in (("PLANE WAVES", self.plane_waves),
                              ("POINT SOURCES", self.point_sources)):
            if srcs:
                out += [keyword]
                out += [f"{int(s.nr)} {' '.join(_g(v) for v in s.vec)} "
                        f"{_g(s.real)} {int(s.curve_real)} "
                        f"{_g(s.imag)} {int(s.curve_imag)}" for s in srcs]
                out += ["##"]

        if self.curves:
            out += ["CURVES",
                    f"{len(self.curves)} {max(len(c.points) for c in self.curves)}"]
            for c in self.curves:
                out += [f"{int(c.id)} {len(c.points)}"]
                out += [f"{_g(hz)} {_g(val)} 0.0" for hz, val in c.points]
            out += ["##"]

        out += ["POST PROCESS", "##", "END"]

        path = Path(path)
        # Trailing newline is mandatory: NC_GetLine drops the last character of
        # every line, assuming it is one.
        path.write_text("\n".join(out) + "\n")

    # ------------------------------------------------------------------ #

    @classmethod
    def read(cls, path):
        """Parse an ``NC.inp`` file back into a Deck.

        Applies NumCalc's own comment rule (a first term containing ``#``) so a
        hand-edited deck is read exactly as the solver would read it.

        Caveat on frequencies: the step-to-Hz mapping is recomputed with
        ``np.interp``, which agrees with ``NC_ComputeFrequencies`` to within
        floating-point association order, not bit-for-bit. It is exact for decks
        :meth:`write` produced (one curve node per step, so every step lands on
        a node), and can differ in the last bits when reading a deck that
        encodes a fine grid with a coarse curve — which is precisely why we
        write one node per step.
        """
        path = Path(path)
        raw = path.read_text().splitlines()

        header, seen_data = [], False
        rows = []
        for line in raw:
            stripped = line.strip()
            is_comment = (not stripped) or stripped.split(" ")[0].find("#") >= 0
            if is_comment:
                if not seen_data:
                    header.append(line)
                continue
            seen_data = True
            rows.append(stripped.split())

        if not rows:
            raise ValueError(f"{path}: no data lines")
        # The header is only the block above the version line; comments further
        # down are ours to regenerate and are not round-tripped.
        cur = _DeckParser(rows, path)

        version = cur.take()
        if version[0] != "Mesh2HRTF":
            raise ValueError(f"{path}: expected 'Mesh2HRTF' version line, "
                             f"got {' '.join(version)!r}")
        title = " ".join(cur.take())
        cur.take()                                   # Controlparameter I
        ctrl2 = cur.take()                           # Controlparameter II
        n_freq = int(ctrl2[1])
        increment, first = float(ctrl2[2]), float(ctrl2[3])

        n_curve_points = int(cur.take()[1])          # Load Frequency Curve
        curve_t, curve_f = [], []
        for _ in range(n_curve_points):
            t = cur.take()
            curve_t.append(float(t[0]))
            curve_f.append(float(t[1]))
        steps = first + (np.arange(n_freq) + 1) * increment
        # np.interp clamps flat outside the range, exactly like NumCalc's
        # NC_ComputeFrequencies (NC_Input.cpp:1686).
        frequencies = np.interp(steps, curve_t, curve_f)

        mp1 = cur.take()
        mp2 = cur.take()
        cur.take()                                   # Main Parameters III
        mp4 = cur.take()

        deck = cls(
            title=title,
            frequencies=frequencies,
            n_element_groups=int(mp1[0]),
            n_total_elements=int(mp1[1]),
            n_total_nodes=int(mp1[2]),
            min_expansion_terms=int(mp1[3]),
            bem_type=int(mp1[7]),
            isolver=int(mp1[8]) if len(mp1) > 8 else 0,
            is_internal=int(mp2[2]),
            cluster_edge_length=float(mp2[3]),
            preconditioner=int(mp2[4]),
            speed_of_sound=float(mp4[0]),
            air_density=float(mp4[1]),
            time_factor=float(mp4[2]),
            header=header,
        )

        while not cur.done():
            kw = cur.take()
            head = kw[0].upper()
            if head == "NODES":
                deck.node_files = cur.read_files()
            elif head == "ELEMENTS":
                deck.element_files = cur.read_files()
            elif head == "SYMMETRY":
                deck.symmetry_planes = [int(v) for v in cur.take()]
                deck.symmetry_coordinates = [float(v) for v in cur.take()]
            elif head == "BOUNDARY":
                deck.boundary = cur.read_boundary()
            elif head in ("PLANE", "POINT"):
                srcs = cur.read_sources()
                if head == "PLANE":
                    deck.plane_waves = srcs
                else:
                    deck.point_sources = srcs
            elif head == "CURVES":
                deck.curves = cur.read_curves()
            elif head in ("POST", "END"):
                if head == "END":
                    break
            else:
                raise ValueError(f"{path}: unexpected keyword {' '.join(kw)!r}")
        # Sections are assigned after construction, so canonicalise once more.
        deck._normalize()
        return deck

    # ------------------------------------------------------------------ #

    def __eq__(self, other):
        if not isinstance(other, Deck):
            return NotImplemented
        if not np.array_equal(np.asarray(self.frequencies, dtype=float),
                              np.asarray(other.frequencies, dtype=float)):
            return False
        skip = {"frequencies", "header"}
        return all(getattr(self, f) == getattr(other, f)
                   for f in self.__dataclass_fields__ if f not in skip)


class _DeckParser:
    """Positional reader over the deck's tokenised data lines.

    The deck is read strictly in order, and the section readers below advance
    the same position as the preamble, so they live here rather than as loose
    functions taking a cursor.
    """

    def __init__(self, rows, path):
        self.rows, self.i, self.path = rows, 0, path

    def take(self):
        if self.i >= len(self.rows):
            raise ValueError(f"{self.path}: unexpected end of file")
        self.i += 1
        return self.rows[self.i - 1]

    def peek(self):
        return self.rows[self.i] if self.i < len(self.rows) else None

    def done(self):
        return self.i >= len(self.rows)

    def read_boundary(self):
        """Read ELEM lines up to RETU."""
        out = []
        while True:
            row = self.take()
            if row[0].upper() == "RETU":
                return out
            out.append(parse_bc_line(row, where=f"{self.path}: "))

    def read_files(self):
        """Read mesh-file names until the next section keyword.

        The count is not fixed at two. A boundary-only bundle has no
        EvalGrid, and then NODES/ELEMENTS name one file each -- see
        ``Deck.node_files``.
        """
        out = []
        while (row := self.peek()) is not None and row[0].upper() not in _KEYWORDS:
            out.append(" ".join(self.take()))
        return tuple(out)

    def read_sources(self):
        """Read 8-term source rows until the next section keyword."""
        out = []
        while (row := self.peek()) is not None and row[0].upper() not in _KEYWORDS:
            row = self.take()
            if len(row) != 8:
                raise ValueError(
                    f"{self.path}: source row {' '.join(row)!r} must have 8 terms")
            out.append(Source(nr=int(row[0]),
                              vec=tuple(float(v) for v in row[1:4]),
                              real=float(row[4]), curve_real=int(row[5]),
                              imag=float(row[6]), curve_imag=int(row[7])))
        return out

    def read_curves(self):
        """Read the CURVES section: a count line, then id/npoints + point rows."""
        n_curves = int(self.take()[0])
        out = []
        for _ in range(n_curves):
            head = self.take()
            cid, npoints = int(head[0]), int(head[1])
            points = [(float(r[0]), float(r[1]))
                      for r in (self.take() for _ in range(npoints))]
            out.append(Curve(id=cid, points=points))
        return out


def parse_bc_line(tokens, where=""):
    """Parse one tokenised ``ELEM lo TO hi KIND re curve imag curve`` line.

    Shared with the lenient section scanner in :mod:`numcalckit.readers`, so
    the element ranges that determine ``A_vib`` are extracted by exactly the
    same code that reads a full deck.

    Raises:
        ValueError: on anything NumCalc would reject (it needs at least 9
            terms, ``NC_Input.cpp:833``).
    """
    if (len(tokens) < 9 or tokens[0].upper() != "ELEM"
            or tokens[2].upper() != "TO"):
        raise ValueError(
            f"{where}malformed BOUNDARY line {' '.join(tokens)!r} — expected "
            f"'ELEM lo TO hi KIND re curve imag curve' (NumCalc requires at "
            f"least 9 terms, NC_Input.cpp:833)")
    return BC(start=int(tokens[1]), end=int(tokens[3]), kind=tokens[4].upper(),
              real=float(tokens[5]), curve_real=int(tokens[6]),
              imag=float(tokens[7]), curve_imag=int(tokens[8]))
