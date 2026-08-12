"""A second, in-process backend: bempp-cl.

It exists to *cross-check* NumCalc, not to replace it. NumCalc is a batch
binary with a bundle directory; this one assembles dense matrices in the
current process and hands back a :class:`~bemkit.results.results.Results` over
a ``MemoryStore``. Two backends that share nothing but the neutral
:class:`~bemkit.model.Model` are what turn "the Model is solver-neutral" from
a claim into a measurement — see
``packages/bemkit/tests/solvers/test_cross_validation.py``.

Because it is a cross-check, it supports a deliberately small subset and
raises :class:`NotImplementedError` — naming the feature — for everything
else. A backend that quietly solves a *different* problem than the one it was
handed is worse than no second backend at all: it would make a disagreement
look like a bug in NumCalc and an agreement look like a proof.

Supported
    Exterior radiation, uniform (possibly complex) :class:`Velocity`
    conditions, field-point and boundary output.

Refused
    Interior problems, ``Pressure``/``Admittance``/``Impedance`` conditions,
    frequency-dependent conditions, incident sources (plane waves, point
    sources).

Formulation
    Direct (non-regularised) exterior Neumann BIE, Galerkin, piecewise
    constant (``DP`` 0) on both traces — the same one-value-per-element
    discretization NumCalc's collocation scheme uses, so the comparison is
    between two solvers rather than between two element types.

    With ``n`` pointing out of the body, the exterior representation is
    ``p = -S[t] + D[p]`` and its exterior trace gives::

        (1/2 I - K) p = -S t,        t = dp/dn

    A Burton-Miller combined operator would be immune to the irregular
    frequencies of this formulation, but bempp's ``hypersingular`` refuses a
    piecewise-constant space outright (*"Domain shapeset must be of type
    'p1_discontinuous'"*), so it would mean discretizing on linear elements
    and comparing two element types as well as two solvers. The direct form
    is used instead, and the irregular frequencies are avoided rather than
    regularised: they are the interior
    Dirichlet eigenvalues of the surface, whose lowest for a sphere of radius
    ``a`` is ``ka = pi`` (``f = c/2a``: 3400 Hz for the 50 mm sphcap sphere).
    Stay well below the first one, or use NumCalc.

Time convention
    bempp's Helmholtz kernel is ``exp(+ikr)/(4*pi*r)``
    (``bempp_cl/core/numba_kernels.py``), i.e. an outgoing wave under
    ``exp(-i*omega*t)``. bemkit's own convention is the opposite one, and this
    was *measured*, not assumed: at ``time_factor = -1.0`` NumCalc's radiated
    field goes as ``exp(-ikr)`` (verified on the sphcap on-axis field: the
    phase advances by ``-k*dr``), the analytic reference
    ``bemkit.model.analytic`` is built on ``h_n^(2)`` and so does the same,
    and NumCalc run at ``time_factor = +1.0`` returns exactly the complex
    conjugate. So a bempp solve is conjugated on the way out for
    ``time_factor = -1.0`` and left alone for ``+1.0``. See the module test
    ``test_cross_validation.py`` for the pinned evidence.
"""

import numpy as np

from bemkit.results.results import Results
from bemkit.results.store import MemoryStore
from bemkit.solvers import fieldpoints

#: Step, in metres, for the central difference that turns the pressure
#: potential into a particle velocity. bempp exposes no gradient potential
#: operator, so the gradient is differenced. At 1e-5 m the truncation error is
#: O((k*h)^2) — below 1e-8 anywhere in the audio band — and the cancellation
#: error is ~1e-11 relative, so the result is exact to far more digits than
#: the discretization it is differentiating.
_GRAD_STEP = 1e-5


def _refuse(model):
    """Raise NotImplementedError for every part of ``model`` this cannot solve.

    Named, one feature at a time. The alternative — solving the parts it
    understands and ignoring the rest — is the failure this whole backend
    exists to rule out.
    """
    if not model.exterior:
        raise NotImplementedError(
            "BemppSolver solves exterior problems only; this Model has "
            "exterior=False (an interior problem). Use NumCalcSolver.")

    if model.sources:
        kinds = sorted({type(s).__name__ for s in model.sources})
        raise NotImplementedError(
            f"BemppSolver does not support incident sources "
            f"({', '.join(kinds)}) — it solves the radiation problem only. "
            f"Use NumCalcSolver.")

    for bc in model.bcs:
        cond = bc.condition
        if cond.kind != "VELO":
            raise NotImplementedError(
                f"BemppSolver supports Velocity conditions only; this Model "
                f"has a {type(cond).__name__} ({cond.kind}). Use "
                f"NumCalcSolver.")
        if cond.is_frequency_dependent:
            raise NotImplementedError(
                "BemppSolver does not support frequency-dependent "
                "conditions. Use NumCalcSolver.")


def _normal_velocity(model):
    """(M,) complex prescribed normal velocity, one value per element.

    An element that no condition names stays at 0 — sound-hard, which is what
    NumCalc means by an element with no BOUNDARY line, so both backends
    default the same way.

    Raises:
        ValueError: a condition selects no elements, or two conditions land
            on the same element. Both mirror ``NumCalcSolver``: an empty
            selector is a BC that vanished without a trace, and two velocities
            on one element cannot both hold.
    """
    vn = np.zeros(model.mesh.n_elements, dtype=complex)
    claimed = np.zeros(model.mesh.n_elements, dtype=bool)

    for rbc in model.resolved_bcs():
        elements = np.unique(np.asarray(rbc.elements, dtype=int))
        if elements.size == 0:
            raise ValueError(
                "a VELO boundary condition selects no elements — check the "
                "selector (an empty physical group, or an empty literal "
                "list, resolves to nothing)")
        clash = elements[claimed[elements]]
        if clash.size:
            raise ValueError(
                f"element {int(clash[0])} has two VELO boundary conditions "
                f"assigned; one would have to overwrite the other, so this "
                f"is always a modelling mistake.")
        claimed[elements] = True
        vn[elements] = complex(rbc.condition.at(model.frequencies)[0])

    return vn


def _signed_volume(mesh):
    """Divergence-theorem volume. Positive iff the winding is outward."""
    v = mesh.vertices[mesh.triangles]
    return float(np.einsum(
        "ij,ij->i", v[:, 0], np.cross(v[:, 1] - v[:, 0],
                                      v[:, 2] - v[:, 0])).sum() / 6.0)


def _grid(model):
    """A bempp Grid over ``model.mesh``, checked for the two ways it can lie.

    bempp derives element normals from the vertex winding, so an inward-wound
    mesh would still assemble — and silently solve the *interior* problem. And
    the boundary conditions are indexed by element, so a Grid that reordered
    elements would scramble which ones vibrate. Neither shows up in the
    numbers as anything but a wrong answer, so both are checked here.
    """
    import bempp_cl.api as bempp

    volume = _signed_volume(model.mesh)
    if volume <= 0:
        raise ValueError(
            f"the mesh is not a closed, outward-wound surface (signed volume "
            f"{volume:.4g} m^3). bempp takes element normals from the vertex "
            f"winding, so an inverted mesh would quietly solve the interior "
            f"problem instead.")

    grid = bempp.Grid(np.ascontiguousarray(model.mesh.vertices.T, dtype=float),
                      np.ascontiguousarray(model.mesh.triangles.T,
                                           dtype=np.uint32))
    if grid.number_of_elements != model.mesh.n_elements:
        raise RuntimeError(
            f"bempp built a grid with {grid.number_of_elements} elements from "
            f"a {model.mesh.n_elements}-element mesh")
    drift = np.abs(np.asarray(grid.centroids) - model.mesh.centroids).max()
    if drift > 1e-9:
        raise RuntimeError(
            f"bempp reordered the mesh elements (centroids differ by "
            f"{drift:.3g} m); the per-element boundary conditions would no "
            f"longer line up")
    return grid



def _field_points(model):
    """Validated field points and their per-set offsets.

    Shares ``bemkit.solvers.fieldpoints`` with NumCalcSolver on purpose:
    the two backends must drop the *same* points, or their Results would
    have different rows under the same set names and could not be
    compared at all.

    Unlike NumCalcSolver a model with no field points is fine here (an
    in-process solve has no EvalGrid to be empty), but a model whose
    points were all dropped still raises — that is a modelling mistake in
    either backend.
    """
    raw, _ = fieldpoints.combine(model.fields)
    if len(raw) == 0:
        return raw, {name: np.zeros(0, dtype=int) for name in model.fields}

    mask = fieldpoints.validate(raw, model.mesh, exterior=model.exterior)
    points, offsets = fieldpoints.combine(model.fields, mask)
    if len(points) == 0:
        raise ValueError(
            f"all {len(raw)} field point(s) were dropped by validation "
            f"(on the wrong side of the surface, or too close to it) — "
            f"check the field-point sets against the mesh and "
            f"Model.exterior")
    return points, offsets


def _solve_one(space, identity, k, t_coeff, points, rho, omega):
    """One frequency, in bempp's own ``exp(-i*omega*t)`` convention.

    ``identity`` is the frequency-independent mass matrix, assembled once by
    the caller.

    Returns ``(p_surface, p_points, v_points)``: the surface pressure per
    element, the pressure at ``points``, and the particle velocity there.
    """
    import bempp_cl.api as bempp
    from bempp_cl.api.operators.boundary import helmholtz as boundary
    from bempp_cl.api.operators.potential import helmholtz as potential

    dlp = boundary.double_layer(space, space, space, k).weak_form().to_dense()
    slp = boundary.single_layer(space, space, space, k).weak_form().to_dense()

    # (1/2 I - K) p = -S t, the exterior trace of p = -S[t] + D[p].
    p_surface = np.linalg.solve(0.5 * identity - dlp, -(slp @ t_coeff))

    if len(points) == 0:
        empty_p = np.zeros(0, dtype=complex)
        return p_surface, empty_p, np.zeros((0, 3), dtype=complex)

    t_fun = bempp.GridFunction(space, coefficients=t_coeff)
    p_fun = bempp.GridFunction(space, coefficients=p_surface)

    def field(at):
        return (-potential.single_layer(space, at.T, k).evaluate(t_fun)
                + potential.double_layer(space, at.T, k).evaluate(p_fun)).ravel()

    p_points = field(points)

    # No gradient potential operator exists, so difference the field. Euler's
    # equation in bempp's convention (-i*omega*rho*v = -grad p) gives
    # v = grad(p) / (i*omega*rho).
    gradient = np.empty((len(points), 3), dtype=complex)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = _GRAD_STEP
        gradient[:, axis] = ((field(points + step) - field(points - step))
                             / (2 * _GRAD_STEP))
    return p_surface, p_points, gradient / (1j * omega * rho)


class BemppSolver:
    """Solve a Model in-process with bempp-cl.

    Deliberately not a NumCalc replacement — see the module docstring for the
    supported subset and the formulation.
    """

    name = "bempp"

    def solve(self, model, *, workdir=None, bins=None, jobs=1):
        """Solve ``model`` and return a :class:`Results` over a MemoryStore.

        Args:
            model: the Model. Must fall inside the supported subset; anything
                outside it raises NotImplementedError naming the feature.
            workdir: accepted for the Solver protocol and **ignored**. This
                backend keeps no bundle and writes nothing to disk; use
                ``Results.save`` if you want one. (:meth:`status` says the
                same thing from the other side rather than inventing a
                Status for bins that were never persisted.)
            bins: 1-based frequency indices to compute (None = all). Taken
                literally, as ``NumCalcSolver`` does. Bins that are not
                requested come back as NaN — "not computed", never 0.
            jobs: accepted for the protocol and ignored; the assembly and
                solve run in this process.

        Raises:
            NotImplementedError: the Model uses something this backend does
                not support.
            ValueError: a bad ``bins`` selection, a boundary condition that
                selects nothing or collides with another, or every field
                point dropped by validation.
        """
        _refuse(model)

        frequencies = model.frequencies
        nf = len(frequencies)
        if bins is None:
            wanted = list(range(1, nf + 1))
        else:
            wanted = sorted({int(b) for b in bins})
            out_of_range = [b for b in wanted if not 1 <= b <= nf]
            if out_of_range:
                raise ValueError(
                    f"bins {out_of_range} are outside 1..{nf} (the model has "
                    f"{nf} frequencies)")

        points, offsets = _field_points(model)
        vn = _normal_velocity(model)
        grid = _grid(model)

        import bempp_cl.api as bempp
        from bempp_cl.api.operators.boundary import sparse

        space = bempp.function_space(grid, "DP", 0)
        # Frequency-independent, so assembled once rather than per bin.
        identity = sparse.identity(space, space, space).weak_form().to_dense()

        n_elements = model.mesh.n_elements
        n_points = len(points)
        p_boundary = np.full((n_elements, nf), np.nan + 0j, dtype=complex)
        v_boundary = np.full((n_elements, nf), np.nan + 0j, dtype=complex)
        p_eval = np.full((n_points, nf), np.nan + 0j, dtype=complex)
        v_eval = np.full((n_points, nf, 3), np.nan + 0j, dtype=complex)

        # bemkit's convention (time_factor -1.0) is the conjugate of bempp's;
        # see the module docstring for how that was established. Conjugating
        # a solve means conjugating what goes in as well as what comes out —
        # the same physical velocity is the conjugate phasor in the other
        # convention, and for a complex Velocity() that is not a no-op.
        conjugate = model.time_factor == -1.0
        vn_native = np.conj(vn) if conjugate else vn

        for b in wanted:
            f = float(frequencies[b - 1])
            omega = 2 * np.pi * f
            k = omega / model.medium.c
            # dp/dn = i*omega*rho*v_n under exp(-i*omega*t), the convention
            # the operators below are assembled in.
            t_coeff = 1j * omega * model.medium.rho * vn_native
            surface, pressure, velocity = _solve_one(
                space, identity, k, t_coeff, points, model.medium.rho, omega)
            if conjugate:
                surface, pressure, velocity = (np.conj(surface),
                                               np.conj(pressure),
                                               np.conj(velocity))
            p_boundary[:, b - 1] = surface
            # The prescribed normal velocity, echoed back exactly as NumCalc
            # does — it is boundary *data*, not something either solver
            # computes.
            v_boundary[:, b - 1] = vn
            p_eval[:, b - 1] = pressure
            v_eval[:, b - 1, :] = velocity

        store = MemoryStore({"pBoundary": p_boundary, "vBoundary": v_boundary,
                             "pEvalGrid": p_eval, "vEvalGrid": v_eval})
        return Results(store, frequencies, offsets)

    def status(self, workdir):
        """Always raises: this backend persists nothing.

        A ``Status`` here could only be a vacuous one — no bins present, none
        missing, therefore "complete" — describing a run that left no trace
        on disk at all. Saying so is better than answering.
        """
        raise NotImplementedError(
            "BemppSolver keeps no working directory: it solves in-process and "
            "returns a materialised Results, so there is no run in "
            f"{workdir} whose bins could be present or missing. Use "
            "Results.save()/Results.load() to persist a solve, or "
            "NumCalcSolver for a resumable bundle.")
