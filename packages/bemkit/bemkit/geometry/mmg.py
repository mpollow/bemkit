"""Talking to ``mmgs``: the Medit files it reads, and running it.

``mmgs`` adapts an existing triangulated surface to a target size field rather
than meshing from scratch, which is what makes it the right tool once a mesh
exists and only part of it should change. It communicates entirely through two
Medit files — a ``.mesh`` carrying the surface and a ``.sol`` carrying one
target size per vertex — so this module is the writers, the reader, and the
subprocess call.

The size field itself is built in :mod:`bemkit.geometry.metric`;
``curvature_sizes`` is re-exported here for the callers that predate that split.

**Why the ``.mesh`` writer is ours rather than meshio's.** meshio emits only
``Vertices``/``Edges``/``Triangles``, and its reader does not recognise the
``Required*`` keywords at all. ``mmgs`` (checked against mmg 5.8.0) does read
``RequiredVertices``, ``RequiredEdges`` and ``RequiredTriangles``, and those are
the only mechanism for telling it *not* to touch part of a surface. Freezing a
region is the whole reason this module exists in the form it does — a head mesh
that is remeshed differently for each device configuration cannot be compared
against itself at the ear — so we write the file.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from bemkit.geometry.metric import curvature as _curvature
from bemkit.model import Mesh

#: The mmgs binary. Distributions name it after the optimisation level;
#: ``mmgs_O3`` is what Arch and Debian ship.
DEFAULT_BINARY = "mmgs_O3"


def curvature_sizes(mesh, hmin, hmax):
    """Deprecated alias for :func:`bemkit.geometry.metric.curvature`."""
    return _curvature(mesh, hmin, hmax)


def write_sol(path, sizes):
    """Write an mmgs isotropic metric (.sol) — one size per vertex."""
    with open(path, "w") as f:
        f.write("MeshVersionFormatted 2\n\nDimension 3\n\nSolAtVertices\n")
        f.write(f"{len(sizes)}\n1 1\n\n")
        f.writelines(f"{s:.10g}\n" for s in sizes)
        f.write("\nEnd\n")


def write_mesh(path, mesh, *, refs=None, required_triangles=None,
               required_vertices=None):
    """Write a Medit ``.mesh`` surface, optionally with frozen entities.

    Args:
        path: destination ``.mesh`` file.
        mesh: a :class:`~bemkit.model.Mesh`, metres.
        refs: optional ``(M,)`` integer reference per triangle. mmgs preserves
            references and treats the boundary between two of them as a
            constrained edge, so tagging regions is a second, softer way to
            stop it merging across a line you care about. Defaults to all 1.
        required_triangles: element indices mmgs must leave alone.
        required_vertices: vertex indices mmgs must not move or remove.
            Marking a triangle required does **not** pin its corners, so a
            region that has to come back bit-identical needs both — pass the
            triangles and their vertices.

    Notes:
        Medit indices are 1-based; the conversion happens here so nothing above
        this line has to think about it.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=int)

    if refs is None:
        refs = np.ones(len(triangles), dtype=int)
    refs = np.asarray(refs, dtype=int)
    if len(refs) != len(triangles):
        raise ValueError(
            f"write_mesh: refs has {len(refs)} entries for "
            f"{len(triangles)} triangles")

    lines = ["MeshVersionFormatted 2", "", "Dimension 3", "",
             "Vertices", str(len(vertices))]
    #: trailing 0 is Medit's per-vertex reference
    lines += [f"{x:.17g} {y:.17g} {z:.17g} 0" for x, y, z in vertices]

    lines += ["", "Triangles", str(len(triangles))]
    lines += [f"{a + 1} {b + 1} {c + 1} {r}"
              for (a, b, c), r in zip(triangles, refs)]

    for keyword, idx in (("RequiredVertices", required_vertices),
                         ("RequiredTriangles", required_triangles)):
        if idx is None:
            continue
        idx = np.asarray(idx, dtype=int).ravel()
        if idx.size == 0:
            continue
        lines += ["", keyword, str(len(idx))]
        lines += [str(i + 1) for i in idx]

    lines += ["", "End", ""]
    Path(path).write_text("\n".join(lines))


def _section(tokens, keyword, width):
    """Locate ``keyword`` in a token stream and return its rows as ``(n, width)``.

    Medit sections are all ``<keyword> <count> <count*width values>``, so a
    section can be read without parsing anything before or after it. That is
    what lets this ignore the blocks we have no use for — ``Corners``,
    ``Ridges``, ``Normals``, and the ``Required*`` blocks mmgs echoes back.
    Keywords are whole tokens, so ``"Vertices"`` never matches inside
    ``"RequiredVertices"``.

    Returns None when the keyword is absent.
    """
    try:
        at = tokens.index(keyword)
    except ValueError:
        return None
    n = int(tokens[at + 1])
    start = at + 2
    return np.array(tokens[start:start + n * width]).reshape(n, width)


def read_mesh(path):
    """Read a Medit ``.mesh`` surface into a :class:`~bemkit.model.Mesh`.

    Triangle references become groups named ``"ref<N>"``, which is how a region
    tagged on the way in is recovered on the way out — mmgs renumbers vertices
    and triangles freely, so a reference is the only label that survives a
    remesh.

    **Not meshio.** meshio's Medit reader does not know ``RequiredTriangles``
    and, on meeting it, calls ``sys.exit(1)`` — not an exception, an interpreter
    exit, which would take down a build script with no traceback. Since
    ``mmgs`` echoes the ``Required*`` blocks it was given straight into its
    output, every frozen remesh would hit that. The format is four numbers per
    row, so we read it ourselves.
    """
    tokens = Path(path).read_text().split()

    verts = _section(tokens, "Vertices", 4)
    tris = _section(tokens, "Triangles", 4)
    if verts is None or tris is None:
        raise ValueError(f"read_mesh: no Vertices/Triangles section in {path}")

    vertices = verts[:, :3].astype(float)
    triangles = tris[:, :3].astype(int) - 1          # Medit is 1-based
    refs = tris[:, 3].astype(int)

    groups = {f"ref{int(r)}": np.nonzero(refs == r)[0] for r in np.unique(refs)}
    return Mesh(vertices, triangles, groups)


def remesh(mesh, sizes=None, *, hgrad=1.1, hmin=None, hmax=None, hausd=None,
           refs=None, required_triangles=None, required_vertices=None,
           binary=DEFAULT_BINARY, workdir=None, extra_args=()):
    """Adapt a surface to a target size field with ``mmgs``.

    Args:
        mesh: the :class:`~bemkit.model.Mesh` to remesh, metres.
        sizes: ``(N,)`` target size per vertex from
            :mod:`bemkit.geometry.metric`. ``None`` runs mmgs with no metric,
            which is only useful together with ``hmin``/``hmax``.
        hgrad: gradation — the most one element may grow relative to its
            neighbour. 1.1 is tight; mmgs defaults to 1.3.
        hmin, hmax, hausd: passed through as ``-hmin``/``-hmax``/``-hausd``.
            **A command-line ``hmin`` overrides the metric's own floor**, so
            passing one lets mmgs go finer than the size field asked for
            anywhere on the mesh — the trap in
            ``hutubs21_glassesearbuds/__pre_2_make_metric.sh``, which pairs a
            5 mm metric
            floor with ``-hmin 0.0005``.
        refs, required_triangles, required_vertices: see :func:`write_mesh`.
        binary: mmgs executable name or path.
        workdir: where to put the intermediate Medit files. A temporary
            directory is used and cleaned up when omitted; pass one to keep
            them for inspection when mmgs does something surprising.
        extra_args: additional mmgs flags, e.g. ``("-nr",)`` to disable angle
            detection or ``("-hgradreq", "2.0")`` to loosen how size grades
            away from frozen entities.

    Returns:
        A new :class:`~bemkit.model.Mesh`. Vertices and triangles are
        renumbered by mmgs — nothing about the input indexing survives, which
        is why regions have to be carried across as ``refs`` and drivers
        addressed geometrically (``model.select.Near``) rather than by index.

    Raises:
        FileNotFoundError: ``binary`` is not on PATH.
        RuntimeError: mmgs exited non-zero, with its output attached.
    """
    if shutil.which(binary) is None:
        raise FileNotFoundError(
            f"remesh: {binary!r} not found on PATH. Install mmg (the surface "
            f"remesher) or pass binary= with the full path.")
    if sizes is not None and len(sizes) != mesh.n_nodes:
        raise ValueError(
            f"remesh: sizes has {len(sizes)} entries for {mesh.n_nodes} "
            f"vertices")

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(workdir) if workdir is not None else Path(tmp)
        d.mkdir(parents=True, exist_ok=True)
        src, dst = d / "in.mesh", d / "out.mesh"

        write_mesh(src, mesh, refs=refs,
                   required_triangles=required_triangles,
                   required_vertices=required_vertices)

        cmd = [binary, str(src), str(dst), "-hgrad", str(hgrad)]
        if sizes is not None:
            write_sol(d / "in.sol", sizes)
            cmd += ["-met", str(d / "in.sol")]
        for flag, value in (("-hmin", hmin), ("-hmax", hmax),
                            ("-hausd", hausd)):
            if value is not None:
                cmd += [flag, str(value)]
        cmd += [str(a) for a in extra_args]

        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0 or not dst.exists():
            raise RuntimeError(
                f"remesh: {' '.join(cmd)} failed with code "
                f"{done.returncode}\n{done.stdout}\n{done.stderr}")

        return read_mesh(dst)
