"""Pack NumCalc ASCII results into a single compact ``be.npz`` per variant.

The NumCalc output ``be.out/be.{i}/{pBoundary,vBoundary,pEvalGrid,vEvalGrid}``
is plain text and dominates disk usage (>99% of a variant directory). Packing
into a single complex64 ``.npz`` typically yields a 4× shrink and a 6× faster
load, with relative error well below NumCalc's 6-digit text precision.

The packed file layout is::

    numcalc/{variant}/be.npz
        pBoundary    (M, nf)    complex64
        vBoundary    (M, nf)    complex64
        pEvalGrid    (N, nf)    complex64
        vEvalGrid    (N, nf, 3) complex64
        bin_indices  (k,)       int32  (1-based bins actually present)

Missing bins are NaN-filled, matching the behaviour of the ASCII loader.

``SimulationData`` automatically prefers ``be.npz`` over ``be.out/`` when
present, so existing code paths require no changes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from bemkit.numcalc.readers import (
    QUANTITIES,
    _read_numcalc_data,
    _read_numcalc_velocity_data,
    complete_bins,
    list_bins,
    load_bins_ascii,
)


def is_packed(nc_dir) -> bool:
    """True if ``nc_dir/be.npz`` exists."""
    return (Path(nc_dir) / "be.npz").exists()


def has_results(nc_dir) -> bool:
    """True if ``nc_dir`` has either ``be.npz`` or ``be.out/``."""
    nc_dir = Path(nc_dir)
    return (nc_dir / "be.npz").exists() or (nc_dir / "be.out").is_dir()


def pack_variant(nc_dir, *, nf: int | None = None, dtype=np.complex64,
                 delete_original: bool = False, verbose: bool = True) -> Path:
    """Pack ``nc_dir/be.out/`` into ``nc_dir/be.npz``.

    Parameters
    ----------
    nc_dir : Path or str
        Variant directory containing ``be.out/``.
    nf : int, optional
        Expected number of frequency bins (``config.n_frequencies``). Bin
        directories numbered outside ``1..nf`` are not real results and are
        skipped — NumCalc has been observed to leave a stray ``be.{i}/`` with
        an out-of-range index behind after an aborted run. Without this the
        array width would be taken from the stray index (a single ``be.999/``
        widens every array to 999 mostly-NaN columns). Defaults to the highest
        bin present, which is correct whenever no stray directory exists.
    dtype : numpy dtype
        Storage dtype. Default ``complex64`` (~5e-8 rel. error vs ASCII).
    delete_original : bool
        If True, remove ``be.out/`` after a verified round-trip across every
        bin and quantity.
    verbose : bool
        Print progress.

    Returns
    -------
    Path to the written ``be.npz``.
    """
    nc_dir = Path(nc_dir)
    be_out = nc_dir / "be.out"
    if not be_out.is_dir():
        raise FileNotFoundError(f"No be.out/ in {nc_dir}")

    bins = list_bins(be_out)
    if not bins:
        raise FileNotFoundError(f"No be.{{i}}/ subdirs in {be_out}")
    if nf is not None:
        stray = [b for b in bins if b < 1 or b > nf]
        if stray:
            print(f"  ignoring out-of-range bin dirs (nf={nf}): "
                  f"{', '.join(f'be.{b}' for b in stray)}")
            bins = [b for b in bins if 1 <= b <= nf]
            if not bins:
                raise FileNotFoundError(
                    f"No in-range be.{{i}}/ subdirs (1..{nf}) in {be_out}")
    else:
        nf = max(bins)

    if verbose:
        print(f"Packing {nc_dir} ({len(bins)} bins, nf={nf})")

    arrays: dict[str, np.ndarray] = {}
    for name in QUANTITIES:
        arr = load_bins_ascii(be_out, name, nf, bins=bins)
        if arr is None:
            if verbose:
                print(f"  {name}: no data — skipped")
            continue
        arrays[name] = arr.astype(dtype, copy=False)
        if verbose:
            print(f"  {name}: {arr.shape}  ->  {arrays[name].dtype}")

    if not arrays:
        raise FileNotFoundError(f"No quantity files found in {be_out}")

    # Claim only fully-written bins: a half-written bin (killed mid-write)
    # is packed as NaN columns but stays "missing" so a resume recomputes it.
    # complete_bins() judges that from file sizes, which a *truncated* file
    # can pass while the reader still rejects it -- so intersect with what
    # actually landed in the arrays. Claiming an all-NaN bin would be worse
    # than dropping a good one: a resume would consider it done for good.
    in_range = set(bins)
    claimed = [
        b for b in complete_bins(nc_dir)
        if b in in_range
        and not any(np.all(np.isnan(a[:, b - 1])) for a in arrays.values())
    ]
    arrays["bin_indices"] = np.asarray(claimed, dtype=np.int32)

    npz_path = nc_dir / "be.npz"
    tmp = nc_dir / "_be.partial.npz"  # np.savez requires .npz extension
    np.savez_compressed(tmp, **arrays)
    tmp.replace(npz_path)

    # A fresh pack supersedes any backup clear_pack moved aside on the last run.
    (nc_dir / "be.npz.bak").unlink(missing_ok=True)

    size_mb = npz_path.stat().st_size / 1e6
    if verbose:
        print(f"  wrote {npz_path.name}: {size_mb:.1f} MB")

    if delete_original:
        _verify_round_trip(be_out, npz_path, bins, dtype, verbose=verbose)
        shutil.rmtree(be_out)
        if verbose:
            print(f"  removed {be_out}")

    return npz_path


def _verify_round_trip(be_out: Path, npz_path: Path, bins: list[int], dtype,
                       *, verbose: bool = True) -> None:
    """Re-read every bin/quantity from ASCII and compare against the packed file.

    Independently parses each ``be.{i}/{quantity}`` file the same way
    ``SimulationData`` will, and checks it matches the corresponding slice in
    ``be.npz``. Bins/quantities with no ASCII source must be NaN in the packed
    array. Raises (keeping ``be.out/``) on any mismatch.

    A source file that is present but truncated (NumCalc killed mid-write)
    counts as "no ASCII source" rather than an error: the reader signals that
    by raising FileNotFoundError, and ``pack_variant`` keeps such a bin out of
    ``bin_indices``, so it is packed as NaN and stays missing for a later
    resume. Letting the reader's exception escape here would abort the whole
    pack and leave ``be.out/`` in place — a truncated bin says nothing about
    the fidelity of the bins that did pack.
    """
    n_checked = 0
    with np.load(npz_path) as nz:
        for name, vector in QUANTITIES.items():
            present = name in nz.files
            reader = _read_numcalc_velocity_data if vector else _read_numcalc_data
            for bi in bins:
                f = be_out / f"be.{bi}" / name
                has_ascii = f.exists() and f.stat().st_size > 10
                if has_ascii:
                    try:
                        ascii_data = reader(f).astype(dtype)
                    except FileNotFoundError as e:
                        print(f"  skipping incomplete {name} bin {bi}: {e}")
                        has_ascii = False
                packed = (nz[name][:, bi - 1, :] if vector else nz[name][:, bi - 1]) \
                    if present else None

                if not has_ascii:
                    # No ASCII source: packed slice (if any) must be NaN-filled.
                    if present and not np.all(np.isnan(packed)):
                        raise RuntimeError(
                            f"Round-trip check failed: {name} bin {bi} has no ASCII "
                            f"source but packed data is not NaN in {npz_path.parent}; "
                            f"keeping be.out/")
                    continue

                if not present:
                    raise RuntimeError(
                        f"Round-trip check failed: {name} present in be.out/ but missing "
                        f"from {npz_path.name}; keeping be.out/")

                if not np.allclose(ascii_data, packed, rtol=1e-5, atol=1e-30,
                                   equal_nan=True):
                    raise RuntimeError(
                        f"Round-trip verification failed at {name} bin {bi} for "
                        f"{npz_path.parent}; keeping be.out/")
                n_checked += 1
    if verbose:
        print(f"  verified {n_checked} bin/quantity slices against be.out/")


def clear_pack(nc_dir, *, restore: bool = False, verbose: bool = True) -> bool:
    """Remove ``nc_dir/be.npz`` so fresh ``be.out/`` results are not shadowed.

    ``be.npz`` takes precedence over ``be.out/`` everywhere (status, loaders),
    so a re-run writing into ``be.out/`` would otherwise be invisible behind
    stale packed data. With ``restore=True`` the packed bins are first
    unpacked back into ``be.out/`` so a resumed run merges with them.

    Returns True if a ``be.npz`` was removed.
    """
    nc_dir = Path(nc_dir)
    npz_path = nc_dir / "be.npz"
    if not npz_path.exists():
        return False
    if restore:
        unpack_variant(nc_dir, verbose=False)
    # Move aside rather than delete, so an interrupted/failed re-run does not
    # lose the previously packed results. A later successful `pack` overwrites
    # be.npz; the stale .bak is cleaned up then (see pack_variant).
    bak_path = nc_dir / "be.npz.bak"
    bak_path.unlink(missing_ok=True)
    npz_path.replace(bak_path)
    if verbose:
        action = "unpacked into be.out/ and moved to be.npz.bak" if restore \
            else "moved to be.npz.bak"
        print(f"  {nc_dir.name}: be.npz {action} (it would shadow new "
              f"results) — re-run 'bem pack' afterwards")
    return True


def unpack_variant(nc_dir, *, verbose: bool = True) -> Path:
    """Recreate ``nc_dir/be.out/`` ASCII tree from ``nc_dir/be.npz``.

    Useful as an escape hatch — packing is otherwise one-way for delete_original=True
    callers. The written files match NumCalc's original 6-digit scientific format
    closely enough to round-trip through ``SimulationData``.
    """
    nc_dir = Path(nc_dir)
    npz_path = nc_dir / "be.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No be.npz in {nc_dir}")

    be_out = nc_dir / "be.out"
    be_out.mkdir(exist_ok=True)

    with np.load(npz_path) as nz:
        bins = list(nz["bin_indices"])
        for bi in bins:
            d = be_out / f"be.{bi}"
            d.mkdir(exist_ok=True)
            for name, vector in QUANTITIES.items():
                if name not in nz.files:
                    continue
                col = nz[name][:, bi - 1] if not vector else nz[name][:, bi - 1, :]
                if np.all(np.isnan(col)):
                    continue
                _write_ascii_quantity(d / name, col, vector)
    if verbose:
        print(f"Unpacked {npz_path} -> {be_out} ({len(bins)} bins)")
    return be_out


def _write_ascii_quantity(path: Path, data: np.ndarray, vector: bool) -> None:
    """Write one bin of one quantity in NumCalc's text format."""
    n = data.shape[0]
    with open(path, "w") as f:
        f.write("Mesh2HRTF 1.0.0\n    1\n    1 %d\n" % n)
        if vector:
            for j in range(n):
                vx, vy, vz = data[j]
                f.write(f"{j:6d} {vx.real: .6E} {vx.imag: .6E} "
                        f"{vy.real: .6E} {vy.imag: .6E} "
                        f"{vz.real: .6E} {vz.imag: .6E}\n")
        else:
            for j in range(n):
                v = data[j]
                f.write(f"{j:6d} {v.real: .6E} {v.imag: .6E}\n")
