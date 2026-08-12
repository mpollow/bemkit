"""auralizer-bundle exporter (formatVersion 1).

Converts solved variants into the static "auralizer-bundle" format consumed
by the bemkit-auralizer web framework; the format contract lives in that
repo (docs/bundle-format.md), pinned there by a golden-manifest test against
real `bem export-web` output. All signal processing (normalization, EQ
derivation, iFFT to 48 kHz, quantization) happens here — the browser only
dequantizes and plays.

Pure numeric helpers first (unit-tested), then the per-variant writers and
the export_web orchestrator. All binary payloads are little-endian and
headerless; manifest.json carries exact byte lengths (the web loader rejects
any mismatch).

``export_web`` itself currently refuses: it was written against the
config.toml stack and has not been ported to the model.py spine (see its
docstring). Everything else in this module is intact and still tested — the
signal processing was never the config-coupled part.
"""

import warnings

import numpy as np

FORMAT = "auralizer-bundle"
FORMAT_VERSION = 1


# -- Pure numeric helpers ---------------------------------------------------

def synth_irs(spectra, df_hz, ir_fs):
    """Synthesize time-domain IRs from linear-grid half spectra.

    Parameters
    ----------
    spectra : (N, nf) complex ndarray
        Transfer functions on the linear frequency grid f_k = (k+1)*df_hz
        (DC excluded, as in BEMResult).
    df_hz : float
        Bin spacing in Hz.
    ir_fs : float
        Output sample rate. Must be an integer multiple of 2*df_hz so the
        IR length 1/df_hz lands on the sample grid (48000/250 -> 192 taps).

    Returns
    -------
    (N, n_taps) float ndarray with n_taps = ir_fs / df_hz.

    The half-spectrum is zero-padded above the simulated band before the
    irfft — exact band-limited resampling, silent above the simulation's
    Nyquist. DC and output Nyquist are 0, matching BEMResult.impulse_response.
    """
    spectra = np.asarray(spectra)
    n_half = ir_fs / 2 / df_hz
    if abs(n_half - round(n_half)) > 1e-9:
        raise ValueError(
            f"ir_fs {ir_fs} is not a multiple of 2*df ({2 * df_hz}); "
            "IR length would not land on the sample grid")
    n_half = int(round(n_half))
    nf = spectra.shape[1]
    if n_half < nf:
        raise ValueError(f"ir_fs {ir_fs} too low for {nf} bins of {df_hz} Hz")
    spec = np.zeros((spectra.shape[0], n_half + 1), dtype=complex)
    spec[:, 1:nf + 1] = spectra
    return np.fft.irfft(spec, axis=1)


def advance_spectra(spectra, frequencies, seconds):
    """Advance spectra in time: multiply by e^{+j 2 pi f t}.

    Used to remove the bulk propagation delay to the evaluation radius
    (r/c, ~2.9 ms for 1 m) which would otherwise wrap around the short
    1/df IR window. A safety margin keeps slightly-longer paths causal.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    return spectra * np.exp(2j * np.pi * frequencies * seconds)[None, :]


def relative_spectra(spectra, ref_index, floor_db=-40.0):
    """Divide all spectra by the reference direction's, regularized.

    H_rel = H * conj(H_ref) / max(|H_ref|^2, floor^2) with
    floor = 10^(floor_db/20) * max|H_ref| — deep reference notches are
    clamped instead of amplified. The reference direction itself comes out
    flat (|H_rel| = 1) wherever the floor is inactive.
    """
    href = spectra[ref_index]
    floor = 10.0 ** (floor_db / 20.0) * np.abs(href).max()
    denom = np.maximum(np.abs(href) ** 2, floor ** 2)
    return spectra * np.conj(href)[None, :] / denom[None, :]


def quantize_irs(irs):
    """Quantize float IRs to int16 with per-direction float32 scales.

    Returns (scales (N,) float32, taps (N, n_taps) int16) with
    float = int16 * scale and scale = peak/32767 (0 for an all-zero IR).
    """
    irs = np.asarray(irs)
    peaks = np.abs(irs).max(axis=1)
    scales = (peaks / 32767.0).astype(np.float32)
    safe = np.where(scales == 0, 1.0, scales.astype(np.float64))
    taps = np.round(irs / safe[:, None]).astype(np.int16)
    taps[scales == 0] = 0
    return scales, taps


def encode_db_u8(mag_db, min_db, max_db):
    """Encode dB values to u8: 0..254 spans [min_db, max_db], NaN -> 255."""
    mag_db = np.asarray(mag_db, dtype=np.float64)
    t = (mag_db - min_db) / (max_db - min_db)
    v = np.clip(np.round(t * 254), 0, 254)
    v = np.where(np.isnan(mag_db), 255, v)
    return v.astype(np.uint8)


def magnitude_db_rel(p):
    """20*log10|p| relative to the array-wide maximum (NaN-safe)."""
    mag = np.abs(p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # log of 0/NaN handled below
        db = 20.0 * np.log10(mag)
    ref = np.nanmax(db)
    return db - ref


def detect_plane(points, tol=1e-6):
    """Recover regular-grid parameters from flattened plane points.

    Parameters
    ----------
    points : (n, 3) float ndarray
        The ORIGINAL (pre-validation) fieldpoint set, row-major with the
        u axis varying fastest — as produced by make_fields_plane.

    Returns
    -------
    dict with keys w, h, origin, u_axis, v_axis, spacing_m. Raises if the
    grid is irregular or anisotropic (format v1 has a single spacingM).
    """
    points = np.asarray(points, dtype=np.float64)
    d0 = points[1] - points[0]
    spacing = np.linalg.norm(d0)
    if spacing == 0:
        raise ValueError("plane: first two points coincide")
    u_axis = d0 / spacing
    steps = np.linalg.norm(np.diff(points, axis=0) - d0, axis=1)
    wraps = np.nonzero(steps > spacing * 1e-3)[0]
    w = int(wraps[0]) + 1 if wraps.size else len(points)
    if len(points) % w != 0:
        raise ValueError(f"plane: {len(points)} points not divisible by row length {w}")
    h = len(points) // w
    if h > 1:
        dv = points[w] - points[0]
        v_spacing = np.linalg.norm(dv)
        if abs(v_spacing - spacing) > spacing * 1e-3:
            raise ValueError("plane: anisotropic spacing not supported in format v1")
        v_axis = dv / v_spacing
    else:
        v_axis = np.zeros(3)
    # Verify the reconstruction on all points.
    iu = np.arange(len(points)) % w
    iv = np.arange(len(points)) // w
    rebuilt = points[0] + np.outer(iu, u_axis * spacing) + np.outer(iv, v_axis * spacing)
    if np.abs(rebuilt - points).max() > max(spacing * 1e-3, tol):
        raise ValueError("plane: points do not form a regular grid")
    return {"w": w, "h": h, "origin": points[0], "u_axis": u_axis,
            "v_axis": v_axis, "spacing_m": spacing}


def pick_up_axis(front):
    """A global axis orthogonal-ish to front, re-orthogonalized exactly."""
    front = np.asarray(front, dtype=np.float64)
    front = front / np.linalg.norm(front)
    candidate = np.array([0.0, 0.0, 1.0])
    if abs(front @ candidate) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    up = candidate - (candidate @ front) * front
    return up / np.linalg.norm(up)


# -- Orchestrator -----------------------------------------------------------

def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)


def _export_variant(sim, out_dir, vid, dirs_unit, ref_index, directions_set,
                    planes, plane_geoms, ir_fs, db_range, floor_db,
                    advance_s, margin_s):
    """Write one variant's payload files; returns its manifest entry."""
    nf = len(sim.frequencies)
    df_hz = float(sim.frequencies[1] - sim.frequencies[0])
    vdir = out_dir / "variants" / vid

    # Directivity: acceleration-normalized spectra at the direction set.
    spectra = sim.pressure(directions_set, norm="acc")
    if np.isnan(spectra).any():
        raise ValueError(f"{vid}: NaN in {directions_set} spectra "
                         "(missing bins or dropped directions)")
    # Remove the bulk propagation delay so the IR fits the 1/df window.
    spectra = advance_spectra(spectra, sim.frequencies, advance_s)

    irs_physical = synth_irs(spectra, df_hz, ir_fs)
    cal = float(1.0 / np.sqrt(np.sum(irs_physical[ref_index] ** 2)))
    irs_physical *= cal
    # The division removes the (already compensated) bulk delay entirely, so
    # re-apply the causality margin: it keeps slightly-early off-axis
    # arrivals from wrapping AND time-aligns both EQ modes' main arrivals —
    # they play simultaneously during the EQ crossfade.
    rel = relative_spectra(spectra, ref_index, floor_db)
    rel = advance_spectra(rel, sim.frequencies, -margin_s)
    irs_relative = synth_irs(rel, df_hz, ir_fs)

    ir_entries = {}
    for name, irs in (("relative", irs_relative), ("physical", irs_physical)):
        scales, taps = quantize_irs(irs)
        payload = scales.astype("<f4").tobytes() + taps.astype("<i2").tobytes()
        n = _write(vdir / f"ir_{name}.bin", payload)
        ir_entries[name] = {"file": f"variants/{vid}/ir_{name}.bin", "byteLength": n}
    ir_entries["physical"]["calibrationGain"] = cal

    # Mesh (as solved, meters, aligned index-for-index with pBoundary).
    nodes = np.asarray(sim.boundary_nodes, dtype="<f4")
    elements = np.asarray(sim.boundary_elements, dtype="<u4")
    mesh_bytes = nodes.tobytes() + elements.tobytes()
    n = _write(vdir / "mesh.bin", mesh_bytes)
    mesh_entry = {"file": f"variants/{vid}/mesh.bin",
                  "nVerts": int(len(nodes)), "nTris": int(len(elements)),
                  "byteLength": n}

    # Surface pressure map, dB re variant max, per element x bin.
    surf_db = magnitude_db_rel(sim.boundary_pressure(norm="acc"))
    n = _write(vdir / "surface.bin",
               encode_db_u8(surf_db, -db_range, 0.0).tobytes())
    surface_entry = {"file": f"variants/{vid}/surface.bin",
                     "minDb": -db_range, "maxDb": 0.0, "byteLength": n}

    # Field planes: rows are the original grid order with NaN at dropped
    # points (split_by_offsets reconstruction) -> u8 with 255 = invalid.
    field_entries = []
    for plane in planes:
        p = sim.pressure(plane, norm="acc")
        geom = plane_geoms[plane]
        if len(p) != geom["w"] * geom["h"]:
            raise ValueError(f"{vid}: {plane} has {len(p)} rows, grid needs "
                             f"{geom['w'] * geom['h']}")
        db = magnitude_db_rel(p)
        n = _write(vdir / f"field_{plane}.bin",
                   encode_db_u8(db, -db_range, 0.0).tobytes())
        field_entries.append({"planeId": plane,
                              "file": f"variants/{vid}/field_{plane}.bin",
                              "minDb": -db_range, "maxDb": 0.0, "byteLength": n})

    title = vid.replace("_", " ").replace("-", " ").strip().title()
    return ({"id": vid, "title": title, "mesh": mesh_entry, "irs": ir_entries,
             "surface": surface_entry, "fields": field_entries},
            {"nf": nf, "irs_physical": irs_physical, "irs_relative": irs_relative})


def export_web(model_dir, out_dir, variants=None, directions="g30_1m",
               planes=("planeXZ",), front=None, ir_fs=48000, db_range=60.0,
               floor_db=-40.0, delay_margin_ms=0.5, check=False, verbose=True):
    """Export solved variants of a model as an auralizer web bundle.

    Parameters
    ----------
    model_dir : Path or str
    out_dir : Path or str
        Bundle directory (manifest.json + shared/ + variants/).
    variants : list of str or None
        Variant names; None exports every variant with results.
    directions : str
        Fieldpoint set used for auralization (must be a full sphere of
        equal-radius points, e.g. g30_1m).
    planes : sequence of str
        Fieldpoint plane sets to include as visualization heatmaps.
    front : (3,) array-like or None
        Model front axis. None auto-detects from the 'onaxis' fieldpoint
        set direction, falling back to +z.
    ir_fs : int
        IR sample rate (the web player forces its AudioContext to this).
    db_range : float
        u8 window below the variant maximum for surface/plane maps.
    floor_db : float
        Regularization floor for the relative EQ division.
    delay_margin_ms : float
        The bulk propagation delay radius/c is removed from the IRs (it
        would wrap the short 1/df window); this margin is left in so paths
        slightly longer than the radius stay causal.
    check : bool
        Print quantization/aliasing diagnostics per variant.

    Raises
    ------
    NotImplementedError
        Always, for now — see the guard below. The parameters above are the
        contract the port has to honour, so they are documented rather than
        removed.
    """
    raise NotImplementedError(
        "export_web is not yet ported to Python studies.\n"
        "  It was written against the config.toml stack and still needs, "
        "from a solved study, the declared field-point sets *before* "
        "validation dropped any: it exports the full sphere geometry, not "
        "the surviving rows, and bemkit.view.bundle deliberately hands back "
        "only what was solved.\n"
        "  The signal processing below (synth_irs / relative_spectra / "
        "quantize_irs / encode_db_u8 / detect_plane / pick_up_axis) is "
        "unchanged, still unit-tested, and is what the port reuses.")


def _self_validate(out_dir, manifest):
    """Re-check every declared byteLength against the file on disk."""
    refs = [manifest["directions"]]
    for v in manifest["variants"]:
        refs += [v["mesh"], v["irs"]["relative"], v["irs"]["physical"],
                 v["surface"], *v["fields"]]
    for ref in refs:
        actual = (out_dir / ref["file"]).stat().st_size
        if actual != ref["byteLength"]:
            raise RuntimeError(f"self-validation failed: {ref['file']} is "
                               f"{actual} bytes, manifest says {ref['byteLength']}")


def _print_check(vid, extras, ir_fs):
    """Quantization + time-aliasing diagnostics for one variant."""
    for name in ("relative", "physical"):
        irs = extras[f"irs_{name}"]
        scales, taps = quantize_irs(irs)
        deq = taps.astype(np.float64) * scales[:, None].astype(np.float64)
        num = np.sum((deq - irs) ** 2)
        den = np.sum(irs ** 2)
        snr = 10 * np.log10(den / num) if num > 0 else np.inf
        # Tail energy in the last quarter of the (circular) IR — a proxy for
        # time-alias wrap-around risk.
        n = irs.shape[1]
        tail = np.sum(irs[:, -n // 4:] ** 2, axis=1)
        total = np.sum(irs ** 2, axis=1)
        ratio = np.max(tail / np.maximum(total, 1e-30))
        peak = int(np.argmax(np.abs(irs[0])))
        print(f"  {vid}/{name}: quant SNR {snr:.0f} dB, worst tail-energy "
              f"ratio {10 * np.log10(max(ratio, 1e-30)):.0f} dB, "
              f"ref IR peak at tap {peak}/{n}")
