"""2D plot subcommand (matplotlib; requires bemkit[viz])."""

import json
from pathlib import Path

PLOT_TYPES = ["freq", "ir", "polar", "plane", "phase", "etc", "sphere", "contour"]


def register(sub):
    p = sub.add_parser("plot", help="2D result plots")
    p.add_argument("plot_type", nargs="?", choices=PLOT_TYPES, default=None,
                   help="Plot type (default: freq + ir)")
    p.add_argument("set_name", nargs="?", default=None,
                   help="Fieldpoint set (default: all sets)")
    from bemkit.cli import VariantAction
    p.add_argument("--variant", nargs="+", metavar="PREFIX", default=None,
                   action=VariantAction)
    p.add_argument("--freq", type=float, metavar="HZ",
                   help="Frequency in Hz (required for plane/sphere/contour)")
    p.add_argument("--freqs", metavar="F1,F2,...",
                   help="Comma-separated frequencies for polar (default: octave bands)")
    p.add_argument("--value-func", default="magnitude_db", dest="value_func",
                   choices=["magnitude_db", "magnitude", "phase", "real", "imag"])
    p.add_argument("--filled", action="store_true", help="Filled contours")
    p.set_defaults(func=cmd_plot)


def cmd_plot(model_dir, args):
    from bemkit.cli import require
    require("matplotlib", "viz")
    _plot_sets(model_dir, args.plot_type, args.set_name, args, prefixes=args.variant)


def _plot_sets(model_dir, plot_type, set_name, args, prefixes=None):
    """Per-set plot loop (transplanted from process.py PlotStage.run)."""
    from bemkit.sim.discovery import discover_variants
    model_dir = Path(model_dir)
    variants = discover_variants(model_dir, prefixes, need_be_out=True)
    if not variants:
        raise SystemExit(f"  No variants with results found in {model_dir}")

    # Collect unique set_names (preserving order) across all variants
    seen = set()
    ordered_sets = []
    for v in variants:
        offsets_file = model_dir / "numcalc" / v / "eval_offsets.json"
        if offsets_file.exists():
            with open(offsets_file) as f:
                offsets_sets = json.load(f).get("sets", {}).keys()
            for sn in offsets_sets:
                if sn not in seen:
                    seen.add(sn)
                    ordered_sets.append(sn)
    if not ordered_sets:
        ordered_sets = [None]

    for sn in ordered_sets:
        effective_sn = set_name if set_name is not None else sn
        print(f"\n--- Plot: {effective_sn or 'auto'} ---")
        try:
            if plot_type is None or plot_type == 'freq':
                _run_plot('freq', model_dir, effective_sn, args)
            if plot_type is None or plot_type == 'ir':
                _run_plot('ir', model_dir, effective_sn, args)
            if plot_type and plot_type not in ('freq', 'ir'):
                _run_plot(plot_type, model_dir, effective_sn, args)
        except SystemExit as e:
            print(f"  Skipped: {e}")


# ---------------------------------------------------------------------------
# Transplanted from process.py: _CBAR_LABELS, _save_fig, _run_plot
# ---------------------------------------------------------------------------

_CBAR_LABELS = {
    "magnitude_db": "Magnitude [dB re 1 Pa]",
    "magnitude": "Magnitude [Pa]",
    "phase": "Phase [degrees]",
    "real": "Real Part [Pa]",
    "imag": "Imaginary Part [Pa]",
}


def _save_fig(fig, output_file):
    import matplotlib.pyplot as plt
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {output_file}")
    plt.show()


def _run_plot(plot_type, model_dir, set_name, args):
    """Unified plot dispatcher."""
    import matplotlib.pyplot as plt
    import numpy as np

    from bemkit.analysis.result import load_result
    from bemkit.viz import (
        freq_magnitude, impulse_response, polar_directivity,
        heatmap, freq_phase, energy_time_curve, spherical_heatmap, contour,
    )
    from bemkit.viz._style import bem_style

    model_dir = Path(model_dir)
    freq = args.freq

    # Validate --freq requirement
    if plot_type in ('plane', 'sphere', 'contour') and freq is None:
        raise SystemExit(f"--freq is required for 'plot {plot_type}'")

    try:
        result, variant = load_result(model_dir, set_name)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    print(f"Loaded: {result}")

    # Freq range validation for freq-specific plots
    if freq is not None and plot_type in ('plane', 'sphere', 'contour'):
        if freq < result.frequencies[0] or freq > result.frequencies[-1]:
            raise SystemExit(
                f"Frequency {freq} Hz out of range "
                f"[{result.frequencies[0]:.0f}, {result.frequencies[-1]:.0f}] Hz")

    # Grid validation for spatial plots
    if plot_type in ('plane', 'contour'):
        if not result.is_grid:
            raise SystemExit(f"Data is not grid-structured. "
                             f"{'Heatmap' if plot_type == 'plane' else 'Contour plot'} "
                             f"requires planar grid fieldpoints.")
    if plot_type == 'sphere':
        if not result.is_grid:
            raise SystemExit("Data is not grid-structured. "
                             "Spherical heatmap requires spherical grid fieldpoints.")
        if result._grid_info["type"] != "spherical":
            raise SystemExit(
                f"Grid type is '{result._grid_info['type']}', not 'spherical'.")

    stem = f"{variant}_{set_name or result.name}"
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    with bem_style():
        if plot_type == 'freq':
            print(f"Frequencies: {result.n_freq} steps from {result.frequencies[0]:.0f} "
                  f"to {result.frequencies[-1]:.0f} Hz")
            fig, ax = plt.subplots(figsize=(12, 6))
            freq_magnitude(ax, result, label_fmt="x={x:.2f} m")
            ax.set_title(f"{result.name}: Field Pressure (volume_acc=1.0 m³/s²)",
                         fontsize=13, fontweight="bold")
            _save_fig(fig, plots_dir / f"{stem}_field_pressure.png")

        elif plot_type == 'ir':
            fig, ax = plt.subplots(figsize=(12, 6))
            impulse_response(ax, result, time_range=(0, 5), label_fmt="z={z:.2f} m")
            ax.set_title(f"{result.name}: Impulse Responses",
                         fontsize=13, fontweight="bold")
            _save_fig(fig, plots_dir / f"{stem}_IR.png")

        elif plot_type == 'polar':
            freqs_list = ([float(f) for f in args.freqs.split(',')]
                          if args.freqs else [500, 1000, 2000, 4000, 8000])
            valid_freqs = [f for f in freqs_list
                           if result.frequencies[0] <= f <= result.frequencies[-1]]
            if not valid_freqs:
                raise SystemExit(
                    f"None of the requested frequencies {freqs_list} are in range "
                    f"[{result.frequencies[0]:.0f}, {result.frequencies[-1]:.0f}] Hz")
            print(f"Plotting polar directivity at {len(valid_freqs)} frequencies: {valid_freqs} Hz")
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
            ax.set_theta_offset(np.pi / 2)
            polar_directivity(ax, result, freq=valid_freqs, angle_key="auto")
            ax.set_title(f"{result.name}: Polar Directivity",
                         fontsize=13, fontweight="bold", pad=20)
            _save_fig(fig, plots_dir / f"{stem}_polar_directivity.png")

        elif plot_type == 'phase':
            print(f"Frequencies: {result.n_freq} steps from {result.frequencies[0]:.0f} "
                  f"to {result.frequencies[-1]:.0f} Hz")
            fig, ax = plt.subplots(figsize=(12, 6))
            freq_phase(ax, result, unwrap=True, degrees=True, label_fmt="z={z:.2f} m")
            ax.set_title(f"{result.name}: Phase Response",
                         fontsize=13, fontweight="bold")
            _save_fig(fig, plots_dir / f"{stem}_phase.png")

        elif plot_type == 'etc':
            fig, ax = plt.subplots(figsize=(12, 6))
            energy_time_curve(ax, result, db=True, label_fmt="z={z:.2f} m")
            ax.set_title(f"{result.name}: Energy Time Curve (Schroeder Integral)",
                         fontsize=13, fontweight="bold")
            _save_fig(fig, plots_dir / f"{stem}_ETC.png")

        elif plot_type == 'plane':
            value_func = args.value_func
            print(f"Plotting {value_func} heatmap at {freq} Hz")
            fig, ax = plt.subplots(figsize=(10, 8))
            mesh_obj = heatmap(ax, result, freq=freq, value_func=value_func)
            cbar = fig.colorbar(mesh_obj, ax=ax)
            cbar.set_label(_CBAR_LABELS.get(value_func, value_func), fontsize=11)
            ax.set_title(f"{result.name}: {value_func.replace('_', ' ').title()} at {freq:.0f} Hz",
                         fontsize=13, fontweight="bold")
            ax.set_aspect("equal")
            _save_fig(fig, plots_dir / f"{stem}_plane_{value_func}_{freq:.0f}Hz.png")

        elif plot_type == 'sphere':
            print(f"Plotting spherical heatmap at {freq} Hz")
            fig, ax = plt.subplots(figsize=(10, 6))
            mesh_obj = spherical_heatmap(ax, result, freq=freq)
            cbar = fig.colorbar(mesh_obj, ax=ax)
            cbar.set_label("Magnitude [dB re 1 Pa]", fontsize=11)
            ax.set_title(f"{result.name}: Spherical Directivity at {freq:.0f} Hz",
                         fontsize=13, fontweight="bold")
            _save_fig(fig, plots_dir / f"{stem}_sphere_{freq:.0f}Hz.png")

        elif plot_type == 'contour':
            filled = args.filled
            print(f"Plotting {'filled ' if filled else ''}contours at {freq} Hz")
            fig, ax = plt.subplots(figsize=(10, 8))
            contour_obj = contour(ax, result, freq=freq, filled=filled)
            cbar = fig.colorbar(contour_obj, ax=ax)
            cbar.set_label("Magnitude [dB re 1 Pa]", fontsize=11)
            ax.set_title(
                f"{result.name}: {'Filled ' if filled else ''}Contours at {freq:.0f} Hz",
                fontsize=13, fontweight="bold")
            ax.set_aspect("equal")
            suffix = "_filled" if filled else ""
            _save_fig(fig, plots_dir / f"{stem}_contour{suffix}_{freq:.0f}Hz.png")
