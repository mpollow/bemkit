"""Interactive 3D results viewer extracted from viz_results.py."""

import numpy as np

from bemkit.sim.discovery import numcalc_dir
from bemkit.analysis.simulation import SimulationData
from bemkit.viz._style import cmap_for_quantity


PHASE_DB_RANGE = 80.0  # dynamic range: max → opaque, max-80dB → transparent

_CLIM_QUANTITIES = {"magnitude_db", "real", "imag"}
_SYMMETRIC_QUANTITIES = {"real", "imag"}


def _phase_rgba(pressure, db_ref=None):
    """Compute (N, 4) uint8 RGBA from complex pressure for phase display.

    Hue = phase angle (hsv colormap), fully opaque.
    Low-amplitude regions fade to white: ref → phase color, ref-PHASE_DB_RANGE → white.
    If db_ref is None, uses the max of the data as the reference.
    """
    import matplotlib.pyplot as plt
    mag_db = 20.0 * np.log10(np.maximum(np.abs(pressure), 1e-12))
    if db_ref is None:
        db_ref = mag_db.max()
    db_floor = db_ref - PHASE_DB_RANGE
    weight = np.clip((mag_db - db_floor) / PHASE_DB_RANGE, 0.0, 1.0)
    phase_norm = (np.degrees(np.angle(pressure)) + 180.0) / 360.0
    phase_rgb = plt.cm.hsv(phase_norm)[:, :3]
    rgb = weight[:, np.newaxis] * phase_rgb + (1.0 - weight[:, np.newaxis])
    alpha = np.ones(len(pressure))
    return (np.column_stack([rgb, alpha]) * 255).astype(np.uint8)


def _compute_impedance_scalars(points, pressure, velocity, rho0, c0, quantity):
    """Compute impedance scalars from pressure and velocity data.

    Returns (N,) float array of the requested quantity.
    """
    # Radial unit vector from origin to each point
    r_norm = np.linalg.norm(points, axis=1, keepdims=True)
    r_hat = points / np.maximum(r_norm, 1e-30)

    # Radial velocity component (complex scalar)
    v_r = np.sum(velocity * r_hat, axis=1)

    # Specific impedance, normalized: Z₀ = Z / (ρ₀c₀)
    Z = pressure / np.where(np.abs(v_r) > 1e-30, v_r, np.nan + 0j)
    Z0 = Z / (rho0 * c0)

    if quantity == "real":
        return Z0.real
    elif quantity == "imag":
        return Z0.imag
    elif quantity == "magnitude":
        return np.abs(Z0)
    elif quantity == "magnitude_db":
        return 20.0 * np.log10(np.maximum(np.abs(Z), 1e-30))
    elif quantity == "phase":
        return np.degrees(np.angle(Z))
    else:
        raise ValueError(f"Unknown quantity: {quantity!r}")


_LABELS = {
    "magnitude_db": "dB (acc)", "magnitude": "|p| (acc)",
    "phase": "Phase (°)", "real": "Re(p)", "imag": "Im(p)",
}


class ResultsViewer:
    """Interactive 3D viewer for BEM simulation results.

    Manages scene actors, frequency/variant sliders, and clim key bindings.
    """

    def __init__(self, scene, model_dir, config, freqs, args):
        self.scene = scene
        self.model_dir = model_dir
        self.config = config
        self.freqs = freqs
        self.args = args

        self.modelname = config["modelname"]
        self.nbins = config["nbins"]
        self.sr = config["sr"]
        self.rho0 = config.get("air_density", 1.2)
        self.c0 = config.get("speed_of_sound", 340.0)

        # Resolve per-quantity settings
        self.surface_quantity = args.surface_quantity or args.quantity
        self.points_quantity = args.points_quantity or args.quantity

        # What to show
        self.show_surface = not args.no_surface
        self.show_points = not args.no_points and args.set_name is not None

        # RGBA phase mode
        phase_fade = not args.no_phase_fade
        self.surface_rgba = (self.surface_quantity == "phase") and phase_fade
        self.points_rgba = (self.points_quantity == "phase") and phase_fade and not args.plane

        # Mutable state
        self.variant = None
        self.bin_idx = 0
        self.nc_dir = None
        self.sim = None
        self.surface_actor = None
        self.mesh_actor = None
        self.points_actor = None
        self.intensity_actor = None
        self.impedance_actor = None
        self.result = None
        self.A_vib = None
        self.share_colorbar = False

        # Cached mesh data (loaded once per variant)
        self._nodes = None
        self._elements = None

        # Clim state
        self.db_actors = []
        self.db_symmetric = []
        self.initial_clims = []
        self.clim_state = [0.0, 0.0]

    def _get_eval_field(self, bin_idx, set_name):
        """Return acc-normalized (pts, p, v) at a freq bin, NaN rows removed.

        Returns (None, None, None) if data is unavailable or not yet computed
        for this bin (partial simulation).
        """
        be_index = bin_idx + 1  # bin_idx is 0-based; pressure_at uses 1-based
        try:
            pts = self.sim.nodes(set_name)
            p = self.sim.pressure_at(be_index, domain="eval",
                                     set_name=set_name, norm="acc")
            v = self.sim.velocity_at(be_index, domain="eval",
                                     set_name=set_name, norm="acc")
        except (FileNotFoundError, KeyError):
            return None, None, None
        valid = ~np.any(np.isnan(pts), axis=1)
        return pts[valid], p[valid], v[valid]

    def _surface_scalars(self, pressure, quantity, norm, freq, A_vib):
        if norm == "acc" and A_vib is not None:
            pressure = pressure / (1j * self.rho0 * 2.0 * np.pi * freq * A_vib)
        if quantity == "magnitude_db":
            return 20.0 * np.log10(np.maximum(np.abs(pressure), 1e-12))
        elif quantity == "magnitude":
            return np.abs(pressure)
        elif quantity == "phase":
            return np.degrees(np.angle(pressure))
        elif quantity == "real":
            return pressure.real
        else:  # imag
            return pressure.imag

    def _read_pBoundary(self, be_index):
        """Read pBoundary for a frequency bin, or return None."""
        try:
            return self.sim.pressure_at(be_index, domain="boundary", norm="raw")
        except FileNotFoundError:
            return None

    def _compute_surface_data(self, be_index, freq, A_vib):
        """Compute surface scalars or RGBA for the given frequency bin.

        Returns (scalars, rgba_flag) or (None, False) if data missing.
        """
        p_surf = self._read_pBoundary(be_index)
        if p_surf is None:
            return None, False

        if self.surface_rgba:
            if self.args.norm == "acc" and A_vib is not None:
                p_surf = p_surf / (1j * self.rho0 * 2.0 * np.pi * freq * A_vib)
            return _phase_rgba(p_surf), True
        else:
            scalars = self._surface_scalars(
                p_surf, self.surface_quantity, self.args.norm, freq, A_vib)
            return scalars, False

    def build_variant(self, variant, bin_idx):
        """Build actors for a variant at a given freq bin."""
        args = self.args
        scene = self.scene
        freqs = self.freqs

        from bemkit.sim.packing import has_results

        nc_dir = numcalc_dir(self.model_dir, variant)
        has_data = has_results(nc_dir)
        be_index = bin_idx + 1

        # Update instance state early (needed by helper methods)
        self.variant = variant
        self.bin_idx = bin_idx
        self.nc_dir = nc_dir
        self.sim = SimulationData(nc_dir, self.config)

        v_show_surface = self.show_surface and has_data
        v_show_points = self.show_points and args.set_name is not None and has_data

        result = None
        if v_show_points:
            result = self.sim.result(args.set_name)

        # Cache mesh data from SimulationData
        if v_show_surface or has_data:
            self._nodes = self.sim.boundary_nodes
            self._elements = self.sim.boundary_elements

        A_vib = None
        if v_show_surface:
            A_vib = self.sim.A_vib
            if A_vib is not None:
                print(f"Vibrating area: {A_vib:.6g} m²")

        # Share colorbar when both visible with same scalar quantity
        share_colorbar = (v_show_surface and v_show_points
                          and self.surface_quantity == self.points_quantity
                          and not self.surface_rgba)

        # Joint dB reference for phase fade
        actual_freq = freqs[bin_idx]
        phase_alpha_surf = PHASE_DB_RANGE if self.surface_rgba else None
        phase_alpha_pts = PHASE_DB_RANGE if self.points_rgba else None
        if self.surface_rgba and v_show_surface and self.points_rgba and v_show_points:
            p_surf = self._read_pBoundary(be_index)
            if p_surf is not None:
                if args.norm == "acc" and A_vib is not None:
                    p_surf = p_surf / (1j * self.rho0 * 2.0 * np.pi * actual_freq * A_vib)
                db_surf = 20.0 * np.log10(np.maximum(np.abs(p_surf), 1e-12)).max()
                fi = result.at_freq(actual_freq)
                p_p = result.data("pressure", args.norm)[:, fi][result.valid_mask]
                db_pts = 20.0 * np.log10(np.maximum(np.abs(p_p), 1e-12)).max()
                db_ref = max(db_surf, db_pts)
                phase_alpha_surf = (db_ref, PHASE_DB_RANGE)
                phase_alpha_pts = (db_ref, PHASE_DB_RANGE)

        # Add surface
        surface_actor = None
        mesh_actor = None
        if v_show_surface:
            scalars, is_rgba = self._compute_surface_data(be_index, actual_freq, A_vib)
            if scalars is None:
                # No data yet — fill with placeholder
                n = len(self._elements)
                if self.surface_rgba:
                    scalars = np.full((n, 4), 128, dtype=np.uint8)
                    scalars[:, 3] = 255
                    is_rgba = True
                else:
                    scalars = np.full(n, np.nan)
                    is_rgba = False

            surface_actor = scene.add_surface_pressure(
                self._nodes, self._elements, scalars,
                actual_freq=actual_freq,
                quantity=self.surface_quantity,
                cmap=cmap_for_quantity(self.surface_quantity),
                opacity=args.mesh_opacity,
                show_scalar_bar=not share_colorbar,
                rgba=is_rgba,
            )
        elif has_data:
            # --no-surface: show geometry without pressure coloring
            mesh_actor = scene.add_object_mesh(
                self._nodes, self._elements, opacity=args.mesh_opacity)

        # Add eval points
        points_actor = None
        if v_show_points:
            _pts_kwargs = dict(
                freq=freqs[bin_idx],
                value_func=self.points_quantity,
                norm=args.norm,
                cmap=cmap_for_quantity(self.points_quantity),
                show_scalar_bar=not share_colorbar,
                scalar_bar_args={
                    "title": f"{_LABELS.get(self.points_quantity, self.points_quantity)}\n@ {freqs[bin_idx]:.1f} Hz",
                    "position_x": 0.85, "position_y": 0.50,
                    "vertical": True, "height": 0.20,
                },
            )
            if args.plane:
                points_actor = scene.add_field_surface(result, **_pts_kwargs)
            else:
                points_actor = scene.add_field_points(
                    result, point_size=args.point_size,
                    alpha_from_db=phase_alpha_pts, **_pts_kwargs
                )

        # Unify clim + shared colorbar
        if share_colorbar:
            vmin = min(surface_actor.mapper.scalar_range[0], points_actor.mapper.scalar_range[0])
            vmax = max(surface_actor.mapper.scalar_range[1], points_actor.mapper.scalar_range[1])
            surface_actor.mapper.scalar_range = (vmin, vmax)
            points_actor.mapper.scalar_range = (vmin, vmax)
            scene._plotter.add_scalar_bar(
                title=f"{_LABELS.get(self.surface_quantity, self.surface_quantity)}\n@ {actual_freq:.1f} Hz",
                mapper=surface_actor.mapper,
                position_x=0.85, position_y=0.72, vertical=True, height=0.20,
            )

        # Add intensity arrows
        intensity_actor = None
        if args.intensity:
            pts, p, v = self._get_eval_field(bin_idx, args.set_name)
            if pts is not None:
                intensity_actor = scene.add_intensity_arrows(
                    pts, p, v,
                    actual_freq=actual_freq,
                    scale=args.intensity_scale,
                    length_mode=args.intensity_length,
                    length_clim=args.intensity_clim,
                )
            if intensity_actor is not None:
                print(f"  + Intensity arrows @ {freqs[bin_idx]:.1f} Hz "
                      f"(scale={args.intensity_scale}, length={args.intensity_length})")
            else:
                print(f"  (intensity: vEvalGrid not found, skipping)")

        # Add impedance points
        impedance_actor = None
        if args.impedance and args.set_name is not None:
            pts, p, v = self._get_eval_field(bin_idx, args.set_name)
            if pts is not None:
                imp_scalars = _compute_impedance_scalars(
                    pts, p, v, self.rho0, self.c0, args.impedance_quantity)
                impedance_actor = scene.add_impedance_points(
                    pts, imp_scalars,
                    actual_freq=actual_freq,
                    quantity=args.impedance_quantity,
                    point_size=args.point_size,
                    as_surface=args.plane,
                )
            if impedance_actor is not None:
                print(f"  + Impedance Z₀ ({args.impedance_quantity}) @ {freqs[bin_idx]:.1f} Hz")
            else:
                print(f"  (impedance: pEvalGrid/vEvalGrid not found, skipping)")

        # Update remaining instance state
        self.surface_actor = surface_actor
        self.mesh_actor = mesh_actor
        self.points_actor = points_actor
        self.intensity_actor = intensity_actor
        self.impedance_actor = impedance_actor
        self.result = result
        self.A_vib = A_vib
        self.share_colorbar = share_colorbar

    def _update_db_actors(self, first=False):
        """Collect scalar actors eligible for clim key events."""
        self.db_actors = []
        self.db_symmetric = []
        if self.surface_actor is not None and self.surface_quantity in _CLIM_QUANTITIES:
            self.db_actors.append(self.surface_actor)
            self.db_symmetric.append(self.surface_quantity in _SYMMETRIC_QUANTITIES)
        if self.points_actor is not None and self.points_quantity in _CLIM_QUANTITIES:
            self.db_actors.append(self.points_actor)
            self.db_symmetric.append(self.points_quantity in _SYMMETRIC_QUANTITIES)
        if first:
            self.initial_clims = [tuple(a.mapper.scalar_range) for a in self.db_actors]
            self.clim_state = [0.0, 0.0]

    def _apply_clim(self):
        shift, half_exp = self.clim_state
        for actor, (v0, v1), sym in zip(
                self.db_actors, self.initial_clims, self.db_symmetric):
            if sym:
                half = max(abs(v0), abs(v1)) + half_exp
                actor.mapper.scalar_range = (-half, half)
            else:
                actor.mapper.scalar_range = (v0 + shift - half_exp,
                                             v1 + shift + half_exp)
        self.scene._plotter.render()
        if self.initial_clims:
            v0, v1 = self.initial_clims[0]
            sym0 = self.db_symmetric[0]
            if sym0:
                half = max(abs(v0), abs(v1)) + half_exp
                print(f"  clim: [{-half:.3g}, {half:.3g}]")
            else:
                print(f"  clim: [{v0 + shift - half_exp:.3g}, "
                      f"{v1 + shift + half_exp:.3g}]")

    def update_freq(self, freq_hz):
        """Frequency slider callback."""
        new_bin = int(np.argmin(np.abs(self.freqs - freq_hz)))
        if new_bin == self.bin_idx:
            return
        self.bin_idx = new_bin
        freq = self.freqs[new_bin]
        be_index = new_bin + 1
        args = self.args

        # Read surface pressure once
        p_surf = None
        surf_missing = False
        if self.surface_actor is not None:
            p_surf = self._read_pBoundary(be_index)
            if p_surf is not None:
                if self.surface_rgba and args.norm == "acc" and self.A_vib is not None:
                    p_surf = p_surf / (1j * self.rho0 * 2.0 * np.pi * freq * self.A_vib)
            else:
                surf_missing = True
                print(f"  bin {be_index} ({freq:.1f} Hz): pBoundary not yet computed")

        # Joint dB reference for phase
        _db_ref = None
        if self.surface_rgba and p_surf is not None and self.points_actor is not None and self.points_rgba:
            fi = self.result.at_freq(freq)
            p_p = self.result.data("pressure", args.norm)[:, fi][self.result.valid_mask]
            _db_ref = max(
                20.0 * np.log10(np.maximum(np.abs(p_surf), 1e-12)).max(),
                20.0 * np.log10(np.maximum(np.abs(p_p), 1e-12)).max(),
            )

        if self.surface_actor is not None:
            cell_data = self.surface_actor.mapper.dataset.cell_data
            n_cells = len(cell_data["surface_pressure"])
            if surf_missing:
                if self.surface_rgba:
                    cell_data["surface_pressure"] = np.full((n_cells, 4), 128, dtype=np.uint8)
                else:
                    cell_data["surface_pressure"] = np.full(n_cells, np.nan)
            elif self.surface_rgba:
                cell_data["surface_pressure"] = _phase_rgba(p_surf, db_ref=_db_ref)
            else:
                cell_data["surface_pressure"] = (
                    self._surface_scalars(p_surf, self.surface_quantity, args.norm, freq, self.A_vib)
                )
            self.surface_actor.mapper.dataset.Modified()

        if self.points_actor is not None:
            result = self.result
            if self.points_rgba:
                fi = result.at_freq(freq)
                p = result.data("pressure", args.norm)[:, fi]
                self.points_actor.mapper.dataset.point_data["values"] = (
                    _phase_rgba(p[result.valid_mask], db_ref=_db_ref)
                )
            else:
                from bemkit.viz.field import _get_values
                vals = _get_values(result, freq, self.points_quantity, "pressure", args.norm)
                self.points_actor.mapper.dataset.point_data["values"] = vals[result.valid_mask]
            self.points_actor.mapper.dataset.Modified()

        if args.intensity:
            if self.intensity_actor is not None:
                self.scene._plotter.remove_actor(self.intensity_actor)
                for t in list(self.scene._plotter.scalar_bars.keys()):
                    if "Intensity" in t:
                        self.scene._plotter.remove_scalar_bar(t, render=False)
            pts, p, v = self._get_eval_field(new_bin, args.set_name)
            if pts is not None:
                self.intensity_actor = self.scene.add_intensity_arrows(
                    pts, p, v,
                    actual_freq=freq,
                    scale=args.intensity_scale,
                    length_mode=args.intensity_length,
                    length_clim=args.intensity_clim,
                )

        if args.impedance:
            if self.impedance_actor is not None:
                self.scene._plotter.remove_actor(self.impedance_actor)
                for t in list(self.scene._plotter.scalar_bars.keys()):
                    if "Impedance" in t:
                        self.scene._plotter.remove_scalar_bar(t, render=False)
            pts, p, v = self._get_eval_field(new_bin, args.set_name)
            if pts is not None:
                imp_scalars = _compute_impedance_scalars(
                    pts, p, v, self.rho0, self.c0, args.impedance_quantity)
                self.impedance_actor = self.scene.add_impedance_points(
                    pts, imp_scalars,
                    actual_freq=freq,
                    quantity=args.impedance_quantity,
                    point_size=args.point_size,
                    as_surface=args.plane,
                )

        self._apply_clim()

        # Update scalar bar titles with new frequency
        freq_label = f"@ {freq:.1f} Hz"
        for title, actor in list(self.scene._plotter.scalar_bars.items()):
            if "@ " in title and title.rstrip().endswith("Hz"):
                actor.SetTitle(title.rsplit("@ ", 1)[0] + freq_label)
        self.scene._plotter.render()

    def update_variant(self, v):
        """Variant slider callback."""
        if v == self.variant:
            return
        # Remove old actors
        for actor in (self.surface_actor, self.mesh_actor, self.points_actor,
                      self.intensity_actor, self.impedance_actor):
            if actor is not None:
                self.scene._plotter.remove_actor(actor)
        # Remove stale scalar bars
        for title in list(self.scene._plotter.scalar_bars.keys()):
            self.scene._plotter.remove_scalar_bar(title, render=False)
        # Build new actors at current freq bin
        self.build_variant(v, self.bin_idx)
        self._update_db_actors()
        self._apply_clim()
        print(f"  → variant: {v}")

    def setup_interactivity(self, variants):
        """Set up freq slider, variant slider, clim keys, screenshot hotkey."""
        args = self.args
        scene = self.scene

        # Initialize clim tracking
        self._update_db_actors(first=True)

        # Clim key events
        if not args.screenshot:
            step = 10.0

            def shift_clim(delta):
                if not all(self.db_symmetric):
                    self.clim_state[0] += delta
                    self._apply_clim()

            def expand_range(delta):
                self.clim_state[1] += delta / 2
                self._apply_clim()

            # Register both KP_* (native X11 backend) and symbol names (Qt backend)
            for key in ("KP_Add", "plus"):
                scene._plotter.add_key_event(key, lambda: shift_clim(-step))
            for key in ("KP_Subtract", "minus"):
                scene._plotter.add_key_event(key, lambda: shift_clim(+step))
            for key in ("KP_Multiply", "asterisk"):
                scene._plotter.add_key_event(key, lambda: expand_range(+step))
            for key in ("KP_Divide", "slash"):
                scene._plotter.add_key_event(key, lambda: expand_range(-step))
            print("  [*/÷ or KP_*/÷] expand/contract clim range"
                  + ("  [+/- or KP_+/-] shift window" if not all(self.db_symmetric) else ""))

        # Frequency slider
        if not args.screenshot and (self.show_surface or self.show_points):
            actual_freq = self.freqs[self.bin_idx]
            scene._plotter.add_slider_widget(
                self.update_freq,
                rng=[self.freqs[0], self.freqs[-1]],
                value=actual_freq,
                title="Frequency (Hz)",
                pointa=(0.1, 0.05),
                pointb=(0.9, 0.05),
                style="modern",
                fmt="%.0f",
                interaction_event="always",
            )
            print("  [slider] drag to change frequency")

        # Variant slider
        if not args.screenshot and len(variants) > 1 and variants[0] is not None:
            scene.add_variant_slider(variants, self.update_variant)

        # Camera
        scene.set_camera(args.camera)

        # Screenshot hotkey
        if not args.screenshot:
            scene.add_screenshot_hotkey()
