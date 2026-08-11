"""Interactive 3D results viewer extracted from viz_results.py."""

import numpy as np

from bemkit.sim.discovery import numcalc_dir
from bemkit.analysis.simulation import SimulationData
from bemviz.viz._style import cmap_for_quantity
from bemviz.viz._symmetry import EDGE_STYLE


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


def _quantity_values(pressure, quantity):
    """Elementwise real-valued transform of complex pressure for a
    clim-eligible quantity (magnitude_db/real/imag — the members of
    _CLIM_QUANTITIES)."""
    if quantity == "magnitude_db":
        return 20.0 * np.log10(np.maximum(np.abs(pressure), 1e-12))
    elif quantity == "real":
        return pressure.real
    else:  # imag
        return pressure.imag


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
        self.nbins = len(self.sim.frequencies) + 1
        self.sr = config["sr"]
        self.rho0 = config.get("air_density", 1.2)
        self.c0 = config.get("speed_of_sound", 340.0)

        # Resolve per-quantity settings
        self.surface_quantity = args.surface_quantity or args.quantity
        self.points_quantity = args.points_quantity or args.quantity

        # What to show (args.set_name=None means all fieldpoint sets)
        self.show_surface = not args.no_surface
        self.show_points = not args.no_points

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
        self.ghost_actor = None
        self.mesh_actor = None
        self.points_actor = None
        self.intensity_actor = None
        self.impedance_actor = None
        self.result = None
        self.A_vib = None
        self.share_colorbar = False

        # Cached per-variant data (see _load_variant)
        self._nodes = None
        self._elements = None
        self._ghost = None
        self._has_data = False
        self._v_show_surface = False
        self._v_show_points = False
        self._clim_cache = None
        self._pb_cache = None

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
                                     set_name=set_name, norm="auto")
            v = self.sim.velocity_at(be_index, domain="eval",
                                     set_name=set_name, norm="auto")
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
        """Read pBoundary for a frequency bin, or return None.

        Cached for the duration of one frame only (`_rebuild` clears it):
        both `_joint_db_ref` and `_compute_surface_data` want the same bin,
        and parsing it is ~18 ms on a 36k-element mesh. Deliberately not
        cached across frames -- a bin that NumCalc is still computing must
        be re-read, or it would stay grey for the rest of the session.
        """
        if self._pb_cache is not None and self._pb_cache[0] == be_index:
            return self._pb_cache[1]
        try:
            p = self.sim.pressure_at(be_index, domain="boundary", norm="raw")
        except FileNotFoundError:
            p = None
        self._pb_cache = (be_index, p)
        return p

    def _compute_surface_data(self, be_index, freq, A_vib, db_ref=None,
                              ghost=None):
        """Compute surface scalars or RGBA for the given frequency bin.

        With ``ghost``, computes them for the mirrored copies instead: each
        ghost element takes its source element's pressure, sign-flipped once
        per antisymmetric plane that copy crossed. Applying the sign to the
        complex pressure rather than to the finished scalars gets every
        quantity right at once -- a flip is a pi phase shift, invisible in
        magnitude but not in real/imag/phase.

        Returns (scalars, rgba_flag) or (None, False) if data missing.
        """
        p_surf = self._read_pBoundary(be_index)
        if p_surf is None:
            if ghost is None:
                print(f"  bin {be_index} ({freq:.1f} Hz): pBoundary not yet computed")
            return None, False

        if ghost is not None:
            p_surf = p_surf[ghost.source] * ghost.sign

        if self.surface_rgba:
            if self.args.norm == "acc" and A_vib is not None:
                p_surf = p_surf / (1j * self.rho0 * 2.0 * np.pi * freq * A_vib)
            return _phase_rgba(p_surf, db_ref=db_ref), True
        else:
            scalars = self._surface_scalars(
                p_surf, self.surface_quantity, self.args.norm, freq, A_vib)
            return scalars, False

    def _surface_placeholder(self, n_cells):
        """Neutral surface coloring for a bin that isn't computed yet."""
        if self.surface_rgba:
            scalars = np.full((n_cells, 4), 128, dtype=np.uint8)
            scalars[:, 3] = 255
            return scalars, True
        return np.full(n_cells, np.nan), False

    def _joint_db_ref(self, be_index, freq, A_vib, result):
        """Shared dB reference for the phase fade when surface and eval points
        are both drawn as RGBA phase, so one fade scale spans both.

        Returns None when that doesn't apply (``result`` is None, either side
        isn't in RGBA phase mode, or the bin has no surface data yet), which
        makes each actor fall back to its own max.
        """
        if result is None or not (self.surface_rgba and self.points_rgba):
            return None
        p_surf = self._read_pBoundary(be_index)
        if p_surf is None:
            return None
        if self.args.norm == "acc" and A_vib is not None:
            p_surf = p_surf / (1j * self.rho0 * 2.0 * np.pi * freq * A_vib)
        p_pts = result.data("pressure", self.args.norm)[:, result.at_freq(freq)]
        return max(
            20.0 * np.log10(np.maximum(np.abs(p_surf), 1e-12)).max(),
            20.0 * np.log10(np.maximum(np.abs(p_pts[result.valid_mask]), 1e-12)).max(),
        )

    def _load_variant(self, variant):
        """Load everything that depends on the variant but not the frequency:
        the boundary mesh, the eval-point result, and the vibrating area.

        Kept apart from actor building because a frequency step reuses all of
        it. Reloading it per frame (SimulationData drops its caches, so
        `result` re-reads every bin) is what made stepping frequency cost
        ~0.5 s instead of ~0.05 s.
        """
        from pynumcalc.packing import has_results

        nc_dir = numcalc_dir(self.model_dir, variant)
        self.variant = variant
        self.nc_dir = nc_dir
        self.sim = SimulationData(nc_dir, self.config)
        self._has_data = has_results(nc_dir)
        self._v_show_surface = self.show_surface and self._has_data
        self._v_show_points = self.show_points and self._has_data

        self.result = (self.sim.result(self.args.set_name)
                       if self._v_show_points else None)
        if self._v_show_surface or self._has_data:
            self._nodes = self.sim.boundary_nodes
            self._elements = self.sim.boundary_elements
            # The copies NumCalc reflects internally. Read from the bundle's
            # NC.inp, not config.toml, so the picture shows what was actually
            # exported and solved.
            self._ghost = None
            if not self.args.no_mirror:
                from pynumcalc.readers import read_nc_inp_symmetry
                from bemviz.viz._symmetry import ghost_geometry, describe_symmetry
                symmetry = read_nc_inp_symmetry(nc_dir / "NC.inp")
                self._ghost = ghost_geometry(self._nodes, self._elements, symmetry)
                if self._ghost is not None:
                    print(f"Symmetry ghosts: {describe_symmetry(symmetry)}")

        self.A_vib = None
        if self._v_show_surface:
            self.A_vib = self.sim.A_vib
            if self.A_vib is not None:
                print(f"Vibrating area: {self.A_vib:.6g} m²")

        # Clim spans all bins of this variant, so it changes only here.
        self._clim_cache = None

    def build_variant(self, variant, bin_idx):
        """Build actors for a variant at a given freq bin.

        Per-variant data is loaded only when the variant actually changes --
        stepping the frequency reuses it (see `_load_variant`).
        """
        args = self.args
        scene = self.scene
        freqs = self.freqs

        if variant != self.variant or self.sim is None:
            self._load_variant(variant)
        self.bin_idx = bin_idx
        be_index = bin_idx + 1

        has_data = self._has_data
        v_show_surface = self._v_show_surface
        v_show_points = self._v_show_points
        result = self.result
        A_vib = self.A_vib

        # Share colorbar when both visible with same scalar quantity
        share_colorbar = (v_show_surface and v_show_points
                          and self.surface_quantity == self.points_quantity
                          and not self.surface_rgba)

        # Joint dB reference for phase fade
        actual_freq = freqs[bin_idx]
        db_ref = self._joint_db_ref(
            be_index, actual_freq, A_vib,
            result if (v_show_surface and v_show_points) else None)
        phase_alpha_pts = PHASE_DB_RANGE if self.points_rgba else None
        if db_ref is not None:
            phase_alpha_pts = (db_ref, PHASE_DB_RANGE)

        # Add surface
        surface_actor = None
        ghost_actor = None
        mesh_actor = None
        if v_show_surface:
            scalars, is_rgba = self._compute_surface_data(
                be_index, actual_freq, A_vib, db_ref=db_ref)
            if scalars is None:
                scalars, is_rgba = self._surface_placeholder(len(self._elements))

            edges = not args.no_edges
            surface_actor = scene.add_surface_pressure(
                self._nodes, self._elements, scalars,
                actual_freq=actual_freq,
                quantity=self.surface_quantity,
                cmap=cmap_for_quantity(self.surface_quantity),
                opacity=args.mesh_opacity,
                show_scalar_bar=not share_colorbar,
                rgba=is_rgba,
                show_edges=edges,
                **(EDGE_STYLE if edges else {}),
            )

            # Ghosts get their own actor so the real surface keeps its exact
            # colorbar and clim handling; it is excluded from db_actors for
            # the same reason -- its range is the real one's. Same opacity,
            # no wireframe: the edges are what say "this part was solved".
            if self._ghost is not None:
                g_scalars, g_rgba = self._compute_surface_data(
                    be_index, actual_freq, A_vib, db_ref=db_ref,
                    ghost=self._ghost)
                if g_scalars is None:
                    g_scalars, g_rgba = self._surface_placeholder(
                        len(self._ghost.elements))
                ghost_actor = scene.add_surface_pressure(
                    self._ghost.nodes, self._ghost.elements, g_scalars,
                    actual_freq=actual_freq,
                    quantity=self.surface_quantity,
                    cmap=cmap_for_quantity(self.surface_quantity),
                    opacity=args.mesh_opacity,
                    clim=surface_actor.mapper.scalar_range if not g_rgba else None,
                    show_scalar_bar=False,
                    rgba=g_rgba,
                )
        elif has_data:
            # --no-surface: show geometry without pressure coloring
            edges = not args.no_edges
            mesh_actor = scene.add_object_mesh(
                self._nodes, self._elements, opacity=args.mesh_opacity,
                show_edges=edges, **(EDGE_STYLE if edges else {}))
            if self._ghost is not None:
                ghost_actor = scene.add_object_mesh(
                    self._ghost.nodes, self._ghost.elements,
                    opacity=args.mesh_opacity)

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
        if args.impedance:
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
        self.ghost_actor = ghost_actor
        self.mesh_actor = mesh_actor
        self.points_actor = points_actor
        self.intensity_actor = intensity_actor
        self.impedance_actor = impedance_actor
        # result/A_vib already live on self, set by _load_variant
        self.share_colorbar = share_colorbar

    def _global_range(self, actor):
        """(min, max) of an actor's clim quantity across ALL frequency bins
        of the current variant — not just whichever bin is on screen, so the
        color scale doesn't clip to near-constant color as the frequency
        slider moves away from wherever it started."""
        if actor is self.surface_actor:
            pressure = self.sim.boundary_pressure(norm=self.args.norm)
            values = _quantity_values(pressure, self.surface_quantity)
        else:
            pressure = self.result.data("pressure", self.args.norm)[self.result.valid_mask]
            values = _quantity_values(pressure, self.points_quantity)
        return float(np.nanmin(values)), float(np.nanmax(values))

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
        # Spans every bin of the variant, so it survives a frequency step but
        # must be recomputed per variant (_load_variant clears the cache) --
        # otherwise a second variant would reuse the first one's range.
        if self._clim_cache is None:
            self._clim_cache = [self._global_range(a) for a in self.db_actors]
        self.initial_clims = self._clim_cache
        if first:
            self.clim_state = [0.0, 0.0]

    def _apply_clim(self):
        shift, half_exp = self.clim_state
        for actor, (v0, v1), sym in zip(
                self.db_actors, self.initial_clims, self.db_symmetric):
            if sym:
                half = max(abs(v0), abs(v1)) + half_exp
                rng = (-half, half)
            else:
                rng = (v0 + shift - half_exp, v1 + shift + half_exp)
            actor.mapper.scalar_range = rng
            # The ghosts share the real surface's scale by definition -- they
            # are the same elements. Kept out of db_actors (their range would
            # be counted twice) but dragged along whenever it changes.
            if actor is self.surface_actor and self.ghost_actor is not None:
                self.ghost_actor.mapper.scalar_range = rng
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

    def _rebuild(self, variant, bin_idx):
        """Remove all actors/scalar bars and rebuild them fresh.

        Shared by the frequency and variant sliders. `build_variant` only
        reloads per-variant data when the variant actually changed, so a
        frequency step here rebuilds actors from cached arrays rather than
        re-reading every bin off disk.
        """
        plotter = self.scene._plotter
        self._pb_cache = None
        # Every remove_actor/add_mesh/add_scalar_bar renders on its own,
        # which cost ~9 full render passes per frequency step. Draw once,
        # at the end (_apply_clim renders).
        plotter.suppress_rendering = True
        try:
            for actor in (self.surface_actor, self.ghost_actor, self.mesh_actor,
                          self.points_actor, self.intensity_actor,
                          self.impedance_actor):
                if actor is not None:
                    plotter.remove_actor(actor)
            for title in list(plotter.scalar_bars.keys()):
                plotter.remove_scalar_bar(title, render=False)
            self.build_variant(variant, bin_idx)
            self._update_db_actors()
        finally:
            plotter.suppress_rendering = False
        self._apply_clim()

    def update_freq(self, freq_hz):
        """Frequency slider callback.

        Goes through the same remove-and-re-add path as the variant slider.
        Updating the actors' scalar arrays in place is faster still, but it
        does not reliably repaint under every pyvista/VTK build (notably
        RGBA phase colors under pyvista 0.48), so this uses the path that is
        known to work everywhere. The per-variant data is cached, which is
        what keeps it affordable per frame.
        """
        new_bin = int(np.argmin(np.abs(self.freqs - freq_hz)))
        if new_bin == self.bin_idx:
            return
        self._rebuild(self.variant, new_bin)

    def update_variant(self, v):
        """Variant slider callback."""
        if v == self.variant:
            return
        self._rebuild(v, self.bin_idx)
        print(f"  → variant: {v}")

    def setup_interactivity(self, variants):
        """Set up freq slider, variant slider, clim keys, screenshot hotkey."""
        args = self.args
        scene = self.scene

        # Initialize clim tracking — apply immediately so the very first
        # frame (including --screenshot) already uses the global range,
        # not just whichever bin happened to be built first.
        self._update_db_actors(first=True)
        self._apply_clim()

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
