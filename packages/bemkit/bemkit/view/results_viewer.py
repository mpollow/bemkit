"""The interactive 3D results viewer behind `bem STUDY results`.

Rebuilt on the model.py spine. What changed from the ``SimulationData``-era
viewer, and why:

* **Geometry comes from the bundle, not from the results.** A
  :class:`~bemkit.results.Results` is store-backed and solver-neutral: it has
  no workdir, and one from ``solve(workdir=None)`` has no files at all. The
  boundary mesh, the eval-point coordinates and the boundary conditions are
  read straight out of ``runs/<solver>/<variant>/`` — see
  :mod:`bemkit.view.bundle`.
* **Acceleration normalization is a scale factor.** ``--norm acc`` is
  ``Results.scaled_by(1 / (j rho omega A_vib))`` with ``A_vib`` from
  ``Model.vibrating_area``, applied once when the bundle loads rather than on
  every read. A model with no velocity drive says so and stops, instead of
  showing raw Pascals under an acc label.
* **The frequency slider only offers computed bins.** A partial band leaves
  the rest NaN, and NaN reaches the colour mapper as "below the range" — a
  confident, wrong colour. ``Bundle.bins`` is the list the slider snaps to.

The interactive machinery — frequency slider, variant slider, colour-limit
keys, per-frame actor rebuild — is carried over as it was; it is the part
that was worth keeping.
"""

import numpy as np

from bemkit.view._quantities import (
    CLIM_QUANTITIES, LABELS, PHASE_DB_RANGE, SYMMETRIC_QUANTITIES,
    impedance_scalars, magnitude_db, phase_rgba, quantity_values,
)
from bemkit.view._style import cmap_for_quantity
from bemkit.view._symmetry import EDGE_STYLE, describe_symmetry
from bemkit.view.bundle import Bundle, nearest_bin


def _label(quantity, norm):
    """Scalar-bar label: the quantity plus which normalization it is in."""
    return f"{LABELS.get(quantity, quantity)} ({norm})"


class ResultsViewer:
    """Actors, sliders and colour limits for one study's solved variants.

    Args:
        scene: a :class:`~bemkit.view.scene.BEMScene`.
        workdir_for: ``variant -> run directory`` (from
            ``bundle.prepared_bundles``).
        models: ``{variant: Model}`` — the medium and the vibrating area come
            from here; everything drawn comes from the bundle on disk.
        args: the parsed `bem STUDY results` namespace.
    """

    def __init__(self, scene, workdir_for, models, args):
        self.scene = scene
        self.workdir_for = workdir_for
        self.models = models
        self.args = args

        self.surface_quantity = args.surface_quantity or args.quantity
        self.points_quantity = args.points_quantity or args.quantity

        self.show_surface = not args.no_surface
        self.show_points = not args.no_points

        phase_fade = not args.no_phase_fade
        self.surface_rgba = (self.surface_quantity == "phase") and phase_fade
        self.points_rgba = ((self.points_quantity == "phase") and phase_fade
                            and not args.plane)

        # Mutable state
        self.variant = None
        self.bundle = None
        self.bin_idx = 0
        self.surface_actor = None
        self.ghost_actor = None
        self.mesh_actor = None
        self.points_actor = None
        self.intensity_actor = None
        self.impedance_actor = None
        self.share_colorbar = False

        # Colour-limit state (spans every computed bin of the variant)
        self.db_actors = []
        self.db_symmetric = []
        self.initial_clims = []
        self.clim_state = [0.0, 0.0]
        self._clim_cache = None

    # ── per-variant loading ────────────────────────────────────────────────

    def load_variant(self, variant):
        """Everything that depends on the variant but not the frequency."""
        model = self.models[variant]
        self.bundle = Bundle(self.workdir_for(variant), model,
                             norm=self.args.norm,
                             mirror=not self.args.no_mirror)
        self.variant = variant
        self._clim_cache = None

        if self.bundle.ghost is not None:
            print(f"  Symmetry ghosts: "
                  f"{describe_symmetry(self.bundle.symmetry)}")
        if self.args.norm == "acc":
            print(f"  Vibrating area: {model.vibrating_area:.6g} m²")
        n, nf = len(self.bundle.bins), len(self.bundle.frequencies)
        if n < nf:
            print(f"  Partial band: {n} of {nf} bins computed — the slider "
                  f"offers only those")

    def initial_bin(self, freq_hz):
        """The bin to open on: nearest computed bin to ``freq_hz``."""
        return nearest_bin(self.bundle.frequencies, freq_hz, self.bundle.bins)

    # ── scalars ────────────────────────────────────────────────────────────

    def _eval_field(self, set_name):
        """``(points, pressure, velocity)`` at the current bin, or Nones.

        Velocity is missing when the bundle has no ``vEvalGrid`` — an older
        run, or one whose EvalGrid velocity NumCalc did not write.
        """
        b = self.bundle
        if not b.has("vEvalGrid") or not b.has("pEvalGrid"):
            return None, None, None
        return (b.points(set_name),
                b.pressure(self.bin_idx, set_name),
                b.velocity(self.bin_idx, set_name))

    def _surface_scalars(self, db_ref=None, ghost=None):
        """(scalars, is_rgba) for the surface at the current bin.

        With ``ghost``, computes them for the mirrored copies instead: each
        ghost element takes its source element's pressure, sign-flipped once
        per antisymmetric plane that copy crossed. The sign goes on the
        *complex* pressure, which gets every quantity right at once — a flip
        is a pi phase shift, invisible in magnitude but not in
        real/imag/phase.
        """
        p = self.bundle.boundary_pressure(self.bin_idx)
        if ghost is not None:
            p = p[ghost.source] * ghost.sign
        if self.surface_rgba:
            return phase_rgba(p, db_ref=db_ref), True
        return quantity_values(p, self.surface_quantity), False

    def _joint_db_ref(self):
        """Shared dB reference for the phase fade when the surface and the
        eval points are both drawn as RGBA phase, so one fade scale spans
        both. None when that does not apply, which makes each actor fall back
        to its own max."""
        if not (self.surface_rgba and self.points_rgba):
            return None
        if not (self.show_surface and self.bundle.has("pBoundary")):
            return None
        if not (self.show_points and self.bundle.has("pEvalGrid")):
            return None
        p_surf = self.bundle.boundary_pressure(self.bin_idx)
        p_pts = self.bundle.pressure(self.bin_idx, self.args.set_name)
        return max(magnitude_db(p_surf).max(), magnitude_db(p_pts).max())

    # ── actor building ─────────────────────────────────────────────────────

    def build_variant(self, variant, bin_idx):
        """Build every actor for ``variant`` at the 0-based ``bin_idx``."""
        args = self.args
        scene = self.scene

        if variant != self.variant or self.bundle is None:
            self.load_variant(variant)
        self.bin_idx = bin_idx
        freq = float(self.bundle.frequencies[bin_idx])

        # Decided on what will actually be *drawn*, not on what was asked
        # for: a bundle missing pBoundary draws no surface, and suppressing
        # both actors' own colour bars for a shared one that then never gets
        # added would leave the picture with no scale at all.
        draw_surface = self.show_surface and self.bundle.has("pBoundary")
        draw_points = self.show_points and self.bundle.has("pEvalGrid")
        share_colorbar = (draw_surface and draw_points
                          and self.surface_quantity == self.points_quantity
                          and not self.surface_rgba)

        db_ref = self._joint_db_ref()
        phase_alpha_pts = PHASE_DB_RANGE if self.points_rgba else None
        if db_ref is not None:
            phase_alpha_pts = (db_ref, PHASE_DB_RANGE)

        edges = not args.no_edges
        surface_actor = ghost_actor = mesh_actor = None

        if draw_surface:
            scalars, is_rgba = self._surface_scalars(db_ref=db_ref)
            surface_actor = scene.add_surface_pressure(
                self.bundle.nodes, self.bundle.elements, scalars,
                actual_freq=freq,
                quantity=_label(self.surface_quantity, args.norm),
                cmap=cmap_for_quantity(self.surface_quantity),
                opacity=args.mesh_opacity,
                show_scalar_bar=not share_colorbar,
                rgba=is_rgba, show_edges=edges,
                **(EDGE_STYLE if edges else {}),
            )
            # Ghosts get their own actor so the real surface keeps its exact
            # colorbar and clim handling; kept out of db_actors for the same
            # reason -- their range *is* the real one's. Same opacity, no
            # wireframe: the edges are what say "this part was solved".
            if self.bundle.ghost is not None:
                g_scalars, g_rgba = self._surface_scalars(
                    db_ref=db_ref, ghost=self.bundle.ghost)
                ghost_actor = scene.add_surface_pressure(
                    self.bundle.ghost.nodes, self.bundle.ghost.elements,
                    g_scalars, actual_freq=freq,
                    quantity=_label(self.surface_quantity, args.norm),
                    cmap=cmap_for_quantity(self.surface_quantity),
                    opacity=args.mesh_opacity,
                    clim=(surface_actor.mapper.scalar_range
                          if not g_rgba else None),
                    show_scalar_bar=False, rgba=g_rgba,
                )
        else:
            # --no-surface (or a boundary-less run): geometry, no pressure.
            mesh_actor = scene.add_object_mesh(
                self.bundle.nodes, self.bundle.elements,
                opacity=args.mesh_opacity, show_edges=edges,
                **(EDGE_STYLE if edges else {}))
            if self.bundle.ghost is not None:
                ghost_actor = scene.add_object_mesh(
                    self.bundle.ghost.nodes, self.bundle.ghost.elements,
                    opacity=args.mesh_opacity)

        points_actor = None
        if draw_points:
            pts = self.bundle.points(args.set_name)
            vals = self.bundle.pressure(self.bin_idx, args.set_name)
            kw = dict(
                value_func=self.points_quantity,
                cmap=cmap_for_quantity(self.points_quantity),
                show_scalar_bar=not share_colorbar,
                scalar_bar_args={
                    "title": f"{_label(self.points_quantity, args.norm)}"
                             f"\n@ {freq:.1f} Hz",
                    "position_x": 0.85, "position_y": 0.50,
                    "vertical": True, "height": 0.20,
                },
            )
            if args.plane:
                points_actor = scene.add_field_surface(pts, vals, **kw)
            else:
                points_actor = scene.add_field_points(
                    pts, vals, point_size=args.point_size,
                    alpha_from_db=phase_alpha_pts, **kw)

        if share_colorbar:
            vmin = min(surface_actor.mapper.scalar_range[0],
                       points_actor.mapper.scalar_range[0])
            vmax = max(surface_actor.mapper.scalar_range[1],
                       points_actor.mapper.scalar_range[1])
            surface_actor.mapper.scalar_range = (vmin, vmax)
            points_actor.mapper.scalar_range = (vmin, vmax)
            scene._plotter.add_scalar_bar(
                title=f"{_label(self.surface_quantity, args.norm)}"
                      f"\n@ {freq:.1f} Hz",
                mapper=surface_actor.mapper,
                position_x=0.85, position_y=0.72, vertical=True, height=0.20)

        intensity_actor = None
        if args.intensity:
            pts, p, v = self._eval_field(args.set_name)
            if pts is None:
                print("  (intensity: vEvalGrid not in this run, skipping)")
            else:
                intensity_actor = scene.add_intensity_arrows(
                    pts, p, v, actual_freq=freq, scale=args.intensity_scale,
                    length_mode=args.intensity_length,
                    length_clim=args.intensity_clim)
                print(f"  + Intensity arrows @ {freq:.1f} Hz "
                      f"(scale={args.intensity_scale}, "
                      f"length={args.intensity_length})")

        impedance_actor = None
        if args.impedance:
            pts, p, v = self._eval_field(args.set_name)
            if pts is None:
                print("  (impedance: vEvalGrid not in this run, skipping)")
            else:
                medium = self.models[variant].medium
                imp = impedance_scalars(pts, p, v, medium.rho, medium.c,
                                        args.impedance_quantity)
                impedance_actor = scene.add_impedance_points(
                    pts, imp, actual_freq=freq,
                    quantity=args.impedance_quantity,
                    point_size=args.point_size, as_surface=args.plane)
                print(f"  + Impedance Z₀ ({args.impedance_quantity}) "
                      f"@ {freq:.1f} Hz")

        self.surface_actor = surface_actor
        self.ghost_actor = ghost_actor
        self.mesh_actor = mesh_actor
        self.points_actor = points_actor
        self.intensity_actor = intensity_actor
        self.impedance_actor = impedance_actor
        self.share_colorbar = share_colorbar

    # ── colour limits ──────────────────────────────────────────────────────

    def _global_range(self, actor):
        """(min, max) of an actor's quantity across every *computed* bin of
        the current variant — not just the one on screen, so the colour scale
        does not clip to a near-constant colour as the slider moves away from
        wherever it started. Uncomputed bins are NaN and are excluded, which
        is why this is nanmin/nanmax and not min/max."""
        b = self.bundle
        if actor is self.surface_actor:
            values = quantity_values(b.results.boundary_pressure(),
                                     self.surface_quantity)
        else:
            values = quantity_values(b.results.pressure(self.args.set_name),
                                     self.points_quantity)
        return float(np.nanmin(values)), float(np.nanmax(values))

    def _update_db_actors(self, first=False):
        self.db_actors = []
        self.db_symmetric = []
        for actor, quantity in ((self.surface_actor, self.surface_quantity),
                                (self.points_actor, self.points_quantity)):
            if actor is not None and quantity in CLIM_QUANTITIES:
                self.db_actors.append(actor)
                self.db_symmetric.append(quantity in SYMMETRIC_QUANTITIES)
        # Spans every bin of the variant, so it survives a frequency step but
        # must be recomputed per variant (load_variant clears the cache) --
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
            if self.db_symmetric[0]:
                half = max(abs(v0), abs(v1)) + half_exp
                print(f"  clim: [{-half:.3g}, {half:.3g}]")
            else:
                print(f"  clim: [{v0 + shift - half_exp:.3g}, "
                      f"{v1 + shift + half_exp:.3g}]")

    # ── interactivity ──────────────────────────────────────────────────────

    def _rebuild(self, variant, bin_idx):
        """Remove every actor and scalar bar and rebuild them fresh.

        Shared by the frequency and variant sliders. ``build_variant`` only
        reloads the bundle when the variant actually changed, so a frequency
        step rebuilds actors from arrays already in memory.
        """
        plotter = self.scene._plotter
        # Every remove_actor/add_mesh/add_scalar_bar renders on its own,
        # which cost ~9 full render passes per frequency step. Draw once, at
        # the end (_apply_clim renders).
        plotter.suppress_rendering = True
        try:
            for actor in (self.surface_actor, self.ghost_actor,
                          self.mesh_actor, self.points_actor,
                          self.intensity_actor, self.impedance_actor):
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
        """Frequency slider callback. Snaps to a *computed* bin."""
        new_bin = nearest_bin(self.bundle.frequencies, freq_hz,
                              self.bundle.bins)
        if new_bin == self.bin_idx:
            return
        self._rebuild(self.variant, new_bin)
        print(f"  → {self.bundle.frequencies[new_bin]:.1f} Hz "
              f"(bin {new_bin + 1})")

    def update_variant(self, v):
        """Variant slider callback."""
        if v == self.variant:
            return
        # A new variant may have a different set of computed bins; keep the
        # frequency rather than the index, which would silently jump.
        freq = float(self.bundle.frequencies[self.bin_idx])
        self.load_variant(v)
        self._rebuild(v, self.initial_bin(freq))
        print(f"  → variant: {v}")

    def setup_interactivity(self, variants):
        """Frequency slider, variant slider, clim keys, screenshot hotkey."""
        args = self.args
        scene = self.scene

        # Apply immediately so the very first frame (including --screenshot)
        # already uses the global range, not just the bin built first.
        self._update_db_actors(first=True)
        self._apply_clim()

        if not args.screenshot:
            step = 10.0

            def shift_clim(delta):
                if not all(self.db_symmetric):
                    self.clim_state[0] += delta
                    self._apply_clim()

            def expand_range(delta):
                self.clim_state[1] += delta / 2
                self._apply_clim()

            # Both KP_* (native X11 backend) and symbol names (Qt backend).
            for key in ("KP_Add", "plus"):
                scene._plotter.add_key_event(key, lambda: shift_clim(-step))
            for key in ("KP_Subtract", "minus"):
                scene._plotter.add_key_event(key, lambda: shift_clim(+step))
            for key in ("KP_Multiply", "asterisk"):
                scene._plotter.add_key_event(key, lambda: expand_range(+step))
            for key in ("KP_Divide", "slash"):
                scene._plotter.add_key_event(key, lambda: expand_range(-step))
            if self.db_actors:
                print("  [*/÷ or KP_*/÷] expand/contract clim range"
                      + ("  [+/- or KP_+/-] shift window"
                         if not all(self.db_symmetric) else ""))

        # Frequency slider, spanning the computed bins only.
        bins = self.bundle.bins
        if (not args.screenshot and len(bins) > 1
                and (self.show_surface or self.show_points)):
            freqs = self.bundle.frequencies
            scene._plotter.add_slider_widget(
                self.update_freq,
                rng=[float(freqs[bins[0]]), float(freqs[bins[-1]])],
                value=float(freqs[self.bin_idx]),
                title="Frequency (Hz)",
                pointa=(0.1, 0.05), pointb=(0.9, 0.05),
                style="modern", fmt="%.0f", interaction_event="always",
            )
            print(f"  [slider] drag to change frequency "
                  f"({len(bins)} computed bin(s))")

        if not args.screenshot and len(variants) > 1:
            scene.add_variant_slider(variants, self.update_variant)

        scene.set_camera(args.camera)

        if not args.screenshot:
            scene.add_screenshot_hotkey()
