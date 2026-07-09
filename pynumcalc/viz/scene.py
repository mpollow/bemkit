"""3D visualization with PyVista (lazy import)."""

import numpy as np


def _make_polydata(pv, nodes, elements):
    """Build a PyVista PolyData from nodes and triangle elements."""
    n = len(elements)
    faces = np.column_stack([np.full(n, 3), elements]).ravel()
    return pv.PolyData(nodes, faces)


class BEMScene:
    """Interactive 3D scene for BEM meshes and field data.

    PyVista is imported lazily — only when this class is instantiated.

    Parameters
    ----------
    **plotter_kwargs
        Passed to ``pv.Plotter()``.
    """

    def __init__(self, **plotter_kwargs):
        import pyvista as pv
        self._pv = pv
        self._plotter = pv.Plotter(**plotter_kwargs)

    def add_mesh(self, nodes, elements, *, color="lightblue", opacity=1.0,
                 show_edges=False, scalars=None, cmap=None, **kwargs):
        """Add a triangular mesh to the scene.

        Parameters
        ----------
        nodes : (N, 3) array
            Vertex coordinates in meters.
        elements : (M, 3) array
            Triangle vertex indices.
        color, opacity, show_edges, scalars, cmap
            PyVista mesh display options.
        **kwargs
            Passed to ``plotter.add_mesh()``.

        Returns
        -------
        pyvista Actor
        """
        poly = _make_polydata(self._pv, nodes, elements)

        kw = {"color": color, "show_edges": show_edges}
        if scalars is not None:
            kw["scalars"] = scalars
        if cmap is not None:
            kw["cmap"] = cmap
        kw.update(kwargs)

        actor = self._plotter.add_mesh(poly, **kw)
        actor.prop.opacity = opacity
        return actor

    def add_field_points(self, result, freq, *, quantity="pressure", norm="acc",
                         value_func="magnitude_db", cmap="magma", clim=None,
                         point_size=5.0, alpha_from_db=None, **kwargs):
        """Add field evaluation points colored by value.

        Parameters
        ----------
        result : BEMResult
        freq : float
            Frequency in Hz.
        quantity, norm : str
        value_func : str or callable
        cmap : str
        clim : (vmin, vmax) or None
        point_size : float
        alpha_from_db : float or None
            When set and value_func=="phase", encode phase as hue (hsv) and
            magnitude_db as alpha, over a dynamic range of ``alpha_from_db`` dB
            relative to the max in the data (max → opaque, max-db_range → transparent).
        **kwargs
            Passed to ``plotter.add_points()``.

        Returns
        -------
        pyvista Actor
        """
        from pynumcalc.viz.field import _get_values
        pv = self._pv

        use_rgba = (value_func == "phase" and alpha_from_db is not None)
        valid = result.valid_mask
        pts = result.eval_points[valid]

        if use_rgba:
            import matplotlib.pyplot as plt
            fi = result.at_freq(freq)
            p = result.data(quantity, norm)[:, fi][valid]
            mag_db = 20.0 * np.log10(np.maximum(np.abs(p), 1e-12))
            if isinstance(alpha_from_db, tuple):
                db_ref, db_range = alpha_from_db
                db_floor = db_ref - db_range
            else:
                db_range = alpha_from_db
                db_floor = mag_db.max() - db_range
            weight = np.clip((mag_db - db_floor) / db_range, 0.0, 1.0)
            phase_norm = (np.degrees(np.angle(p)) + 180.0) / 360.0
            phase_rgb = plt.cm.hsv(phase_norm)[:, :3]
            rgb = weight[:, np.newaxis] * phase_rgb + (1.0 - weight[:, np.newaxis])
            v = (np.column_stack([rgb, np.ones(len(p))]) * 255).astype(np.uint8)
        else:
            vals = _get_values(result, freq, value_func, quantity, norm)
            v = vals[valid]

        cloud = pv.PolyData(pts)
        cloud["values"] = v

        if use_rgba:
            show_sb = kwargs.pop("show_scalar_bar", True)
            sb_args = kwargs.pop("scalar_bar_args", {})
            kwargs.pop("cmap", None)
            kwargs.pop("clim", None)
            kw = {"scalars": "values", "rgba": True, "point_size": point_size,
                  "render_points_as_spheres": True, "show_scalar_bar": False}
            kw.update(kwargs)
            kw["show_scalar_bar"] = False
        else:
            kw = {"scalars": "values", "cmap": cmap, "point_size": point_size,
                  "render_points_as_spheres": True}
            if clim is not None:
                kw["clim"] = clim
            kw.update(kwargs)

        actor = self._plotter.add_points(cloud, **kw)

        if use_rgba and show_sb:
            _sb = {"title": "Phase (°)", "position_x": 0.85, "position_y": 0.50,
                   "vertical": True, "height": 0.20}
            _sb.update(sb_args)
            dummy = pv.PolyData(np.zeros((2, 3)))
            dummy["_phase_cb"] = np.array([-180.0, 180.0])
            d = self._plotter.add_points(dummy, scalars="_phase_cb", cmap="hsv",
                                         clim=(-180, 180), show_scalar_bar=True,
                                         scalar_bar_args=_sb, point_size=0.0001,
                                         render_points_as_spheres=False)
            d.prop.opacity = 0.0

        return actor

    def add_field_surface(self, result, freq, *, quantity="pressure", norm="acc",
                          value_func="magnitude_db", cmap="magma", clim=None,
                          **kwargs):
        """Add a Delaunay-triangulated surface from field points.

        Works best with structured grids (e.g. planeXZ).

        Parameters
        ----------
        result : BEMResult
        freq, quantity, norm, value_func, cmap, clim
            See ``add_field_points``.
        **kwargs
            Passed to ``plotter.add_mesh()``.

        Returns
        -------
        pyvista Actor
        """
        from pynumcalc.viz.field import _get_values
        pv = self._pv

        vals = _get_values(result, freq, value_func, quantity, norm)
        valid = result.valid_mask
        pts = result.eval_points[valid]
        v = vals[valid]

        cloud = pv.PolyData(pts)
        cloud["values"] = v
        surf = cloud.delaunay_2d()

        kw = {"scalars": "values", "cmap": cmap,
              "scalar_bar_args": {"position_x": 0.85, "position_y": 0.50,
                                  "vertical": True, "height": 0.20}}
        if clim is not None:
            kw["clim"] = clim
        kw.update(kwargs)

        return self._plotter.add_mesh(surf, **kw)

    def add_points(self, points, *, color="red", point_size=8.0, **kwargs):
        """Add raw points to the scene.

        Parameters
        ----------
        points : (N, 3) array
        color : str
        point_size : float
        **kwargs
            Passed to ``plotter.add_points()``.

        Returns
        -------
        pyvista Actor
        """
        pv = self._pv
        cloud = pv.PolyData(np.asarray(points))
        kw = {"color": color, "point_size": point_size,
              "render_points_as_spheres": True}
        kw.update(kwargs)
        return self._plotter.add_points(cloud, **kw)

    def add_object_mesh(self, nodes, elements, *, color="lightblue", opacity=1.0,
                        show_edges=False, scalars=None, cmap=None,
                        color_by_bc=False, bc_elements=None, bc_colors=None,
                        **kwargs):
        """Add ObjectMesh triangular surface to the scene.

        Parameters
        ----------
        nodes : (N, 3) array
            Vertex coordinates in meters.
        elements : (M, 3) array
            Triangle vertex indices.
        color : str
            Mesh color (default: "lightblue"). Ignored if color_by_bc=True.
        opacity : float
            Mesh opacity 0-1 (default: 1.0).
        show_edges : bool
            Show triangle edges (default: False).
        scalars : array-like or None
            Per-vertex or per-cell scalar values for coloring.
        cmap : str or None
            Colormap for scalars.
        color_by_bc : bool
            If True, color elements by boundary condition.
            Rigid elements → gray, Vibrating elements → red.
        bc_elements : dict or None
            Boundary condition element groups, e.g.
            ``{"vibrating": np.array([0, 1, 2, ...])}``.
            Required when color_by_bc=True.
        bc_colors : dict or None
            Custom colors for boundary conditions,
            e.g., {"rigid": "gray", "vibrating": "red"}.
        **kwargs
            Passed to ``plotter.add_mesh()``.

        Returns
        -------
        pyvista Actor
        """
        pv = self._pv
        n = len(elements)
        poly = _make_polydata(pv, nodes, elements)

        # Handle boundary condition coloring
        if color_by_bc:
            if not bc_elements or "vibrating" not in bc_elements:
                print("Warning: No vibrating boundary conditions found, cannot color by BC")
            else:
                if bc_colors is None:
                    bc_colors = {"rigid": "gray", "vibrating": "red"}

                bc_scalars = np.zeros(n, dtype=int)
                vibrating_indices = bc_elements.get("vibrating", np.array([], dtype=np.int32))
                bc_scalars[vibrating_indices] = 1

                poly["boundary_condition"] = bc_scalars

                rigid_color = bc_colors.get("rigid", "gray")
                vibrating_color = bc_colors.get("vibrating", "red")

                kw = {"scalars": "boundary_condition", "show_edges": show_edges,
                      "cmap": [rigid_color, vibrating_color], "show_scalar_bar": False}
                kw.update(kwargs)
                actor = self._plotter.add_mesh(poly, **kw)
                actor.prop.opacity = opacity
                return actor

        # Build kwargs (standard coloring)
        kw = {"color": color, "show_edges": show_edges}
        if scalars is not None:
            kw["scalars"] = scalars
        if cmap is not None:
            kw["cmap"] = cmap
        kw.update(kwargs)

        actor = self._plotter.add_mesh(poly, **kw)
        actor.prop.opacity = opacity
        return actor

    def add_eval_grid(self, points, *, color="red", point_size=8.0,
                      render_points_as_spheres=True, **kwargs):
        """Add EvalGrid evaluation points to the scene.

        Parameters
        ----------
        points : (N, 3) array
            Evaluation point coordinates in meters.
        color : str
            Point color (default: "red").
        point_size : float
            Point size (default: 8.0).
        render_points_as_spheres : bool
            Render as 3D spheres vs flat circles (default: True).
        **kwargs
            Passed to ``plotter.add_points()``.

        Returns
        -------
        pyvista Actor
        """
        pv = self._pv
        cloud = pv.PolyData(np.asarray(points))
        kw = {"color": color, "point_size": point_size,
              "render_points_as_spheres": render_points_as_spheres}
        kw.update(kwargs)
        return self._plotter.add_points(cloud, **kw)

    def add_surface_pressure(self, nodes, elements, scalars, *,
                             actual_freq=None, quantity="magnitude_db",
                             cmap="magma", clim=None, show_edges=False,
                             opacity=1.0, rgba=False, **kwargs):
        """Add ObjectMesh surface colored by pre-computed scalars.

        Parameters
        ----------
        nodes : (N, 3) array
            Vertex coordinates in meters.
        elements : (M, 3) array
            Triangle vertex indices.
        scalars : (M,) array or (M, 4) uint8
            Per-cell scalar values or RGBA colors.
        actual_freq : float or None
            Frequency label for the scalar bar title.
        quantity : str
            Label for the scalar bar (e.g. "magnitude_db").
        cmap : str
            Colormap (default "magma").
        clim : (vmin, vmax) or None
            Color limits.
        show_edges : bool
            Show triangle edges (default False).
        opacity : float
            Mesh opacity (default 1.0).
        rgba : bool
            If True, ``scalars`` is (M, 4) uint8 RGBA (no colormap).
        **kwargs
            Passed to ``plotter.add_mesh()``.

        Returns
        -------
        pyvista Actor
        """
        pv = self._pv
        poly = _make_polydata(pv, nodes, elements)
        poly.cell_data["surface_pressure"] = scalars

        freq_label = f" @ {actual_freq:.1f} Hz" if actual_freq is not None else ""

        if rgba:
            show_sb = kwargs.pop("show_scalar_bar", True)
            sb_args = kwargs.pop("scalar_bar_args", {})
            kwargs.pop("cmap", None)
            kwargs.pop("clim", None)
            kw = {"scalars": "surface_pressure", "rgba": True, "show_edges": show_edges,
                  "show_scalar_bar": False}
            kw.update(kwargs)
            kw["show_scalar_bar"] = False
        else:
            kw = {"scalars": "surface_pressure", "cmap": cmap, "show_edges": show_edges,
                  "scalar_bar_args": {"title": f"Surface pressure\n({quantity}){freq_label}",
                                      "position_x": 0.85, "position_y": 0.72,
                                      "vertical": True, "height": 0.20}}
            if clim is not None:
                kw["clim"] = clim
            kw.update(kwargs)

        actor = self._plotter.add_mesh(poly, **kw)
        actor.prop.opacity = opacity

        if rgba and show_sb:
            _sb = {"title": "Phase (°)", "position_x": 0.85, "position_y": 0.72,
                   "vertical": True, "height": 0.20}
            _sb.update(sb_args)
            dummy = pv.PolyData(np.zeros((2, 3)))
            dummy["_phase_cb"] = np.array([-180.0, 180.0])
            d = self._plotter.add_points(dummy, scalars="_phase_cb", cmap="hsv",
                                         clim=(-180, 180), show_scalar_bar=True,
                                         scalar_bar_args=_sb, point_size=0.0001,
                                         render_points_as_spheres=False)
            d.prop.opacity = 0.0

        return actor

    def add_intensity_arrows(self, points, pressure, velocity, *,
                              actual_freq=None, scale=1.0,
                              length_mode="db", length_clim=None,
                              cmap="hot", clim=None, **kwargs):
        """Add acoustic intensity arrows I = 1/2 Re(p v*) at eval grid points.

        Parameters
        ----------
        points : (N, 3) array
            Evaluation point coordinates in meters.
        pressure : (N,) complex array
            Scalar pressure at each point.
        velocity : (N, 3) complex array
            Velocity vector at each point.
        actual_freq : float or None
            Frequency label for the scalar bar title.
        scale : float
            Arrow length relative to 5% of the eval grid extent (default 1.0).
        length_mode : str
            "db" (default): arrow length scaled by normalized dB value.
            "magnitude": length proportional to I (W/m²), dynamic per bin.
            "uniform": all arrows the same length.
        length_clim : (lo, hi) or None
            dB range for length_mode="db" normalization.
        cmap : str
            Colormap for intensity level (default "hot").
        clim : (vmin, vmax) or None
            Color limits in dB.
        **kwargs
            Passed to ``plotter.add_mesh()``.

        Returns
        -------
        pyvista Actor
        """
        pv = self._pv

        # Active intensity I = ½ Re(p · v*)  [W/m²]
        I = 0.5 * np.real(pressure[:, None] * np.conj(velocity))  # (N, 3)
        I_mag = np.linalg.norm(I, axis=1)
        I_db = 10.0 * np.log10(np.maximum(I_mag, 1e-30))
        I_dir = I / np.maximum(I_mag[:, None], 1e-30)

        extent = float(np.ptp(points, axis=0).max()) if len(points) > 1 else 1.0
        arrow_len = scale * extent * 0.05

        if length_mode == "uniform":
            lengths = np.full(len(I_dir), arrow_len)
        elif length_mode == "magnitude":
            peak = I_mag.max() or 1.0
            lengths = (I_mag / peak) * arrow_len
        else:  # "db"
            if length_clim is not None:
                db_lo, db_hi = length_clim
            else:
                db_hi = I_db.max()
                db_lo = db_hi - 60.0
            db_range = db_hi - db_lo or 1.0
            t = np.clip((I_db - db_lo) / db_range, 0, 1)
            lengths = t * arrow_len

        cloud = pv.PolyData(points)
        cloud["intensity_db"] = I_db
        cloud["vectors"] = I_dir.astype(np.float64)
        cloud.set_active_vectors("vectors")
        cloud["lengths"] = lengths.astype(np.float64)
        cloud.set_active_scalars("lengths")

        arrows = cloud.glyph(orient=True, scale=True, factor=1.0)

        freq_label = f" @ {actual_freq:.1f} Hz" if actual_freq is not None else ""
        kw = {"scalars": "intensity_db", "cmap": cmap,
              "scalar_bar_args": {
                  "title": f"Intensity\n(dBW/m²){freq_label}",
                  "position_x": 0.85, "position_y": 0.28,
                  "vertical": True, "height": 0.20,
              }}
        if clim is not None:
            kw["clim"] = clim
        kw.update(kwargs)
        return self._plotter.add_mesh(arrows, **kw)

    def add_impedance_points(self, points, scalars, *,
                              actual_freq=None, quantity="real",
                              cmap=None, clim=None, point_size=5.0,
                              as_surface=False, **kwargs):
        """Add eval grid points colored by pre-computed impedance scalars.

        Parameters
        ----------
        points : (N, 3) array
            Evaluation point coordinates in meters.
        scalars : (N,) array
            Pre-computed impedance values (already converted to the
            desired quantity: real, imag, magnitude, magnitude_db, phase).
        actual_freq : float or None
            Frequency label for the scalar bar title.
        quantity : str
            One of "real", "imag", "magnitude", "magnitude_db", "phase".
            Used for default cmap/clim selection and scalar bar title.
        cmap : str or None
            Colormap (auto-selected by quantity if None).
        clim : (vmin, vmax) or None
        point_size : float
        as_surface : bool
            If True, render as Delaunay-triangulated surface.
        **kwargs
            Passed to ``plotter.add_points()`` or ``plotter.add_mesh()``.

        Returns
        -------
        pyvista Actor
        """
        pv = self._pv

        _defaults = {
            "real":         ("RdBu_r", (0.0,   2.0)),
            "imag":         ("RdBu_r", (-2.0,  2.0)),
            "magnitude":    ("magma",  (0.0,   2.0)),
            "magnitude_db": ("magma",  None),
            "phase":        ("hsv",    (-180.0, 180.0)),
        }

        default_cmap, default_clim = _defaults.get(quantity, ("magma", None))
        if cmap is None:
            cmap = default_cmap
        if clim is None:
            clim = default_clim

        cloud = pv.PolyData(np.asarray(points))
        cloud["values"] = scalars

        freq_label = f" @ {actual_freq:.1f} Hz" if actual_freq is not None else ""
        sb_args = {
            "title": f"Impedance Z\u2080\n({quantity}){freq_label}",
            "position_x": 0.85, "position_y": 0.06,
            "vertical": True, "height": 0.20,
        }

        if as_surface:
            surf = cloud.delaunay_2d()
            kw = {"scalars": "values", "cmap": cmap,
                  "scalar_bar_args": sb_args}
            if clim is not None:
                kw["clim"] = clim
            kw.update(kwargs)
            return self._plotter.add_mesh(surf, **kw)
        else:
            kw = {"scalars": "values", "cmap": cmap, "point_size": point_size,
                  "render_points_as_spheres": True, "scalar_bar_args": sb_args}
            if clim is not None:
                kw["clim"] = clim
            kw.update(kwargs)
            return self._plotter.add_points(cloud, **kw)

    def set_camera(self, position):
        """Set camera to a preset or custom position.

        Parameters
        ----------
        position : str or tuple
            Presets: "xz", "xy", "yz", "iso".
            Or a PyVista camera position tuple.
        """
        presets = {
            "xz": [(0, -1, 0), (0, 0, 0), (0, 0, 1)],
            "xy": [(0, 0, 1), (0, 0, 0), (0, 1, 0)],
            "yz": [(1, 0, 0), (0, 0, 0), (0, 0, 1)],
            "iso": [(1, -1, 1), (0, 0, 0), (0, 0, 1)],
        }
        if isinstance(position, str):
            position = presets[position]
        self._plotter.camera_position = position

    def show(self, **kwargs):
        """Display the scene."""
        self._plotter.show(**kwargs)

    def screenshot(self, path, **kwargs):
        """Save a screenshot."""
        self._plotter.screenshot(str(path), **kwargs)

    def close(self):
        """Close the plotter."""
        self._plotter.close()

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def add_screenshot_hotkey(self):
        """Register 's' key to save a timestamped screenshot."""
        import datetime

        def _save():
            path = f"screenshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self._plotter.screenshot(path)
            print(f"Screenshot saved: {path}")

        self._plotter.add_key_event("s", _save)
        print("  [s] save screenshot")

    def add_variant_slider(self, variants, callback):
        """Add a text slider widget for variant selection.

        Parameters
        ----------
        variants : list of str
        callback : callable(variant_name)
        """
        self._plotter.add_text_slider_widget(
            callback,
            variants,
            value=0,
            pointa=(0.25, 0.92),
            pointb=(0.75, 0.92),
            style="modern",
            interaction_event="always",
        )
        print("  [slider] variant selector at top")

    def show_or_screenshot(self, path):
        """Show interactive window or save screenshot.

        Parameters
        ----------
        path : str or None
            If not None, save screenshot to this path.
        """
        if path:
            print(f"Saving screenshot: {path}")
            self.screenshot(path)
            print("Done!")
        else:
            print("Opening viewer...")
            self.show()


def setup_scene(screenshot=False):
    """Create a BEMScene with optional off-screen rendering."""
    kwargs = {}
    if screenshot:
        kwargs["off_screen"] = True
    return BEMScene(**kwargs)
