"""`bem check` over a built Model.

The diagnostics themselves are tested in tests/geometry/test_quality.py; what
matters here is that the CLI feeds them the right things off the Model — the
band's top frequency, the medium's speed of sound, and *point* source
positions only.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from bemkit.cli import main

REPO = Path(__file__).resolve().parents[4]
SPHCAP = REPO / "models" / "sphcap"


@pytest.fixture()
def sphcap(tmp_path):
    dst = tmp_path / "sphcap"
    shutil.copytree(SPHCAP, dst,
                    ignore=shutil.ignore_patterns("__pycache__", "runs"))
    return dst


def test_check_reports_the_coarse_mesh_underresolved_at_16k(sphcap, capsys):
    """sphcap's band tops out at 16 kHz, where the coarse mesh's 22.7 mm
    edges are longer than the 21.2 mm wavelength — about 1 element per
    wavelength against the ~6 the paper asks for. An error, so a bare
    `check` exits non-zero."""
    with pytest.raises(SystemExit) as e:
        main([str(sphcap), "check", "--variant", "coarse"])
    assert e.value.code == 1

    out = capsys.readouterr().out
    assert "coarse  (284 nodes, 564 elements, up to 16000 Hz)" in out
    assert "✗ elements_per_wavelength" in out
    assert "16000 Hz" in out
    assert "✓ watertight" in out
    assert "✓ orientation" in out


def test_check_covers_every_variant_by_default(sphcap, capsys):
    with pytest.raises(SystemExit):
        main([str(sphcap), "check"])
    out = capsys.readouterr().out
    for v in ("coarse", "standard", "fine"):
        assert f"\n{v}  (" in out


# ── exit codes ─────────────────────────────────────────────────────────────

MODEL = """\
import numpy as np
from bemkit.model import BC, Elements, Mesh, Model, PointSource, Velocity

# A regular icosahedron: watertight, outward, and regular enough that no
# check fires at the low band this model uses.
P = (1 + 5 ** 0.5) / 2
V = np.array([[-1, P, 0], [1, P, 0], [-1, -P, 0], [1, -P, 0],
              [0, -1, P], [0, 1, P], [0, -1, -P], [0, 1, -P],
              [P, 0, -1], [P, 0, 1], [-P, 0, -1], [-P, 0, 1]], dtype=float)
V /= np.linalg.norm(V, axis=1)[:, None]
V *= 0.1
T = np.array([
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
FREQS = np.array({freqs})
SOURCES = {sources}


def build():
    return Model(Mesh(V, T), FREQS,
                 bcs=[BC(Elements([0]), Velocity(1.0))],
                 sources=SOURCES,
                 fields={{"pts": np.array([[5., 5., 5.]])}})
"""


def write(path, freqs="[100., 200.]", sources="[]"):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.py").write_text(MODEL.format(freqs=freqs, sources=sources))
    return path


def test_a_clean_model_exits_zero(tmp_path, capsys):
    study = write(tmp_path / "clean")
    assert main([str(study), "check"]) is None      # no SystemExit
    out = capsys.readouterr().out
    assert "✗" not in out and "!" not in out


def test_strict_turns_a_warning_into_a_non_zero_exit(tmp_path, capsys):
    """800 Hz gives ~4 elements per wavelength on this icosahedron: below
    the ~6 rule of thumb but above the half-of-target error floor, so it
    warns — exactly the band --strict exists for."""
    study = write(tmp_path / "warn", freqs="[100., 800.]")
    main([str(study), "check"])                     # warning alone: exit 0
    assert "!" in capsys.readouterr().out

    with pytest.raises(SystemExit) as e:
        main([str(study), "check", "--strict"])
    assert e.value.code == 1


# ── what reaches check_mesh ────────────────────────────────────────────────


def _captured(monkeypatch, study):
    """Run `check` with check_mesh replaced, returning its kwargs."""
    from bemkit.cli import _check
    seen = {}

    def spy(vertices, triangles, **kw):
        seen.update(vertices=vertices, triangles=triangles, **kw)
        return []

    monkeypatch.setattr("bemkit.geometry.quality.check_mesh", spy)
    main([str(study), "check"])
    return seen


def test_the_top_of_the_band_and_the_medium_come_from_the_model(
        tmp_path, monkeypatch):
    study = write(tmp_path / "m", freqs="[100., 200., 900.]")
    seen = _captured(monkeypatch, study)
    assert seen["max_frequency"] == 900.0
    assert seen["speed_of_sound"] == 343.0          # Medium's default


def test_point_sources_are_passed_as_positions(tmp_path, monkeypatch):
    study = write(tmp_path / "src",
                  sources="[PointSource((0.5, 0., 0.)), "
                          "PointSource((0., 0.5, 0.))]")
    seen = _captured(monkeypatch, study)
    assert seen["sources"] == [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0)]


def test_plane_waves_are_not_passed_as_coordinates(tmp_path, monkeypatch):
    """A plane wave carries a direction of travel. Hint 3 measures a source
    *position* against the average edge length, so feeding it a direction
    would score a unit vector as a point 1 m from the origin — a clearance
    verdict about a source that does not exist there."""
    (tmp_path / "pw").mkdir()
    text = MODEL.format(freqs="[100., 200.]",
                        sources="[PlaneWave((0., 0., 1.))]")
    text = text.replace("PointSource, Velocity",
                        "PlaneWave, PointSource, Velocity")
    (tmp_path / "pw" / "model.py").write_text(text)
    seen = _captured(monkeypatch, tmp_path / "pw")
    assert seen["sources"] == []


def test_mesh_arrays_come_straight_off_the_model(tmp_path, monkeypatch):
    study = write(tmp_path / "mesh")
    seen = _captured(monkeypatch, study)
    from bemkit.cli.study import load_study
    mesh = load_study(study)["default"].mesh
    assert np.array_equal(seen["vertices"], mesh.vertices)
    assert np.array_equal(seen["triangles"], mesh.triangles)


def test_an_unmatched_variant_names_what_exists(sphcap):
    with pytest.raises(SystemExit) as e:
        main([str(sphcap), "check", "--variant", "nope"])
    assert "coarse" in str(e.value)
