"""Tests for pynumcalc.sim.config_snapshot and SimulationData's use of it."""

import warnings

import numpy as np

from pynumcalc.analysis.simulation import SimulationData
from pynumcalc.sim.config_snapshot import read_snapshot, write_snapshot


def test_write_and_read_roundtrip(tmp_path):
    cfg = {"sr": 48000, "nbins": 65, "air_density": 1.2,
           "speed_of_sound": 340.0, "bem_type": 4}
    write_snapshot(tmp_path, cfg)
    snap = read_snapshot(tmp_path)
    assert snap["sr"] == 48000 and snap["nbins"] == 65
    assert snap["bem_type"] == 4
    assert "pynumcalc_version" in snap and "exported_at" in snap


def test_read_missing_returns_none(tmp_path):
    assert read_snapshot(tmp_path) is None


def test_records_mesh_and_boundary(tmp_path):
    mesh = tmp_path / "m.stl"
    mesh.write_bytes(b"solid\nendsolid\n")
    write_snapshot(tmp_path, {"sr": 8000, "nbins": 5},
                   mesh_file=mesh, resolved_boundary={"ear": [(3, 3, 1.0)]})
    snap = read_snapshot(tmp_path)
    assert snap["mesh_file"] == "m.stl" and len(snap["mesh_sha256"]) == 64
    assert snap["resolved_boundary"] == {"ear": [[3, 3, 1.0]]}


def test_simulationdata_prefers_snapshot(numcalc_dir):
    # Snapshot says sr=48000/nbins=3; pass a *different* live config.
    write_snapshot(numcalc_dir, {"sr": 48000, "nbins": 3,
                                 "air_density": 1.2, "speed_of_sound": 340.0})
    live = {"sr": 8000, "nbins": 9}  # would give different frequencies
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = SimulationData(numcalc_dir, live)
    expected = np.linspace(0, 48000 / 2, 3)[1:]
    np.testing.assert_allclose(sim.frequencies, expected)


def test_simulationdata_warns_on_drift(numcalc_dir):
    write_snapshot(numcalc_dir, {"sr": 48000, "nbins": 3,
                                 "air_density": 1.2, "speed_of_sound": 340.0})
    live = {"sr": 8000, "nbins": 3}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SimulationData(numcalc_dir, live)
    assert any("differs from current vars.toml" in str(w.message) for w in caught)


def test_no_snapshot_uses_live_config(numcalc_dir):
    # Fixture has no config.json — live config governs, no warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = SimulationData(numcalc_dir, {"sr": 16000, "nbins": 3})
    assert not any("vars.toml" in str(w.message) for w in caught)
    np.testing.assert_allclose(sim.frequencies, np.linspace(0, 8000, 3)[1:])
