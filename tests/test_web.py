"""Unit tests for the pure numeric helpers of the web bundle exporter."""

import numpy as np
import pytest

from bemkit.web import (
    detect_plane,
    encode_db_u8,
    pick_up_axis,
    quantize_irs,
    relative_spectra,
    synth_irs,
)


class TestSynthIrs:
    def test_matches_plain_irfft_at_native_rate(self):
        # nf bins at df on an sr grid: synth at ir_fs = sr reproduces
        # BEMResult.impulse_response's plain irfft (DC prepended as 0).
        rng = np.random.default_rng(0)
        nf, df = 64, 250.0
        spectra = rng.normal(size=(3, nf)) + 1j * rng.normal(size=(3, nf))
        irs = synth_irs(spectra, df, 32000)
        spec_full = np.concatenate([np.zeros((3, 1), complex), spectra], axis=1)
        expected = np.fft.irfft(spec_full, axis=1)
        assert irs.shape == (3, 128)
        np.testing.assert_allclose(irs, expected, atol=1e-12)

    def test_pure_delay_peaks_at_the_delay_sample(self):
        nf, df, fs = 64, 250.0, 48000
        m = 10  # delay in output samples
        f = df * np.arange(1, nf + 1)
        spectra = np.exp(-2j * np.pi * f * m / fs)[None, :]
        ir = synth_irs(spectra, df, fs)[0]
        assert ir.shape == (192,)
        assert np.argmax(np.abs(ir)) == m

    def test_energy_matches_parseval_at_both_rates(self):
        # Same spectral content synthesized at 32k and 48k: Parseval fixes
        # the energies exactly. At 32k the top bin (16 kHz) is the Nyquist
        # bin — counted once, and irfft keeps only its real part; at 48k it
        # is an interior bin (counted twice, fully complex).
        rng = np.random.default_rng(1)
        spectra = rng.normal(size=(1, 64)) + 1j * rng.normal(size=(1, 64))
        s_int = np.sum(np.abs(spectra) ** 2)
        top = spectra[0, -1]
        e32 = np.sum(synth_irs(spectra, 250.0, 32000) ** 2)
        e48 = np.sum(synth_irs(spectra, 250.0, 48000) ** 2)
        expected32 = (2 * (s_int - np.abs(top) ** 2) + top.real ** 2) / 128
        np.testing.assert_allclose(e32, expected32, rtol=1e-9)
        np.testing.assert_allclose(e48, 2 * s_int / 192, rtol=1e-9)

    def test_rejects_off_grid_rate(self):
        with pytest.raises(ValueError, match="multiple"):
            synth_irs(np.zeros((1, 64), complex), 250.0, 44100)

    def test_rejects_rate_below_band(self):
        with pytest.raises(ValueError, match="too low"):
            synth_irs(np.zeros((1, 64), complex), 250.0, 16000)


class TestRelativeSpectra:
    def test_reference_direction_is_flat(self):
        rng = np.random.default_rng(2)
        spectra = rng.normal(size=(5, 16)) + 1j * rng.normal(size=(5, 16))
        rel = relative_spectra(spectra, ref_index=2)
        np.testing.assert_allclose(np.abs(rel[2]), 1.0, rtol=1e-12)
        np.testing.assert_allclose(np.angle(rel[2]), 0.0, atol=1e-12)

    def test_floor_caps_amplification_at_reference_notches(self):
        href = np.ones(8, complex)
        href[3] = 1e-8  # deep notch
        spectra = np.vstack([np.ones(8, complex), href])
        rel = relative_spectra(spectra, ref_index=1, floor_db=-40.0)
        # Without the floor this bin would be ~1e8; the floor caps it.
        assert np.abs(rel[0, 3]) <= 10 ** (80 / 20) + 1e-6
        np.testing.assert_allclose(np.abs(rel[0, :3]), 1.0, rtol=1e-9)


class TestQuantizeIrs:
    def test_roundtrip_within_one_step(self):
        rng = np.random.default_rng(3)
        irs = rng.normal(size=(4, 32)) * np.geomspace(1e-3, 10, 4)[:, None]
        scales, taps = quantize_irs(irs)
        back = taps.astype(np.float64) * scales[:, None].astype(np.float64)
        peaks = np.abs(irs).max(axis=1)
        assert np.all(np.abs(back - irs) <= (peaks / 32767)[:, None] * 1.001)

    def test_zero_ir_gets_zero_scale(self):
        scales, taps = quantize_irs(np.zeros((1, 8)))
        assert scales[0] == 0
        assert not taps.any()


class TestEncodeDbU8:
    def test_window_endpoints_and_nan(self):
        v = encode_db_u8(np.array([-60.0, 0.0, -120.0, 12.0, np.nan]), -60, 0)
        assert list(v) == [0, 254, 0, 254, 255]

    def test_linear_midpoint(self):
        assert encode_db_u8(np.array([-30.0]), -60, 0)[0] == 127


class TestDetectPlane:
    def _grid(self, w, h, spacing, origin, u, v):
        iu, iv = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
        pts = (np.asarray(origin)[None, :]
               + iu.reshape(-1, 1) * spacing * np.asarray(u)[None, :]
               + iv.reshape(-1, 1) * spacing * np.asarray(v)[None, :])
        return pts

    def test_recovers_grid_parameters(self):
        pts = self._grid(11, 7, 0.01, [-0.05, 0, -0.03], [1, 0, 0], [0, 0, 1])
        g = detect_plane(pts)
        assert (g["w"], g["h"]) == (11, 7)
        np.testing.assert_allclose(g["spacing_m"], 0.01)
        np.testing.assert_allclose(g["origin"], [-0.05, 0, -0.03])
        np.testing.assert_allclose(g["u_axis"], [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(g["v_axis"], [0, 0, 1], atol=1e-12)

    def test_rejects_anisotropic_spacing(self):
        iu, iv = np.meshgrid(np.arange(4), np.arange(4), indexing="xy")
        pts = np.column_stack([iu.ravel() * 0.01, np.zeros(16), iv.ravel() * 0.02])
        with pytest.raises(ValueError, match="anisotropic"):
            detect_plane(pts)

    def test_rejects_irregular_points(self):
        pts = self._grid(5, 5, 0.01, [0, 0, 0], [1, 0, 0], [0, 0, 1])
        pts[13] += 0.004
        with pytest.raises(ValueError, match="regular"):
            detect_plane(pts)


class TestPickUpAxis:
    def test_front_z_gets_up_y(self):
        np.testing.assert_allclose(pick_up_axis([0, 0, 1]), [0, 1, 0], atol=1e-12)

    def test_front_x_gets_up_z(self):
        np.testing.assert_allclose(pick_up_axis([1, 0, 0]), [0, 0, 1], atol=1e-12)

    def test_up_is_always_orthonormal(self):
        rng = np.random.default_rng(4)
        for _ in range(20):
            front = rng.normal(size=3)
            front /= np.linalg.norm(front)
            up = pick_up_axis(front)
            assert abs(up @ front) < 1e-9
            assert abs(np.linalg.norm(up) - 1) < 1e-9
