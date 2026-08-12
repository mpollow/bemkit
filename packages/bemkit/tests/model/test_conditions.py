"""Conditions: *what* a boundary condition is."""

import numpy as np
import pytest

from bemkit.model import Admittance, Impedance, Pressure, Velocity

FREQS = np.array([100.0, 200.0, 400.0])


def test_velocity_scalar():
    v = Velocity(1.0)
    np.testing.assert_allclose(v.at(FREQS), [1.0, 1.0, 1.0])


def test_velocity_accepts_complex():
    np.testing.assert_allclose(Velocity(1 + 0.5j).at(FREQS)[0], 1 + 0.5j)


def test_velocity_rejects_a_spectrum_and_says_why():
    """Excitation is linear: solve at unit amplitude and scale the results."""
    with pytest.raises(ValueError, match="scaled_by"):
        Velocity(np.array([1.0, 2.0, 3.0]))


def test_velocity_rejects_a_length_one_array_too():
    """A length-1 array still carries no way to know the model's band length
    at construction time, so it is treated as a spectrum, not a scalar."""
    with pytest.raises(ValueError, match="scaled_by"):
        Velocity(np.array([1.0]))


def test_pressure_defaults_to_sound_soft():
    assert Pressure().at(FREQS)[0] == 0


def test_admittance_accepts_a_spectrum():
    y = Admittance(np.array([0.1, 0.2, 0.4]))
    np.testing.assert_allclose(y.at(FREQS), [0.1, 0.2, 0.4])


def test_admittance_accepts_a_callable():
    y = Admittance(lambda f: f / 1000.0)
    np.testing.assert_allclose(y.at(FREQS), [0.1, 0.2, 0.4])


def test_admittance_spectrum_length_must_match():
    with pytest.raises(ValueError, match="3 frequencies"):
        Admittance(np.array([0.1, 0.2])).at(FREQS)


def test_admittance_callable_wrong_length_fails_clearly():
    """A callable ignoring its frequency argument must not silently
    broadcast into nonsense — it should fail the same shape check as an
    array."""
    with pytest.raises(ValueError, match="3 frequencies"):
        Admittance(lambda f: np.array([0.1, 0.2])).at(FREQS)


def test_admittance_rejects_si_units():
    """NumCalc wants Y/(rho0*c); an SI value is wrong by ~410 and looks fine."""
    with pytest.raises(ValueError, match="normalised"):
        Admittance(408.0)


def test_impedance_rejects_si_units():
    with pytest.raises(ValueError, match="normalised"):
        Impedance(400.0 + 120.0j)


def test_admittance_callable_si_units_caught_at_at_time():
    """_check_magnitude can't inspect a callable's output before it has been
    evaluated against real frequencies, so the SI guard for a callable
    admittance is enforced when at() actually calls it — it must not be a
    silent bypass of the guard."""
    y = Admittance(lambda f: 408.0 * np.ones_like(f))
    with pytest.raises(ValueError, match="normalised"):
        y.at(FREQS)


def test_is_frequency_dependent():
    assert Admittance(np.array([0.1, 0.2, 0.4])).is_frequency_dependent
    assert not Admittance(0.4).is_frequency_dependent
    assert not Velocity(1.0).is_frequency_dependent
