"""Model -> NumCalc Deck. The only place NumCalc-isms are allowed."""

import numpy as np
import pytest

from bemkit.model import (
    BC, Admittance, Elements, Group, Impedance, Medium, Mesh, Model,
    PlaneWave, PointSource, Pressure, Velocity,
)
from bemkit.solvers.numcalc import model_to_deck

VERT = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
TRI = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
MESH = Mesh(VERT, TRI, groups={"vibrating": np.array([0, 1])})
FREQS = np.array([250.0, 500.0, 750.0])


def _model(**kw):
    kw.setdefault("bcs", [BC(Group("vibrating"), Velocity(1.0))])
    return Model(MESH, FREQS, Medium(), **kw)


def test_frequencies_carry_over_exactly():
    deck = model_to_deck(_model(), n_eval_nodes=1, n_eval_elements=1)
    np.testing.assert_array_equal(deck.frequencies, FREQS)


def test_velocity_becomes_a_velo_range():
    deck = model_to_deck(_model(), n_eval_nodes=1, n_eval_elements=1)
    assert [(b.start, b.end, b.kind) for b in deck.boundary] == [(0, 1, "VELO")]


def test_pressure_becomes_pres():
    m = _model(bcs=[BC(Group("vibrating"), Pressure(0.0))])
    deck = model_to_deck(m, n_eval_nodes=1, n_eval_elements=1)
    assert deck.boundary[0].kind == "PRES"


def test_constant_admittance_needs_no_curves():
    m = _model(bcs=[BC(Group("vibrating"), Admittance(0.4))])
    deck = model_to_deck(m, n_eval_nodes=1, n_eval_elements=1)
    assert deck.curves == []
    assert (deck.boundary[0].curve_real, deck.boundary[0].curve_imag) == (-1, -1)


def test_frequency_dependent_admittance_emits_two_curves():
    """Real and imaginary parts scale independently, and NumCalc refuses a line
    that mixes curve refs with constants."""
    y = np.array([0.1, 0.2, 0.4]) - 0.05j
    m = _model(bcs=[BC(Group("vibrating"), Admittance(y))])
    deck = model_to_deck(m, n_eval_nodes=1, n_eval_elements=1)
    assert [c.id for c in deck.curves] == [1, 2]
    assert deck.curves[0].points == [(250.0, 0.1), (500.0, 0.2), (750.0, 0.4)]
    bc = deck.boundary[0]
    assert (bc.curve_real, bc.curve_imag) == (1, 2)


def test_curve_constants_are_both_one():
    """NumCalc multiplies each part by its own curve, so a 0.0 imaginary
    constant would silently zero the imaginary curve."""
    m = _model(bcs=[BC(Group("vibrating"),
                       Admittance(np.array([0.1, 0.2, 0.4])))])
    deck = model_to_deck(m, n_eval_nodes=1, n_eval_elements=1)
    assert (deck.boundary[0].real, deck.boundary[0].imag) == (1.0, 1.0)


def test_frequency_dependent_impedance_is_rejected():
    """NumCalc hard-exits (exit(-1)) on IMPE with a CURVES reference — this
    must be caught here, not discovered at runtime by the solver binary."""
    m = _model(bcs=[BC(Group("vibrating"),
                       Impedance(np.array([1.0, 1.2, 1.4])))])
    with pytest.raises(ValueError, match="IMPE"):
        model_to_deck(m, n_eval_nodes=1, n_eval_elements=1)


def test_plane_wave_carries_direction():
    m = _model(sources=[PlaneWave((0, 0, 1), amplitude=2.0)])
    deck = model_to_deck(m, n_eval_nodes=1, n_eval_elements=1)
    assert deck.plane_waves[0].vec == (0.0, 0.0, 1.0)
    assert deck.plane_waves[0].real == 2.0


def test_source_numbering_is_per_section_not_global():
    """nr numbers a source within its own PLANE WAVES / POINT SOURCES section
    (NumCalc reads the two as independent lists) — a global index across
    model.sources would number the second source '2' in the wrong section."""
    m = _model(sources=[PointSource((2, 0, 0), amplitude=1.0),
                        PlaneWave((0, 0, 1), amplitude=2.0)])
    deck = model_to_deck(m, n_eval_nodes=1, n_eval_elements=1)
    assert [s.nr for s in deck.plane_waves] == [1]
    assert [s.nr for s in deck.point_sources] == [1]
    assert deck.plane_waves[0].vec == (0.0, 0.0, 1.0)
    assert deck.point_sources[0].vec == (2.0, 0.0, 0.0)


def test_totals_are_object_plus_eval():
    deck = model_to_deck(_model(), n_eval_nodes=3, n_eval_elements=2)
    assert deck.n_total_nodes == 4 + 3
    assert deck.n_total_elements == 4 + 2


def test_out_of_range_bc_is_impossible_by_construction():
    """Selectors resolve against the mesh, so a stale index cannot survive."""
    m = _model(bcs=[BC(Elements([99]), Velocity(1.0))])
    with pytest.raises(ValueError, match="out of range"):
        model_to_deck(m, n_eval_nodes=1, n_eval_elements=1)


def test_duplicate_indices_within_one_bc_collapse_harmlessly():
    """A selector that names the same element twice (e.g. two overlapping
    literal ranges) must not turn into two boundary lines for one element —
    that would make deck.validate() see nothing wrong, but write() would
    still emit a spurious duplicate ELEM line."""
    m = _model(bcs=[BC(Elements([0, 0, 1]), Velocity(1.0))])
    deck = model_to_deck(m, n_eval_nodes=1, n_eval_elements=1)
    assert [(b.start, b.end) for b in deck.boundary] == [(0, 1)]


def test_two_bcs_of_the_same_kind_on_the_same_element_is_rejected():
    """NumCalc reads BOUNDARY lines in file order; a later line for an
    element that already has one silently overwrites the earlier value.
    Two overlapping BCs of the same kind can never be valid input — one of
    them would just vanish."""
    m = _model(bcs=[BC(Elements([0, 1]), Velocity(1.0)),
                    BC(Elements([1, 2]), Velocity(0.5))])
    with pytest.raises(ValueError, match="element 1"):
        model_to_deck(m, n_eval_nodes=1, n_eval_elements=1)


def test_velo_and_admi_may_share_an_element():
    """Unlike two BCs of the same kind, VELO and ADMI together on one element
    is documented, valid NumCalc input."""
    m = _model(bcs=[BC(Elements([0, 1]), Velocity(1.0)),
                    BC(Elements([1, 2]), Admittance(0.4))])
    deck = model_to_deck(m, n_eval_nodes=1, n_eval_elements=1)
    kinds = sorted((b.start, b.end, b.kind) for b in deck.boundary)
    assert kinds == [(0, 1, "VELO"), (1, 2, "ADMI")]


def test_bc_selecting_no_elements_is_rejected():
    """An empty selector match (e.g. a physical group with no members) would
    otherwise silently vanish into a deck with one fewer boundary line than
    the model author intended."""
    m = _model(bcs=[BC(Elements([]), Velocity(1.0))])
    with pytest.raises(ValueError, match="no elements"):
        model_to_deck(m, n_eval_nodes=1, n_eval_elements=1)
