"""Analysis layer: NumPy results interface, normalization, analytic reference.

Everything that interprets NumCalc output physically lives here, on top of
the raw I/O in pynumcalc.sim — nothing below it imports it.
"""

from pynumcalc.analysis.normalization import normalize_pressure
from pynumcalc.analysis.result import BEMResult, load_result
from pynumcalc.analysis.simulation import SimulationData

# sphcap_pressure (analytic reference solver) is intentionally not re-exported:
# it's a standalone validation utility — import it explicitly from
# pynumcalc.analysis.analytic when needed.
