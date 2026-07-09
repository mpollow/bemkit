"""The restructured export must reproduce the pre-restructure outputs
captured in tests/golden/ (Task 1), byte-identical modulo the date stamp."""

import re
import shutil
from pathlib import Path

from pynumcalc.config import ModelConfig
from pynumcalc.sim.export import export_variant

GOLDEN = Path(__file__).parent / "golden"


def _strip_date(text):
    return re.sub(r"## Date: .*", "## Date: X", text)


def test_export_matches_pre_restructure_golden(tmp_path):
    model = tmp_path / "model"
    shutil.copytree(GOLDEN / "model", model)
    config = ModelConfig.from_dir(model)
    export_variant(config, "tetra", verbose=False)

    nc = model / "numcalc" / "tetra"
    expected = GOLDEN / "expected"
    for rel in ("ObjectMesh/Nodes.txt", "ObjectMesh/Elements.txt",
                "EvalGrid/Nodes.txt", "EvalGrid/Elements.txt",
                "eval_offsets.json"):
        assert (nc / rel).read_text() == (expected / rel).read_text(), rel
    assert _strip_date((nc / "NC.inp").read_text()) == \
        _strip_date((expected / "NC.inp").read_text())
