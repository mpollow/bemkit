"""`bem metric` — standalone mmgs .sol metric generation (no model dir)."""


def main(argv=None):
    from pynumcalc.meshing.metric import main as metric_main
    return metric_main(argv)
