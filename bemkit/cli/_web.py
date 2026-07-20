"""`bem MODEL export-web` — write an auralizer web bundle."""


def register(sub):
    p = sub.add_parser(
        "export-web",
        help="Export solved variants as a web auralizer bundle")
    p.add_argument("--out", required=True, metavar="DIR",
                   help="Output bundle directory")
    from bemkit.cli import VariantAction
    p.add_argument("--variant", nargs="+", metavar="NAME", default=None,
                   action=VariantAction,
                   help="Variants to export (default: all with results)")
    p.add_argument("--directions", default="g30_1m", metavar="SET",
                   help="Spherical fieldpoint set for auralization")
    p.add_argument("--planes", default="planeXZ", metavar="SET,SET",
                   help="Comma-separated plane sets (default: planeXZ)")
    p.add_argument("--front", default=None, metavar="X,Y,Z",
                   help="Model front axis (default: auto from 'onaxis' set)")
    p.add_argument("--ir-fs", type=int, default=48000, dest="ir_fs",
                   help="IR sample rate (default: 48000)")
    p.add_argument("--db-range", type=float, default=60.0, dest="db_range",
                   help="dB window for surface/plane maps (default: 60)")
    p.add_argument("--floor-db", type=float, default=-40.0, dest="floor_db",
                   help="Regularization floor for relative EQ (default: -40)")
    p.add_argument("--delay-margin-ms", type=float, default=0.5,
                   dest="delay_margin_ms",
                   help="Causality margin kept when removing the bulk "
                        "propagation delay (default: 0.5 ms)")
    p.add_argument("--check", action="store_true",
                   help="Print quantization/aliasing diagnostics")
    p.set_defaults(func=cmd_export_web)


def cmd_export_web(model_dir, args):
    from bemkit.web import export_web
    front = None
    if args.front:
        front = [float(x) for x in args.front.split(",")]
    export_web(
        model_dir, args.out,
        variants=args.variant,
        directions=args.directions,
        planes=[p for p in args.planes.split(",") if p],
        front=front,
        ir_fs=args.ir_fs,
        db_range=args.db_range,
        floor_db=args.floor_db,
        delay_margin_ms=args.delay_margin_ms,
        check=args.check,
    )
    return 0
