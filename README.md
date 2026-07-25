# bemkit

Python pipeline for [Mesh2HRTF NumCalc](https://github.com/Any2HRTF/Mesh2HRTF)
BEM simulations, aimed at audio acoustics (loudspeakers, HRTFs, microphone
arrays): mesh generation (gmsh), NC.inp export, parallel/resumable runs,
result loading with physical normalization, an analytic spherical-cap
reference solver for validation, 2D/3D visualization, and a web-bundle
exporter for the `bemkit-auralizer` browser player (`bem MODEL export-web`).

## Install

    pip install -e ".[all]"          # or pick extras: viz, viz3d, meshing, analytic

NumCalc itself is a separate program (part of Mesh2HRTF, EUPL-licensed);
`bemkit` writes its inputs, runs it as a subprocess, and reads its outputs.

## Quickstart

    bem examples/sphcap mesh         # STEP -> triangle mesh (needs gmsh)
    bem examples/sphcap export       # fieldpoints + mesh + NC.inp
    bem examples/sphcap run          # run NumCalc (resumable; -j N for parallel)
    bem examples/sphcap run --preview 8   # or solve 8 bins first for a fast look
    bem examples/sphcap plot         # frequency response + impulse response
    bem examples/sphcap              # status

MODEL_DIR defaults to the current directory, so from inside a model
directory the above can be shortened to `bem mesh`, `bem export`, `bem`, etc.
(a model directory named the same as a subcommand, e.g. `mesh/`, needs a
path prefix to disambiguate: `bem ./mesh`).

## Validation

`bemkit.analysis.analytic` computes the exact solution for a vibrating
spherical cap; the test suite compares BEM output against it. The
`examples/sphcap` model is that validation case.

## Extending the CLI

Installed packages can add `bem` subcommands via the `bemkit.cli`
entry-point group; each entry point is a `register(subparsers)` callable.

## Contributing

See CONTRIBUTING.md — sign-off (DCO) required.

## License

GPL-3.0-only. See LICENSE.
