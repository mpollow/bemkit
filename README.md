# bemkit

Python library and `bem` CLI for [Mesh2HRTF NumCalc](https://github.com/Any2HRTF/Mesh2HRTF)
BEM simulations, aimed at audio acoustics (loudspeakers, HRTFs, microphone
arrays). A study is a Python `model.py`, not a config grammar: mesh generation
(gmsh), NumCalc export, parallel/resumable runs, result loading with physical
normalization, an analytic spherical-cap reference solver for validation, a
second in-process solver (bempp) for cross-checking NumCalc, 3D visualization,
and a web-bundle exporter for the `bemkit-auralizer` browser player (`bem
STUDY export-web`, currently being re-wired onto the new spine).

## Install

    pip install -e packages/pynumcalc -e "packages/bemkit[all]"
    #   ...or pick extras: bemkit[view] for the 3D viewers, bemkit[meshing]
    #   for gmsh, bemkit[analytic] for the sphcap closed-form reference

NumCalc itself is a separate program (part of Mesh2HRTF, EUPL-licensed);
`bemkit` writes its inputs, runs it as a subprocess, and reads its outputs.

## Quickstart

    bem models/sphcap check          # mesh + setup diagnostics
    bem models/sphcap solve          # run NumCalc (resumable; -j N for parallel)
    bem models/sphcap solve --preview 8   # or solve 8 bins first for a fast look
    bem models/sphcap results onaxis --freq 1000   # 3D field inspection
    bem models/sphcap              # status

`STUDY_DIR` defaults to the current directory, so from inside a study
directory the above can be shortened to `bem check`, `bem solve`, `bem`, etc.
(a study directory named the same as a subcommand, e.g. `check/`, needs a
path prefix to disambiguate: `bem ./check`). See `HOWTO.md` for the full
walkthrough from a `model.py` you write to solved results, and `CLAUDE.md` for
the architecture.

## Validation

`bemkit.model.analytic` computes the exact solution for a vibrating spherical
cap; the test suite compares both solver backends (NumCalc and bempp) against
it. `models/sphcap` is that validation case.

## Extending the CLI

Installed packages can add `bem` subcommands via the `bemkit.cli`
entry-point group; each entry point is a `register(subparsers)` callable.

## Contributing

See CONTRIBUTING.md — sign-off (DCO) required.

## License

GPL-3.0-only. See LICENSE.
