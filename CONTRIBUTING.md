# Contributing

Bug reports and pull requests are welcome.

All contributions must be signed off (Developer Certificate of Origin,
https://developercertificate.org/): commit with `git commit -s`. This
certifies you have the right to submit the code under the GPL-3.0 license and
keeps relicensing decisions possible for the project.

Run the tests with `pytest` before submitting. The layer rules are enforced
by `tests/test_boundaries.py`: `config`/`meshing`/`sim` never import
`analysis`/`viz`; glue modules only lazily.

Changing `bemkit/web.py` output? The bundle format is a cross-repo contract:
the bemkit-auralizer repo pins a golden manifest (real `bem export-web`
output) in `packages/bundle/test-fixtures/small-ls.manifest.json` and
documents the format in its `docs/bundle-format.md`. An intentional format
change means bumping `FORMAT_VERSION` on both sides and refreshing that
golden fixture from a fresh export.
