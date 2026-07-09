# Contributing

Bug reports and pull requests are welcome.

All contributions must be signed off (Developer Certificate of Origin,
https://developercertificate.org/): commit with `git commit -s`. This
certifies you have the right to submit the code under the GPL-3.0 license and
keeps relicensing decisions possible for the project.

Run the tests with `pytest` before submitting. The layer rules are enforced
by `tests/test_boundaries.py`: `config`/`meshing`/`sim` never import
`analysis`/`viz`; glue modules only lazily.
