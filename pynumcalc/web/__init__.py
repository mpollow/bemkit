"""Web bundle export for the BEM auralizer.

Converts solved variants into the static "auralizer-bundle" format consumed
by the 2026_cc_BEM-auralizer web framework. The format contract lives in
that repo: docs/bundle-format.md (formatVersion 1). All signal processing
(normalization, EQ derivation, iFFT to 48 kHz, quantization) happens here —
the browser only dequantizes and plays.
"""

from pynumcalc.web.export import export_web

__all__ = ["export_web"]
