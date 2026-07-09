"""Allow `python -m pynumcalc.cli` as an uninstalled alternative to `bem`."""

import sys

from pynumcalc.cli import main

sys.exit(main())
