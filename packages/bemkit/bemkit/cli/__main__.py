"""Allow `python -m bemkit.cli` as an uninstalled alternative to `bem`."""

import sys

from bemkit.cli import main

sys.exit(main())
