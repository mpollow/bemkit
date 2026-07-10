#!/usr/bin/env python3
"""Generate fieldpoint .npy files for sphcap."""
from pathlib import Path
import numpy as np
from bemkit import make_fields_onaxis, make_fields_polar, make_fields_plane

fp_dir = Path(__file__).parent / "fieldpoints"
fp_dir.mkdir(exist_ok=True)
np.save(fp_dir / "onaxis.npy",  make_fields_onaxis(dim='z', radii=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0]))
np.save(fp_dir / "polar.npy",   make_fields_polar(dims='zx', res=5))
np.save(fp_dir / "planeXZ.npy", make_fields_plane(dims='xz', limits=[-500, 500, -500, 500], res=10))
