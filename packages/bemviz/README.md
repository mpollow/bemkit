# bemviz

Plotting and web export for [bemkit](../bemkit) results — everything that needs
matplotlib or pyvista, kept out of bemkit so that running simulations and reading
numbers back never drags a display stack in.

Installing it adds `plot`, `geometry`, `results` and `export-web` to the `bem`
CLI automatically, through the `bemkit.cli` entry-point group.

```python
import matplotlib.pyplot as plt
from bemkit import SimulationData
from bemviz import freq_magnitude

sim = SimulationData.from_model("examples/sphcap")
fig, ax = plt.subplots()
freq_magnitude(ax, sim.result("onaxis"))
```
