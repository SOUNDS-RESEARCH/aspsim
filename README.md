# ASPSIM : Audio / Adaptive Signal Processing SIMulator

## Introduction
The package is intended to help evaluate and aid development of adaptive audio processing algorithms. 


## Installation
I recommend an editable install. That basically means that python pretends that the package is installed like any other package, but it's instead just a link to a folder on your computer with the code. Then you can use the package from anywhere at the same time as you are developing it. It can be done by opening your favorite terminal and writing (changing out cd for the appropriate change-directory command, and the path to whatever you want) 
```
cd c:/folder/parent_folder_to_code
git clone https://github.com/SOUNDS-RESEARCH/aspsim.git
pip install -e ./aspsim
```

### Required depencies
The required non-standard dependencies are the packages aspcore and pyroomacoustics. Numpy, Scipy, Matplotlib, Numba and dill are necessary for basic functionality. The unit tests are relying on hypothesis and pytest. 

### Non-required dependecies
Some extra dependencies can be necessary for particular non-mandatory features. \
To make use of functions for loading audio files, the package soundfile is used. \
To make use of the "tikz" option for plotting, the package tikzplotlib is used. \
The collections modules, namely diagnostics/diagnosticscollection.py and signal/sourcescollection.py are allowed to depend on the package aspcol, which is then only required if you make use of classes or functions from a collections module. 

## Usage

### Example: Minimal setup
**Not currently possible - A processor is currently required!**\
Here a simulation is run for a single source and microphone with a default config. No data is saved to file.
```python
from aspsim.simulator import SimulatorSetup
import signal.sources as src

setup = SimulatorSetup()
setup.add_free_source("source", [[1,0,0]], src.WhiteNoiseSource(1,1))
setup.add_mics("mic", [[0,0,0]])

sim = setup.create_simulator()
sim.run_simulation()
```

### Example: Minimal setup - Working in current version
Here a simulation is run for a single source and microphone with a default config. No data is saved to file.
```python
import numpy as np
from aspsim.simulator import SimulatorSetup
from aspsim.processor import AudioProcessor
import aspsim.signal.sources as src

class DoNothingProcessor(AudioProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def process(self, num_samples):
        pass

setup = SimulatorSetup()
setup.add_free_source("source", np.array([[1,0,0]]), src.WhiteNoiseSource(1,1))
setup.add_mics("mic", np.array([[0,0,0]]))

sim = setup.create_simulator()
proc = DoNothingProcessor(sim.sim_info, sim.arrays, 1024)
sim.add_processor(proc)
sim.run_simulation()
```


### Audio reproduction example
```python
import numpy as np
from pathlib import Path

from aspsim.simulator import SimulatorSetup
from aspsim.processor import AudioProcessor
import aspsim.signal.sources as src
import aspsim.diagnostics.diagnostics as dg

class PlaySoundProcessor(AudioProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.src = src.WhiteNoiseSource(1,1)
        self.diag.add_diagnostic("loudspeaker power", dg.SignalPower("ls", self.sim_info, self.block_size))
    
    def process(self, num_samples):
        self.sig["ls"][:, self.idx:self.idx+num_samples] = self.src.get_samples(num_samples)

bs = 64

setup = SimulatorSetup(Path(__file__).parent.joinpath("figs"))
setup.add_controllable_source("ls", np.array([[1,0,0]]))
setup.add_mics("mic", np.array([[0,0,0]]))

sim = setup.create_simulator()
proc = PlaySoundProcessor(sim.sim_info, sim.arrays, bs)
proc.diag.add_diagnostic("loudspeaker signal", dg.RecordSignal("ls", sim.sim_info, bs))

sim.add_processor(proc)
sim.run_simulation()
```