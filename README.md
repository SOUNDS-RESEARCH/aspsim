# ASPSIM : Audio signal processing simulator

## Introduction
The package simulates sound in a reverberant space. The simulator supports moving microphones and moving sources. The simulator supports adaptive processing, where the sound of the sources are determined by the sound in the room. This is useful for instance for active noise control and similar adaptive processing tasks. 

The package currently uses the image-source method implementation of [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics) to generate the room impulse responses. 


## Installation
Requires python version that satisfies 3.8 <= version <= 3.11, as those are supported by Numba.
### Quick installation
1. Install ASPCORE by following the instructions at https://github.com/SOUNDS-RESEARCH/aspcore
2. Navigate into an appropriate folder and run
```
git clone https://github.com/SOUNDS-RESEARCH/aspsim.git
pip install ./aspsim
pip install -r aspsim/requirements.txt
```

### Installing the package
The package can be installed by opening your favorite terminal and writing (changing out cd for the appropriate change-directory command, and the path to whatever you want)
```
cd c:/folder/parent_folder_to_code
git clone https://github.com/SOUNDS-RESEARCH/aspsim.git
pip install ./aspsim
```

If you intend to work on the development, it might be preferable to get the develop branch directly instead, and to use an editable install. Use the commands
```
cd c:/folder/parent_folder_to_code
git clone --branch develop https://github.com/SOUNDS-RESEARCH/aspsim.git
pip install -e ./aspsim
```

### Dependencies
To install all (except aspcore) dependencies use the command
```
pip install -r requirements.txt
```

It is required to have [aspcore](https://github.com/SOUNDS-RESEARCH/aspcore) installed, which does not exist on PYPI as of yet (i.e. you can not install with pip install the regular way). \
The collections modules, namely diagnostics/diagnosticscollection.py and signal/sourcescollection.py, are allowed to depend on the package [aspcol](https://github.com/SOUNDS-RESEARCH/aspcol). It is only required if you make use of classes or functions from a collections module. 


## Usage

### Example: Minimal setup
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

### Example: Minimal setup with processor
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