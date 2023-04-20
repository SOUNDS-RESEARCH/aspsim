# ASPSIM : Audio / Adaptive Signal Processing SIMulatior

## Introduction
The package is intended to help evaluate and aid development of adaptive audio processing algorithms. 


## Installation

### Required depencies
An attempt has been made to keep the necessary dependencies few. The only required non-standard dependencies are the packages aspcore and pyroomacoustics. Numpy, Scipy and Matplotlib are necessary for basic functionality. The unit tests are relying on hypothesis and pytest. 

### Non-required dependecies
Some extra dependencies can be necessary for particular non-mandatory features. \
To make use of functions for loading audio files, the package soundfile is used. \
To make use of the "tikz" option for plotting, the package tikzplotlib is used. \
The collections modules, namely diagnostics/diagnosticscollection.py and signal/sourcescollection.py are allowed to depend on the package aspcol. 

## Usage

### Example: Minimal setup
**Not currently possible - A processor is currently required**
Here a simulation is run for a single source and microphone with a default config. No data is saved to file.
```
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
```
from aspsim.simulator import SimulatorSetup
from aspsim.processor import AudioProcessor
import signal.sources as src

class DoNothingProcessor(AudioProcessor):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def process(self, num_samples):
        pass

setup = SimulatorSetup()
setup.add_free_source("source", [[1,0,0]], src.WhiteNoiseSource(1,1))
setup.add_mics("mic", [[0,0,0]])

sim = setup.create_simulator()
proc = DoNothingProcessor()
sim.add_processor(proc)
sim.run_simulation()
```


### Audio reproduction example
```
from aspsim.simulator import SimulatorSetup
from aspsim.processor import AudioProcessor
import signal.sources as src





```


## Room impulse responses
Room impulse responses can 
