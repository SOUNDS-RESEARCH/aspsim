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