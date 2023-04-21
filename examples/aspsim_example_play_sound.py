import numpy as np
from aspsim.simulator import SimulatorSetup
from aspsim.processor import AudioProcessor
import aspsim.signal.sources as src

class PlaySoundProcessor(AudioProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.src = src.WhiteNoiseSource(1,1)
    
    def process(self, num_samples):
        self.sig["ls"][:, self.idx:self.idx+num_samples] = self.src.get_samples(num_samples)

setup = SimulatorSetup()
setup.add_controllable_source("ls", np.array([[1,0,0]]))
setup.add_mics("mic", np.array([[0,0,0]]))

sim = setup.create_simulator()
proc = PlaySoundProcessor(sim.sim_info, sim.arrays, 64)
sim.add_processor(proc)
sim.run_simulation()