import numpy as np
from aspsim.simulator import SimulatorSetup
from aspsim.processor import AudioProcessor
import aspsim.signal.sources as src
import aspsim.diagnostics.diagnostics as dg

class PlaySoundProcessor(AudioProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.src = src.WhiteNoiseSource(1,1) 
    
    def process(self, num_samples):
        # this can be replaced with any other source or function, as long as some values
        # are added to the buffer between self.sig.idx and self.sig.idx+num_samples
        self.sig["ls"][:, self.sig.idx:self.sig.idx+num_samples] = self.src.get_samples(num_samples) 

setup = SimulatorSetup()
setup.add_controllable_source("ls", np.array([[1,0,0]]))
setup.add_mics("mic", np.array([[0,0,0]]))
sim = setup.create_simulator()

block_size = 64
proc = PlaySoundProcessor(sim.sim_info, sim.arrays, block_size)
sim.add_processor(proc)

sim.diag.add_diagnostic("loudspeaker_signal", dg.RecordSignal("ls", sim.sim_info))
sim.diag.add_diagnostic("microphone_signal", dg.RecordSignal("mic", sim.sim_info))

sim.run_simulation()