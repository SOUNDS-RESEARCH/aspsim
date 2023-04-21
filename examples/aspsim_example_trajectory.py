import numpy as np
from pathlib import Path

from aspsim.simulator import SimulatorSetup
from aspsim.processor import AudioProcessor
import aspsim.signal.sources as src
import aspsim.diagnostics.diagnostics as dg
import aspsim.room.trajectory as traj

class PlaySoundProcessor(AudioProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.src = src.WhiteNoiseSource(1,1)
        self.diag.add_diagnostic("loudspeaker power", dg.SignalPower("ls", self.sim_info, self.block_size))
    
    def process(self, num_samples):
        self.sig["ls"][:, self.idx:self.idx+num_samples] = self.src.get_samples(num_samples)

bs = 64

setup = SimulatorSetup(Path(__file__).parent.joinpath("figs"))
setup.sim_info.samplerate = 1000
setup.sim_info.rt60 = 0.6

setup.sim_info.array_update_freq = 1
setup.add_controllable_source("ls", traj.LinearTrajectory([[1,0,0], [1,1,0], [0,1,0]], 10, setup.sim_info.samplerate))
setup.add_mics("mic", np.array([[0,0,0]]))

sim = setup.create_simulator()
proc = PlaySoundProcessor(sim.sim_info, sim.arrays, bs)
proc.diag.add_diagnostic("loudspeaker signal", dg.RecordSignal("ls", sim.sim_info, bs))

sim.add_processor(proc)
sim.run_simulation()