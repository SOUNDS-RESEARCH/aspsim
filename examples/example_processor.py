"""Example custom processor."""

import numpy as np

import aspsim.diagnostics.diagnostics as dg
import aspsim.signal.sources as src
from aspsim.processor import AudioProcessor
from aspsim.simulator import SimulatorSetup


class PlaySoundProcessor(AudioProcessor):
    """Processor that plays a source signal."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.src = src.WhiteNoiseSource(1, 1)

    def process(self, num_samples):
        """Process a block of samples."""
        # this can be replaced with any other source or function, as long as some values
        # are added to the buffer between self.sig.idx and self.sig.idx+num_samples
        self.sig["ls"][:, self.sig.idx : self.sig.idx + num_samples] = (
            self.src.get_samples(num_samples)
        )


setup = SimulatorSetup()
setup.add_controllable_source("ls", np.array([[1, 0, 0]]))
setup.add_mics("mic", np.array([[0, 0, 0]]))
sim = setup.create_simulator()

block_size = 64
proc = PlaySoundProcessor(sim.sim_info, sim.arrays, block_size)
sim.add_processor(proc)

sim.diag.add_diagnostic("loudspeaker_signal", dg.RecordSignal("ls", sim.sim_info))
sim.diag.add_diagnostic("microphone_signal", dg.RecordSignal("mic", sim.sim_info))

sim.run_simulation()
