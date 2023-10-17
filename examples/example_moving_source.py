import numpy as np
from pathlib import Path

from aspsim.simulator import SimulatorSetup
from aspsim.processor import AudioProcessor
import aspsim.signal.sources as src
import aspsim.diagnostics.diagnostics as dg
import aspsim.room.trajectory as traj

# Choose where figures should be saved and create a SimulatorSetup object
fig_path = Path(__file__).parent.joinpath("figs")
fig_path.mkdir(exist_ok=True)
setup = SimulatorSetup(fig_path)

# Adjust config values
setup.sim_info.tot_samples = 2000
setup.sim_info.samplerate = 1000
setup.sim_info.rt60 = 0.6
setup.sim_info.max_room_ir_length = 1000
setup.sim_info.array_update_freq = 1

setup.sim_info.export_frequency = 2000
setup.sim_info.plot_output = "pdf"

#Setup sources and microphones
sound_src = src.WhiteNoiseSource(1,1)
setup.add_free_source("ls", traj.LinearTrajectory([[1,0,0], [1,1,0], [0,1,0]], 10, setup.sim_info.samplerate), sound_src)
setup.add_mics("mic", np.array([[0,0,0]]))
sim = setup.create_simulator()

# Choose which signals should be saved to files
sim.diag.add_diagnostic("loudspeaker_signal", dg.RecordSignal("ls", sim.sim_info))
sim.diag.add_diagnostic("microphone_signal", dg.RecordSignal("mic", sim.sim_info))

sim.run_simulation()