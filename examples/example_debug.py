import numpy as np
from pathlib import Path

from aspsim.simulator import SimulatorSetup
from aspsim.processor import DebugProcessor
import aspsim.signal.sources as src
import aspsim.diagnostics.diagnostics as dg

bs = 4

setup = SimulatorSetup(Path(__file__).parent.joinpath("figs"))
setup.sim_info.tot_samples = 20
setup.sim_info.sim_buffer = 20
setup.sim_info.export_frequency = 6
setup.sim_info.sim_chunk_size = 20

setup.add_free_source("src", np.array([[1,0,0]]), src.WhiteNoiseSource(1,1))
setup.add_controllable_source("loudspeaker", np.array([[1,0,0]]))
setup.add_mics("mic", np.array([[0,0,0]]))

setup.arrays.path_type["loudspeaker"]["mic"] = "none"
setup.arrays.path_type["src"]["mic"] = "identity"

sim = setup.create_simulator()
proc = DebugProcessor(sim.sim_info, sim.arrays, bs)
#sim.diag.add_diagnostic("loudspeaker_signal", dg.RecordSignal("loudspeaker", sim.sim_info, bs, export_func="npz"))
sim.diag.add_diagnostic("mic", dg.RecordSignal("mic", sim.sim_info, bs, export_func="npz"))

sim.add_processor(proc)
sim.run_simulation()
signal_log = np.load(sim.folder_path.joinpath(f"mic_{sim.sim_info.tot_samples}.npz"))["mic"]
l = list(sim.folder_path.iterdir())
signal_true = sim.processors[0].mic
print (np.allclose(signal_log, signal_true))