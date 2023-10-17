import numpy as np
import pathlib
from aspsim.simulator import SimulatorSetup
import aspsim.signal.sources as src
import aspsim.diagnostics.diagnostics as dg

# Choose where figures should be saved and create a SimulatorSetup object
fig_path = pathlib.Path(__file__).parent.joinpath("figs")
fig_path.mkdir(exist_ok=True)
setup = SimulatorSetup()

# Setup sources and microphones
setup.add_free_source("source", np.array([[1,0,0]]), src.WhiteNoiseSource(1,1))
setup.add_mics("mic", np.array([[0,0,0]]))
sim = setup.create_simulator()

# Choose to save the microphone signal and run simulation
sim.diag.add_diagnostic("mic", dg.RecordSignal("mic", sim.sim_info))
sim.run_simulation()