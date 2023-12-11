import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import copy

from aspsim.simulator import SimulatorSetup
import aspsim.signal.sources as src
import aspsim.diagnostics.diagnostics as dg
import aspsim.room.generatepoints as gp

REVERB = True

# Choose where figures should be saved and create a SimulatorSetup object
fig_path = Path(__file__).parent.joinpath("figs")
fig_path.mkdir(exist_ok=True)
setup = SimulatorSetup(fig_path)

# Adjust config values
setup.sim_info.tot_samples = 5000
setup.sim_info.samplerate = 1000
setup.sim_info.max_room_ir_length = 1000
setup.sim_info.array_update_freq = 1
setup.sim_info.save_source_contributions = True
setup.sim_info.export_frequency = setup.sim_info.tot_samples
setup.sim_info.plot_output = "pdf"
if REVERB:
    setup.sim_info.rt60 = 0.25
else:
    setup.sim_info.rt60 = 0.0

#Setup sources and microphones
num_src = 50
source_angles = np.linspace(0, 2*np.pi, num_src)
(x, y) = gp.pol2cart(0.5*np.ones(num_src), source_angles)
pos_src = np.zeros((num_src, 3))
pos_src[:,0] = x
pos_src[:,1] = y

directivity_dir = np.array([[0,1,0]])
sound_src = src.WhiteNoiseSource(1, 1)
for i in range(num_src):
    setup.add_free_source(f"ls_{i}", pos_src[i:i+1,:], copy.deepcopy(sound_src))
setup.add_mics("mic", np.array([[0,0,0]]), directivity_type=["cardioid"], directivity_dir=directivity_dir)
sim = setup.create_simulator()

# Choose which signals should be saved to files
for i in range(num_src):
    sim.diag.add_diagnostic(f"recorded_ls_sig_{i}", dg.RecordSignal(f"ls_{i}~mic", sim.sim_info, export_func="npz"))
sim.diag.add_diagnostic("microphone_signal", dg.RecordSignal("mic", sim.sim_info, export_func="npz"))

sim.run_simulation()

# Calculate average power from each loudspeaker recorded by the microphone
recorded_power = []
for i in range(num_src):
    rec_sig = np.load(sim.folder_path.joinpath(f"recorded_ls_sig_{i}_{sim.sim_info.tot_samples}.npz"))
    rec_sig = rec_sig[f"recorded_ls_sig_{i}"]
    recorded_power.append(np.mean(rec_sig**2))
recorded_power = recorded_power / np.max(recorded_power)

# Calculate what the true power distribution should be from a cardioid microphone
num_true_cardioid_angles = 300
true_cardioid_angles = np.linspace(0, 2*np.pi, num_true_cardioid_angles)
(x, y) = gp.pol2cart(np.ones(num_true_cardioid_angles), true_cardioid_angles)
pos_true_cardioid = np.zeros((num_true_cardioid_angles, 3))
pos_true_cardioid[:,0] = x
pos_true_cardioid[:,1] = y
true_cardoid_power = (0.5 + 0.5 * np.sum(directivity_dir * pos_true_cardioid, axis=-1))**2

# Plot the results
fig, ax = plt.subplots(1,1, figsize=(8,6), subplot_kw={'projection': 'polar'})
ax.plot(true_cardioid_angles, true_cardoid_power, label="True cardioid power", alpha=0.4, linewidth=5)
ax.plot(source_angles, recorded_power, "-o", label="Recorded power", alpha=0.9)
ax.legend()
plt.show()
