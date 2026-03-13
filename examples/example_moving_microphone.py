import numpy as np
from pathlib import Path
import json

from aspsim.simulator import SimulatorSetup
from aspsim.processor import AudioProcessor
import aspsim.signal.sources as src
import aspsim.diagnostics.diagnostics as dg
import aspsim.room.trajectory as traj
import aspsim.signal.sources as sources
import aspsim.room.region as reg
import aspcore.pseq as pseq

import exp_funcs_ideal_sampling as exis


RT60 = 0.2
RIRLEN = 1000
SAMPLERATE = 2000

def main():
    rirs, sig = native_moving_mic()
    rirs_verified, sig_verified = verified_scripts()

    print("MSE RIR: ", np.mean((rirs - rirs_verified)**2))
    print("MSE signal: ", np.mean((sig - sig_verified)**2))

def native_moving_mic():
    # Choose where figures should be saved and create a SimulatorSetup object
    fig_path = Path(__file__).parent.joinpath("figs")
    fig_path.mkdir(exist_ok=True)
    setup = SimulatorSetup(fig_path)

    # Adjust config values
    initial_delay = RIRLEN
    post_delay = 0
    setup.sim_info.tot_samples = initial_delay + RIRLEN + post_delay
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [5.4, 4.3, 3.2]
    setup.sim_info.room_center = [0.8, 0.2, 0.1]
    setup.sim_info.rt60 = RT60
    setup.sim_info.max_room_ir_length = RIRLEN
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = RIRLEN
    setup.sim_info.extra_delay = 40
    setup.sim_info.plot_output = "pdf"
    setup.sim_info.start_sources_before_0 = True
    setup.sim_info.save_source_contributions = True
    setup.sim_info.highpass_cutoff = 0


    #Setup sources and microphones
    sound_src = src.WhiteNoiseSource(1,1)
    setup.add_free_source("ls", traj.LinearTrajectory([[1,0,0], [1,1,0], [0,1,0]], 10, setup.sim_info.samplerate), sound_src)
    setup.add_mics("mic", np.array([[0,0,0]]))
    sim = setup.create_simulator()

    # Choose which signals should be saved to files
    sim.diag.add_diagnostic("loudspeaker_signal", dg.RecordSignal("ls", sim.sim_info))
    sim.diag.add_diagnostic("microphone_signal", dg.RecordSignal("mic", sim.sim_info))

    sim.run_simulation()

    signal_paths = exis.get_signal_paths(sim.folder_path)
    sig = exis.load_npz(signal_paths)

    rirs = sim.arrays.paths["ls"]["mic"]
    return rirs, sig["ls"]

def verified_scripts():
    fig_folder = exis.generate_signals_3d()
    sig, sim_info, arrays, pos_dyn, seq_len, extra_params = exis.load_session(fig_folder)

    return arrays.paths["src"]["mic_dynamic"], sig["mic_dynamic"][0,...]





def generate_signals_3d():
    side_len = 1 #0.75
    height = 0.25
    seq_len = RIRLEN

    center = np.zeros((1,3))

    pos_src = np.array([[2,0,0]])

    setup = SimulatorSetup()
    setup.sim_info.samplerate = SAMPLERATE

    speed_factor = 0.5
    tot_trajectory_samples = 32 * seq_len
    freq_factors = np.array([[1.8, 3.8, 2.1]])
    traj_amp = np.array([[side_len/2, side_len/2, height/2]])
    trajectory = exis.LissajousTrajectoryConstantSpeed(traj_amp, speed_factor * freq_factors / SAMPLERATE, center, SAMPLERATE, speed_factor, tot_trajectory_samples)
    traj_pos = np.concatenate([trajectory.current_pos(t) for t in range(tot_trajectory_samples)], axis=0)

    speed = np.linalg.norm(traj_pos[1:,:] - traj_pos[:-1,:], axis=-1) * SAMPLERATE
    pos_mic = traj_pos[seq_len//2::seq_len,:]


    initial_delay = seq_len
    post_delay = 0
    setup.sim_info.tot_samples = initial_delay + seq_len + post_delay
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [5.4, 4.3, 3.2]
    setup.sim_info.room_center = [0.8, 0.2, 0.1]
    setup.sim_info.rt60 = RT60
    setup.sim_info.max_room_ir_length = seq_len
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = seq_len
    setup.sim_info.extra_delay = 40
    setup.sim_info.plot_output = "pdf"
    setup.sim_info.start_sources_before_0 = True
    setup.sim_info.save_source_contributions = True
    setup.sim_info.highpass_cutoff = 0

    sequence = pseq.create_pseq(seq_len)
    sequence_src = sources.Sequence(sequence)

    setup.add_mics("mic", pos_mic)
    setup.add_free_source("src", pos_src, sequence_src)
    setup.add_mics("mic_dynamic", traj_pos)

    sim = setup.create_simulator()

    exis.run_and_save(sim)
    with open(sim.folder_path.joinpath("extra_parameters.json"), "w") as f:
        json.dump({"seq_len" : seq_len, 
                    "initial_delay" : initial_delay,
                   "post_delay" : post_delay,
                   "max_sweep_freq" : SAMPLERATE // 2,
                    "center" : center.tolist(), 
                    "downsampling_factor" : 1,
                    "freq_factors" : freq_factors.tolist(),
                    "speed_factor" : speed_factor,
                    "speed min" : np.min(speed),
                    "speed max" : np.max(speed),
                    "speed mean" : np.mean(speed),
                    } ,f)
    return sim.folder_path


if __name__ == "__main__":
    main()