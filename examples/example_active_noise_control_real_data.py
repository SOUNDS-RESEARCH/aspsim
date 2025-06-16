""" Active noise control using real data as impulse responses. 

The IRs from the secondary sources are from the dataset, while the primary source uses the image-source method. 

The example requires the MeshRIR dataset, which can be downloaded from sh01.org/MeshRIR.
The example also uses the samplerate package, which can be installed via pip.
"""
import numpy as np
import pathlib

import anc_processors

from aspsim.simulator import SimulatorSetup
from aspsim.processor import AudioProcessor
import aspsim.signal.sources as src
import aspsim.diagnostics.diagnostics as dg
import aspsim.room.generatepoints as gp

import samplerate

# Specify the path to the MeshRIR dataset folder S32-M441_npy, which should contain the irutilities.py module.
MESHRIR_DATASET_PATH = pathlib.Path("c:/research/datasets/S32-M441_npy")
import sys
sys.path.append(str(MESHRIR_DATASET_PATH))

import irutilities


def resample_multichannel(ir, ratio):
    #if ir.ndim == 3:
    #    assert ir.shape[0] == 1
    #    ir = ir[0,...]
    print(f"Resampling IR from {ir.shape[-1]} to {int(ir.shape[-1] * ratio)} samples")
    ir_out = []
    for ls_idx in range(ir.shape[0]):
        ir_out.append([])
        for mic_idx in range(ir.shape[1]):
            #print(f"Resampling IR for loudspeaker {ls_idx}, microphone {mic_idx}")
            ir_out[ls_idx].append(samplerate.resample(ir[ls_idx, mic_idx,...], ratio, "sinc_fastest"))
        ir_out[ls_idx] = np.stack(ir_out[ls_idx], axis=0)
    ir_out = np.stack(ir_out, axis=0)
    return ir_out.astype(float)



def main():
    # Choose where figures should be saved and create a SimulatorSetup object
    fig_path = pathlib.Path(__file__).parent.joinpath("figs")
    fig_path.mkdir(exist_ok=True)
    setup = SimulatorSetup(fig_path)
    rng = np.random.default_rng(123456)  # For reproducibility

    # Choose the simulation parameters
    #   These relate to the simulation
    setup.sim_info.tot_samples = 10000
    setup.sim_info.samplerate = 1000
    setup.sim_info.sim_buffer = setup.sim_info.samplerate
    setup.sim_info.sim_chunk_size = 10 * setup.sim_info.samplerate
    setup.sim_info.start_sources_before_0 = True
    setup.sim_info.save_source_contributions = True
    #   These are the only RIR-related parameters that should be set for real data
    setup.sim_info.max_room_ir_length = setup.sim_info.samplerate // 2
    setup.sim_info.c = 343.0
    setup.sim_info.extra_delay = 8
    #   These relate to the output files
    setup.sim_info.export_frequency = setup.sim_info.tot_samples # only saves the data at the end of the simulation
    setup.sim_info.plot_output = "pdf"
    setup.sim_info.output_smoothing = 512
    
    # Add a noise source, loudspeakers, error microphones and reference microphones
    # The names of the arrays must correspond to the names used in the processor
    pos_eval, pos_src, ir = irutilities.loadIR(MESHRIR_DATASET_PATH)
    num_eval = pos_eval.shape[0]

    num_mic = 5
    mic_idxs = rng.choice(num_eval, size=num_mic, replace=False)
    pos_mic = pos_eval[mic_idxs, :]

    num_ls = 3
    ls_idxs = rng.choice(pos_src.shape[0], size=num_ls, replace=False)
    pos_ls = pos_src[ls_idxs, :]

    ir_eval = ir[ls_idxs, ...]
    ir_eval = resample_multichannel(ir_eval, setup.sim_info.samplerate / 48000)
    ir_eval = ir_eval[..., int(0.1 * setup.sim_info.samplerate):]  # Remove the first 100 ms of IR since it is mostly silence
    ir_eval = ir_eval[..., :setup.sim_info.max_room_ir_length]  # Make the IR length correspond to the settings chosen earlier
    ir_mic = ir_eval[:, mic_idxs, :]

    setup.add_mics("error", pos_mic)
    setup.add_mics("eval", pos_eval)
    setup.add_mics("ref", np.array([[0,0,0]]))
    setup.add_controllable_source("loudspeaker", pos_ls)
    setup.add_free_source("noise", np.array([[-3,0,0]]), src.BandlimitedNoiseSource(1,1, (40, 300), setup.sim_info.samplerate, rng=rng))

    setup.set_path("loudspeaker", "error", ir_mic) # use real IRs
    setup.set_path("loudspeaker", "eval", ir_eval) # use real IRs
    setup.arrays.path_type["loudspeaker"]["ref"] = "none" # no feedback from the loudspeakers to the reference microphones
    setup.arrays.path_type["noise"]["ref"] = "direct" #reference signal is the noise signal directly

    sim = setup.create_simulator()

    # Create a FxLMS processor which will generate the anti-noise signal
    block_size = 1
    update_block_size = 512
    control_filt_len = 512
    step_size = 0.05
    beta = 1e-6
    sec_path_ir = sim.arrays.paths["loudspeaker"]["error"]
    #proc = anc_processors.FxLMS(sim.sim_info, sim.arrays, block_size, update_block_size, control_filt_len, step_size, beta, sec_path_ir, normalization="sample")
    proc = anc_processors.FastBlockFxLMS(sim.sim_info, sim.arrays, block_size, control_filt_len, step_size, beta, sec_path_ir)
    sim.add_processor(proc)

    # Choose which signals and diagnostics to save 
    anc_processors.add_anc_diagnostics(sim)

    # Run the simulation
    sim.run_simulation()



if __name__ == "__main__":
    main()