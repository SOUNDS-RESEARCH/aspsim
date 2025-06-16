import numpy as np
import pathlib

import anc_processors

from aspsim.simulator import SimulatorSetup
from aspsim.processor import AudioProcessor
import aspsim.signal.sources as src
import aspsim.diagnostics.diagnostics as dg
import aspsim.room.generatepoints as gp


def main():
    # Choose where figures should be saved and create a SimulatorSetup object
    fig_path = pathlib.Path(__file__).parent.joinpath("figs")
    fig_path.mkdir(exist_ok=True)
    setup = SimulatorSetup(fig_path)

    # Choose the simulation parameters
    #   These relate to the simulation
    setup.sim_info.tot_samples = 10000
    setup.sim_info.samplerate = 1000
    setup.sim_info.sim_buffer = setup.sim_info.samplerate
    setup.sim_info.sim_chunk_size = 10 * setup.sim_info.samplerate
    setup.sim_info.start_sources_before_0 = True
    setup.sim_info.save_source_contributions = True
    #   These relate to the RIR generation
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 5, 2.5]
    setup.sim_info.room_center = [-1, 0, 0]
    setup.sim_info.rt60 = 0.3
    setup.sim_info.max_room_ir_length = setup.sim_info.samplerate // 2
    setup.sim_info.c = 343.0
    setup.sim_info.extra_delay = 8
    #   These relate to the output files
    setup.sim_info.export_frequency = setup.sim_info.tot_samples # only saves the data at the end of the simulation
    setup.sim_info.plot_output = "pdf"
    setup.sim_info.output_smoothing = 512
    
    # Add a noise source, loudspeakers, error microphones and reference microphones
    # The names of the arrays must correspond to the names used in the processor
    rng = np.random.default_rng(123456)
    num_mics = 2
    num_ls = 3
    setup.add_mics("error", rng.uniform(-0.5, 0.5, (num_mics, 3)))
    setup.add_mics("ref", np.array([[0,0,0]]))
    setup.add_controllable_source("loudspeaker", gp.equidistant_rectangle(num_ls, side_lengths=(2, 2), z = 0))
    setup.add_free_source("noise", np.array([[-3,0,0]]), src.BandlimitedNoiseSource(1,1, (40, 300), setup.sim_info.samplerate, rng=rng))

    setup.arrays.path_type["loudspeaker"]["ref"] = "none" # no feedback from the loudspeakers to the reference microphones
    setup.arrays.path_type["noise"]["ref"] = "direct" #reference signal is the noise signal directly

    sim = setup.create_simulator()

    # Create a FxLMS processor which will generate the anti-noise signal
    block_size = 1
    update_block_size = 512
    control_filt_len = 512
    step_size = 0.2
    beta = 1e-6
    sec_path_ir = sim.arrays.paths["loudspeaker"]["error"]
    proc = anc_processors.FxLMS(sim.sim_info, sim.arrays, block_size, update_block_size, control_filt_len, step_size, beta, sec_path_ir, normalization="sample")
    #proc = anc_processors.FastBlockFxLMS(sim.sim_info, sim.arrays, block_size, control_filt_len, step_size, beta, sec_path_ir)
    sim.add_processor(proc)

    # Choose which signals and diagnostics to save 
    anc_processors.add_anc_diagnostics(sim)
    sim.diag.add_diagnostic("loudspeaker_signal", dg.RecordSignal("loudspeaker", sim.sim_info, num_channels=num_ls))
    sim.diag.add_diagnostic("microphone_signal", dg.RecordSignal("error", sim.sim_info, num_channels = num_mics))

    # Run the simulation
    sim.run_simulation()



if __name__ == "__main__":
    main()