import numpy as np
import pytest

from aspsim.simulator import SimulatorSetup
import aspsim.diagnostics.diagnostics as dia
import aspsim.signal.sources as sources
import aspsim.signal.filterclasses as fc

@pytest.fixture(scope="session")
def fig_folder(tmp_path_factory):
    return tmp_path_factory.mktemp("figs")


def _setup_ism(fig_folder, samplerate):
    setup = SimulatorSetup(fig_folder)
    setup.sim_info.samplerate = samplerate
    setup.sim_info.tot_samples =  samplerate
    setup.sim_info.sim_chunk_size = 2*samplerate
    setup.sim_info.sim_buffer = samplerate
    setup.sim_info.export_frequency =  samplerate
    setup.sim_info.save_source_contributions = False
    setup.sim_info.randomized_ism = False
    setup.sim_info.start_sources_before_0 = False

    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 5, 5]
    setup.sim_info.room_center = [0, 0, 0]
    setup.sim_info.rt60 = 0.25
    setup.sim_info.max_room_ir_length = samplerate // 2
    return setup

def test_generated_rirs_for_different_positions_are_not_identical(fig_folder):
    sr = 1000
    rng = np.random.default_rng()

    pos_mic = rng.uniform(-1, 1, size=(2,3)) 
    pos_src = np.array([[0,0,0]])

    sim_setup = _setup_ism(fig_folder, sr)
    sim_setup.add_mics("mic", pos_mic)
    sim_setup.add_controllable_source("src", pos_src)
    sim = sim_setup.create_simulator()
    rir1 = sim.arrays.paths["src"]["mic"][0,0,:]
    rir2 = sim.arrays.paths["src"]["mic"][0,1,:]

    assert not np.allclose(rir1, rir2)

def test_generated_rirs_for_same_position_is_consistent_with_more_than_one_simulator(fig_folder):
    sr = 1000
    rng = np.random.default_rng()

    pos_mic = rng.uniform(-1, 1, size=(1,3))
    pos_src = rng.uniform(-1, 1, size=(1,3))

    sim_setup = _setup_ism(fig_folder, sr)
    sim_setup.add_mics("mic", pos_mic)
    sim_setup.add_controllable_source("src", pos_src)
    sim = sim_setup.create_simulator()
    rir1 = sim.arrays.paths["src"]["mic"]

    sim_setup = _setup_ism(fig_folder, sr)
    sim_setup.add_mics("mic", pos_mic)
    sim_setup.add_controllable_source("src", pos_src)
    sim = sim_setup.create_simulator()
    rir2 = sim.arrays.paths["src"]["mic"]
    assert np.allclose(rir1, rir2)

def test_generated_rir_satisfies_reciprocity(fig_folder):
    sr = 1000
    rng = np.random.default_rng()

    pos1 = np.array([[1,0,0]]) #rng.uniform(-1, 1, size=(1,3))
    pos2 = np.array([[0,1,0]]) #¤rng.uniform(-1, 1, size=(1,3))

    sim_setup = _setup_ism(fig_folder, sr)
    sim_setup.add_mics("mic", pos1)
    #sim_setup.add_free_source("src", pos2, sources.WhiteNoiseSource(1, 1, rng))
    sim_setup.add_controllable_source("src", pos2)
    sim = sim_setup.create_simulator()
    rir1 = sim.arrays.paths["src"]["mic"]

    sim_setup = _setup_ism(fig_folder, sr)
    sim_setup.add_mics("mic", pos2)
    sim_setup.add_controllable_source("src", pos1)
    sim = sim_setup.create_simulator()
    rir2 = sim.arrays.paths["src"]["mic"]

    assert np.allclose(rir1, rir2)






def test_simulation_equals_direct_convolution(fig_folder):
    rng = np.random.default_rng()
    sr = 1000

    setup = SimulatorSetup(fig_folder)
    setup.sim_info.samplerate = sr

    setup.sim_info.tot_samples = sr
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 4, 2]
    setup.sim_info.room_center = [2, 0.1, 0.1]
    setup.sim_info.rt60 =  0.25
    setup.sim_info.max_room_ir_length = sr // 2
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // 2
    setup.sim_info.extra_delay = 60
    setup.sim_info.plot_output = "pdf"

    src_sig = rng.normal(size=(1, setup.sim_info.tot_samples))

    setup.add_mics("mic", np.zeros((1,3)))
    setup.add_free_source("src", np.array([[1,0,0]]), sources.Sequence(src_sig))

    sim = setup.create_simulator()
    sim.diag.add_diagnostic("mic", dia.RecordSignal("mic", sim.sim_info, 1, export_func="npz"))
    sim.run_simulation()

    sig_sim = np.load(sim.folder_path.joinpath(f"mic_{sim.sim_info.tot_samples}.npz"))["mic"]

    filt = fc.create_filter(ir=sim.arrays.paths["src"]["mic"])
    direct_mic_sig = filt.process(src_sig)

    assert np.allclose(sig_sim, direct_mic_sig)