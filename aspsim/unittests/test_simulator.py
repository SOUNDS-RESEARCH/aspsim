import numpy as np
import hypothesis as hyp
#from hypothesis import given
import hypothesis.strategies as st
import pytest
import copy

from aspsim.simulator import SimulatorSetup
import aspsim.array as ar
import aspsim.processor as bse
import aspsim.diagnostics.core as diacore
import aspsim.diagnostics.diagnostics as dia
import aspsim.signal.sources as sources
import aspsim.room.trajectory as tr
import aspcore.filterclasses as fc

import aspsim.configutil as cu

def default_sim_info():

    sim_info = cu.load_default_config()
    sim_info.tot_samples = 20
    sim_info.sim_buffer = 20
    sim_info.export_frequency = 20
    sim_info.sim_chunk_size = 20
    sim_info.max_room_ir_length = 8

    sim_info.start_sources_before_0 = False
    return sim_info


@pytest.fixture(scope="session")
def fig_folder(tmp_path_factory):
    return tmp_path_factory.mktemp("figs")

def setup_with_debug_processor(fig_folder):
    setup = SimulatorSetup(fig_folder)
    setup.sim_info.tot_samples = 20
    setup.sim_info.sim_buffer = 20
    setup.sim_info.export_frequency = 20
    setup.sim_info.sim_chunk_size = 20

    setup.add_free_source("src", np.array([[1,0,0]]), sources.WhiteNoiseSource(1,1))
    setup.add_controllable_source("loudspeaker", np.array([[1,0,0]]))
    setup.add_mics("mic", np.zeros((1,3)))

    setup.arrays.path_type["loudspeaker"]["mic"] = "none"
    setup.arrays.path_type["src"]["mic"] = "direct"
    return setup


def setup_ism(fig_folder, samplerate):
    setup = SimulatorSetup(fig_folder)
    setup.sim_info.samplerate = samplerate
    setup.sim_info.tot_samples =  samplerate
    setup.sim_info.sim_chunk_size = 2*samplerate
    setup.sim_info.sim_buffer = samplerate
    setup.sim_info.export_frequency =  samplerate
    setup.sim_info.save_source_contributions = False
    setup.sim_info.randomized_ism = False

    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [4, 3, 3]
    setup.sim_info.room_center = [-1, 0, 0]
    setup.sim_info.rt60 = 0.25
    setup.sim_info.max_room_ir_length : samplerate // 2
    return setup


@pytest.fixture(scope="session")
def sim_setup(tmp_path_factory):
    setup = SimulatorSetup(tmp_path_factory.mktemp("figs"), None)
    return setup


@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_minimum_of_tot_samples_are_processed(fig_folder, bs):
    sim_setup = setup_with_debug_processor(fig_folder)
    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()
    assert sim.processors[0].processed_samples >= sim.sim_info.tot_samples

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_consecutive_simulators_give_same_values(fig_folder, bs):
    # change this to a free source instead without processors
    sim_setup = setup_with_debug_processor(fig_folder)
    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()

    sig1 = sim.sig["mic"]

    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()

    sig2 = sim.sig["mic"]
    assert np.allclose(sig1, sig2)

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_minimum_value_for_sim_buffer(fig_folder, bs):
    assert False # not implemented yet
    sim_setup = setup_with_debug_processor(fig_folder)
    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()
    assert sim.processors[0].processed_samples >= sim.sim_info.tot_samples

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_correct_processing_delay(fig_folder, bs):
    sim_setup = setup_with_debug_processor(fig_folder)
    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()

    proc = sim.processors[0]
    assert np.allclose(proc.mic[:,:-bs], proc.ls[:,bs:])



@hyp.settings(deadline=None)
@hyp.given(num_src = st.integers(min_value=1, max_value=3))
def test_simulation_equals_direct_convolution_multiple_sources(fig_folder, num_src):
    rng = np.random.default_rng()
    setup = SimulatorSetup(fig_folder)
    setup.sim_info = default_sim_info()

    src_sig = [rng.normal(size=(1, setup.sim_info.tot_samples)) for s in range(num_src)]

    setup.add_mics("mic", np.zeros((1,3)))
    for s in range(num_src):    
        setup.add_free_source(f"src_{s}", np.zeros((1,3)), sources.Sequence(src_sig[s]))
        setup.arrays.path_type[f"src_{s}"]["mic"] = "random"

    sim = setup.create_simulator()
    sim.diag.add_diagnostic("mic", dia.RecordSignal("mic", sim.sim_info, 1, export_func="npz"))
    sim.run_simulation()

    sig_sim = np.load(sim.folder_path.joinpath(f"mic_{sim.sim_info.tot_samples}.npz"))["mic"]

    filt = [fc.create_filter(ir=sim.arrays.paths[f"src_{s}"]["mic"]) for s in range(num_src)]
    direct_mic_sig = np.sum(np.concatenate([filt[s].process(src_sig[s]) for s in range(num_src)], axis=0), axis=0)

    assert np.allclose(sig_sim, direct_mic_sig)


#@hyp.settings(deadline=None)
def test_simulation_equals_direct_convolution(fig_folder):
    rng = np.random.default_rng()
    setup = SimulatorSetup(fig_folder)
    setup.sim_info = default_sim_info()

    src_sig = rng.normal(size=(1, setup.sim_info.tot_samples))

    setup.add_mics("mic", np.zeros((1,3)))
    setup.add_free_source("src", np.array([[1,0,0]]), sources.Sequence(src_sig))
    setup.arrays.path_type["src"]["mic"] = "random"

    sim = setup.create_simulator()
    sim.diag.add_diagnostic("mic", dia.RecordSignal("mic", sim.sim_info, 1, export_func="npz"))
    sim.run_simulation()

    sig_sim = np.load(sim.folder_path.joinpath(f"mic_{sim.sim_info.tot_samples}.npz"))["mic"]

    filt = fc.create_filter(ir=sim.arrays.paths["src"]["mic"])
    direct_mic_sig = filt.process(src_sig)

    assert np.allclose(sig_sim, direct_mic_sig)

#@hyp.settings(deadline=None)
# def test_trajectory_mic(fig_folder):
#     bs = 1
#     sim_setup = setup_simple(fig_folder)
#     sim_setup.sim_info.reverb = "ism"
#     sim_setup.sim_info.rt60 = 0.15
#     #room_size = [4, 3, 3]
#     sim_setup.sim_info.max_room_ir_length = 256
#     sim_setup.arrays = ar.ArrayCollection()
#     mic_traj = tr.Trajectory.linear_interpolation_const_speed([[0,0,0], [1,1,1], [0,1,0], [1,0,1]], 1, sim_setup.config["samplerate"])
#     sim_setup.add_mics("mic", mic_traj)
#     sim_setup.add_controllable_source("loudspeaker", np.array([[-1, -1, -1]]))

#     sim = sim_setup.create_simulator()
#     sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
#     sim.run_simulation()

#     assert False


def test_unmoving_trajectory_same_as_static(fig_folder):
    sr = 500
    sim_setup = setup_ism(fig_folder, sr)
    def zero_pos_func(time):
        return np.zeros((1,3))

    mic_traj = tr.Trajectory(zero_pos_func)
    sim_setup.add_mics("mic", np.zeros((1,3)))
    sim_setup.add_mics("trajectory", mic_traj)
    sim_setup.add_free_source("source", np.array([[0, -1, -1]]), sources.WhiteNoiseSource(1, 1, np.random.default_rng(1)))

    sim = sim_setup.create_simulator()
    sim.diag.add_diagnostic("mic", dia.RecordSignal("mic", sim.sim_info, 1, export_func="npz"))
    sim.diag.add_diagnostic("trajectory", dia.RecordSignal("trajectory", sim.sim_info, 1, export_func="npz"))
    sim.run_simulation()

    sig_mic = np.load(sim.folder_path.joinpath(f"mic_{sim.sim_info.tot_samples}.npz"))["mic"]
    
    sig_traj = np.load(sim.folder_path.joinpath(f"trajectory_{sim.sim_info.tot_samples}.npz"))["trajectory"]
    assert np.allclose(sig_mic, sig_traj)



def test_generated_rir_satisfies_reciprocity(fig_folder):
    sr = 500
    rng = np.random.default_rng()

    pos1 = rng.uniform(-1, 1, size=(1,3))
    pos2 = rng.uniform(-1, 1, size=(1,3))

    sim_setup = setup_ism(fig_folder, sr)
    sim_setup.add_mics("mic", pos1)
    sim_setup.add_free_source("src", pos2, sources.WhiteNoiseSource(1, 1, rng))
    sim = sim_setup.create_simulator()
    rir1 = sim.arrays.paths["src"]["mic"]

    sim_setup = setup_ism(fig_folder, sr)
    sim_setup.add_mics("mic", pos2)
    sim_setup.add_free_source("src", pos1, sources.WhiteNoiseSource(1, 1, rng))
    sim = sim_setup.create_simulator()
    rir2 = sim.arrays.paths["src"]["mic"]

    assert np.allclose(rir1, rir2)

def test_moving_microphone_equals_moving_source(fig_folder):
    """ A moving microphone should give the same output as a moving source, if the
    positions are switched. This is an extension of the reciprocity test, but with
    moving sources. 

    Actually, I don't think this is true...

    Currently gives an sample-wise error of around 1e-5, which is unclear if it is 
    within tolerance for numerical errors or not. Probably a valid result, just that
    the test is actually not true.
    """
    sr = 500
    rng = np.random.default_rng()

    pos_stationary = np.zeros((1,3))
    pos_moving = tr.LinearTrajectory([[1,1,1], [0,1,0], [1,0,1]], 1, sr)

    sim_setup = setup_ism(fig_folder, sr)
    src_sig = rng.random(size=(1, sim_setup.sim_info.tot_samples*2))

    
    sim_setup.add_mics("mic", pos_stationary)
    sim_setup.add_free_source("src", pos_moving, sources.Sequence(src_sig))
    sim = sim_setup.create_simulator()
    sim.diag.add_diagnostic("mic1", dia.RecordSignal("mic", sim.sim_info, 1, export_func="npz"))
    sim.run_simulation()
    sig1 = np.load(sim.folder_path.joinpath(f"mic1_{sim.sim_info.tot_samples}.npz"))["mic1"]

    sim_setup = setup_ism(fig_folder, sr)
    sim_setup.add_mics("mic", pos_moving)
    sim_setup.add_free_source("src", pos_stationary, sources.Sequence(src_sig))
    sim = sim_setup.create_simulator()
    sim.diag.add_diagnostic("mic2", dia.RecordSignal("mic", sim.sim_info, 1, export_func="npz"))
    sim.run_simulation()

    sig2 = np.load(sim.folder_path.joinpath(f"mic2_{sim.sim_info.tot_samples}.npz"))["mic2"]
    assert np.allclose(sig1, sig2)

def test_moving_microphone_is_using_expected_positions(fig_folder):
    rng = np.random.default_rng()

    setup = SimulatorSetup(fig_folder)
    setup.sim_info = default_sim_info()

    pos_src = np.zeros((1,3))
    pos_mic = tr.LinearTrajectory([[1,1,1], [0,1,0], [1,0,1]], 1, setup.sim_info.samplerate)

    setup.add_mics("mic", pos_mic)
    setup.add_free_source("src", pos_src, sources.WhiteNoiseSource(1,1, rng))
    sim = setup.create_simulator()
    sim.run_simulation()
    pos_all = np.array(sim.arrays["mic"].pos_all)
    time_all = np.array(sim.arrays["mic"].time_all)

    pos_compare = np.array([pos_mic.current_pos(t) for t in range(setup.sim_info.tot_samples)])
    assert np.allclose(pos_all[:setup.sim_info.tot_samples], pos_compare)
   


def test_moving_microphone_has_same_rirs_as_stationary_microphones_on_trajectory(fig_folder):
    rng = np.random.default_rng()
    sr = 500

    setup = setup_ism(fig_folder, sr)
    src_sig = rng.random(size=(1, setup.sim_info.tot_samples*2))

    pos_src = np.zeros((1,3))
    traj = tr.LinearTrajectory([[1,1,1], [0,1,0], [1,0,1]], 1, setup.sim_info.samplerate)
    all_pos = np.array([traj.current_pos(t) for t in range(setup.sim_info.tot_samples)])[:,0,:]

    setup.add_mics("traj", traj)
    setup.add_mics("mic", all_pos)
    setup.add_free_source("src", pos_src, sources.Sequence(src_sig))
    sim = setup.create_simulator()
    
    sim.run_simulation()

    rir1 = np.array(sim.arrays._rir_dynamic_all)[:setup.sim_info.tot_samples,0,0,:]
    rir2 = sim.arrays.paths["src"]["mic"][0,:,:]
    #sim.diag.add_diagnostic("mic", dia.RecordSignal("mic", sim.sim_info, num_channels = all_pos.shape[0], export_func="npz"))
    #sim.diag.add_diagnostic("traj_rir", dia.RecordState("traj", sim.sim_info, num_channels = 1, export_func="npz"))
    #sim.run_simulation()


    #sig_mic = np.load(sim.folder_path.joinpath(f"mic_{sim.sim_info.tot_samples}.npz"))["mic"]
    #sig_traj_reconstruct = np.array([sig_mic[i,i] for i in range(sig_mic.shape[0])])[None,:]
    #sig_traj = np.load(sim.folder_path.joinpath(f"traj_{sim.sim_info.tot_samples}.npz"))["traj"]

    assert np.allclose(rir1, rir2)



def test_moving_microphone_gives_same_output_as_pointwise_stationary_convolutions(fig_folder):
    """The equivalence is currently for the range (-1, tot_samples-1), which is not the 
    intended behaviour. The sim should be changed to correctly give equivalence for (0, tot_samples)
    """
    rng = np.random.default_rng()
    sr = 500

    setup = setup_ism(fig_folder, sr)
    src_sig = rng.random(size=(1, setup.sim_info.tot_samples*2))

    pos_src = np.zeros((1,3))
    traj = tr.LinearTrajectory([[1,1,1], [0,1,0], [1,0,1]], 1, setup.sim_info.samplerate)
    all_pos = np.array([traj.current_pos(t) for t in range(-1, setup.sim_info.tot_samples-1)])[:,0,:]

    setup.add_mics("traj", traj)
    setup.add_mics("mic", all_pos)
    setup.add_free_source("src", pos_src, sources.Sequence(src_sig))
    sim = setup.create_simulator()
    sim.diag.add_diagnostic("mic", dia.RecordSignal("mic", sim.sim_info, num_channels = all_pos.shape[0], export_func="npz"))
    sim.diag.add_diagnostic("traj", dia.RecordSignal("traj", sim.sim_info, num_channels = 1, export_func="npz"))
    sim.run_simulation()


    sig_mic = np.load(sim.folder_path.joinpath(f"mic_{sim.sim_info.tot_samples}.npz"))["mic"]
    sig_traj_reconstruct = np.array([sig_mic[i,i] for i in range(sig_mic.shape[0])])[None,:]
    sig_traj = np.load(sim.folder_path.joinpath(f"traj_{sim.sim_info.tot_samples}.npz"))["traj"]

    assert np.allclose(sig_traj, sig_traj_reconstruct)

