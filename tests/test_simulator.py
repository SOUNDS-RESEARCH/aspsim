import numpy as np
import hypothesis as hyp
import hypothesis.strategies as st
import pytest

from aspsim.simulator import SimulatorSetup
import aspsim.diagnostics.diagnostics as dia
import aspsim.signal.sources as sources
import aspsim.room.trajectory as tr
import aspcore.filter as fc


import aspsim.configutil as cu

def _default_sim_info():

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


def _setup_ism(fig_folder, samplerate):
    setup = SimulatorSetup(fig_folder)
    setup.sim_info.samplerate = samplerate
    setup.sim_info.tot_samples =  samplerate
    setup.sim_info.sim_chunk_size = 2*samplerate
    setup.sim_info.sim_buffer = samplerate
    setup.sim_info.export_frequency =  samplerate
    setup.sim_info.save_source_contributions = False
    setup.sim_info.randomized_ism = False

    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 5, 5]
    setup.sim_info.room_center = [0, 0, 0]
    setup.sim_info.rt60 = 0.25
    setup.sim_info.max_room_ir_length = samplerate // 2
    return setup


def test_unmoving_trajectory_same_as_static(fig_folder):
    sr = 500
    sim_setup = _setup_ism(fig_folder, sr)
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

    sim_setup = _setup_ism(fig_folder, sr)
    src_sig = rng.random(size=(1, sim_setup.sim_info.tot_samples*2))

    
    sim_setup.add_mics("mic", pos_stationary)
    sim_setup.add_free_source("src", pos_moving, sources.Sequence(src_sig))
    sim = sim_setup.create_simulator()
    sim.diag.add_diagnostic("mic1", dia.RecordSignal("mic", sim.sim_info, 1, export_func="npz"))
    sim.run_simulation()
    sig1 = np.load(sim.folder_path.joinpath(f"mic1_{sim.sim_info.tot_samples}.npz"))["mic1"]

    sim_setup = _setup_ism(fig_folder, sr)
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
    setup.sim_info = _default_sim_info()

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

    setup = _setup_ism(fig_folder, sr)
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

    setup = _setup_ism(fig_folder, sr)
    src_sig = rng.random(size=(1, setup.sim_info.tot_samples*2))

    pos_src = np.zeros((1,3))
    traj = tr.LinearTrajectory([[1,1,1], [0,1,0], [1,0,1]], 1, setup.sim_info.samplerate)
    all_pos = np.array([traj.current_pos(t) for t in range(-1, setup.sim_info.tot_samples-1)])[:,0,:]
    all_pos[0,:] += 2 # position at index 0 does not matter
    all_pos[1,:] += 2 # position at index 1 does not matter

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

