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

@pytest.fixture(scope="session")
def fig_folder(tmp_path_factory):
    return tmp_path_factory.mktemp("figs")

def setup_simple(fig_folder):
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
    setup.sim_info.tot_samples = 10 * samplerate
    setup.sim_info.sim_chunk_size = 20*samplerate
    setup.sim_info.sim_buffer = samplerate
    setup.sim_info.export_frequency = 10 * samplerate
    setup.sim_info.save_source_contributions = False

    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 5, 5]
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
    sim_setup = setup_simple(fig_folder)
    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()
    assert sim.processors[0].processed_samples >= sim.sim_info.tot_samples

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_consecutive_simulators_give_same_values(fig_folder, bs):
    # change this to a free source instead without processors
    sim_setup = setup_simple(fig_folder)
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
    sim_setup = setup_simple(fig_folder)
    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()
    assert sim.processors[0].processed_samples >= sim.sim_info.tot_samples

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_correct_processing_delay(fig_folder, bs):
    sim_setup = setup_simple(fig_folder)
    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()

    proc = sim.processors[0]
    assert np.allclose(proc.mic[:,:-bs], proc.ls[:,bs:])



@hyp.settings(deadline=None)
@hyp.given(num_src = st.integers(min_value=1, max_value=3))
def test_multiple_free_sources(sim_setup, num_src):
    """
    The likely problem is that in the simulator case, the random source is used
    in the prepare() step, so when they reach time 0, the seeds are different. 
    """
    block_size = 5
    rng = np.random.default_rng()
    seed = rng.integers(0, 100000)
    sr = 500
    reset_sim_setup_ism(sim_setup, sr)

    for s in range(num_src):
        sim_setup.add_free_source(f"src_{s}", rng.uniform(-2, 2, size=(1,3)), sources.WhiteNoiseSource(1, 1, rng))

    sim_setup.add_mics("mic", rng.uniform(-2, 2, size=(1,3)))
    sim_setup.add_controllable_source("loudspeaker", rng.uniform(-2, 2, size=(1,3)))
    sim_setup.arrays.path_type["loudspeaker"]["mic"] = "none"

    sim = sim_setup.create_simulator()

    irs = np.concatenate([sim.arrays.paths[f"src_{s}"]["mic"] for s in range(num_src)], axis=0)
    filt = fc.create_filter(ir=irs)
    
    srcs = [copy.deepcopy(sim.arrays[f"src_{s}"].source) for s in range(num_src)]
    signals = np.concatenate([s.get_samples(sim.sim_info.tot_samples+sim.sim_info.sim_buffer) for s in srcs], axis=0)
    sig_filt = filt.process(signals)

    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, block_size))
    sim.run_simulation()

    import matplotlib.pyplot as plt
    plt.plot(sig_filt[:,sim.sim_info.sim_buffer:].T)
    plt.plot(sim.processors[0].mic.T)
    plt.show()

    assert np.allclose(sig_filt[:,sim.sim_info.sim_buffer:-block_size], sim.processors[0].mic[:,block_size:])


#@hyp.settings(deadline=None)
def test_simulation_equals_direct_convolution(fig_folder):
    rng = np.random.default_rng()
    setup = SimulatorSetup(fig_folder)
    setup.sim_info.tot_samples = 20
    setup.sim_info.sim_buffer = 10
    setup.sim_info.export_frequency = 20
    setup.sim_info.sim_chunk_size = 20
    setup.sim_info.start_sources_before_0 = False

    setup.sim_info.max_room_ir_length = 8

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
    bs = 1
    sr = 500
    sim_setup = setup_ism(fig_folder, sr)
    sim_setup.sim_info.rt60 = 0.15
    room_size = [4, 3, 3]
    sim_setup.sim_info.max_room_ir_length = 256
    sim_setup.arrays = ar.ArrayCollection()
    def zero_pos_func(time):
        return np.zeros((1,3))

    mic_traj = tr.Trajectory(zero_pos_func)
    sim_setup.add_mics("mic", mic_traj)
    sim_setup.add_controllable_source("loudspeaker", np.array([[-1, -1, -1]]))
    sim_setup.add_free_source("source", np.array([[0, -1, -1]]), sources.WhiteNoiseSource(1, 1, np.random.default_rng(1)))

    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
    diagnostics={"mic":dia.RecordSignal("mic", sim.sim_info, bs, export_func="npz")}))
    sim.run_simulation()


    sim_setup = setup_ism(fig_folder, sr)
    sim_setup.sim_info.rt60 = 0.15
    room_size = [4, 3, 3]
    sim_setup.sim_info.max_room_ir_length = 256
    sim_setup.arrays = ar.ArrayCollection()
    sim_setup.add_mics("mic", np.array([[0,0,0]]))
    sim_setup.add_controllable_source("loudspeaker", np.array([[-1, -1, -1]]))
    sim_setup.add_free_source("source", np.array([[0, -1, -1]]), sources.WhiteNoiseSource(1, 1, np.random.default_rng(1)))

    sim2 = sim_setup.create_simulator()
    sim2.add_processor(bse.DebugProcessor(sim2.sim_info, sim2.arrays, bs, 
    diagnostics={"mic":dia.RecordSignal("mic", sim.sim_info, bs, export_func="npz")}))
    sim2.run_simulation()

    for f, f2 in zip(sim.folder_path.iterdir(), sim2.folder_path.iterdir()):
        assert f.name == f2.name
        if f.suffix == ".npy" or f.suffix == ".npz":
            saved_data = np.load(f)
            saved_data2 = np.load(f2)
            for data, data2 in zip(saved_data.values(), saved_data2.values()):
                assert np.allclose(data, data2)



#Kolla manuellt på noiset i rir_extimation_exp, och se att det är det jag förväntar mig. Efterssom
# icke-modifierade noise correlation matrix är annorlunda. 



# def test_same_with_multiple_processors():
#     assert False