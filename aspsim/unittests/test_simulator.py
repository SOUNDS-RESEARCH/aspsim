import numpy as np
import hypothesis as hyp
#from hypothesis import given
import hypothesis.strategies as st
import pytest
import copy
#import sys
#sys.path.append("c:/skola/utokyo_lab/ancsim/ancsim")

from aspsim.simulator import SimulatorSetup
import aspsim.array as ar
import aspsim.processor as bse
import aspsim.diagnostics.core as diacore
import aspsim.diagnostics.diagnostics as dia
import aspsim.signal.sources as sources
import aspsim.room.trajectory as tr
import aspcore.filterclasses as fc

def reset_sim_setup(setup):
    setup.arrays = ar.ArrayCollection()
    setup.sim_info.tot_samples = 100
    setup.sim_info.sim_chunk_size = 10
    setup.sim_info.sim_buffer = 10
    setup.sim_info.export_frequency = 100
    setup.sim_info.save_source_contributions = False
    setup.use_preset("debug")

def reset_sim_setup_realistic(setup, samplerate):
    setup.arrays = ar.ArrayCollection()
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


@pytest.fixture(scope="session")
def sim_setup(tmp_path_factory):
    setup = SimulatorSetup(tmp_path_factory.mktemp("figs"), None)
    return setup

# @hyp.settings(deadline=None)
# @hyp.given(bs = st.integers(min_value=1, max_value=5))
# def test_minimum_of_tot_samples_are_processed(sim_setup, bs):
#     sim = sim_setup.create_simulator()
#     sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
#     sim.run_simulation()
#     print(sim.n_tot >= sim.sim_info.tot_samples)
#     assert sim.n_tot >= sim.sim_info.tot_samples

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_minimum_of_tot_samples_are_processed(sim_setup, bs):
    reset_sim_setup(sim_setup)
    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()
    assert sim.processors[0].processor.processed_samples >= sim.sim_info.tot_samples
    #assert sim.n_tot >= sim.sim_info.tot_samples

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_consecutive_simulators_give_same_values(sim_setup, bs):
    reset_sim_setup(sim_setup)
    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()

    sig1 = sim.processors[0].processor.sig["mic"]

    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()

    sig2 = sim.processors[0].processor.sig["mic"]

    assert np.allclose(sig1, sig2)

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_correct_processing_delay(sim_setup, bs):
    reset_sim_setup(sim_setup)
    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()

    proc = sim.processors[0].processor
    assert np.allclose(proc.mic[:,:-bs], proc.ls[:,bs:])



@hyp.settings(deadline=None)
@hyp.given(num_src = st.integers(min_value=1, max_value=3))
def test_multiple_free_sources(sim_setup, num_src):
    rng = np.random.default_rng()
    sr = 500
    reset_sim_setup_realistic(sim_setup, sr)

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

    block_size = 5

    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, block_size))
    sim.run_simulation()

    import matplotlib.pyplot as plt
    plt.plot(sig_filt[:,sim.sim_info.sim_buffer:].T)
    plt.plot(sim.processors[0].processor.mic.T)
    plt.show()

    assert np.allclose(sig_filt[:,sim.sim_info.sim_buffer:-block_size], sim.processors[0].processor.mic[:,block_size:])

def test_same_with_multiple_processors():
    assert False

#@hyp.settings(deadline=None)
def test_trajectory_mic(sim_setup):
    bs = 1
    reset_sim_setup(sim_setup)
    sim_setup.sim_info.reverb = "ism"
    sim_setup.sim_info.rt60 = 0.15
    #room_size = [4, 3, 3]
    sim_setup.sim_info.max_room_ir_length = 256
    sim_setup.arrays = ar.ArrayCollection()
    mic_traj = tr.Trajectory.linear_interpolation_const_speed([[0,0,0], [1,1,1], [0,1,0], [1,0,1]], 1, sim_setup.config["samplerate"])
    sim_setup.add_mics("mic", mic_traj)
    sim_setup.add_controllable_source("loudspeaker", np.array([[-1, -1, -1]]))

    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()

    assert False


def test_unmoving_trajectory_same_as_static(sim_setup):
    bs = 1
    reset_sim_setup(sim_setup)
    sim_setup.sim_info.reverb = "ism"
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


    reset_sim_setup(sim_setup)
    sim_setup.sim_info.reverb = "ism"
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