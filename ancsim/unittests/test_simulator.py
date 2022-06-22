import numpy as np
import hypothesis as hyp
#from hypothesis import given
import hypothesis.strategies as st
import pytest
#import sys
#sys.path.append("c:/skola/utokyo_lab/ancsim/ancsim")

from ancsim.simulator import SimulatorSetup
import ancsim.array as ar
import ancsim.processor as bse
import ancsim.diagnostics.core as diacore
import ancsim.diagnostics.diagnostics as dia
import ancsim.diagnostics.diagnosticutils as diautil
import ancsim.signal.sources as sources

def reset_sim_setup(setup):
    setup.arrays = ar.ArrayCollection()
    setup.config["tot_samples"] = 100
    setup.config["sim_chunk_size"] = 10
    setup.config["sim_buffer"] = 10
    setup.config["chunk_per_export"] = 1
    setup.config["save_source_contributions"] = False
    setup.usePreset("debug")

@pytest.fixture(scope="session")
def sim_setup(tmp_path_factory):
    setup = SimulatorSetup(tmp_path_factory.mktemp("figs"), None)
    return setup

# @hyp.settings(deadline=None)
# @hyp.given(bs = st.integers(min_value=1, max_value=5))
# def test_minimum_of_tot_samples_are_processed(sim_setup, bs):
#     sim = sim_setup.createSimulator()
#     sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
#     sim.runSimulation()
#     print(sim.n_tot >= sim.sim_info.tot_samples)
#     assert sim.n_tot >= sim.sim_info.tot_samples

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_minimum_of_tot_samples_are_processed(sim_setup, bs):
    reset_sim_setup(sim_setup)
    sim = sim_setup.createSimulator()
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.runSimulation()
    assert sim.processors[0].processor.processed_samples >= sim.sim_info.tot_samples
    #assert sim.n_tot >= sim.sim_info.tot_samples

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_consecutive_simulators_give_same_values(sim_setup, bs):
    reset_sim_setup(sim_setup)
    sim = sim_setup.createSimulator()
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.runSimulation()

    sig1 = sim.processors[0].processor.sig["mic"]

    sim = sim_setup.createSimulator()
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.runSimulation()

    sig2 = sim.processors[0].processor.sig["mic"]

    assert np.allclose(sig1, sig2)


def test_correct_processing_delay():
    assert False


def test_multiple_free_sources():
    assert False

def test_same_with_multiple_processors():
    assert False

#@hyp.settings(deadline=None)
def test_trajectory_mic(sim_setup):
    bs = 1
    reset_sim_setup(sim_setup)
    sim_setup.config["reverb"] = "ism"
    sim_setup.config["rt60"] = 0.15
    #room_size = [4, 3, 3]
    sim_setup.config["max_room_ir_length"] = 256
    sim_setup.arrays = ar.ArrayCollection()
    mic_traj = ar.Trajectory.linear_interpolation_const_speed([[0,0,0], [1,1,1], [0,1,0], [1,0,1]], 1, sim_setup.config["samplerate"])
    sim_setup.addMics("mic", mic_traj)
    sim_setup.addControllableSource("loudspeaker", np.array([[-1, -1, -1]]))

    sim = sim_setup.createSimulator()
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.runSimulation()

    assert False


def test_unmoving_trajectory_same_as_static(sim_setup):
    bs = 1
    reset_sim_setup(sim_setup)
    sim_setup.config["reverb"] = "ism"
    sim_setup.config["rt60"] = 0.15
    room_size = [4, 3, 3]
    sim_setup.config["max_room_ir_length"] = 256
    sim_setup.arrays = ar.ArrayCollection()
    def zero_pos_func(time):
        return np.zeros((1,3))

    mic_traj = ar.Trajectory(zero_pos_func)
    sim_setup.addMics("mic", mic_traj)
    sim_setup.addControllableSource("loudspeaker", np.array([[-1, -1, -1]]))
    sim_setup.addFreeSource("source", np.array([[0, -1, -1]]), sources.WhiteNoiseSource(1, 1, np.random.default_rng(1)))

    sim = sim_setup.createSimulator()
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
    diagnostics={"mic":dia.RecordSignal("mic", sim.sim_info, bs, export_func="npz")}))
    sim.runSimulation()


    reset_sim_setup(sim_setup)
    sim_setup.config["reverb"] = "ism"
    sim_setup.config["rt60"] = 0.15
    room_size = [4, 3, 3]
    sim_setup.config["max_room_ir_length"] = 256
    sim_setup.arrays = ar.ArrayCollection()
    sim_setup.addMics("mic", np.array([[0,0,0]]))
    sim_setup.addControllableSource("loudspeaker", np.array([[-1, -1, -1]]))
    sim_setup.addFreeSource("source", np.array([[0, -1, -1]]), sources.WhiteNoiseSource(1, 1, np.random.default_rng(1)))

    sim2 = sim_setup.createSimulator()
    sim2.addProcessor(bse.DebugProcessor(sim2.sim_info, sim2.arrays, bs, 
    diagnostics={"mic":dia.RecordSignal("mic", sim.sim_info, bs, export_func="npz")}))
    sim2.runSimulation()

    for f, f2 in zip(sim.folderPath.iterdir(), sim2.folderPath.iterdir()):
        assert f.name == f2.name
        if f.suffix == ".npy" or f.suffix == ".npz":
            saved_data = np.load(f)
            saved_data2 = np.load(f2)
            for data, data2 in zip(saved_data.values(), saved_data2.values()):
                assert np.allclose(data, data2)



#Kolla manuellt på noiset i rir_extimation_exp, och se att det är det jag förväntar mig. Efterssom
# icke-modifierade noise correlation matrix är annorlunda. 