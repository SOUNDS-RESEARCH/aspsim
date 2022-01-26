import numpy as np
import hypothesis as hyp
#from hypothesis import given
import hypothesis.strategies as st
import pytest
#import sys
#sys.path.append("c:/skola/utokyo_lab/ancsim/ancsim")

from ancsim.simulator import SimulatorSetup
import ancsim.array as ar
import ancsim.adaptivefilter.base as bse
import ancsim.diagnostics.core as diacore
import ancsim.diagnostics.diagnostics as dia
import ancsim.diagnostics.diagnosticutils as diautil

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