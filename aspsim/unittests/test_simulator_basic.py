import numpy as np
import hypothesis as hyp
import hypothesis.strategies as st
import pytest

from aspsim.simulator import SimulatorSetup
import aspsim.processor as bse
import aspsim.diagnostics.diagnostics as dia
import aspsim.signal.sources as sources
import aspsim.signal.filterclasses as fc

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

def _setup_with_debug_processor(fig_folder):
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

@pytest.fixture(scope="session")
def _sim_setup(tmp_path_factory):
    setup = SimulatorSetup(tmp_path_factory.mktemp("figs"), None)
    return setup


@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_minimum_of_tot_samples_are_processed(fig_folder, bs):
    sim_setup = _setup_with_debug_processor(fig_folder)
    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()
    assert sim.processors[0].processed_samples >= sim.sim_info.tot_samples

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_consecutive_simulators_give_same_values(fig_folder, bs):
    # change this to a free source instead without processors
    sim_setup = _setup_with_debug_processor(fig_folder)
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
    _sim_setup = _setup_with_debug_processor(fig_folder)
    sim = _sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()
    assert sim.processors[0].processed_samples >= sim.sim_info.tot_samples

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_correct_processing_delay(fig_folder, bs):
    sim_setup = _setup_with_debug_processor(fig_folder)
    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()

    proc = sim.processors[0]
    assert np.allclose(proc.mic[:,:-bs], proc.ls[:,bs:])

def test_simulation_equals_direct_convolution_random_rirs(fig_folder):
    rng = np.random.default_rng()
    setup = SimulatorSetup(fig_folder)
    setup.sim_info = _default_sim_info()

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

@hyp.settings(deadline=None)
@hyp.given(num_src = st.integers(min_value=1, max_value=3))
def test_simulation_equals_direct_convolution_random_rirs_multiple_sources(fig_folder, num_src):
    rng = np.random.default_rng()
    setup = SimulatorSetup(fig_folder)
    setup.sim_info = _default_sim_info()

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










