import numpy as np
import hypothesis as hyp
#from hypothesis import given
import hypothesis.strategies as st
import pytest
#import sys
#sys.path.append("c:/skola/utokyo_lab/ancsim/ancsim")

from ancsim.simulator import SimulatorSetup
import ancsim.adaptivefilter.base as bse
import ancsim.diagnostics.core as diacore
import ancsim.diagnostics.diagnostics as dia

@pytest.fixture(scope="session")
def sim_setup(tmp_path_factory):
    setup = SimulatorSetup(tmp_path_factory.mktemp("figs"), None)
    setup.config["tot_samples"] = 100
    setup.config["sim_chunk_size"] = 10
    setup.config["sim_buffer"] = 10
    setup.config["chunk_per_export"] = 1
    setup.config["save_source_contributions"] = False
    setup.usePreset("debug")
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
    sim = sim_setup.createSimulator()
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.runSimulation()
    assert sim.processors[0].processor.processed_samples >= sim.sim_info.tot_samples
    #assert sim.n_tot >= sim.sim_info.tot_samples


@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_consecutive_simulators_give_same_values(sim_setup, bs):
    sim = sim_setup.createSimulator()
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.runSimulation()

    sig1 = sim.processors[0].processor.sig["mic"]

    sim = sim_setup.createSimulator()
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.runSimulation()

    sig2 = sim.processors[0].processor.sig["mic"]

    assert np.allclose(sig1, sig2)


@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_all_samples_saved_for_signal_diagnostics(sim_setup, bs):
    sim = sim_setup.createSimulator()
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                   diagnostics={"mic":dia.RecordSignal("mic", sim.sim_info, bs, export_func="npz")}))

    sim.runSimulation()
    
    one_file_saved = False
    for f in sim.folderPath.iterdir():
        if f.stem.startswith("mic"):
            one_file_saved = True
            saved_data = np.load(f)
            for proc_name, data in saved_data.items():
                assert np.allclose(data, np.arange(sim.sim_info.sim_buffer, 
                            sim.sim_info.sim_buffer+sim.sim_info.tot_samples))
    assert one_file_saved
    

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_export_file_naming_interval_diagnostics(sim_setup, bs):
    sim = sim_setup.createSimulator()

    save_intervals = ((32,46), (68,69), (71, 99))
    diag_name = "mic"
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                   diagnostics={"mic":dia.RecordSignal(diag_name, sim.sim_info, bs, 
                   export_at = [iv[1] for iv in save_intervals],
                    save_at=diacore.IntervalCounter(save_intervals), 
                    export_func="npz")}))
    sim.runSimulation()

    for iv in save_intervals:
        assert sim.folderPath.joinpath(f"{diag_name}_{iv[1]}.npz").exists()



@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_correct_samples_saved_for_interval_diagnostics(sim_setup, bs):
    sim = sim_setup.createSimulator()

    save_intervals = ((32,46), (68,69), (71, 99))
    diag_name = "mic"
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                   diagnostics={"mic":dia.RecordSignal(
                       diag_name, sim.sim_info, bs, 
                   export_at = [iv[1] for iv in save_intervals],
                    save_at = diacore.IntervalCounter(save_intervals), 
                    export_func="npz")
                    }))

    sim.runSimulation()

    expected = np.full((sim.sim_info.tot_samples), np.nan)
    for iv in save_intervals:
        saved_data = np.load(sim.folderPath.joinpath(f"{diag_name}_{iv[1]}.npz"))
        for proc_name, data in saved_data.items():
            expected[iv[0]:iv[1]] = np.arange(iv[0]+sim.sim_info.sim_buffer, 
                                                iv[1]+sim.sim_info.sim_buffer)

            assert np.allclose(
                data,
                expected,
                equal_nan=True
            )

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_correct_samples_saved_for_instant_diagnostics(sim_setup, bs):
    sim = sim_setup.createSimulator()

    
    save_at = (1,)
    save_intervals = ((1,2), (3,4), (5,6))
    diag_name = "filt"
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                   diagnostics={diag_name:dia.RecordFilter(
                       "filt.ir", sim.sim_info, bs, save_at = save_at, export_func="npz"),
                       "mic":dia.RecordSignal(
                       "mic", sim.sim_info, bs, 
                   export_at = [iv[1] for iv in save_intervals],
                    save_at = diacore.IntervalCounter(save_intervals), 
                    export_func="npz")
                    }))

    sim.runSimulation()

    for idx in save_at:
        saved_data = np.load(sim.folderPath.joinpath(f"{diag_name}_{idx}.npz"))
        for proc_name, data in saved_data.items():
            assert np.allclose(data, np.zeros_like(data)+idx)