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
import ancsim.fileutilities as fu

# @pytest.fixture(scope="session")
# def sim_setup(tmp_path_factory):
#     setup = SimulatorSetup(tmp_path_factory.mktemp("figs"), None)
#     setup.config["tot_samples"] = 100
#     setup.config["sim_chunk_size"] = 10
#     setup.config["sim_buffer"] = 10
#     setup.config["chunk_per_export"] = 1
#     setup.config["save_source_contributions"] = False
#     setup.usePreset("debug")
#     return setup

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



def test_signal_diagnostics_simplest(sim_setup):
    reset_sim_setup(sim_setup)
    bs = 2
    sim_setup.config["tot_samples"] = 13
    sim_setup.config["sim_chunk_size"] = 5
    sim_setup.config["sim_buffer"] = 5

    sim = sim_setup.createSimulator()

    #save_at = diacore.IntervalCounter(np.arange(1,sim.sim_info.tot_samples+1))
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                diagnostics={"mic":dia.RecordSignal("mic", sim.sim_info, bs, export_func="npz")}))

    sim.runSimulation()
    
    one_file_saved = False
    for f in sim.folderPath.iterdir():
        if f.stem.startswith("mic"):
            one_file_saved = True
            idx = fu.find_index_in_name(f.stem)
            saved_data = np.load(f)
            for proc_name, data in saved_data.items():
                assert np.allclose(data, np.arange(sim.sim_info.sim_buffer, 
                            sim.sim_info.sim_buffer+idx))
    assert one_file_saved



@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5),
            buf_size = st.integers(min_value=10, max_value=30),
            num_proc = st.integers(min_value=1, max_value=3))
def test_all_samples_saved_for_signal_diagnostics(sim_setup, bs, buf_size, num_proc):
    reset_sim_setup(sim_setup)
    sim_setup.config["sim_buffer"] = buf_size
    sim = sim_setup.createSimulator()
    for _ in range(num_proc):
        sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                    diagnostics={"mic":dia.RecordSignal("mic", sim.sim_info, bs, export_func="npz")}))

    sim.runSimulation()
    
    at_least_one_file_saved = False
    for f in sim.folderPath.iterdir():
        if f.stem.startswith("mic"):
            at_least_one_file_saved = True
            saved_data = np.load(f)
            for proc_name, data in saved_data.items():
                assert np.allclose(data, np.arange(sim.sim_info.sim_buffer, 
                            sim.sim_info.sim_buffer+sim.sim_info.tot_samples))
    assert at_least_one_file_saved



@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5),
            buf_size = st.integers(min_value=10, max_value=30))
def test_correct_intermediate_samples_saved_for_signal_diagnostics(sim_setup, bs, buf_size):
    reset_sim_setup(sim_setup)
    sim_setup.config["sim_buffer"] = buf_size
    sim = sim_setup.createSimulator()
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                   diagnostics={"mic":dia.RecordSignal("mic", sim.sim_info, bs, export_func="npz", keep_only_last_export=False)}))

    sim.runSimulation()
    
    for f in sim.folderPath.iterdir():
        if f.stem.startswith("mic"):
            idx = fu.find_index_in_name(f.stem)
            saved_data = np.load(f)
            for proc_name, data in saved_data.items():
                assert np.allclose(data, np.arange(sim.sim_info.sim_buffer, 
                            sim.sim_info.sim_buffer+idx))
    

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_export_file_naming_interval_diagnostics(sim_setup, bs):
    reset_sim_setup(sim_setup)
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
    reset_sim_setup(sim_setup)
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

    expected = np.zeros(0)
    for iv in save_intervals:
        saved_data = np.load(sim.folderPath.joinpath(f"{diag_name}_{iv[1]}.npz"))
        for proc_name, data in saved_data.items():
            #expected[iv[0]:iv[1]] = np.arange(iv[0]+sim.sim_info.sim_buffer, 
            #                                    iv[1]+sim.sim_info.sim_buffer)
            expected = np.concatenate((expected, np.arange(iv[0]+sim.sim_info.sim_buffer, 
                                                        iv[1]+sim.sim_info.sim_buffer)))

            assert np.allclose(
                data,
                expected,
                equal_nan=True
            )
            



@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5),
            buf_size = st.integers(min_value=10, max_value=30),
            num_proc = st.integers(min_value=1, max_value=3))
def test_all_samples_saved_state_diagnostics(sim_setup, bs, buf_size, num_proc):
    reset_sim_setup(sim_setup)
    #bs = 1
    #sim_setup.config["tot_samples"] = 13
    #sim_setup.config["sim_chunk_size"] = 5
    sim_setup.config["sim_buffer"] = buf_size

    sim = sim_setup.createSimulator()

    #save_at = diacore.IntervalCounter(np.arange(1,sim.sim_info.tot_samples+1))
    for _ in range(num_proc):
        sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                    diagnostics={"state":dia.RecordState("processed_samples", 1, sim.sim_info, bs, export_func="npz")}))

    sim.runSimulation()
    
    one_file_saved = False
    for f in sim.folderPath.iterdir():
        if f.stem.startswith("state"):
            one_file_saved = True
            idx = fu.find_index_in_name(f.stem)
            saved_data = np.load(f)
            for proc_name, data in saved_data.items():
                assert np.allclose(data, np.arange(bs, idx+1, bs))
    assert one_file_saved

















@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_correct_samples_saved_for_instant_diagnostics(sim_setup, bs):
    reset_sim_setup(sim_setup)
    sim = sim_setup.createSimulator()

    #save_at = np.arange(bs, sim.sim_info.tot_samples, bs)#(bs,)
    #save_at = [bs*i for i in range(1, sim.sim_info.tot_samples//bs)]
    save_at = (bs, 2*bs, 5*bs)
    #save_intervals = ((1,2), (3,4), (5,6))
    diag_name = "filt"
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                   diagnostics={diag_name:dia.RecordFilter(
                       "filt.ir", sim.sim_info, bs, save_at = save_at, export_func="npz")}))
                #        "mic":dia.RecordSignal(
                #        "mic", sim.sim_info, bs, 
                #    export_at = [iv[1] for iv in save_intervals],
                #     save_at = diacore.IntervalCounter(save_intervals), 
                #     export_func="npz")
                #     }))

    sim.runSimulation()

    for idx in save_at:
        saved_data = np.load(sim.folderPath.joinpath(f"{diag_name}_{idx}.npz"))
        for proc_name, data in saved_data.items():
            assert np.allclose(data, np.zeros_like(data)+idx)

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_correct_samples_saved_for_instant_diagnostics_savefreq(sim_setup, bs):
    reset_sim_setup(sim_setup)
    sim = sim_setup.createSimulator()

    save_at = bs
    diag_name = "filt"
    sim.addProcessor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                   diagnostics={diag_name:dia.RecordFilter(
                       "filt.ir", sim.sim_info, bs, save_at = save_at, export_func="npz")}))

    sim.runSimulation()

    for idx in range(save_at, sim.sim_info.tot_samples+1, save_at):
        saved_data = np.load(sim.folderPath.joinpath(f"{diag_name}_{idx}.npz"))
        for proc_name, data in saved_data.items():
            assert np.allclose(data, np.zeros_like(data)+idx)










@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_two_processors_with_different_diagnostics(sim_setup, bs):
    reset_sim_setup(sim_setup)
    sim = sim_setup.createSimulator()

    proc1 = bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
            diagnostics = {"common" : dia.RecordSignal("mic", sim.sim_info, bs, export_func = "npz", keep_only_last_export=False),
                            "individual1" : dia.RecordSignal("mic", sim.sim_info, bs, export_func = "npz", keep_only_last_export=False),
            }
            )
    proc2 = bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
            diagnostics = {"common" : dia.RecordSignal("mic", sim.sim_info, bs, export_func = "npz", keep_only_last_export=False),
                            "individual2" : dia.RecordSignal("mic", sim.sim_info, bs, export_func = "npz", keep_only_last_export=False)
                }
            )

    sim.addProcessor(proc1)
    sim.addProcessor(proc2)
    sim.runSimulation()

    for f in sim.folderPath.iterdir():
        if f.stem.startswith("mic"):
            idx = fu.find_index_in_name(f.stem)
            saved_data = np.load(f)
            for proc_name, data in saved_data.items():
                assert np.allclose(data[:idx+1], np.arange(sim.sim_info.sim_buffer, 
                            sim.sim_info.sim_buffer+idx+1))