import numpy as np
import hypothesis as hyp
#from hypothesis import given
import hypothesis.strategies as st
import pytest
#import sys
#sys.path.append("c:/skola/utokyo_lab/ancsim/ancsim")

from aspsim.simulator import SimulatorSetup
import aspsim.array as ar
import aspsim.processor as bse
import aspsim.diagnostics.core as diacore
import aspsim.diagnostics.diagnostics as dia
import aspsim.fileutilities as fu
import aspsim.signal.sources as sources


@pytest.fixture(scope="session")
def fig_folder(tmp_path_factory):
    return tmp_path_factory.mktemp("figs")

def simple_setup(fig_folder):
    setup = SimulatorSetup(fig_folder)
    setup.sim_info.tot_samples = 20
    setup.sim_info.sim_buffer = 20
    setup.sim_info.export_frequency = 20
    setup.sim_info.sim_chunk_size = 20

    setup.add_free_source("src", np.array([[1,0,0]]), sources.WhiteNoiseSource(1,1))
    setup.add_controllable_source("loudspeaker", np.array([[1,0,0]]))
    setup.add_mics("mic", np.array([[0,0,0]]))

    setup.arrays.path_type["loudspeaker"]["mic"] = "none"
    setup.arrays.path_type["src"]["mic"] = "direct"
    return setup

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=10), 
           export_freq = st.integers(min_value=1, max_value=10))
def test_processor_sees_same_mic_samples_as_is_logged_in_record_signal(fig_folder, bs, export_freq):
    setup = simple_setup(fig_folder)
    setup.sim_info.export_frequency = export_freq

    sim = setup.create_simulator()
    proc = bse.DebugProcessor(sim.sim_info, sim.arrays, bs)
    sim.diag.add_diagnostic("mic", dia.RecordSignal("mic", sim.sim_info, export_func="npz"))
    sim.add_processor(proc)
    sim.run_simulation()

    final_export_idx = sim.sim_info.export_frequency * (sim.sim_info.tot_samples // sim.sim_info.export_frequency)
    signal_log = np.load(sim.folder_path.joinpath(f"mic_{final_export_idx}.npz"))["mic"]
    signal_true = sim.processors[0].mic
    assert np.allclose(signal_log, signal_true[:,:final_export_idx])



@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=10))
def test_record_signal_is_same_with_or_without_a_processor(fig_folder, bs):
    setup = simple_setup(fig_folder)
    final_export_idx = setup.sim_info.export_frequency * (setup.sim_info.tot_samples // setup.sim_info.export_frequency)

    sim = setup.create_simulator()
    sim.diag.add_diagnostic("mic", dia.RecordSignal("mic", sim.sim_info, export_func="npz"))
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()

    sig_with_proc = np.load(sim.folder_path.joinpath(f"mic_{final_export_idx}.npz"))["mic"]

    sim = setup.create_simulator()
    sim.diag.add_diagnostic("mic", dia.RecordSignal("mic", sim.sim_info, export_func="npz"))
    sim.run_simulation()

    sig_without_proc = np.load(sim.folder_path.joinpath(f"mic_{final_export_idx}.npz"))["mic"]

    assert np.allclose(sig_with_proc, sig_without_proc)


@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=10),
           export_freq = st.integers(min_value=2, max_value=10),
           tot_samples = st.integers(min_value=10, max_value=30),
           )
def test_signal_diagnostics_correct_files_saved(fig_folder, bs, export_freq, tot_samples):
    sim_setup = simple_setup(fig_folder)
    sim_setup.sim_info.tot_samples = tot_samples
    sim_setup.sim_info.export_frequency = export_freq
    sim_setup.sim_info.sim_chunk_size = 5
    sim_setup.sim_info.sim_buffer = 20

    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.diag.add_diagnostic("mic", dia.RecordSignal("mic", sim.sim_info, export_func="npz"))
    sim.run_simulation()

    all_saved_files = list(sim.folder_path.iterdir())
    num_files_to_save = sim.sim_info.tot_samples // sim_setup.sim_info.export_frequency
    expected_files = [sim.folder_path.joinpath(f"mic_{i*sim_setup.sim_info.export_frequency}.npz") for i in range(1, num_files_to_save+1)]

    # Check all expected files exist
    for f in expected_files:
        assert f.exists()

    # Check there are no other npz files except the expected files
    for f in all_saved_files:
        if f.suffix == ".npz":
            assert f in expected_files


@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5),
            buf_size = st.integers(min_value=10, max_value=30),
            num_proc = st.integers(min_value=1, max_value=3))
def test_all_samples_saved_for_signal_diagnostics(fig_folder, bs, buf_size, num_proc):
    sim_setup = simple_setup(fig_folder)
    sim_setup.sim_info.sim_buffer = buf_size
    sim = sim_setup.create_simulator()
    for _ in range(num_proc):
        sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                    diagnostics={"mic":dia.RecordSignal("mic", sim.sim_info, bs, export_func="npz")}))

    sim.run_simulation()
    
    at_least_one_file_saved = False
    for f in sim.folder_path.iterdir():
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
def test_correct_intermediate_samples_saved_for_signal_diagnostics(fig_folder, bs, buf_size):
    sim_setup = simple_setup(fig_folder)
    sim_setup.sim_info.sim_buffer = buf_size
    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                   diagnostics={"mic":dia.RecordSignal("mic", sim.sim_info, bs, export_func="npz", keep_only_last_export=False)}))

    sim.run_simulation()
    
    for f in sim.folder_path.iterdir():
        if f.stem.startswith("mic"):
            idx = fu.find_index_in_name(f.stem)
            saved_data = np.load(f)
            for proc_name, data in saved_data.items():
                assert np.allclose(data, np.arange(sim.sim_info.sim_buffer, 
                            sim.sim_info.sim_buffer+idx))
    

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_export_file_naming_interval_diagnostics(fig_folder, bs):
    sim_setup = simple_setup(fig_folder)
    sim = sim_setup.create_simulator()

    save_intervals = ((32,46), (68,69), (71, 99))
    diag_name = "mic"
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                   diagnostics={"mic":dia.RecordSignal(diag_name, sim.sim_info, bs, 
                   export_at = [iv[1] for iv in save_intervals],
                    save_at=diacore.IntervalCounter(save_intervals), 
                    export_func="npz")}))
    sim.run_simulation()

    for iv in save_intervals:
        assert sim.folder_path.joinpath(f"{diag_name}_{iv[1]}.npz").exists()



@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_correct_samples_saved_for_interval_diagnostics(fig_folder, bs):
    sim_setup = simple_setup(fig_folder)
    sim = sim_setup.create_simulator()

    save_intervals = ((32,46), (68,69), (71, 99))
    diag_name = "mic"
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                   diagnostics={"mic":dia.RecordSignal(
                       diag_name, sim.sim_info, bs, 
                   export_at = [iv[1] for iv in save_intervals],
                    save_at = diacore.IntervalCounter(save_intervals), 
                    export_func="npz")
                    }))

    sim.run_simulation()

    expected = np.zeros(0)
    for iv in save_intervals:
        saved_data = np.load(sim.folder_path.joinpath(f"{diag_name}_{iv[1]}.npz"))
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
def test_all_samples_saved_state_diagnostics(fig_folder, bs, buf_size, num_proc):
    sim_setup = simple_setup(fig_folder)
    #bs = 1
    #sim_setup.sim_info.tot_samples = 13
    #sim_setup.sim_info.sim_chunk_size = 5
    sim_setup.sim_info.sim_buffer = buf_size

    sim = sim_setup.create_simulator()

    #save_at = diacore.IntervalCounter(np.arange(1,sim.sim_info.tot_samples+1))
    for _ in range(num_proc):
        sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                    diagnostics={"state":dia.RecordState("processed_samples", 1, sim.sim_info, bs, export_func="npz")}))

    sim.run_simulation()
    
    one_file_saved = False
    for f in sim.folder_path.iterdir():
        if f.stem.startswith("state"):
            one_file_saved = True
            idx = fu.find_index_in_name(f.stem)
            saved_data = np.load(f)
            for proc_name, data in saved_data.items():
                assert np.allclose(data, np.arange(bs, idx+1, bs))
    assert one_file_saved


@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_correct_samples_saved_for_instant_diagnostics(fig_folder, bs):
    sim_setup = simple_setup(fig_folder)
    sim = sim_setup.create_simulator()

    #save_at = np.arange(bs, sim.sim_info.tot_samples, bs)#(bs,)
    #save_at = [bs*i for i in range(1, sim.sim_info.tot_samples//bs)]
    save_at = (bs, 2*bs, 5*bs)
    #save_intervals = ((1,2), (3,4), (5,6))
    diag_name = "filt"
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                   diagnostics={diag_name:dia.RecordFilter(
                       "filt.ir", sim.sim_info, bs, save_at = save_at, export_func="npz")}))
                #        "mic":dia.RecordSignal(
                #        "mic", sim.sim_info, bs, 
                #    export_at = [iv[1] for iv in save_intervals],
                #     save_at = diacore.IntervalCounter(save_intervals), 
                #     export_func="npz")
                #     }))

    sim.run_simulation()

    for idx in save_at:
        saved_data = np.load(sim.folder_path.joinpath(f"{diag_name}_{idx}.npz"))
        for proc_name, data in saved_data.items():
            assert np.allclose(data, np.zeros_like(data)+idx)

@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_correct_samples_saved_for_instant_diagnostics_savefreq(fig_folder, bs):
    sim_setup = simple_setup(fig_folder)
    sim = sim_setup.create_simulator()

    save_at = bs
    diag_name = "filt"
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs, 
                   diagnostics={diag_name:dia.RecordFilter(
                       "filt.ir", sim.sim_info, bs, save_at = save_at, export_func="npz")}))

    sim.run_simulation()

    for idx in range(save_at, sim.sim_info.tot_samples+1, save_at):
        saved_data = np.load(sim.folder_path.joinpath(f"{diag_name}_{idx}.npz"))
        for proc_name, data in saved_data.items():
            assert np.allclose(data, np.zeros_like(data)+idx)



@hyp.settings(deadline=None)
@hyp.given(bs = st.integers(min_value=1, max_value=5))
def test_two_processors_with_different_diagnostics(fig_folder, bs):
    sim_setup = simple_setup(fig_folder)
    sim = sim_setup.create_simulator()

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

    sim.add_processor(proc1)
    sim.add_processor(proc2)
    sim.run_simulation()

    for f in sim.folder_path.iterdir():
        if f.stem.startswith("mic"):
            idx = fu.find_index_in_name(f.stem)
            saved_data = np.load(f)
            for proc_name, data in saved_data.items():
                assert np.allclose(data[:idx+1], np.arange(sim.sim_info.sim_buffer, 
                            sim.sim_info.sim_buffer+idx+1))

