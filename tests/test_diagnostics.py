"""Diagnostics-related tests."""

import hypothesis as hyp
import hypothesis.strategies as st
import numpy as np
import pytest

import aspsim.diagnostics.core as diacore
import aspsim.diagnostics.diagnostics as dia
import aspsim.fileutilities as fu
import aspsim.processor as bse
import aspsim.signal.sources as sources
from aspsim.simulator import SimulatorSetup


@pytest.fixture(scope="session")
def fig_folder(tmp_path_factory):
    """Create a temporary folder for diagnostic figures.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Factory for temporary paths.

    Returns
    -------
    pathlib.Path
        Temporary folder path.
    """
    return tmp_path_factory.mktemp("figs")


def simple_setup(fig_folder):
    """Create a minimal simulator setup for diagnostics tests.

    Parameters
    ----------
    fig_folder : pathlib.Path
        Folder for diagnostic output.

    Returns
    -------
    SimulatorSetup
        Prepared simulator setup.
    """
    setup = SimulatorSetup(fig_folder)
    setup.sim_info.tot_samples = 20
    setup.sim_info.sim_buffer = 20
    setup.sim_info.export_frequency = 20
    setup.sim_info.sim_chunk_size = 20
    setup.sim_info.plot_output = "pdf"

    setup.add_free_source("src", np.array([[1, 0, 0]]), sources.Counter(1))
    setup.add_controllable_source("loudspeaker", np.array([[1, 0, 0]]))
    setup.add_mics("mic", np.array([[0, 0, 0]]))

    setup.arrays.path_type["loudspeaker"]["mic"] = "none"
    setup.arrays.path_type["src"]["mic"] = "direct"
    return setup


@hyp.settings(deadline=None)
@hyp.given(
    bs=st.integers(min_value=1, max_value=10),
    export_freq=st.integers(min_value=1, max_value=10),
)
def test_processor_sees_same_mic_samples_as_is_logged_in_record_signal(
    fig_folder, bs, export_freq
):
    """Check processor mic samples match recorded diagnostics.

    Parameters
    ----------
    fig_folder : pathlib.Path
        Folder for diagnostic output.
    bs : int
        Block size.
    export_freq : int
        Export frequency.
    """
    setup = simple_setup(fig_folder)
    setup.sim_info.export_frequency = export_freq

    sim = setup.create_simulator()
    proc = bse.DebugProcessor(sim.sim_info, sim.arrays, bs)
    sim.diag.add_diagnostic(
        "mic", dia.RecordSignal("mic", sim.sim_info, export_func="npz")
    )
    sim.add_processor(proc)
    sim.run_simulation()

    final_export_idx = sim.sim_info.export_frequency * (
        sim.sim_info.tot_samples // sim.sim_info.export_frequency
    )
    signal_log = np.load(sim.folder_path.joinpath(f"mic_{final_export_idx}.npz"))["mic"]
    signal_true = sim.processors[0].mic
    # signal_log[i] is mic at sim time i; the processor reads before propagate, so
    # signal_true[i] is mic at sim time i-1. Compare with a one-sample shift.
    assert np.allclose(
        signal_log[:, : final_export_idx - 1], signal_true[:, 1:final_export_idx]
    )


@hyp.settings(deadline=None)
@hyp.given(bs=st.integers(min_value=1, max_value=10))
def test_record_signal_is_same_with_or_without_a_processor(fig_folder, bs):
    """Check record signal output matches with or without a processor.

    Parameters
    ----------
    fig_folder : pathlib.Path
        Folder for diagnostic output.
    bs : int
        Block size.
    """
    setup = simple_setup(fig_folder)
    final_export_idx = setup.sim_info.export_frequency * (
        setup.sim_info.tot_samples // setup.sim_info.export_frequency
    )

    sim = setup.create_simulator()
    sim.diag.add_diagnostic(
        "mic", dia.RecordSignal("mic", sim.sim_info, export_func="npz")
    )
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()

    sig_with_proc = np.load(sim.folder_path.joinpath(f"mic_{final_export_idx}.npz"))[
        "mic"
    ]

    sim = setup.create_simulator()
    sim.diag.add_diagnostic(
        "mic", dia.RecordSignal("mic", sim.sim_info, export_func="npz")
    )
    sim.run_simulation()

    sig_without_proc = np.load(sim.folder_path.joinpath(f"mic_{final_export_idx}.npz"))[
        "mic"
    ]

    assert np.allclose(sig_with_proc, sig_without_proc)


@hyp.settings(deadline=None)
@hyp.given(
    bs=st.integers(min_value=1, max_value=10),
    export_freq=st.integers(min_value=2, max_value=10),
    tot_samples=st.integers(min_value=10, max_value=30),
)
def test_signal_diagnostics_correct_files_saved(
    fig_folder, bs, export_freq, tot_samples
):
    """Verify expected diagnostic files are saved.

    Parameters
    ----------
    fig_folder : pathlib.Path
        Folder for diagnostic output.
    bs : int
        Block size.
    export_freq : int
        Export frequency.
    tot_samples : int
        Total number of samples.
    """
    sim_setup = simple_setup(fig_folder)
    sim_setup.sim_info.tot_samples = tot_samples
    sim_setup.sim_info.export_frequency = export_freq
    sim_setup.sim_info.sim_chunk_size = 5
    sim_setup.sim_info.sim_buffer = 20

    sim = sim_setup.create_simulator()
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.diag.add_diagnostic(
        "mic", dia.RecordSignal("mic", sim.sim_info, export_func="npz")
    )
    sim.run_simulation()

    all_saved_files = list(sim.folder_path.iterdir())
    num_files_to_save = sim.sim_info.tot_samples // sim_setup.sim_info.export_frequency
    expected_files = [
        sim.folder_path.joinpath(f"mic_{i * sim_setup.sim_info.export_frequency}.npz")
        for i in range(1, num_files_to_save + 1)
    ]

    # Check all expected files exist
    for f in expected_files:
        assert f.exists()

    # Check there are no other npz files except the expected files
    for f in all_saved_files:
        if f.suffix == ".npz":
            assert f in expected_files


@hyp.settings(deadline=None)
@hyp.given(
    bs=st.integers(min_value=1, max_value=5),
    buf_size=st.integers(min_value=10, max_value=30),
)
def test_all_samples_saved_for_signal_diagnostics(fig_folder, bs, buf_size):
    """Check all samples are saved for signal diagnostics.

    Parameters
    ----------
    fig_folder : pathlib.Path
        Folder for diagnostic output.
    bs : int
        Block size.
    buf_size : int
        Simulation buffer size.
    """
    sim_setup = simple_setup(fig_folder)
    sim_setup.sim_info.sim_buffer = buf_size
    sim_setup.sim_info.export_frequency = sim_setup.sim_info.tot_samples
    sim = sim_setup.create_simulator()
    sim.diag.add_diagnostic(
        "mic", dia.RecordSignal("mic", sim.sim_info, export_func="npz")
    )
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))

    sim.run_simulation()

    at_least_one_file_saved = False
    for f in sim.folder_path.iterdir():
        if f.stem.startswith("mic"):
            at_least_one_file_saved = True
            saved_data = np.load(f)
            for proc_name, data in saved_data.items():
                assert np.allclose(data, np.arange(sim.sim_info.tot_samples))
    assert at_least_one_file_saved


@hyp.settings(deadline=None)
@hyp.given(
    bs=st.integers(min_value=1, max_value=5),
    buf_size=st.integers(min_value=10, max_value=30),
)
def test_correct_intermediate_samples_saved_for_signal_diagnostics(
    fig_folder, bs, buf_size
):
    """Check intermediate samples saved for signal diagnostics.

    Parameters
    ----------
    fig_folder : pathlib.Path
        Folder for diagnostic output.
    bs : int
        Block size.
    buf_size : int
        Simulation buffer size.
    """
    sim_setup = simple_setup(fig_folder)
    sim_setup.sim_info.sim_buffer = buf_size
    export_at = [4, 8, 12, 16, 20]
    sim = sim_setup.create_simulator()
    sim.diag.add_diagnostic(
        "mic",
        dia.RecordSignal(
            "mic",
            sim.sim_info,
            export_at=export_at,
            export_func="npz",
            keep_only_last_export=False,
        ),
    )
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))

    sim.run_simulation()

    at_least_one_file_saved = False
    for f in sim.folder_path.iterdir():
        if f.stem.startswith("mic"):
            at_least_one_file_saved = True
            idx = fu.find_index_in_name(f.stem)
            saved_data = np.load(f)
            for proc_name, data in saved_data.items():
                assert np.allclose(data[:, :idx], np.arange(idx))
    assert at_least_one_file_saved


@hyp.settings(deadline=None)
@hyp.given(bs=st.integers(min_value=1, max_value=5))
def test_export_file_naming_interval_diagnostics(fig_folder, bs):
    """Check export file naming for interval diagnostics.

    Parameters
    ----------
    fig_folder : pathlib.Path
        Folder for diagnostic output.
    bs : int
        Block size.
    """
    sim_setup = simple_setup(fig_folder)
    sim_setup.sim_info.tot_samples = 100
    sim = sim_setup.create_simulator()

    save_intervals = ((32, 46), (68, 69), (71, 99))
    diag_name = "mic"
    sim.diag.add_diagnostic(
        diag_name,
        dia.RecordSignal(
            diag_name,
            sim.sim_info,
            export_at=[iv[1] for iv in save_intervals],
            save_at=diacore.IntervalCounter(save_intervals),
            export_func="npz",
        ),
    )
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))
    sim.run_simulation()

    for iv in save_intervals:
        assert sim.folder_path.joinpath(f"{diag_name}_{iv[1]}.npz").exists()


@hyp.settings(deadline=None)
@hyp.given(bs=st.integers(min_value=1, max_value=5))
def test_correct_samples_saved_for_interval_diagnostics(fig_folder, bs):
    """Check saved samples for interval diagnostics.

    Parameters
    ----------
    fig_folder : pathlib.Path
        Folder for diagnostic output.
    bs : int
        Block size.
    """
    sim_setup = simple_setup(fig_folder)
    sim_setup.sim_info.tot_samples = 100
    sim = sim_setup.create_simulator()

    save_intervals = ((32, 46), (68, 69), (71, 99))
    diag_name = "mic"
    sim.diag.add_diagnostic(
        diag_name,
        dia.RecordSignal(
            diag_name,
            sim.sim_info,
            export_at=[iv[1] for iv in save_intervals],
            save_at=diacore.IntervalCounter(save_intervals),
            export_func="npz",
        ),
    )
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))

    sim.run_simulation()

    expected = np.zeros(0)
    for iv in save_intervals:
        saved_data = np.load(sim.folder_path.joinpath(f"{diag_name}_{iv[1]}.npz"))
        expected = np.concatenate((expected, np.arange(iv[0], iv[1])))
        for proc_name, data in saved_data.items():
            assert np.allclose(data[0, : expected.shape[0]], expected, equal_nan=True)


@hyp.settings(deadline=None)
@hyp.given(
    bs=st.integers(min_value=1, max_value=5),
    buf_size=st.integers(min_value=10, max_value=30),
)
def test_all_samples_saved_state_diagnostics(fig_folder, bs, buf_size):
    """Check all samples saved for state diagnostics.

    Parameters
    ----------
    fig_folder : pathlib.Path
        Folder for diagnostic output.
    bs : int
        Block size.
    buf_size : int
        Simulation buffer size.
    """
    sim_setup = simple_setup(fig_folder)
    sim_setup.sim_info.sim_buffer = buf_size

    sim = sim_setup.create_simulator()

    last_save_idx = (sim.sim_info.tot_samples // bs) * bs
    sim.diag.add_diagnostic(
        "state",
        dia.RecordState(
            "processed_samples",
            1,
            sim.sim_info,
            export_at=last_save_idx,
            save_frequency=bs,
            export_func="npz",
        ),
    )
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))

    sim.run_simulation()

    one_file_saved = False
    for f in sim.folder_path.iterdir():
        if f.stem.startswith("state"):
            one_file_saved = True
            idx = fu.find_index_in_name(f.stem)
            saved_data = np.load(f)
            num_saved = idx // bs
            for proc_name, data in saved_data.items():
                values = data[0, :num_saved]
                assert np.all(np.diff(values) == bs)
                assert values[0] >= bs
                assert values[-1] <= idx + bs
    assert one_file_saved


@hyp.settings(deadline=None)
@hyp.given(bs=st.integers(min_value=1, max_value=5))
def test_correct_samples_saved_for_instant_diagnostics(fig_folder, bs):
    """Check saved samples for instant diagnostics.

    Parameters
    ----------
    fig_folder : pathlib.Path
        Folder for diagnostic output.
    bs : int
        Block size.
    """
    sim_setup = simple_setup(fig_folder)
    sim = sim_setup.create_simulator()

    # save_at = np.arange(bs, sim.sim_info.tot_samples, bs)#(bs,)
    # save_at = [bs*i for i in range(1, sim.sim_info.tot_samples//bs)]
    save_at = (bs, 2 * bs, 3 * bs)
    diag_name = "filt"
    sim.diag.add_diagnostic(
        diag_name,
        dia.RecordFilter(
            "filt.ir", sim.sim_info, save_at=save_at, export_func="npz"
        ),
    )
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))

    sim.run_simulation()

    for idx in save_at:
        saved_data = np.load(sim.folder_path.joinpath(f"{diag_name}_{idx}.npz"))
        expected = ((idx + 1) // bs) * bs
        for proc_name, data in saved_data.items():
            assert np.allclose(data, np.zeros_like(data) + expected)


@hyp.settings(deadline=None)
@hyp.given(bs=st.integers(min_value=1, max_value=5))
def test_correct_samples_saved_for_instant_diagnostics_savefreq(fig_folder, bs):
    """Check saved samples for instant diagnostics with save frequency.

    Parameters
    ----------
    fig_folder : pathlib.Path
        Folder for diagnostic output.
    bs : int
        Block size.
    """
    sim_setup = simple_setup(fig_folder)
    sim = sim_setup.create_simulator()

    save_at = bs
    diag_name = "filt"
    sim.diag.add_diagnostic(
        diag_name,
        dia.RecordFilter(
            "filt.ir", sim.sim_info, save_at=save_at, export_func="npz"
        ),
    )
    sim.add_processor(bse.DebugProcessor(sim.sim_info, sim.arrays, bs))

    sim.run_simulation()

    for idx in range(save_at, sim.sim_info.tot_samples + 1, save_at):
        saved_data = np.load(sim.folder_path.joinpath(f"{diag_name}_{idx}.npz"))
        expected = ((idx + 1) // bs) * bs
        for proc_name, data in saved_data.items():
            assert np.allclose(data, np.zeros_like(data) + expected)
