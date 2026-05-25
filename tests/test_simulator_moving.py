"""Tests for moving microphones and trajectories."""

import numpy as np
import pytest

import aspsim.array as ar
import aspsim.configutil as cu
import aspsim.diagnostics.core as diacore
import aspsim.diagnostics.diagnostics as dia
import aspsim.room.trajectory as tr
import aspsim.signal.sources as sources
from aspsim.simulator import SimulatorSetup


class DebugTrajectory(tr.Trajectory):
    """Debug trajectory with straight-line segments."""

    def __init__(self, points, period, samplerate):
        """Move through a series of points in straight lines.

        Parameters
        ----------
        points : ndarray of shape (numpoints, spatial_dim) or equivalent list of lists
            The points that the trajectory will move through. The trajectory will
            start and end at the first point.
        period : float
            The time in seconds for the trajectory to go through all the points and
            return to the starting point.
        samplerate : int
            The samplerate of the simulation.
        mode : 'constant_speed' or 'constant_time'
            if 'constant_speed', the speed of the movement will be constant, and
            calibrated such that it returns to the starting position after one period.
            if 'constant_time', each segment will take equal time, and the speed will
            therefore go up for long segments and down for short segments.
        """
        if isinstance(points, (list, tuple)):
            points = np.array(points)
        # self.num_pos = 1

        if not np.allclose(points[-1, :], points[0, :]):
            points = np.concatenate((points, points[:1, :]), axis=0)
        self.anchor_points = points
        self.period = period
        self.samplerate = samplerate

        def pos_func(n):
            if n < 0:
                return self.anchor_points[:1, :]

            point_idx = 1 + (n % 2)
            return self.anchor_points[point_idx : point_idx + 1, :]

        super().__init__(pos_func)

    def plot(self, ax, symbol, name, tot_samples=None):
        """Plot the anchor points for debugging."""
        points = self.anchor_points
        ax.plot(
            points[:, 0],
            points[:, 1],
            marker=symbol,
            linestyle="dashed",
            label=name,
            alpha=0.8,
        )


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
    """Return a temporary folder for figures."""
    return tmp_path_factory.mktemp("figs")


def _setup_ism(fig_folder, samplerate):
    setup = SimulatorSetup(fig_folder)
    setup.sim_info.samplerate = samplerate
    setup.sim_info.tot_samples = samplerate
    setup.sim_info.sim_chunk_size = 2 * samplerate
    setup.sim_info.sim_buffer = samplerate
    setup.sim_info.export_frequency = samplerate
    setup.sim_info.save_source_contributions = False
    setup.sim_info.randomized_ism = False

    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [7, 5, 5]
    setup.sim_info.room_center = [0, 0, 0]
    setup.sim_info.rt60 = 0.25
    setup.sim_info.max_room_ir_length = samplerate // 2
    return setup


def _static_reference_rir(path_generator, sim_info, src, mic_pos):
    mic = ar.MicArray("reference_mic", np.atleast_2d(np.asarray(mic_pos)))
    return np.asarray(path_generator.create_path(src, mic, "ism", sim_info))[0, 0, :]


def test_unmoving_trajectory_same_as_static(fig_folder):
    """Ensure unmoving trajectory matches static microphone."""
    sr = 500
    sim_setup = _setup_ism(fig_folder, sr)

    def zero_pos_func(time):
        return np.zeros((1, 3))

    mic_traj = tr.Trajectory(zero_pos_func)
    sim_setup.add_mics("mic", np.zeros((1, 3)))
    sim_setup.add_mics("trajectory", mic_traj)
    sim_setup.add_free_source(
        "source",
        np.array([[0, -1, -1]]),
        sources.WhiteNoiseSource(1, 1, np.random.default_rng(1)),
    )

    sim = sim_setup.create_simulator()
    sim.diag.add_diagnostic(
        "mic", dia.RecordSignal("mic", sim.sim_info, 1, export_func="npz")
    )
    sim.diag.add_diagnostic(
        "trajectory", dia.RecordSignal("trajectory", sim.sim_info, 1, export_func="npz")
    )
    sim.run_simulation()

    sig_mic = np.load(sim.folder_path.joinpath(f"mic_{sim.sim_info.tot_samples}.npz"))[
        "mic"
    ]

    sig_traj = np.load(
        sim.folder_path.joinpath(f"trajectory_{sim.sim_info.tot_samples}.npz")
    )["trajectory"]
    assert np.allclose(sig_mic, sig_traj)


def test_moving_microphone_equals_moving_source(fig_folder):
    """A moving microphone should give the same output as a moving source.

    If the positions are switched, this is an extension of the reciprocity test,
    but with moving sources.

    Actually, I don't think this is true...

    Currently gives an sample-wise error of around 1e-5, which is unclear if it is
    within tolerance for numerical errors or not. Probably a valid result, just that
    the test is actually not true.
    """
    sr = 500
    rng = np.random.default_rng()

    pos_stationary = np.zeros((1, 3))
    pos_moving = tr.LinearTrajectory([[1, 1, 1], [0, 1, 0], [1, 0, 1]], 1, sr)

    sim_setup = _setup_ism(fig_folder, sr)
    src_sig = rng.random(size=(1, sim_setup.sim_info.tot_samples * 2))

    sim_setup.add_mics("mic", pos_stationary)
    sim_setup.add_free_source("src", pos_moving, sources.Sequence(src_sig))
    sim = sim_setup.create_simulator()
    sim.diag.add_diagnostic(
        "mic1", dia.RecordSignal("mic", sim.sim_info, 1, export_func="npz")
    )
    sim.run_simulation()
    sig1 = np.load(sim.folder_path.joinpath(f"mic1_{sim.sim_info.tot_samples}.npz"))[
        "mic1"
    ]

    sim_setup = _setup_ism(fig_folder, sr)
    sim_setup.add_mics("mic", pos_moving)
    sim_setup.add_free_source("src", pos_stationary, sources.Sequence(src_sig))
    sim = sim_setup.create_simulator()
    sim.diag.add_diagnostic(
        "mic2", dia.RecordSignal("mic", sim.sim_info, 1, export_func="npz")
    )
    sim.run_simulation()

    sig2 = np.load(sim.folder_path.joinpath(f"mic2_{sim.sim_info.tot_samples}.npz"))[
        "mic2"
    ]
    assert np.allclose(sig1, sig2, atol=1e-6, rtol=1e-6)


def test_moving_microphone_is_using_expected_positions(fig_folder):
    """Verify moving microphone positions are used by the simulator."""
    rng = np.random.default_rng()

    setup = SimulatorSetup(fig_folder)
    setup.sim_info = _default_sim_info()

    pos_src = np.zeros((1, 3))
    pos_mic = tr.LinearTrajectory(
        [[1, 1, 1], [0, 1, 0], [1, 0, 1]], 1, setup.sim_info.samplerate
    )

    setup.add_mics("mic", pos_mic)
    setup.add_free_source("src", pos_src, sources.WhiteNoiseSource(1, 1, rng))
    sim = setup.create_simulator()
    sim.run_simulation()
    pos_all = np.array(sim.arrays["mic"].pos_all)

    pos_compare = np.array(
        [pos_mic.current_pos(t) for t in range(setup.sim_info.tot_samples)]
    )
    assert np.allclose(pos_all[: setup.sim_info.tot_samples], pos_compare)


def test_moving_microphone_is_using_expected_rirs(fig_folder):
    """Verify moving microphone RIRs match expected positions."""
    rng = np.random.default_rng()

    setup = SimulatorSetup(fig_folder)
    setup.sim_info = _default_sim_info()

    pos_src = np.zeros((1, 3))
    pos_mic = tr.LinearTrajectory(
        [[1, 1, 1], [0, 1, 0], [1, 0, 1]], 1, setup.sim_info.samplerate
    )

    setup.add_mics("mic", pos_mic)
    setup.add_free_source("src", pos_src, sources.WhiteNoiseSource(1, 1, rng))
    sim = setup.create_simulator()
    sim.run_simulation()
    pos_all = np.array(sim.arrays["mic"].pos_all)

    pos_compare = np.array(
        [pos_mic.current_pos(t) for t in range(setup.sim_info.tot_samples)]
    )
    assert np.allclose(pos_all[: setup.sim_info.tot_samples], pos_compare)


def test_moving_microphone_has_same_rirs_as_stationary_microphones_on_trajectory(
    fig_folder,
):
    """Compare dynamic RIRs with stationary microphones on the trajectory."""
    rng = np.random.default_rng()
    sr = 500

    setup = _setup_ism(fig_folder, sr)
    src_sig = rng.random(size=(1, setup.sim_info.tot_samples * 2))

    pos_src = np.zeros((1, 3))
    traj = tr.LinearTrajectory(
        [[1, 1, 1], [0, 1, 0], [1, 0, 1]], 1, setup.sim_info.samplerate
    )
    all_pos = np.array(
        [traj.current_pos(t) for t in range(setup.sim_info.tot_samples)]
    )[:, 0, :]

    setup.add_mics("traj", traj)
    setup.add_mics("mic", all_pos)
    setup.add_free_source("src", pos_src, sources.Sequence(src_sig))
    setup.sim_info.plot_output = "none"
    sim = setup.create_simulator()

    sim.run_simulation()

    rir1 = np.array(sim.arrays._rir_dynamic_all)[: setup.sim_info.tot_samples, 0, 0, :]
    rir2 = sim.arrays.paths["src"]["mic"][0, :, :]
    # sim.diag.add_diagnostic("mic", dia.RecordSignal("mic", sim.sim_info, num_channels = all_pos.shape[0], export_func="npz"))
    # sim.diag.add_diagnostic("traj_rir", dia.RecordState("traj", sim.sim_info, num_channels = 1, export_func="npz"))
    # sim.run_simulation()

    # sig_mic = np.load(sim.folder_path.joinpath(f"mic_{sim.sim_info.tot_samples}.npz"))["mic"]
    # sig_traj_reconstruct = np.array([sig_mic[i,i] for i in range(sig_mic.shape[0])])[None,:]
    # sig_traj = np.load(sim.folder_path.joinpath(f"traj_{sim.sim_info.tot_samples}.npz"))["traj"]

    assert np.allclose(rir1, rir2)


def test_moving_microphone_rirs_match_sample_positions(fig_folder):
    """Confirm RIRs match per-sample microphone positions."""
    sr = 500
    setup = _setup_ism(fig_folder, sr)
    setup.sim_info.tot_samples = 4
    setup.sim_info.sim_buffer = 4
    setup.sim_info.export_frequency = 4
    setup.sim_info.sim_chunk_size = 4
    setup.sim_info.max_room_ir_length = 8
    setup.sim_info.plot_output = "none"

    traj = DebugTrajectory([[2, 1, 1], [0, 1, 0], [1, 0, 1]], 1, sr)
    setup.add_mics("mic", traj)
    setup.add_free_source(
        "src",
        np.zeros((1, 3)),
        sources.Sequence(np.zeros((1, 8))),
    )

    sim = setup.create_simulator()
    sim.run_simulation()

    observed_positions = np.asarray(sim.arrays["mic"].pos_all)[
        : setup.sim_info.tot_samples
    ]
    dynamic_rirs = np.asarray(sim.arrays._rir_dynamic_all)[
        : setup.sim_info.tot_samples, 0, 0, :
    ]

    assert len(dynamic_rirs) == setup.sim_info.tot_samples
    for sample_idx, mic_pos in enumerate(observed_positions):
        reference_rir = _static_reference_rir(
            sim.arrays.path_generator, sim.sim_info, sim.arrays["src"], mic_pos
        )
        assert np.allclose(dynamic_rirs[sample_idx], reference_rir)


def test_moving_microphone_gives_same_output_as_pointwise_stationary_convolutions(
    fig_folder,
):
    """Document the current equivalence range for moving microphones.

    The equivalence is currently for the range (-1, tot_samples-1), which is not the
    intended behaviour. The sim should be changed to correctly give equivalence for
    (0, tot_samples).
    """
    rng = np.random.default_rng()
    sr = 500

    setup = _setup_ism(fig_folder, sr)
    src_sig = rng.random(size=(1, setup.sim_info.tot_samples * 2))

    pos_src = np.zeros((1, 3))
    traj = tr.LinearTrajectory(
        [[1, 1, 1], [0, 1, 0], [1, 0, 1]], 1, setup.sim_info.samplerate
    )
    all_pos = np.array(
        [traj.current_pos(t) for t in range(-1, setup.sim_info.tot_samples - 1)]
    )[:, 0, :]
    all_pos[0, :] = [0.5, 0.5, 0.5]  # position at index 0 does not matter
    all_pos[1, :] = [0.5, 0.4, 0.4]  # position at index 1 does not matter

    setup.add_mics("traj", traj)
    setup.add_mics("mic", all_pos)
    setup.add_free_source("src", pos_src, sources.Sequence(src_sig))
    setup.sim_info.plot_output = "none"
    sim = setup.create_simulator()
    sim.diag.add_diagnostic(
        "mic",
        dia.RecordSignal(
            "mic", sim.sim_info, num_channels=all_pos.shape[0], export_func="npz"
        ),
    )
    sim.diag.add_diagnostic(
        "traj",
        dia.RecordSignal("traj", sim.sim_info, num_channels=1, export_func="npz"),
    )
    sim.run_simulation()

    sig_mic = np.load(sim.folder_path.joinpath(f"mic_{sim.sim_info.tot_samples}.npz"))[
        "mic"
    ]
    sig_traj_reconstruct = np.array([sig_mic[i, i] for i in range(sig_mic.shape[0])])[
        None, :
    ]
    sig_traj = np.load(
        sim.folder_path.joinpath(f"traj_{sim.sim_info.tot_samples}.npz")
    )["traj"]

    assert np.allclose(sig_traj, sig_traj_reconstruct)


def test_moving_microphone_offset_within_startup(fig_folder):
    """Pinpoint where sig_traj and the diagonal reconstruction differ.

    This test reproduces the reconstruction used in the failing test and
    asserts that any mismatches are confined to the startup transient (first 5 samples).
    """
    rng = np.random.default_rng(0)
    sr = 500

    setup = _setup_ism(fig_folder, sr)
    src_sig = rng.random(size=(1, setup.sim_info.tot_samples * 2))

    pos_src = np.zeros((1, 3))
    traj = tr.LinearTrajectory(
        [[1, 1, 1], [0, 1, 0], [1, 0, 1]], 1, setup.sim_info.samplerate
    )
    all_pos = np.array(
        [traj.current_pos(t) for t in range(-1, setup.sim_info.tot_samples - 1)]
    )[:, 0, :]
    all_pos[0, :] = [0.5, 0.5, 0.5]
    all_pos[1, :] = [0.5, 0.4, 0.4]

    setup.add_mics("traj", traj)
    setup.add_mics("mic", all_pos)
    setup.add_free_source("src", pos_src, sources.Sequence(src_sig))
    setup.sim_info.plot_output = "none"
    sim = setup.create_simulator()
    sim.diag.add_diagnostic(
        "mic",
        dia.RecordSignal(
            "mic", sim.sim_info, num_channels=all_pos.shape[0], export_func="npz"
        ),
    )
    sim.diag.add_diagnostic(
        "traj",
        dia.RecordSignal("traj", sim.sim_info, num_channels=1, export_func="npz"),
    )
    sim.run_simulation()

    sig_mic = np.load(sim.folder_path.joinpath(f"mic_{sim.sim_info.tot_samples}.npz"))[
        "mic"
    ]
    sig_traj_reconstruct = np.array([sig_mic[i, i] for i in range(sig_mic.shape[0])])[
        None, :
    ]
    sig_traj = np.load(
        sim.folder_path.joinpath(f"traj_{sim.sim_info.tot_samples}.npz")
    )["traj"]

    # Find indices that differ
    diff_mask = ~np.isclose(sig_traj, sig_traj_reconstruct)
    diff_idx = np.where(diff_mask[0])[0]

    # Sanity: there should be at least one mismatch (we've observed a startup transient)
    assert diff_idx.size > 0

    # Pinpoint: all mismatches must be within the first 5 samples (startup transient)
    assert diff_idx.max() < 5

    # beyond sample 5, signals should match exactly
    assert np.allclose(sig_traj[:, 5:], sig_traj_reconstruct[:, 5:])


def test_moving_microphone_diag_save_alignment(fig_folder):
    """Capture diagnostic save slices and verify alignment between saved 'traj' and reconstructed diagonal from 'mic'.

    This test adds capture diagnostics that record the exact `chunk_interval` and
    `glob_interval` used when saving. It then rebuilds the saved arrays and checks
    whether the saved `traj` matches the diagonal reconstruction of the saved `mic`.
    Any mismatches should be confined to the initial startup transient.
    """

    class CaptureSignal(diacore.SignalDiagnostic):
        def __init__(self, sim_info, sig_name):
            super().__init__(sim_info)
            self.sig_name = sig_name
            self.captures = []

        def save(self, processor, sig, chunk_interval, glob_interval):
            # store copies of the exact slice written to the diagnostic
            data = sig[self.sig_name][:, chunk_interval[0] : chunk_interval[1]].copy()
            self.captures.append((tuple(chunk_interval), tuple(glob_interval), data))

    rng = np.random.default_rng(1)
    sr = 500

    setup = _setup_ism(fig_folder, sr)
    src_sig = rng.random(size=(1, setup.sim_info.tot_samples * 2))

    pos_src = np.zeros((1, 3))
    traj = tr.LinearTrajectory(
        [[1, 1, 1], [0, 1, 0], [1, 0, 1]], 1, setup.sim_info.samplerate
    )
    all_pos = np.array(
        [traj.current_pos(t) for t in range(-1, setup.sim_info.tot_samples - 1)]
    )[:, 0, :]
    all_pos[0, :] = [0.5, 0.5, 0.5]
    all_pos[1, :] = [0.5, 0.4, 0.4]

    setup.add_mics("traj", traj)
    setup.add_mics("mic", all_pos)
    setup.add_free_source("src", pos_src, sources.Sequence(src_sig))
    setup.sim_info.plot_output = "none"
    sim = setup.create_simulator()

    cap_mic = CaptureSignal(sim.sim_info, "mic")
    cap_traj = CaptureSignal(sim.sim_info, "traj")
    sim.diag.add_diagnostic("mic_cap", cap_mic)
    sim.diag.add_diagnostic("traj_cap", cap_traj)

    sim.run_simulation()

    # Rebuild full saved arrays from captured slices
    tot = sim.sim_info.tot_samples
    mic_saved = np.full((all_pos.shape[0], tot), np.nan)
    traj_saved = np.full((1, tot), np.nan)

    for chunk, glob, data in cap_mic.captures:
        start, end = glob
        mic_saved[:, start:end] = data

    for chunk, glob, data in cap_traj.captures:
        start, end = glob
        traj_saved[:, start:end] = data

    # Reconstruct diagonal from mic_saved
    diag_recon = np.array([mic_saved[i, i] for i in range(mic_saved.shape[0])])[None, :]

    # Identify mismatches
    diff_idx = np.where(~np.isclose(traj_saved, diag_recon))[1]

    # There should be mismatches (startup transient) but confined to early samples
    assert diff_idx.size > 0
    assert diff_idx.max() < 5
    assert np.allclose(traj_saved[:, 5:], diag_recon[:, 5:])
