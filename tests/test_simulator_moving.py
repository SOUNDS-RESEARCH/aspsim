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


# def test_moving_microphone_equals_moving_source(fig_folder):
#     """This test is currently incorrect. In order for a moving microphone to be equivalent to a moving source, we also
#

#     If the positions are switched, this is an extension of the reciprocity test,
#     but with moving sources.

#     Currently gives an sample-wise error of around 1e-5, which is unclear if it is
#     within tolerance for numerical errors or not. Probably a valid result, just that
#     the test is actually not true.
#     """
#     sr = 500
#     rng = np.random.default_rng()

#     pos_stationary = np.zeros((1, 3))
#     pos_moving = tr.LinearTrajectory([[1, 1, 1], [0, 1, 0], [1, 0, 1]], 1, sr)

#     sim_setup = _setup_ism(fig_folder, sr)
#     src_sig = rng.random(size=(1, sim_setup.sim_info.tot_samples * 2))

#     sim_setup.add_mics("mic", pos_stationary)
#     sim_setup.add_free_source("src", pos_moving, sources.Sequence(src_sig))
#     sim = sim_setup.create_simulator()
#     sim.diag.add_diagnostic(
#         "mic1", dia.RecordSignal("mic", sim.sim_info, 1, export_func="npz")
#     )
#     sim.run_simulation()
#     sig1 = np.load(sim.folder_path.joinpath(f"mic1_{sim.sim_info.tot_samples}.npz"))[
#         "mic1"
#     ]

#     sim_setup = _setup_ism(fig_folder, sr)
#     sim_setup.add_mics("mic", pos_moving)
#     sim_setup.add_free_source("src", pos_stationary, sources.Sequence(src_sig))
#     sim = sim_setup.create_simulator()
#     sim.diag.add_diagnostic(
#         "mic2", dia.RecordSignal("mic", sim.sim_info, 1, export_func="npz")
#     )
#     sim.run_simulation()

#     sig2 = np.load(sim.folder_path.joinpath(f"mic2_{sim.sim_info.tot_samples}.npz"))[
#         "mic2"
#     ]
#     assert np.allclose(sig1, sig2, atol=1e-6, rtol=1e-6)


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
    mic = sim.arrays["mic"]
    t0 = mic._time_zero_idx
    pos_all = mic.pos_all

    pos_compare = np.array(
        [pos_mic.current_pos(t) for t in range(setup.sim_info.tot_samples)]
    )
    assert np.allclose(pos_all[t0 : t0 + setup.sim_info.tot_samples], pos_compare)


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
    mic = sim.arrays["mic"]
    t0 = mic._time_zero_idx
    pos_all = mic.pos_all

    pos_compare = np.array(
        [pos_mic.current_pos(t) for t in range(setup.sim_info.tot_samples)]
    )
    assert np.allclose(pos_all[t0 : t0 + setup.sim_info.tot_samples], pos_compare)


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

    t0 = sim.arrays["traj"]._time_zero_idx
    rir1 = sim.arrays.rir_all["src"]["traj"][
        t0 : t0 + setup.sim_info.tot_samples, 0, 0, :
    ]
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

    mic = sim.arrays["mic"]
    t0 = mic._time_zero_idx
    observed_positions = mic.pos_all[t0 : t0 + setup.sim_info.tot_samples]
    dynamic_rirs = sim.arrays.rir_all["src"]["mic"][
        t0 : t0 + setup.sim_info.tot_samples, 0, 0, :
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
    """Moving-mic at saved index i must equal static-mic at the trajectory position.

    Specifically, equivalence holds for the full range (0, tot_samples).
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
        [traj.current_pos(t) for t in range(setup.sim_info.tot_samples)]
    )[:, 0, :]

    setup.add_mics("traj", traj)
    setup.add_mics("mic", all_pos)
    setup.add_free_source("src", pos_src, sources.Sequence(src_sig))
    # setup.sim_info.plot_output = "none"
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


def test_adding_sensor_noise_to_moving_microphone_is_equivalent_to_adding_noise_afterwards(
    fig_folder,
):
    """Confirm that adding noise (without an impulse response) to a moving microphone is the same as adding noise after convolution."""
    rng = np.random.default_rng()
    rt60 = 0.0
    sr = 1000
    num_mic = 3
    seq_len_frac_of_sec = 2
    seq_len = sr // seq_len_frac_of_sec

    pos_src = np.array([[2, 0, 0]])

    setup = SimulatorSetup(fig_folder)
    setup.sim_info.samplerate = sr

    target_speed = 0.5
    tot_trajectory_samples = num_mic * seq_len
    freq_factors = rng.uniform(low=1, high=4, size=(1, 3))
    traj_amp = np.array([[1.0, 1.0, 0.2]])
    trajectory = tr.LissajousTrajectoryConstantSpeed(
        traj_amp,
        target_speed * freq_factors / sr,
        np.zeros((1, 3)),
        sr,
        target_speed,
        tot_trajectory_samples,
    )
    setup.sim_info.tot_samples = tot_trajectory_samples
    setup.sim_info.export_frequency = setup.sim_info.tot_samples
    setup.sim_info.reverb = "ism"
    setup.sim_info.room_size = [5.4, 4.3, 3.2]
    setup.sim_info.room_center = [0.8, 0.2, 0.1]
    setup.sim_info.rt60 = rt60
    setup.sim_info.max_room_ir_length = seq_len
    setup.sim_info.array_update_freq = 1
    setup.sim_info.randomized_ism = False
    setup.sim_info.auto_save_load = False
    setup.sim_info.sim_buffer = sr // seq_len_frac_of_sec
    setup.sim_info.extra_delay = 40
    setup.sim_info.plot_output = "pdf"
    setup.sim_info.start_sources_before_0 = False
    setup.sim_info.save_source_contributions = True
    setup.sim_info.highpass_cutoff = 0

    seq_len = setup.sim_info.max_room_ir_length

    sequence_src = sources.Sequence(
        rng.normal(0, 1, size=(1, seq_len)), end_mode="repeat"
    )

    pos_noise = np.array([[0, 1.5, 0]])
    noise_data = rng.normal(0, 1, size=(1, tot_trajectory_samples))
    noise_source = sources.Sequence(noise_data, end_mode="repeat")
    setup.add_free_source("noise", pos_noise, noise_source)
    setup.arrays.path_type["noise"]["mic_with_noise"] = "direct"
    setup.arrays.path_type["noise"]["mic_without_noise"] = "none"

    setup.add_free_source("src", pos_src, sequence_src)
    setup.add_mics("mic_with_noise", trajectory)
    setup.add_mics("mic_without_noise", trajectory)

    sim = setup.create_simulator()

    sim.diag.add_diagnostic(
        "mic_with_noise",
        dia.RecordSignal(
            "mic_with_noise",
            sim.sim_info,
            num_channels=sim.arrays["mic_with_noise"].num,
            export_func="npz",
        ),
    )
    sim.diag.add_diagnostic(
        "mic_without_noise",
        dia.RecordSignal(
            "mic_without_noise",
            sim.sim_info,
            num_channels=sim.arrays["mic_without_noise"].num,
            export_func="npz",
        ),
    )
    sim.diag.add_diagnostic(
        "src",
        dia.RecordSignal(
            "src", sim.sim_info, num_channels=sim.arrays["src"].num, export_func="npz"
        ),
    )
    sim.diag.add_diagnostic(
        "noise",
        dia.RecordSignal(
            "noise",
            sim.sim_info,
            num_channels=sim.arrays["noise"].num,
            export_func="npz",
        ),
    )

    sim.run_simulation()

    sim_sig_mic_with_noise = np.load(
        sim.folder_path / f"mic_with_noise_{sim.sim_info.tot_samples}.npz"
    )["mic_with_noise"]
    sim_sig_mic_without_noise = np.load(
        sim.folder_path / f"mic_without_noise_{sim.sim_info.tot_samples}.npz"
    )["mic_without_noise"]
    sim_sig_noise = np.load(sim.folder_path / f"noise_{sim.sim_info.tot_samples}.npz")[
        "noise"
    ]

    assert np.allclose(sim_sig_noise, noise_data)

    assert np.allclose(
        sim_sig_mic_with_noise, sim_sig_mic_without_noise + sim_sig_noise
    )
