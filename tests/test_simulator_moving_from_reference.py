"""Tests for moving simulations using a reference implementation."""

import copy
import json
import pathlib

import aspcore.fouriertransform as ft
import aspcore.pseq as pseq
import aspcore.utilities as utils
import jax
import numpy as np
import pytest

import aspsim.diagnostics.diagnostics as dg
import aspsim.room.region as reg
import aspsim.room.trajectory as tr
import aspsim.saveloadsession as sls
import aspsim.signal.sources as sources
from aspsim.simulator import SimulatorSetup

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="session")
def parent_folder(tmp_path_factory):
    """Return a temporary folder for figures."""
    return tmp_path_factory.mktemp("figs")


def db2pow(db):
    """Convert a decibel value to power."""
    return 10 ** (db / 10)


def pow2db(power):
    """Convert a power value to decibels."""
    return 10 * np.log10(power)


def generate_constant_speed_trajectory(
    position, velocity, samplerate, target_speed, num_samples, tolerance=0.05
):
    """Generate a constant-speed trajectory.

    Parameters
    ----------
    target_speed : int
        Target speed in meters per second.
    """
    all_pos = np.zeros((num_samples, 3))
    all_pos[0, :] = position(0)

    target_speed_per_sample = target_speed / samplerate

    t = 0
    for n in range(1, num_samples):
        last_pos = all_pos[n - 1, :]
        v = np.linalg.norm(velocity(t))
        timestep = target_speed_per_sample / v

        candidate_pos = position(t + timestep)
        speed = np.linalg.norm(candidate_pos - last_pos)

        if (
            np.abs(speed - target_speed_per_sample) / target_speed_per_sample
            > tolerance
        ):
            if speed > target_speed_per_sample:
                t_low = 0
                t_high = timestep
            else:
                t_low = timestep
                t_high = 2 * timestep
                while (
                    np.linalg.norm(position(t + t_high) - last_pos)
                    <= target_speed_per_sample
                ):
                    t_high = 2 * t_high
            timestep = pos_bifurcation(
                t_low, t_high, t, position, last_pos, target_speed_per_sample, tolerance
            )
            candidate_pos = position(t + timestep)

        t += timestep
        all_pos[n, :] = candidate_pos

    return all_pos


def pos_bifurcation(low, high, offset, pos, previous_pos, desired_val, tolerance):
    """Find a time step that achieves a desired displacement."""
    speed = -1000

    while np.abs(speed - desired_val) / desired_val > tolerance:
        t = (low + high) / 2
        p = pos(offset + t)

        speed = np.linalg.norm(p - previous_pos)

        if speed <= desired_val:
            low = t
        else:
            high = t
    return t


class LissajousTrajectoryConstantSpeed(tr.Trajectory):
    """Lissajous trajectory with constant speed."""

    def __init__(self, amplitude, freq, center, samplerate, speed_factor, num_samples):
        """Define a position function from sample index to position."""
        self.amplitude = amplitude
        self.freq = freq
        assert self.freq.shape == (1, 3)
        # self.period_len = period_len
        self.center = center
        assert self.center.shape == (1, 3)
        self.phase_offset = np.array([[0, np.pi / 2, np.pi / 2]])

        self.samplerate = samplerate
        self.speed_factor = speed_factor
        self.num_samples = num_samples
        # self.pos = np.full((1,3), np.nan)

        self.all_pos = generate_constant_speed_trajectory(
            self.r, self.velocity, self.samplerate, speed_factor, num_samples
        )

    def r(self, t):
        """Return the position at time index t."""
        return self.center + self.amplitude * np.cos(
            2 * np.pi * t * self.freq / self.samplerate + self.phase_offset
        )

    def velocity(self, t):
        """Return the velocity at time index t."""
        return (
            -self.amplitude
            * 2
            * np.pi
            * self.freq
            * np.sin(2 * np.pi * t * self.freq / self.samplerate + self.phase_offset)
            / self.samplerate
        )

    def current_pos(self, time_idx):
        """Return the current position for the given time index."""
        return self.all_pos[time_idx : time_idx + 1, :]

    def plot(self, ax, symbol, name, tot_samples):
        """Plot the trajectory if needed."""
        pass


def run_and_save(sim):
    """Run the simulation and save signals."""
    sim.diag.add_diagnostic(
        "mic",
        dg.RecordSignal(
            "mic", sim.sim_info, num_channels=sim.arrays["mic"].num, export_func="npz"
        ),
    )
    sim.diag.add_diagnostic(
        "eval",
        dg.RecordSignal(
            "eval", sim.sim_info, num_channels=sim.arrays["eval"].num, export_func="npz"
        ),
    )
    sim.diag.add_diagnostic(
        "mic_dynamic",
        dg.RecordSignal(
            "mic_dynamic",
            sim.sim_info,
            num_channels=sim.arrays["mic_dynamic"].num,
            export_func="npz",
        ),
    )
    sim.diag.add_diagnostic(
        "src",
        dg.RecordSignal(
            "src", sim.sim_info, num_channels=sim.arrays["src"].num, export_func="npz"
        ),
    )
    sim.diag.add_diagnostic(
        "src~mic",
        dg.RecordSignal(
            "src~mic",
            sim.sim_info,
            num_channels=sim.arrays["mic"].num,
            export_func="npz",
        ),
    )
    sim.diag.add_diagnostic(
        "noise~mic",
        dg.RecordSignal(
            "noise~mic",
            sim.sim_info,
            num_channels=sim.arrays["mic"].num,
            export_func="npz",
        ),
    )

    if "image" in sim.arrays:
        sim.diag.add_diagnostic(
            "image",
            dg.RecordSignal(
                "image",
                sim.sim_info,
                num_channels=sim.arrays["image"].num,
                export_func="npz",
            ),
        )

    sim.run_simulation()
    sim.arrays.save_to_file(sim.folder_path)
    sim.sim_info.save_to_file(sim.folder_path)


def generate_signals_3d(
    sr, rt60=0.1, num_mic=20, seq_len_frac_of_sec=2, snr=30, parent_folder=None
):
    """Generate 3D signals for a moving-microphone setup."""
    rng = np.random.default_rng(10)
    side_len = 1  # 0.75
    height = 0.25
    # num_eval = 200
    num_mic = num_mic
    seq_len = sr // seq_len_frac_of_sec

    eval_res = 0.05  # 0.03
    eval_region = reg.Cuboid(
        (side_len, side_len, height), (0, 0, 0), (eval_res, eval_res, eval_res)
    )
    pos_eval = eval_region.equally_spaced_points()

    pos_src = np.array([[2, 0, 0]])

    if parent_folder is None:
        parent_folder = pathlib.Path(__file__).parent.joinpath("figs")
    setup = SimulatorSetup(parent_folder)
    setup.sim_info.samplerate = sr

    speed_factor = 0.5
    tot_trajectory_samples = num_mic * seq_len
    freq_factors = np.array([[1.8, 3.8, 2.1]])
    traj_amp = np.array([[side_len / 2, side_len / 2, height / 2]])
    trajectory = LissajousTrajectoryConstantSpeed(
        traj_amp,
        speed_factor * freq_factors / sr,
        np.zeros((1, 3)),
        sr,
        speed_factor,
        tot_trajectory_samples,
    )
    traj_pos = np.concatenate(
        [trajectory.current_pos(t) for t in range(tot_trajectory_samples)], axis=0
    )

    speed = np.linalg.norm(traj_pos[1:, :] - traj_pos[:-1, :], axis=-1) * sr
    # speed2 = np.linalg.norm(traj_pos2[1:,:] - traj_pos2[:-1,:], axis=-1) * sr
    pos_mic = traj_pos[seq_len // 2 :: seq_len, :]
    assert pos_mic.shape[0] == num_mic, (
        "we want the same number of microphones as specified"
    )

    initial_delay = seq_len
    post_delay = 0

    setup.sim_info.tot_samples = initial_delay + seq_len + post_delay
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
    setup.sim_info.start_sources_before_0 = True
    setup.sim_info.save_source_contributions = True
    setup.sim_info.highpass_cutoff = 20

    seq_len = setup.sim_info.max_room_ir_length
    sequence = pseq.create_pseq(seq_len)
    sequence_src = sources.Sequence(sequence)

    # ==== GENERATE NOISE =====
    pos_noise = np.array([[0, 1.5, 0]])
    noise_data = rng.normal(
        0, 1, (1, setup.sim_info.tot_samples + setup.sim_info.sim_buffer)
    )
    noise_source = sources.Sequence(noise_data)
    setup.add_free_source("noise", pos_noise, noise_source)
    setup.arrays.path_type["noise"]["eval"] = "none"

    setup.add_mics("mic", pos_mic)
    setup.add_mics("eval", pos_eval)
    setup.add_free_source("src", pos_src, sequence_src)
    setup.add_mics("mic_dynamic", traj_pos)

    sim = setup.create_simulator()

    # snr = 30
    num_samples_set_noise_power = 5 * setup.sim_info.samplerate
    noise_power_before_compensation = utils.power_of_filtered_signal(
        copy.deepcopy(noise_source),
        sim.arrays.paths["noise"]["mic"],
        num_samples_set_noise_power,
    )
    sig_power = utils.power_of_filtered_signal(
        copy.deepcopy(sequence_src), sim.arrays.paths["src"]["mic"], seq_len
    )
    current_snr = np.mean(sig_power) / np.mean(noise_power_before_compensation)
    if snr != np.inf:
        noise_pow_factor = current_snr / db2pow(snr)
        sim.arrays["noise"].source.amp_factor = sim.arrays[
            "noise"
        ].source.amp_factor * np.sqrt(noise_pow_factor)
    else:
        sim.arrays["noise"].source.amp_factor = (
            sim.arrays["noise"].source.amp_factor * 0
        )

    run_and_save(sim)
    with open(sim.folder_path.joinpath("extra_parameters.json"), "w") as f:
        json.dump(
            {
                "seq_len": seq_len,
                "initial_delay": initial_delay,
                "post_delay": post_delay,
                "max_sweep_freq": sr // 2,
                "downsampling_factor": 1,
                "freq_factors": freq_factors.tolist(),
                "speed_factor": speed_factor,
                "speed min": np.min(speed),
                "speed max": np.max(speed),
                "speed mean": np.mean(speed),
            },
            f,
        )
    return sim.folder_path


def load_npz(signal_paths):
    """Load multiple npz files into a single dict."""
    sig = {}
    for sig_name, sig_path in signal_paths.items():
        loaded_data = np.load(sig_path)
        dict_data = {key: data for key, data in loaded_data.items()}
        for key, data in dict_data.items():
            assert key not in sig
            sig[key] = data
    return sig


def get_signal_paths(fig_folder):
    """Return signal paths keyed by signal name."""
    signal_paths = {}
    for f in fig_folder.iterdir():
        if f.suffix == ".npz":
            sig_name_components = f.stem.split("_")
            sig_name = "_".join(sig_name_components[:-1])
            signal_paths[sig_name] = f
    return signal_paths


def load_session(fig_folder):
    """Load a saved simulation session."""
    with open(fig_folder.joinpath("extra_parameters.json")) as f:
        extra_params = json.load(f)
    # samplerate = int(2 * extra_params["max frequency"] * bandwidth_factor)
    signal_paths = get_signal_paths(fig_folder)
    sig = load_npz(signal_paths)
    sim_info, arrays = sls.load_from_path(fig_folder)
    seq_len = arrays["src"].source.tot_samples
    pos_dyn = arrays["mic_dynamic"].pos[:, None, :]
    sig["mic_dynamic"] = load_dynamic_sig(sig["mic_dynamic"], seq_len)

    extra_params["pseq_start_idx"] = sim_info.sim_buffer + extra_params["initial_delay"]

    # assert extra_params["downsampling"] == (sim_info.samplerate / samplerate)
    return sig, sim_info, arrays, pos_dyn, seq_len, extra_params


def load_dynamic_sig(sig_raw, seq_len):
    """Combine stationary microphones into a moving microphone signal.

    This takes the many statoinary microphones and puts it together into one
    seemingly moving microphone.
    """
    sig_raw = sig_raw[:, :seq_len]
    sig_len = sig_raw.shape[0]

    assert sig_len % seq_len == 0
    num_periods = sig_len // seq_len
    sig_out = np.zeros((1, sig_len))
    for i in range(sig_len):
        sig_out[0, i] = sig_raw[i, i % seq_len]
    return sig_out


def test_reference_implementation_equals_simulator_directly(parent_folder):
    """Compare reference implementation with simulator output."""
    rt60 = 0.0
    sr = 1000
    num_mic = 3
    seq_len_frac_of_sec = 2

    fig_folder = generate_signals_3d(
        sr, rt60, num_mic=num_mic, snr=np.inf, parent_folder=parent_folder
    )
    sig, sim_info, arrays, pos_dyn, seq_len, extra_params = load_session(fig_folder)

    info = {"seq_len": seq_len, "samplerate": sim_info.samplerate, "c": sim_info.c}

    pos = {
        "mic_moving": pos_dyn[:, 0, :],
        "mic": arrays["mic"].pos,
        "eval": arrays["eval"].pos,
    }

    signals = {
        "mic_moving": sig["mic_dynamic"][0, ...],
        "loudspeaker_moving": sig["src"][0, extra_params["initial_delay"] :],
    }

    rir_eval = np.squeeze(arrays.paths["src"]["eval"], axis=0)
    rir_eval_freq = ft.rfft(rir_eval)

    side_len = 1  # 0.75
    height = 0.25
    seq_len = sr // seq_len_frac_of_sec

    eval_res = 0.05  # 0.03
    eval_region = reg.Cuboid(
        (side_len, side_len, height), (0, 0, 0), (eval_res, eval_res, eval_res)
    )
    pos_eval = eval_region.equally_spaced_points()

    pos_src = np.array([[2, 0, 0]])

    setup = SimulatorSetup(parent_folder)
    setup.sim_info.samplerate = sr

    speed_factor = 0.5
    tot_trajectory_samples = num_mic * seq_len
    freq_factors = np.array([[1.8, 3.8, 2.1]])
    traj_amp = np.array([[side_len / 2, side_len / 2, height / 2]])
    trajectory = LissajousTrajectoryConstantSpeed(
        traj_amp,
        speed_factor * freq_factors / sr,
        np.zeros((1, 3)),
        sr,
        speed_factor,
        tot_trajectory_samples,
    )
    traj_pos = np.concatenate(
        [trajectory.current_pos(t) for t in range(tot_trajectory_samples)], axis=0
    )

    pos_mic = traj_pos[seq_len // 2 :: seq_len, :]
    assert pos_mic.shape[0] == num_mic, (
        "we want the same number of microphones as specified"
    )

    initial_delay = seq_len
    post_delay = 0

    setup.sim_info.tot_samples = initial_delay + seq_len + post_delay
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
    setup.sim_info.start_sources_before_0 = True
    setup.sim_info.save_source_contributions = True
    setup.sim_info.highpass_cutoff = 20

    seq_len = setup.sim_info.max_room_ir_length
    sequence = pseq.create_pseq(seq_len)
    sequence_src = sources.Sequence(sequence)

    setup.add_mics("mic", pos_mic)
    setup.add_mics("eval", pos_eval)
    setup.add_free_source("src", pos_src, sequence_src)
    setup.add_mics("mic_dynamic", trajectory)

    sim = setup.create_simulator()

    # Check that the stationary microphones are equivalent, otherwise some simulation parameter is likely different
    assert np.allclose(sim.arrays["eval"].pos, pos["eval"])
    assert np.allclose(np.squeeze(sim.arrays.paths["src"]["eval"], axis=0), rir_eval)

    assert False
