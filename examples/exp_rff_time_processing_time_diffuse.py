"""Diffuse RFF timing experiment."""

import json
import pathlib
from time import process_time, time

import aspcol.plot as aspplot
import aspcol.soundfieldestimation as sfe
import aspcol.soundfieldestimation_jax as sfe_jax
import aspcore.fouriertransform as ft
import aspcore.utilities as utils
import exp_funcs_ideal_sampling as exis
import jax
import load_dataset as ld
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import image

jax.config.update("jax_enable_x64", True)


def plot_trajectory(pos_dyn, fig_folder):
    """Plot the dynamic trajectory."""
    pos_dyn = pos_dyn[:, 0, :]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    DOWNSAMPLE_FACTOR = 50
    pos_dyn = pos_dyn[::DOWNSAMPLE_FACTOR, :]

    axes[0].plot(pos_dyn[:, 0], pos_dyn[:, 1], "x-")
    axes[0].set_xlabel("x(m)")
    axes[0].set_ylabel("y(m)")
    axes[0].set_title("Trajectory")

    axes[1].plot(pos_dyn[:, 0], pos_dyn[:, 2], "x-")
    axes[1].set_xlabel("x(m)")
    axes[1].set_ylabel("z(m)")
    axes[1].set_title("Trajectory")

    for ax in axes:
        utils.set_basic_plot_look(ax)
    utils.save_plot("tikz", fig_folder, "trajectory")


def main_mc(num_mc, fig_folder=None, simulated_data=True):
    """Run Monte Carlo simulations for diffuse timing."""
    rt60 = 0.15
    if fig_folder is None:
        fig_folder = exis.generate_signals_3d(1000, rt60, num_mic=32, snr=np.inf)
    sig, sim_info, arrays, pos_dyn, seq_len, extra_params = exis.load_session(
        fig_folder
    )

    info = {"seq_len": seq_len, "samplerate": sim_info.samplerate, "c": sim_info.c}

    pos = {
        "mic_moving": pos_dyn[:, 0, :],
        "mic": arrays["eval"].pos,
        "image": arrays["image"].pos,
    }

    signals = {
        "mic_moving": sig["mic_dynamic"][0, ...],
        "loudspeaker_moving": sig["src"][0, extra_params["initial_delay"] :],
    }

    rir_eval = np.squeeze(arrays.paths["src"]["eval"], axis=0)
    rir_eval_freq = ft.rfft(rir_eval)
    # rir_eval_freq = np.squeeze(np.moveaxis(np.fft.rfft(rir_eval, axis=-1), 2, 0), axis=1)

    plot_trajectory(pos_dyn, fig_folder)

    for mc_idx in range(num_mc):
        fig_folder_mc = fig_folder / f"mc_{mc_idx}"
        fig_folder_mc.mkdir(exist_ok=True)
        main(fig_folder_mc, mc_idx, info, signals, pos, rir_eval_freq)

        make_mc_plots(fig_folder)


def main(figure_folder, mc_idx, info, signals, pos, rir_eval_freq):
    """Run a single simulation instance."""
    freqs = ft.get_real_freqs(info["seq_len"], info["samplerate"])

    pos_eval = pos["mic"]

    tot_num_samples = signals["mic_moving"].shape[-1]
    tot_num_periods = tot_num_samples // info["seq_len"]

    num_basis_list = [4, 8, 16, 32]
    sig_len_list = [
        info["seq_len"] * num_periods
        for num_periods in [2, 4, 8, 16, 32, 64, 128]
        if num_periods <= tot_num_periods
    ]
    # sig_len_list = [info["seq_len"] * num_periods for num_periods in [16] if num_periods <= tot_num_periods]

    # noise_power = np.mean(noise_moving**2)
    lambda_inv = 0.1  # 0.1
    regularization_mo = 1e-3  # noise_power * lambda_inv
    regularization_rff = 1e-3
    estimates = {}

    times_rff = []
    flops_rff = []

    for sig_len in sig_len_list:
        times_rff.append([])
        flops_rff.append([])
        for nb in num_basis_list:
            print(
                f"moving mic rff with {nb} basis functions and signal length {sig_len}"
            )
            rng = np.random.default_rng(123456 + mc_idx)
            seed = rng.integers(0, 1000000)
            key = jax.random.key(seed)

            sig_moving_temp = np.copy(signals["mic_moving"][:sig_len])
            pos_moving_temp = np.copy(pos["mic_moving"][:sig_len])

            compiled = (
                jax.jit(
                    sfe_jax.krr_moving_mic_rff,
                    static_argnames=["num_basis", "return_params"],
                )
                .lower(
                    sig_moving_temp,
                    pos_moving_temp,
                    pos_eval,
                    signals["loudspeaker_moving"],
                    info["samplerate"],
                    info["c"],
                    regularization_rff,
                    num_basis=nb,
                    key=key,
                    return_params=False,
                )
                .compile()
            )
            flops_rff[-1].append(compiled.cost_analysis()[0]["flops"])

            process_time_start = time()
            estimates[f"moving rff {nb} sig len {sig_len}"] = jax.block_until_ready(
                compiled(
                    sig_moving_temp,
                    pos_moving_temp,
                    pos_eval,
                    signals["loudspeaker_moving"],
                    info["samplerate"],
                    info["c"],
                    regularization_rff,
                    key=key,
                )
            )

            process_time_end = time()
            times_rff[-1].append(process_time_end - process_time_start)

    times_ki = []
    flops_ki = []
    mse_ki = []
    max_periods_ki = 32
    for sig_len in sig_len_list:
        if sig_len > max_periods_ki * info["seq_len"]:
            continue
        print(f"moving mic ki with signal length {sig_len}")

        sig_moving_temp = np.copy(signals["mic_moving"][:sig_len])
        pos_moving_temp = np.copy(pos["mic_moving"][:sig_len])

        compiled = (
            jax.jit(sfe_jax.krr_moving_mic_diffuse, static_argnames=["return_params"])
            .lower(
                sig_moving_temp,
                pos_moving_temp,
                pos_eval,
                signals["loudspeaker_moving"],
                info["samplerate"],
                info["c"],
                regularization_mo,
                return_params=False,
            )
            .compile()
        )
        flops_ki.append(compiled.cost_analysis()[0]["flops"])

        process_time_start = time()
        estimates[f"moving ki {sig_len}"] = jax.block_until_ready(
            compiled(
                sig_moving_temp,
                pos_moving_temp,
                pos_eval,
                signals["loudspeaker_moving"],
                info["samplerate"],
                info["c"],
                regularization_mo,
            )
        )

        process_time_end = time()
        times_ki.append(process_time_end - process_time_start)

    # Calculate MSE per frequency
    mse_per_freq_rff = []
    for sig_len in sig_len_list:
        mse_per_freq_rff.append([])
        for nb in num_basis_list:
            est = estimates[f"moving rff {nb} sig len {sig_len}"]
            mse_per_freq = np.mean(np.abs(est - rir_eval_freq) ** 2, axis=-1) / np.mean(
                np.abs(rir_eval_freq) ** 2, axis=-1
            )
            mse_per_freq_rff[-1].append(mse_per_freq)
    mse_per_freq_rff = np.array(mse_per_freq_rff)
    np.save(figure_folder / "mse_per_freq_rff.npy", mse_per_freq_rff)

    mse_per_freq_ki = []
    for sig_len in sig_len_list:
        if sig_len > max_periods_ki * info["seq_len"]:
            continue
        est = estimates[f"moving ki {sig_len}"]
        mse_per_freq = np.mean(np.abs(est - rir_eval_freq) ** 2, axis=-1) / np.mean(
            np.abs(rir_eval_freq) ** 2, axis=-1
        )
        mse_per_freq_ki.append(mse_per_freq)
    num_ki_estimates = len(mse_per_freq_ki)
    mse_per_freq_ki = np.array(mse_per_freq_ki)
    np.save(figure_folder / "mse_per_freq_ki.npy", mse_per_freq_ki)

    with open(figure_folder / "times_rff.json", "w") as f:
        json.dump(
            {
                "times_rff": times_rff,
                "sig_len_list": sig_len_list,
                "num_basis_list": num_basis_list,
            },
            f,
        )
    with open(figure_folder / "times_ki.json", "w") as f:
        json.dump({"times_ki": times_ki, "sig_len_list": sig_len_list}, f)
    with open(figure_folder / "flops_rff.json", "w") as f:
        json.dump(
            {
                "flops_rff": flops_rff,
                "sig_len_list": sig_len_list,
                "num_basis_list": num_basis_list,
            },
            f,
        )
    with open(figure_folder / "flops_ki.json", "w") as f:
        json.dump({"flops_ki": flops_ki, "sig_len_list": sig_len_list}, f)

    times_ki = np.array(times_ki)
    times_rff = np.array(times_rff)
    flops_rff = np.array(flops_rff)
    flops_ki = np.array(flops_ki)

    fig, ax = plt.subplots(1, 1)
    ax.plot(
        sig_len_list[:num_ki_estimates], times_ki, label="KRR moving mic", marker="o"
    )
    for i, nb in enumerate(num_basis_list):
        ax.plot(
            sig_len_list,
            times_rff[:, i],
            label=f"RFF moving mic {nb} basis",
            marker="o",
        )
    ax.set_xlabel("Signal length")
    ax.set_ylabel("Processing time (s)")
    ax.set_title("Processing time for moving mic KRR and RFF")
    ax.legend()
    utils.save_plot(OUTPUT_METHOD, figure_folder, "processing_time_moving_mic")

    fig, ax = plt.subplots(1, 1)
    ax.plot(
        np.log10(sig_len_list[:num_ki_estimates]),
        np.log10(times_ki),
        label="KRR moving mic",
        marker="o",
    )
    for i, nb in enumerate(num_basis_list):
        ax.plot(
            np.log10(sig_len_list),
            np.log10(times_rff[:, i]),
            label=f"RFF moving mic {nb} basis",
            marker="o",
        )
    ax.set_xlabel("Signal length log10 samples")
    ax.set_ylabel("Processing time (log10 s)")
    ax.set_title("Processing time for moving mic KRR and RFF")
    ax.legend()
    utils.save_plot(OUTPUT_METHOD, figure_folder, "processing_time_moving_mic_log")

    fig, ax = plt.subplots(1, 1)
    ax.plot(
        np.log10(sig_len_list[:num_ki_estimates]),
        np.log10(flops_ki),
        label="KRR moving mic",
        marker="o",
    )
    for i, nb in enumerate(num_basis_list):
        ax.plot(
            np.log10(sig_len_list),
            np.log10(flops_rff[:, i]),
            label=f"RFF moving mic {nb} basis",
            marker="o",
        )
    ax.set_xlabel("Signal length log10 samples")
    ax.set_ylabel("FLOPs (log10)")
    ax.set_title("FLOPs for moving mic KRR and RFF")
    ax.legend()
    utils.save_plot(OUTPUT_METHOD, figure_folder, "flops_moving_mic_log")

    if "image" in pos:
        pos_image = pos["image"]
    else:
        pos_image = None
    aspplot.soundfield_estimation_comparison(
        pos_eval,
        estimates,
        np.copy(rir_eval_freq),
        freqs,
        figure_folder,
        output_method=OUTPUT_METHOD,
        pos_image=pos_image,
        num_examples=4,
        remove_freqs_above=480,
        remove_freqs_below=20,
    )

    with open(figure_folder / "mse_db.json", "r") as f:
        mse_db = json.load(f)
    mse_db_ki = np.array(
        [mse_db[f"moving ki {sl}"] for sl in sig_len_list[:num_ki_estimates]]
    )
    mse_db_rff = np.array(
        [
            [mse_db[f"moving rff {nb} sig len {sl}"] for sl in sig_len_list]
            for nb in num_basis_list
        ]
    )
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        sig_len_list[:num_ki_estimates], mse_db_ki, label="KRR moving mic", marker="o"
    )
    for i, nb in enumerate(num_basis_list):
        ax.plot(
            sig_len_list,
            mse_db_rff[i, :],
            label=f"RFF moving mic {nb} basis",
            marker="o",
        )
    ax.set_xlabel("Signal length")
    ax.set_ylabel("MSE (dB)")
    ax.set_title("MSE for moving mic KRR and RFF")
    ax.legend()
    utils.save_plot(OUTPUT_METHOD, figure_folder, "mse_moving_mic")

    with open(figure_folder / "mse.json", "r") as f:
        mse = json.load(f)
    mse_ki = np.array(
        [mse[f"moving ki {sl}"] for sl in sig_len_list[:num_ki_estimates]]
    )
    mse_rff = np.array(
        [
            [mse[f"moving rff {nb} sig len {sl}"] for sl in sig_len_list]
            for nb in num_basis_list
        ]
    )

    with open(figure_folder / "mse_rff.json", "w") as f:
        json.dump(
            {
                "mse_rff": mse_rff.T.tolist(),
                "sig_len_list": sig_len_list,
                "num_basis_list": num_basis_list,
            },
            f,
        )
    with open(figure_folder / "mse_ki.json", "w") as f:
        json.dump({"mse_ki": mse_ki.tolist(), "sig_len_list": sig_len_list}, f)


def make_mc_plots(fig_folder):
    """Create summary plots for Monte Carlo results."""
    fdrs = [fdr for fdr in fig_folder.iterdir() if fdr.stem.startswith("mc_")]

    # TIME AND MSE PLOTS

    times_rff = []
    times_ki = []
    mse_rff = []
    mse_ki = []
    for fdr in fdrs:
        with open(fdr / "times_rff.json", "r") as f:
            times_rff_single = json.load(f)
            times_rff.append(times_rff_single["times_rff"])
            sig_len_list = times_rff_single["sig_len_list"]
            num_basis_list = times_rff_single["num_basis_list"]
        with open(fdr / "times_ki.json", "r") as f:
            times_ki_single = json.load(f)
            times_ki.append(times_ki_single["times_ki"])
        with open(fdr / "mse_rff.json", "r") as f:
            mse_rff_single = json.load(f)
            mse_rff.append(mse_rff_single["mse_rff"])
        with open(fdr / "mse_ki.json", "r") as f:
            mse_ki_single = json.load(f)
            mse_ki.append(mse_ki_single["mse_ki"])

    # sig_len_list = times_rff[-1]["sig_len_list"]
    # num_basis_list = times_rff[-1]["num_basis_list"]

    times_rff = np.array(times_rff)  # (num_mc, num_sig_len, num_basis)
    times_ki = np.array(times_ki)  # (num_mc, num_sig_len)
    mse_rff = np.array(mse_rff)  # (num_mc, num_sig_len, num_basis)
    mse_ki = np.array(mse_ki)  # (num_mc, num_sig_len)

    num_ki_sig_lens = times_ki.shape[1]

    fig, ax = plt.subplots(1, 1)
    for i, nb in enumerate(num_basis_list):
        mean_time = np.mean(times_rff[..., i], axis=0)
        std_time = np.std(times_rff[..., i], axis=0)
        ax.plot(sig_len_list, mean_time, label=f"RFF {nb} basis", marker="o")
        ax.fill_between(
            sig_len_list, mean_time - std_time, mean_time + std_time, alpha=0.2
        )

    mean_time_ki = np.mean(times_ki, axis=0)
    std_time_ki = np.std(times_ki, axis=0)
    ax.plot(sig_len_list[:num_ki_sig_lens], mean_time_ki, label="KRR", marker="o")
    ax.fill_between(
        sig_len_list[:num_ki_sig_lens],
        mean_time_ki - std_time_ki,
        mean_time_ki + std_time_ki,
        alpha=0.2,
    )

    ax.set_xlabel("Signal length")
    ax.set_ylabel("Processing time (s)")
    ax.legend()
    utils.save_plot(OUTPUT_METHOD, fig_folder, "cpu_vs_sig_len")

    fig, ax = plt.subplots(1, 1)
    for i, nb in enumerate(num_basis_list):
        mean_mse = np.mean(mse_rff[..., i], axis=0)
        std_mse = np.std(mse_rff[..., i], axis=0)
        ax.plot(sig_len_list, mean_mse, label=f"RFF {nb} basis", marker="o")
        ax.fill_between(sig_len_list, mean_mse - std_mse, mean_mse + std_mse, alpha=0.2)

    mean_mse_ki = np.mean(mse_ki, axis=0)
    std_mse_ki = np.std(mse_ki, axis=0)
    ax.plot(sig_len_list[:num_ki_sig_lens], mean_mse_ki, label="KRR", marker="o")
    ax.fill_between(
        sig_len_list[:num_ki_sig_lens],
        mean_mse_ki - std_mse_ki,
        mean_mse_ki + std_mse_ki,
        alpha=0.2,
    )

    ax.set_xlabel("Signal length")
    ax.set_ylabel("NMSE")
    ax.legend()
    utils.save_plot(OUTPUT_METHOD, fig_folder, "mse_vs_sig_len")

    fig, ax = plt.subplots(1, 1)
    for i, nb in enumerate(num_basis_list):
        mean_mse = np.mean(mse_rff[..., i], axis=0)
        std_mse = np.std(mse_rff[..., i], axis=0)
        ax.plot(
            sig_len_list, 10 * np.log10(mean_mse), label=f"RFF {nb} basis", marker="o"
        )
        ax.fill_between(
            sig_len_list,
            10 * np.log10(np.maximum(mean_mse - std_mse, 1e-10)),
            10 * np.log10(mean_mse + std_mse),
            alpha=0.2,
        )

    mean_mse_ki = np.mean(mse_ki, axis=0)
    std_mse_ki = np.std(mse_ki, axis=0)
    ax.plot(
        sig_len_list[:num_ki_sig_lens],
        10 * np.log10(mean_mse_ki),
        label="KRR",
        marker="o",
    )
    ax.fill_between(
        sig_len_list[:num_ki_sig_lens],
        10 * np.log10(np.maximum(mean_mse_ki - std_mse_ki, 1e-10)),
        10 * np.log10(mean_mse_ki + std_mse_ki),
        alpha=0.2,
    )

    ax.set_xlabel("Signal length")
    ax.set_ylabel("NMSE (dB)")
    ax.legend()
    utils.save_plot(OUTPUT_METHOD, fig_folder, "mse_db_vs_sig_len")

    # MSE PER FREQUENCY PLOTS
    mse_per_freq_rff = []
    mse_per_freq_ki = []
    for fdr in fdrs:
        mse_per_freq_rff_single = np.load(fdr / "mse_per_freq_rff.npy")
        mse_per_freq_ki_single = np.load(fdr / "mse_per_freq_ki.npy")
        mse_per_freq_rff.append(mse_per_freq_rff_single)
        mse_per_freq_ki.append(mse_per_freq_ki_single)

    mse_per_freq_rff = np.array(
        mse_per_freq_rff
    )  # (num_mc, num_sig_len, num_basis, num_freqs)
    mse_per_freq_ki = np.array(mse_per_freq_ki)  # (num_mc, num_sig_len, num_freqs)
    # mse_per_freq_rff = mse_per_freq_rff[:,-1, :, :]
    # mse_per_freq_ki = mse_per_freq_ki[:,-1, :]

    mse_per_freq_rff = 10 * np.log10(np.mean(mse_per_freq_rff, axis=0))
    mse_per_freq_ki = 10 * np.log10(np.mean(mse_per_freq_ki, axis=0))

    with open(fig_folder / "extra_parameters.json", "r") as f:
        extra_params = json.load(f)

    with open(fig_folder / "config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    samplerate = config["samplerate"]

    for sg_idx, sg in enumerate(sig_len_list):
        fig, ax = plt.subplots(1, 1)
        freqs = ft.get_real_freqs(
            extra_params["seq_len"], samplerate
        )  # np.arange(mse_per_freq_rff.shape[-1])
        for i, nb in enumerate(num_basis_list):
            ax.plot(freqs, mse_per_freq_rff[sg_idx, i, :], label=f"RFF {nb}")
        if sg_idx < mse_per_freq_ki.shape[0]:
            ax.plot(freqs, mse_per_freq_ki[sg_idx, :], label="KRR")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("MSE (dB)")
        ax.set_title(f"MSE per frequency for moving mic KRR and RFF siglen {sg}")
        ax.legend()
        utils.save_plot(OUTPUT_METHOD, fig_folder, f"mse_per_freq_{sg}")


if __name__ == "__main__":
    OUTPUT_METHOD = "tikz"
    # fig_folder = pathlib.Path(__file__).parent / "figs" / "figs_2025_03_20_11_03_0"
    # _reg_parameter_plot(fig_folder)
    ffdr = pathlib.Path(__file__).parent / "figs" / "figs_2025_11_05_16_26_0"

    main_mc(1, ffdr)
    # make_mc_plots(ffdr)
