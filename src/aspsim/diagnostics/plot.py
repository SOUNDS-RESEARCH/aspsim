"""Plotting and export helpers for diagnostics."""

import json

import aspcore.utilities as utils
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

import aspsim.diagnostics.soundfieldplot as sfplot
import aspsim.fileutilities as fu


def plot_3d_in_2d(ax, pos_data, symbol="x", name=""):
    """Plot 3D positions projected into 2D."""
    unique_z_values = np.unique(pos_data[:, 2].round(decimals=4))

    # if len(uniqueZValues) == 1:
    #     alpha = np.array([1])
    # else:
    #     alpha = np.linspace(0.4, 1, len(uniqueZValues))

    for i, z_val in enumerate(unique_z_values):
        idx = np.where(np.around(pos_data[:, 2], 4) == z_val)[0]

        ax.plot(
            pos_data[idx, 0], pos_data[idx, 1], symbol, label=f"{name}: z = {z_val}"
        )


def function_of_time_plot(
    name, diags, time_idx, folder, preprocess, print_method="pdf"
):
    """Plot diagnostic output as a function of time."""
    # fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    fig.tight_layout(pad=4)

    for proc_name, diag in diags.items():
        output, time_indices = diag.get_processed_output(time_idx, preprocess)

        num_channels = output.shape[0]

        if "label_suffix_channel" in diag.plot_data:
            labels = [
                "_".join((proc_name, suf))
                for suf in diag.plot_data["label_suffix_channel"]
            ]
            assert len(labels) == num_channels
        else:
            labels = [proc_name for _ in range(num_channels)]
        ax = plot_multiple_channels(ax, time_indices, output, labels)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True)
    legend_without_duplicates(ax, "upper right")
    # ax.legend(loc="upper right")

    ax.set_xlabel(diag.plot_data["xlabel"])
    ax.set_ylabel(diag.plot_data["ylabel"])
    ax.set_title(diag.plot_data["title"] + " - " + name)
    utils.save_plot(print_method, folder, name + "_" + str(time_idx))


def plot_multiple_channels(ax, time_idx, signal, labels):
    """Plot multiple channels on a single axis."""
    if signal.shape[-1] < 10:
        marker = "x"
    else:
        marker = ""
    for i, label in enumerate(labels):
        ax.plot(
            np.atleast_2d(time_idx).T,
            signal[i : i + 1, :].T,
            alpha=0.8,
            label=label,
            marker=marker,
        )
    return ax


def legend_without_duplicates(ax, loc):
    """Add a legend without duplicate labels."""
    handles, labels = ax.get_legend_handles_labels()
    new_labels, new_handles = [], []
    for handle, label in zip(handles, labels):
        if label not in new_labels:
            new_labels.append(label.replace("_", "\_"))
            new_handles.append(handle)

    ax.legend(new_handles, new_labels, loc=loc)


def savenpz(name, diags, time_idx, folder, preprocess, print_method="pdf"):
    """Save the latest diagnostics to a compressed npz file.

    Assumes the output of the previous saved diagnostic is not needed, as the same data is contained in the new file.
    """
    outputs = {
        proc_name: diag.get_processed_output(time_idx, preprocess)
        for proc_name, diag in diags.items()
    }
    outputs = {
        key: val[0] if isinstance(val, tuple) else val for key, val in outputs.items()
    }
    # only takes output even if time indices are provided

    # flatOutputs = util.flattenDict(outputs, sep="~")
    np.savez_compressed(folder.joinpath(f"{name}_{time_idx}"), **outputs)

    if list(diags.values())[0].keep_only_last_export:
        earlierFiles = fu.find_all_earlier_files(
            folder, name, time_idx, name_includes_idx=False
        )
        for f in earlierFiles:
            if f.suffix == ".npz":
                f.unlink()


def txt(name, diags, time_idx, folder, preprocess, print_method="pdf"):
    """Save summary values to a JSON file."""
    outputs = {
        proc_name: diag.get_processed_output(time_idx, preprocess)
        for proc_name, diag in diags.items()
    }
    outputs = {
        key: val[0] if isinstance(val, tuple) else val for key, val in outputs.items()
    }
    # only takes output even if time indices are provided
    summary_val = {key: np.mean(val) for key, val in outputs.items()}

    with open(folder.joinpath(f"{name}_{time_idx}.json"), "w") as f:
        json.dump(summary_val, f)
    # np.savez_compressed(folder.joinpath(f"{name}_{time_idx}"), **outputs)


def soundfield(name, diags, time_idx, folder, preprocess, print_method="pdf"):
    """Plot and save a soundfield diagnostic."""
    # outputs = {proc_name: diag.get_processed_output(time_idx, preprocess) for proc_name, diag in diags.items()}

    num_proc = len(diags)
    fig, axes = plt.subplots(1, num_proc, figsize=(num_proc * 5, 5))
    if num_proc == 1:
        axes = [axes]

    sf_all = {
        proc_name: diag.get_processed_output(time_idx, preprocess)
        for proc_name, diag in diags.items()
    }

    max_val = np.max([np.max(sf) for sf in sf_all.values()])
    min_val = np.min([np.min(sf) for sf in sf_all.values()])

    for (proc_name, diag), sf, ax in zip(diags.items(), sf_all.values(), axes):
        pos = diag.pos_mic

        pos_sorted, sf_sorted = sfplot.sort_for_imshow(pos, sf[:, None])
        sfplot.sf_plot(ax, pos_sorted, sf_sorted, proc_name, vminmax=(min_val, max_val))

        if diag.plot_arrays is not None:
            for plot_ar in diag.plot_arrays:
                plot_ar.plot(ax)
        legend_without_duplicates(ax, "upper right")

    utils.save_plot(print_method, folder, f"{name}_{time_idx}")

    # print("a plot would be generated at timeIdx: ", timeIdx, "for diagnostic: ", name)


def plot_ir(name, diags, time_idx, folder, preprocess, print_method="pdf"):
    """Plot impulse responses for diagnostics."""
    num_sets = 0
    for algo_name, diag in diags.items():
        for ir_set in diag.get_output():
            num_irs = ir_set.shape[0]
            num_sets += 1
    num_algo = len(diags)

    # fig, axes = plt.subplots(numSets, numIRs, figsize=(numSets*4, numIRs*4))
    fig, axes = plt.subplots(num_algo, num_irs, figsize=(num_irs * 6, num_algo * 6))
    if num_algo == 1:
        axes = np.expand_dims(axes, 0)
    if num_irs == 1:
        axes = np.expand_dims(axes, -1)
    # axes = np.atleast_2d(axes)
    # if not isinstance(axes, tuple):
    #    axes = [[axes]]
    # elif not isinstance(axes[0], tuple):
    #    axes = [axes]

    for row, (algo_name, diag) in enumerate(diags.items()):
        output = diag.get_processed_output(time_idx, preprocess)
        for set_idx, ir_set in enumerate(output):
            for col in range(ir_set.shape[0]):
                # axes[row, col].plot(irSet[col,:], label=f"{algoName} {metadata['label'][setIdx]}", alpha=0.6)
                axes[row, col].plot(ir_set[col, :], label=f"{algo_name}", alpha=0.6)
                # axes[row, col].set_xlabel(diag.plot_data["xlabel"])
                # axes[row, col].set_ylabel(diag.plot_data["ylabel"])

    for ax in axes.flatten():
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(True)
        legend_without_duplicates(ax, "upper right")
        # ax.legend(loc="upper right")

    utils.save_plot(print_method, folder, f"{name}_{time_idx}")


def matshow(name, diags, time_idx, folder, preprocess, print_method="pdf"):
    """Plot diagnostics output with matshow."""
    outputs = {
        proc_name: diag.get_processed_output(time_idx, preprocess)
        for proc_name, diag in diags.items()
    }

    num_proc = len(diags)
    fig, axes = plt.subplots(1, num_proc, figsize=(num_proc * 5, 5))
    if num_proc == 1:
        axes = [axes]

    for ax, (proc_name, output) in zip(axes, outputs.items()):
        clr = ax.matshow(output)
        fig.colorbar(clr, ax=ax, orientation="vertical")
        ax.set_title(f"{name} - {proc_name}")
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    utils.save_plot(print_method, folder, f"{name}_{time_idx}")


def create_audio_files(name, diags, time_idx, folder, preprocess, print_method=None):
    """Create audio files from diagnostics output."""
    outputs = {
        proc_name: diag.get_processed_output(time_idx, preprocess)
        for proc_name, diag in diags.items()
    }
    outputs = {
        key: val[0] if isinstance(val, tuple) else val for key, val in outputs.items()
    }
    # only takes the signal even if time indices are provided

    samplerate = diags[list(diags.keys())[0]].sim_info.samplerate

    # max_val = np.NINF
    # last_time_idx = np.inf
    # for (proc_name, signal) in outputs.items():
    #        max_val = np.max((max_val, np.max(np.abs(signal[~np.isnan(signal)]))))
    #        last_time_idx = int(np.min((last_time_idx, np.max(np.where(~np.isnan(signal))))))

    # start_idx = np.max((last_time_idx-metadata["maxlength"], 0))

    for proc_name, signal in outputs.items():
        file_name = "_".join((name, proc_name, str(time_idx)))
        file_path = folder.joinpath(f"{file_name}.wav")

        # for channel_idx in range(signal.shape[0]):
        #     if signal.shape[0] > 1:
        #         fileName = "_".join((name, proc_name, audio_name, str(channel_idx), str(time_idx)))
        #     else:
        #         fileName = "_".join((name, proc_name, audio_name, str(time_idx)))
        #     file_path = folder.joinpath(fileName + ".wav")

        signal_to_write = signal / np.max(np.abs(signal))

        ramp_length = min(int(0.2 * samplerate), int(0.1 * signal_to_write.shape[-1]))
        if ramp_length > 0:
            ramp = np.linspace(0, 1, ramp_length)
            signal_to_write[:, :ramp_length] *= ramp
            signal_to_write[:, -ramp_length:] *= 1 - ramp

        sf.write(str(file_path), signal_to_write.T, samplerate)


def spectrum_plot(name, diags, time_idx, folder, preprocess, print_method="pdf"):
    """Plot spectrum diagnostics output."""
    # outputs = {proc_name: diag.get_processed_output(time_idx, preprocess) for proc_name, diag in diags.items()}
    # outputs = {key: val[0] if isinstance(val, tuple) else val for key, val in outputs.items()}

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.tight_layout(pad=4)

    for proc_name, diag in diags.items():
        output = diag.get_processed_output(time_idx, preprocess)
        num_channels = output.shape[0]
        ax.plot(
            output,
            alpha=0.8,
            label=proc_name,
        )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True)
    legend_without_duplicates(ax, "upper right")
    # ax.legend(loc="upper right")

    # ax.set_xlabel(diag.plot_data["xlabel"])
    # ax.set_ylabel(diag.plot_data["ylabel"])
    # ax.set_title(diag.plot_data["title"] + " - " + name)
    utils.save_plot(print_method, folder, name + "_" + str(time_idx))
