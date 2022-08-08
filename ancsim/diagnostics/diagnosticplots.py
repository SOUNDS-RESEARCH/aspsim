import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt
import tikzplotlib
import soundfile as sf

from ancsim.diagnostics.plotscripts import output_plot
import ancsim.fileutilities as fu
import ancsim.utilities as util



def linear(signal):
    return signal

def db_power(signal):
    return 10*np.log10(signal)

def db_amplitude(signal):
    return 20*np.log10(signal)

def natural_log(signal):
    return np.log(signal)

def preset_scaling(scaling):
    if scaling == "linear":
        return linear
    elif scaling == "db_power":
        return db_power
    elif scaling == "db_amplitude":
        return db_amplitude
    elif scaling == "natural_log":
        return natural_log
    else:
        raise ValueError("Invalid scaling string")

def scale_data(signal, scaling):
    if callable(scaling):
        return scaling(signal)
    elif isinstance(scaling, str):
        scaling_func = preset_scaling(scaling)
        return scaling_func(signal)
    else:
        raise ValueError("Invalid scaling type")


def clip(low_lim, high_lim):
    def clip_internal(signal):
        return np.clip(signal, low_lim, high_lim)
    return clip_internal


def smooth(smooth_len):
    ir = np.ones((1, smooth_len)) / smooth_len
    def smooth_internal(signal):
        return spsig.oaconvolve(signal, ir, mode="full", axes=-1)[:,:signal.shape[-1]]
    return smooth_internal













def delete_earlier_tikz_plot(folder, name):
    currentIdx = fu.findIndexInName(name)
    if currentIdx is None:
        return

    earlierFiles = fu.findAllEarlierFiles(folder, name, currentIdx)
    for f in earlierFiles:
        if f.is_dir():
            for plotFile in f.iterdir():
                # assert(plotFile.stem.startswith(startName))
                try:
                    if plotFile.suffix == ".pdf":
                        plotFile.rename(
                            folder.joinpath(plotFile.stem + plotFile.suffix)
                        )
                    else:
                        assert plotFile.suffix == ".tsv" or plotFile.suffix == ".tex"
                        plotFile.unlink()
                except PermissionError:
                    pass
            try:
                f.rmdir()
            except PermissionError:
                pass

def output_plot(printMethod, folder, name="", keepOnlyLatestTikz=True):
    if printMethod == "show":
        plt.show()
    elif printMethod == "tikz":
        if folder is not None:
            nestedFolder = folder.joinpath(name)
            try:
                nestedFolder.mkdir()
            except FileExistsError:
                pass
            tikzplotlib.save(
                str(nestedFolder.joinpath(name + ".tex")),
                externalize_tables=True,
                tex_relative_path_to_data="../figs/" + name + "/",
                float_format=".8g",
            )
            plt.savefig(
                str(nestedFolder.joinpath(name + ".pdf")),
                dpi=300,
                facecolor="w",
                edgecolor="w",
                orientation="portrait",
                format="pdf",
                transparent=True,
                bbox_inches=None,
                pad_inches=0.2,
            )
            if keepOnlyLatestTikz:
                delete_earlier_tikz_plot(folder, name)
    elif printMethod == "pdf":
        if folder is not None:
            plt.savefig(
                str(folder.joinpath(name + ".pdf")),
                dpi=300,
                facecolor="w",
                edgecolor="w",
                orientation="portrait",
                format="pdf",
                transparent=True,
                #bbox_inches=None,
                bbox_inches="tight",
                pad_inches=0.2,
            )
    elif printMethod == "svg":
        if folder is not None:
            plt.savefig(
                str(folder.joinpath(name + ".svg")),
                dpi=300,
                format="svg",
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.2,
            )
    elif printMethod == "none":
        pass
    else:
        raise ValueError
    plt.close("all")



def set_basic_plot_look(ax):
    ax.grid(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def remove_axes_and_labels(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def plot_3d_in_2d(ax, posData, symbol="x", name=""):
    uniqueZValues = np.unique(posData[:,2].round(decimals=4))
    
    # if len(uniqueZValues) == 1:
    #     alpha = np.array([1])
    # else:
    #     alpha = np.linspace(0.4, 1, len(uniqueZValues))

    for i, zVal in enumerate(uniqueZValues):
        idx = np.where(np.around(posData[:,2],4) == zVal)[0]

        ax.plot(posData[idx,0], posData[idx,1], symbol, label=f"{name}: z = {zVal}")










def functionOfTimePlot(name, diags, time_idx, folder, preprocess, printMethod="pdf", scaling="linear"):
    #fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    fig.tight_layout(pad=4)

    for proc_name, diag in diags.items():
        # output = np.atleast_2d(diag.get_output())
        # assert output.ndim == 2
        #num_channels = output.shape[0]

        # if hasattr(diag, "time_indices"):
        #     output, time_indices = get_values_from_selection(output, diag.time_indices, time_idx+1)
        # else:
        #     output, time_indices = get_values_up_to_idx(output, time_idx+1)
        output, time_indices = diag.get_processed_output(time_idx, preprocess)
        #output = scale_data(output, scaling)
        num_channels = output.shape[0]

        if "label_suffix_channel" in diag.plot_data:
            labels = ["_".join((proc_name, suf)) for suf in diag.plot_data["label_suffix_channel"]]
            assert len(labels) == num_channels
        else:
            labels = [proc_name for _ in range(num_channels)]
        ax = plotMultipleChannels(ax, time_indices, output, labels)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True)
    legendWithoutDuplicates(ax, "upper right")
    #ax.legend(loc="upper right")

    ax.set_xlabel(diag.plot_data["xlabel"])
    ax.set_ylabel(diag.plot_data["ylabel"])
    ax.set_title(diag.plot_data["title"] + " - " + name)
    output_plot(printMethod, folder, name + "_" + str(time_idx))

def plotMultipleChannels(ax, time_idx, signal, labels):
    if signal.shape[-1] < 10:
        marker = "x"
    else:
        marker = ""
    for i, label in enumerate(labels):
        ax.plot(
                np.atleast_2d(time_idx).T,
                signal[i:i+1,:].T,
                alpha=0.8,
                label=label,
                marker=marker,
            )
    return ax

def legendWithoutDuplicates(ax, loc):
    handles, labels = ax.get_legend_handles_labels()
    newLabels, newHandles = [], []
    for handle, label in zip(handles, labels):
      if label not in newLabels:
        newLabels.append(label)
        newHandles.append(handle)

    ax.legend(newHandles, newLabels, loc=loc)


def savenpz(name, diags, time_idx, folder, preprocess, printMethod="pdf"):
    """Keeps only the latest save.
    Assumes that the data in previous
    saves is present in the current data"""
    outputs = {proc_name: diag.get_processed_output(time_idx, preprocess) for proc_name, diag in diags.items()}
    outputs = {key: val[0] if isinstance(val, tuple) else val for key, val in outputs.items()} 
    # only takes output even if time indices are provided
  
    #flatOutputs = util.flattenDict(outputs, sep="~")
    np.savez_compressed(folder.joinpath(f"{name}_{time_idx}"), **outputs)

    if list(diags.values())[0].keep_only_last_export:
        earlierFiles = fu.findAllEarlierFiles(folder, name, time_idx, nameIncludesIdx=False)
        for f in earlierFiles:
            if f.suffix == ".npz":
                f.unlink()


def soundfieldPlot(name, outputs, metadata, timeIdx, folder, printMethod="pdf"):
    print("a plot would be generated at timeIdx: ", timeIdx, "for diagnostic: ", name)
	


def plotIR(name, diags, time_idx, folder, preprocess, printMethod="pdf"):
    numSets = 0
    for algoName, diag in diags.items():
        for irSet in diag.get_output():
            numIRs = irSet.shape[0]
            numSets += 1
    numAlgo = len(diags)

    #fig, axes = plt.subplots(numSets, numIRs, figsize=(numSets*4, numIRs*4))
    fig, axes = plt.subplots(numAlgo, numIRs, figsize=(numIRs*6, numAlgo*6))
    if numAlgo == 1:
        axes = np.expand_dims(axes, 0)
    if numIRs == 1:
        axes = np.expand_dims(axes, -1)
    #axes = np.atleast_2d(axes)
    #if not isinstance(axes, tuple):
    #    axes = [[axes]]
    #elif not isinstance(axes[0], tuple):
    #    axes = [axes]

    for row, (algoName, diag) in enumerate(diags.items()):
        output = diag.get_processed_output(time_idx, preprocess)
        for setIdx, irSet in enumerate(output):
            for col in range(irSet.shape[0]):
                #axes[row, col].plot(irSet[col,:], label=f"{algoName} {metadata['label'][setIdx]}", alpha=0.6)
                axes[row, col].plot(irSet[col,:], label=f"{algoName}", alpha=0.6)
                #axes[row, col].set_xlabel(diag.plot_data["xlabel"])
                #axes[row, col].set_ylabel(diag.plot_data["ylabel"])

    for ax in axes.flatten():
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(True)
        legendWithoutDuplicates(ax, "upper right")
        #ax.legend(loc="upper right")
        
    output_plot(printMethod, folder, f"{name}_{time_idx}")

def matshow(name, diags, time_idx, folder, preprocess, printMethod="pdf"):
    outputs = {proc_name: diag.get_processed_output(time_idx, preprocess) for proc_name, diag in diags.items()}
    
    num_proc = len(diags)
    fig, axes = plt.subplots(1,num_proc, figsize=(num_proc*5, 5))
    if num_proc == 1:
        axes = [axes]

    for ax, (proc_name, output) in zip(axes, outputs.items()):
        clr = ax.matshow(output)
        fig.colorbar(clr, ax=ax, orientation='vertical')
        ax.set_title(f"{name} - {proc_name}")
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    output_plot(printMethod, folder, f"{name}_{time_idx}")


def createAudioFiles(name, outputs, metadata, timeIdx, folder, printMethod=None):
    maxValue = np.NINF
    lastTimeIdx = np.Inf
    for output in outputs.values():
        for signal in output.values():
            maxValue = np.max((maxValue, np.max(np.abs(signal[~np.isnan(signal)]))))
            lastTimeIdx = int(np.min((lastTimeIdx, np.max(np.where(~np.isnan(signal))))))

    startIdx = np.max((lastTimeIdx-metadata["maxlength"], 0))

    for algoName, output in outputs.items():
        for audioName, signal in output.items():
            for channelIdx in range(signal.shape[0]):
                if signal.shape[0] > 1:
                    fileName = "_".join((name, algoName, audioName, str(channelIdx), str(timeIdx)))
                else:
                    fileName = "_".join((name, algoName, audioName, str(timeIdx)))
                filePath = folder.joinpath(fileName + ".wav")

                signalToWrite = signal[channelIdx,startIdx:lastTimeIdx]/maxValue
                rampLength = int(0.2*metadata["samplerate"])
                ramp = np.linspace(0,1,rampLength)
                signalToWrite[:rampLength] *= ramp
                signalToWrite[-rampLength:] *= (1-ramp)

                sf.write(str(filePath),signalToWrite, metadata["samplerate"])
