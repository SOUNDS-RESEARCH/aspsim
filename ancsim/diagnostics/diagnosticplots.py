import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import soundfile as sf

from ancsim.experiment.plotscripts import outputPlot
import ancsim.experiment.multiexperimentutils as meu
import ancsim.utilities as util


# class SCALINGTYPE(Enum):
#     linear = 1
#     dbPower = 2
#     dbAmp = 3
#     ln = 4


# def scaleData(scalingType, data):
#     if scalingType == SCALINGTYPE.linear:
#         return data
#     elif scalingType == SCALINGTYPE.dbPower:
#         return 10 * np.log10(data)
#     elif scalingType == SCALINGTYPE.dbAmp:
#         return 20 * np.log10(data)
#     elif scalingType == SCALINGTYPE.ln:
#         return np.log(data)


def get_values_up_to_idx(signal, max_idx):
    """
    gives back signal values that correspond to time_values less than max_idx, 
    and signal values that are not nan

    max_idx is exlusive
    """

    time_indices = np.arange(max_idx)
    signal = signal[:,:max_idx]

    nan_filter = np.logical_not(np.isnan(signal))
    assert np.isclose(nan_filter, nan_filter[0, :]).all()
    nan_filter = nan_filter[0, :]

    time_indices = time_indices[nan_filter]
    signal = signal[:,nan_filter]
    return signal, time_indices

def get_values_from_selection(signal, time_indices, max_idx):
    """
    gives back signal values that correspond to time_values less than max_idx, 
    and signal values that are not nan

    max_idx is exlusive
    """
    nan_filter = np.logical_not(np.isnan(signal))
    assert np.isclose(nan_filter, nan_filter[0, :]).all()
    nan_filter = nan_filter[0, :]

    assert time_indices.shape[-1] == signal.shape[-1]
    time_indices = time_indices[nan_filter]
    signal = signal[:,nan_filter]

    above_max_idx = np.argmax(time_indices >= max_idx)
    if above_max_idx == 0:
        if np.logical_not(above_max_idx).all():
            above_max_idx = len(time_indices)

    time_indices = time_indices[:above_max_idx]
    signal = signal[:,:above_max_idx]
    return signal, time_indices



def functionOfTimePlot(name, diags, time_idx, folder, printMethod="pdf"):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.tight_layout(pad=4)

    for proc_name, diag in diags.items():
        output = np.atleast_2d(diag.get_output())
        assert output.ndim == 2
        num_channels = output.shape[0]

        if hasattr(diag, "time_indices"):
            output, time_indices = get_values_from_selection(output, diag.time_indices, time_idx+1)
            #xValues = diag.time_indices
        else:
            output, time_indices = get_values_up_to_idx(output, time_idx+1)
            #xValues = np.arange(time_idx+1)

        #filterArray = np.logical_not(np.isnan(output))
        #assert np.isclose(filterArray, filterArray[0, :]).all()
        #filterArray = filterArray[0, :]
        #xValues = xValues[filterArray[:time_idx+1]]
        #output = output[:,filterArray]

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
    outputPlot(printMethod, folder, name + "_" + str(time_idx))

def plotMultipleChannels(ax, time_idx, signal, labels):
    for i, label in enumerate(labels):
        ax.plot(
                np.atleast_2d(time_idx).T,
                signal[i:i+1,:].T,
                alpha=0.8,
                label=label,
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


def savenpz(name, diags, timeIdx, folder, printMethod="pdf"):
    """Keeps only the latest save.
    Assumes that the data in previous
    saves is present in the current data"""
    outputs = {proc_name: diag.get_output() for proc_name, diag in diags.items()}
    #flatOutputs = util.flattenDict(outputs, sep="~")
    np.savez_compressed(folder.joinpath(f"{name}_{timeIdx}"), **outputs)

    if list(diags.values())[0].keep_only_last_export:
        earlierFiles = meu.findAllEarlierFiles(folder, name, timeIdx, nameIncludesIdx=False)
        for f in earlierFiles:
            if f.suffix == ".npz":
                f.unlink()


def soundfieldPlot(name, outputs, metadata, timeIdx, folder, printMethod="pdf"):
    print("a plot would be generated at timeIdx: ", timeIdx, "for diagnostic: ", name)
	


def plotIR(name, diags, timeIdx, folder, printMethod="pdf"):
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
        output = diag.get_output()
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
        
    outputPlot(printMethod, folder, f"{name}_{timeIdx}")

def matshow(name, diags, timeIdx, folder, printMethod="pdf"):
    outputs = {proc_name: diag.get_output() for proc_name, diag in diags.items()}
    
    num_proc = len(diags)
    fig, axes = plt.subplots(1,num_proc, figsize=(5, num_proc*5))
    if num_proc == 1:
        axes = [axes]

    for ax, (proc_name, diag) in zip(axes, diags.items()):
        clr = ax.matshow(diag.get_output())
        fig.colorbar(clr, ax=ax, orientation='vertical')
        ax.set_title(f"{name} - {proc_name}")
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    outputPlot(printMethod, folder, f"{name}_{timeIdx}")


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
