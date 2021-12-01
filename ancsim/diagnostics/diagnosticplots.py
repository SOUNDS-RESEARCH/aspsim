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


def functionOfTimePlot(name, diags, timeIdx, folder, printMethod="pdf"):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.tight_layout(pad=4)

    xValues = np.arange(timeIdx)[None, :]
    for proc_name, diag in diags.items():
        output = diag.get_output()
        if output.ndim == 1:
            output = output[None, :timeIdx]
        elif output.ndim == 2:
            output = output[:, :timeIdx]
        else:
            raise NotImplementedError

        num_channels = output.shape[0]

        filterArray = np.logical_not(np.isnan(output))
        assert np.isclose(filterArray, filterArray[0, :]).all()
        filterArray = filterArray[0, :]

        if "label_suffix_channel" in metadata:
            labels = ["_".join((algoName, suf)) for suf in metadata["label_suffix_channel"]]
            assert len(labels) == num_channels
        else:
            labels = [algoName for _ in range(num_channels)]
        ax = plotMultipleChannels(ax, xValues[:,filterArray], output[:,filterArray], labels)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True)
    legendWithoutDuplicates(ax, "upper right")
    #ax.legend(loc="upper right")

    ax.set_xlabel(metadata["xlabel"])
    ax.set_ylabel(metadata["ylabel"])
    ax.set_title(metadata["title"] + " - " + name)
    outputPlot(printMethod, folder, name + "_" + str(timeIdx))

def plotMultipleChannels(ax, timeIdx, signal, labels):
    for i, label in enumerate(labels):
        ax.plot(
                timeIdx.T,
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
    np.savez_compressed(folder.joinpath(name + "_" + str(timeIdx)), **outputs)

    if list(diags.values())[0].keep_only_last_export:
        earlierFiles = meu.findAllEarlierFiles(folder, name, timeIdx, nameIncludesIdx=False)
        for f in earlierFiles:
            if f.suffix == ".npz":
                f.unlink()


def soundfieldPlot(name, outputs, metadata, timeIdx, folder, printMethod="pdf"):
    print("a plot would be generated at timeIdx: ", timeIdx, "for diagnostic: ", name)
	


def plotIR(name, diags, timeIdx, folder, printMethod="pdf"):
    numSets = 0
    for algoName, diag in outputs.items():
        for irSet in diag.get_output():
            numIRs = irSet.shape[0]
            numSets += 1
    numAlgo = len(outputs)

    #fig, axes = plt.subplots(numSets, numIRs, figsize=(numSets*4, numIRs*4))
    fig, axes = plt.subplots(numAlgo, numIRs, figsize=(numIRs*6, numAlgo*6))
    if axes.ndim == 1:
        axes = axes[None,:]

    for row, (algoName, output) in enumerate(outputs.items()):
        for setIdx, irSet in enumerate(output):
            for col in range(irSet.shape[0]):
                axes[row, col].plot(irSet[col,:], label=f"{algoName} {metadata['label'][setIdx]}", alpha=0.6)
    
    for ax in axes.flatten():
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(True)
        legendWithoutDuplicates(ax, "upper right")
        #ax.legend(loc="upper right")
        ax.set_xlabel(metadata["xlabel"])
        ax.set_ylabel(metadata["ylabel"])

    outputPlot(printMethod, folder, name + "_" + str(timeIdx))

	
#plt.plot(trueir[0,:].T, label="true")
# plt.plot(6*est[0,:].T, label="est")
# plt.show()

# plt.plot(np.abs(np.fft.fft(trueir[0,:].T,n=4096)), label="true")
# plt.plot(np.abs(np.fft.fft(3*est[0,:].T,n=4096)), label="est")
# plt.show()


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
