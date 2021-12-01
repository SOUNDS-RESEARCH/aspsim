import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import soundfile as sf
import scipy.signal as spsig

from ancsim.experiment.plotscripts import outputPlot
import ancsim.experiment.multiexperimentutils as meu
import ancsim.utilities as util


# class SCALINGTYPE(Enum):
#     linear = 1
#     dbPower = 2
#     dbAmp = 3
#     ln = 4


def scaleData(scalingType, data):
    if scalingType == "linear":
        return data
    elif scalingType == "dbpow":
        return 10 * np.log10(data)
    elif scalingType == "dbamp":
        return 20 * np.log10(data)
    elif scalingType == "ln":
        return np.log(data)


def functionOfTimePlot(name, outputs, metadata, timeIdx, folder, printMethod="pdf"):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.tight_layout(pad=4)

    xValues = np.arange(timeIdx)[None, :]
    for algoName, output in outputs.items():
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

        xVals = xValues[:,filterArray]
        out = output[:,filterArray]
        if "scaling" in metadata:
            out = scaleData(metadata["scaling"], out)

        if "label_suffix_channel" in metadata:
            labels = ["_".join((algoName, suf)) for suf in metadata["label_suffix_channel"]]
            assert len(labels) == num_channels
        else:
            labels = [algoName for _ in range(num_channels)]
        ax = plotMultipleChannels(ax, xVals, out, labels)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True)
    legendWithoutDuplicates(ax, "upper right")
    #ax.legend(loc="upper right")

    ax.set_xlabel(metadata["xlabel"])
    ax.set_ylabel(metadata["ylabel"])
    ax.set_title(metadata["title"] + " - " + name)
    outputPlot(printMethod, folder, name + "_" + str(timeIdx))

def spectrumPlot(name, outputs, metadata, timeIdx, folder, printMethod="pdf"):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.tight_layout(pad=4)

    for algoName, output in outputs.items():
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

        skip_initial = 2048
        output = output[:,filterArray]
        print(f"spectrum computed for array shape {output[:,skip_initial:].shape}")
        f, spec = spsig.welch(output[:,skip_initial:] ,metadata["samplerate"], nperseg=512, scaling="spectrum", axis=-1)
        spec = 10*np.log10(np.mean(spec,axis=0))
        #for i, label in enumerate(labels):
        ax.plot(
                f,
                spec,
                alpha=0.8,
                label=algoName,
            )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True)
    legendWithoutDuplicates(ax, "upper right")
    #ax.legend(loc="upper right")
    ax.set_xlabel(metadata["xlabel"])
    ax.set_ylabel(metadata["ylabel"])
    ax.set_title(metadata["title"] + " - " + name)
    outputPlot(printMethod, folder, name+ "_" + str(timeIdx))

def spectrumRatioPlot(name, outputs, metadata, timeIdx, folder, printMethod="pdf"):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.tight_layout(pad=4)

    for algoName, (num, denom) in outputs.items():
        if num.ndim == 1:
            num = num[None, :]
        if denom.ndim == 1:
            denom = denom[None,:]
        num = num[:, :timeIdx]
        denom = denom[:, :timeIdx]

        numFilterArray = np.logical_not(np.isnan(num))
        denomFilterArray = np.logical_not(np.isnan(num))
        assert np.isclose(numFilterArray, numFilterArray[0, :]).all()
        assert np.isclose(denomFilterArray, numFilterArray[0, :]).all()
        filterArray = numFilterArray[0, :]

       #skip_initial = 2048
        num_to_average = 30000
        num = num[:,filterArray]
        denom = denom[:,filterArray]
        f, numSpec = spsig.welch(num[:,-num_to_average:] ,metadata["samplerate"], nperseg=256, scaling="spectrum", axis=-1)
        _, denomSpec = spsig.welch(denom[:,-num_to_average:] ,metadata["samplerate"], nperseg=256, scaling="spectrum", axis=-1)
        ratio = np.mean(numSpec,axis=0) / np.mean(denomSpec, axis=0)
        #for i, label in enumerate(labels):
        ax.plot(
                f,
                10*np.log10(ratio),
                alpha=0.8,
                label=algoName,
            )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True)
    legendWithoutDuplicates(ax, "upper right")
    #ax.legend(loc="upper right")
    ax.set_xlabel(metadata["xlabel"])
    ax.set_ylabel(metadata["ylabel"])
    ax.set_title(metadata["title"] + " - " + name)
    outputPlot(printMethod, folder, name+ "_" + str(timeIdx))




# def spectrumPlot(name, outputs, metadata, timeIdx, folder, printMethod="pdf"):
#     fig, ax = plt.subplots(1, 1, figsize=(14, 8))
#     fig.tight_layout(pad=4)

#     xValues = np.arange(timeIdx)[None, :]
#     for algoName, output in outputs.items():
#         if output.ndim == 1:
#             output = output[None, :timeIdx]
#         elif output.ndim == 2:
#             output = output[:, :timeIdx]
#         else:
#             raise NotImplementedError

#         num_channels = output.shape[0]

#         filterArray = np.logical_not(np.isnan(output))
#         assert np.isclose(filterArray, filterArray[0, :]).all()
#         filterArray = filterArray[0, :]

#         # if "label_suffix_channel" in metadata:
#         #     labels = ["_".join((algoName, suf)) for suf in metadata["label_suffix_channel"]]
#         #     assert len(labels) == num_channels
#         # else:
#         #     labels = [algoName for _ in range(num_channels)]

#         if num_channels > 1:
#             raise NotImplementedError

#         skip_initial = 1024
#         output = output[:,filterArray]
#         f, spec = spsig.welch(output[0,skip_initial:] ,metadata["samplerate"], nperseg=512, scaling="spectrum")
#         spec = 10*np.log10(spec)
#         #for i, label in enumerate(labels):
#         ax.plot(
#                 f,
#                 spec,
#                 alpha=0.8,
#                 label=algoName,
#             )

#     ax.spines["right"].set_visible(False)
#     ax.spines["top"].set_visible(False)
#     ax.grid(True)
#     legendWithoutDuplicates(ax, "upper right")
#     #ax.legend(loc="upper right")

#     ax.set_xlabel(metadata["xlabel"])
#     ax.set_ylabel(metadata["ylabel"])
#     ax.set_title(metadata["title"] + " - " + name)
#     outputPlot(printMethod, folder, name+ "_spec_" + str(timeIdx))



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


def savenpz(name, outputs, metadata, timeIdx, folder, printMethod="pdf"):
    """Keeps only the latest save.
    Assumes that the data in previous
    saves is present in the current data"""
    flatOutputs = util.flattenDict(outputs, sep="~")
    np.savez_compressed(folder.joinpath(name + "_" + str(timeIdx)), **flatOutputs)

    earlierFiles = meu.findAllEarlierFiles(folder, name, timeIdx, nameIncludesIdx=False)
    for f in earlierFiles:
        if f.suffix == ".npz":
            f.unlink()


def soundfieldPlot(name, outputs, metadata, timeIdx, folder, printMethod="pdf"):
    print("a plot would be generated at timeIdx: ", timeIdx, "for diagnostic: ", name)
	


def plotIR(name, outputs, metadata, timeIdx, folder, printMethod="pdf"):
    numSets = 0
    for algoName, output in outputs.items():
        for irSet in output:
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
