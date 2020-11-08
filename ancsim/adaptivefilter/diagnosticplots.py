import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import soundfile as sf

from ancsim.experiment.plotscripts import outputPlot
import ancsim.experiment.multiexperimentutils as meu
import ancsim.utilities as util


class SCALINGTYPE(Enum):
    linear = 1
    dbPower = 2
    dbAmp = 3
    ln = 4


def scaleData(scalingType, data):
    if scalingType == SCALINGTYPE.linear:
        return data
    elif scalingType == SCALINGTYPE.dbPower:
        return 10 * np.log10(data)
    elif scalingType == SCALINGTYPE.dbAmp:
        return 20 * np.log10(data)
    elif scalingType == SCALINGTYPE.ln:
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

        filterArray = np.logical_not(np.isnan(output))
        assert np.isclose(filterArray, filterArray[0, :]).all()
        filterArray = filterArray[0, :]

        ax.plot(
            xValues[:, filterArray].T,
            output[:, filterArray].T,
            alpha=0.8,
            label=algoName,
        )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.grid(True)
    ax.legend(loc="upper right")

    ax.set_xlabel(metadata["xlabel"])
    ax.set_ylabel(metadata["ylabel"])
    ax.set_title(metadata["title"] + " - " + name)
    outputPlot(printMethod, folder, name + "_" + str(timeIdx))


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
