import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

from ancsim.experiment.plotscripts import outputPlot
import ancsim.experiment.multiexperimentutils as meu


class SCALINGTYPE(Enum):
    linear = 1
    dbPower = 2
    dbAmp = 3
    ln = 4

def scaleData(scalingType, data):
    if scalingType == SCALINGTYPE.linear:
        return data
    elif scalingType == SCALINGTYPE.dbPower:
        return 10*np.log10(data)
    elif scalingType == SCALINGTYPE.dbAmp:
        return 20*np.log10(data)
    elif scalingType == SCALINGTYPE.ln:
        return np.log(data)

def functionOfTimePlot(name, outputs, metadata, timeIdx, folder, printMethod="pdf"):
    fig, ax = plt.subplots(1,1, figsize=(14,8))
    fig.tight_layout(pad=4)
    
    xValues = np.arange(timeIdx)[None,:]
    for algoName, output in outputs.items():
        if output.ndim == 1:
            output = output[None,:timeIdx]
        elif output.ndim == 2:
            output = output[:,:timeIdx]
        else:
            raise NotImplementedError

        filterArray = np.logical_not(np.isnan(output))
        assert(np.isclose(filterArray, filterArray[0,:]).all())
        filterArray = filterArray[0,:]

        ax.plot(xValues[:,filterArray].T, output[:,filterArray].T, alpha=0.8, label=algoName)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(True)
    ax.legend(loc="upper right")

    ax.set_xlabel(metadata["xlabel"])
    ax.set_ylabel(metadata["ylabel"])
    ax.set_title(metadata["title"] + " - " + name)
    outputPlot(printMethod,folder, name+"_" + str(timeIdx))


def savenpz(name, outputs, metadata, timeIdx, folder, printMethod="pdf"):
    """Keeps only the latest save. 
        Assumes that the data in previous 
        saves is present in the current data"""
    flatOutputs = flattenDict(outputs, sep="~")
    np.savez_compressed(folder.joinpath(name + "_" + str(timeIdx)), **flatOutputs)

    earlierFiles = meu.findAllEarlierFiles(folder, name, timeIdx, nameIncludesIdx=False)
    for f in earlierFiles:
        if f.suffix == ".npz":
            f.unlink()


def soundfieldPlot(name, outputs, metadata, timeIdx, folder, printMethod="pdf"):
    print("a plot would be generated at timeIdx: ", timeIdx, "for diagnostic: ", name)

def flattenDict(dictToFlatten, parentKey="", sep="_"):
    items = []
    for key, value in dictToFlatten.items():
        newKey = parentKey + sep+ key if parentKey else key
        if isinstance(value, dict):
            items.extend(flattenDict(value, newKey, sep).items())
        else:
            items.append((newKey, value))
    newDict = dict(items)
    return newDict

