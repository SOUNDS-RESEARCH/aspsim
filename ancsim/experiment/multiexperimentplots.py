import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import json

from ancsim.experiment.multiexperimentutils import openData
from ancsim.experiment.plotscripts import outputPlot

# selectPlots accepts "latest" and "all" or an index, "all" not yet implemented
def reductionByFrequency(expFolder, selectPlots="latest", plotMethod="tikz"):
    summary, settings, config, filtParams = openData(expFolder, selectPlots)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    indices = np.argsort(config["NOISEFREQ"])
    # freqs = config["NOISEFREQ"][indices]

    dataType = "avgRegionalReduction"
    for filtName in summary[dataType]:
        axes[0].plot(
            np.array(config["NOISEFREQ"])[indices],
            np.array(summary[dataType][filtName])[indices],
            "-x",
        )

    dataType = "avgPointReduction"
    for filtName in summary[dataType]:
        axes[1].plot(
            np.array(config["NOISEFREQ"])[indices],
            np.array(summary[dataType][filtName])[indices],
            "-x",
        )

    axes[0].set_title("Regional Recuction")
    axes[1].set_title("Point Reduction")

    for ax in axes:
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Noise Reduction (dB)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(True)
        ax.legend([filtname for filtname in summary[dataType]])
    # plt.suptitle("Simulation in 2D")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Regional Reduction")
    plotName = "reduction_by_frequency"
    if isinstance(selectPlots, int):
        plotName += "_" + str(selectPlots)
    outputPlot(plotMethod, expFolder, plotName)
    # plt.savefig(expFolder+"/redoverfreq.pdf", dpi=300, facecolor='w', edgecolor='w',
    # orientation='portrait', papertype=None, format="pdf",
    # transparent=False, bbox_inches=None, pad_inches=0.1)


def plotTwoEntries(entry1, entry2, summary, settings, expFolder):
    set1 = settings[entry1]
    set2 = settings[entry2]
    for xVal, yVal in zip(set1, set2):
        summary["avgRegionalReduction"]
        # plt.plot("o")


def plotSingleEntryMetrics(
    entry, summary, settings, expFolder, outputMethod="pdf", filtName="all"
):
    paramVals = settings[entry]

    numMetrics = len(summary)
    fig, axes = plt.subplots(numMetrics, 1, figsize=(10, 4 * numMetrics))
    fig.tight_layout(pad=2)

    for ax, (metricName, metric) in zip(axes, summary.items()):
        if filtName == "all":
            for algoName, result in metric.items():
                ax.plot(paramVals, result, "o", label=algoName)
        else:
            ax.plot(paramVals, metric[filtName], "o", label=filtName)
        if chooseLogScaling(paramVals):
            ax.set_xscale("log")

        ax.set_xlabel(entry)
        ax.set_ylabel(metricName)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(True)
    plt.legend()

    # plt.show()
    outputPlot(outputMethod, expFolder, "metricsOver" + entry + ".pdf")
    # plt.savefig(expFolder.joinpath("metricsOver" + entry +".pdf"), dpi=300, facecolor='w', edgecolor='w',
    # orientation='portrait', papertype=None, format="pdf",
    # transparent=False, bbox_inches=None, pad_inches=0.1)


def getSingleEntryResults(entry, dataName, summary, settings):
    data = summary[dataName]
    paramVals = settings[entry]


def chooseLogScaling(values):
    values = np.array(values)
    valRatio = np.max(np.abs(values)) / np.min(np.abs(values))
    if valRatio >= 100:
        return True
    return False
