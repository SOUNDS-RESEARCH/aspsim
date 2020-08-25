import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import json

from ancsim.experiment.multiexperimentutils import openData
from ancsim.experiment.plotscripts import outputPlot
import ancsim.settings as s

#selectPlots accepts "latest" and "all" or an index, "all" not yet implemented
def reductionByFrequency(expFolder, selectPlots="latest", plotMethod="tikz"):
    summary, settings, config = openData(expFolder, selectPlots)
    fig, axes = plt.subplots(1,2, figsize=(12,5))

    indices = np.argsort(config["NOISEFREQ"])
    #freqs = config["NOISEFREQ"][indices]

    dataType = "avgRegionalReduction"
    for filtName in summary[dataType]:
        axes[0].plot(np.array(config["NOISEFREQ"])[indices], np.array(summary[dataType][filtName])[indices], "-x")
    
    dataType = "avgPointReduction"
    for filtName in summary[dataType]:
        axes[1].plot(np.array(config["NOISEFREQ"])[indices], np.array(summary[dataType][filtName])[indices], "-x")
        
    axes[0].set_title("Regional Recuction")
    axes[1].set_title("Point Reduction")
    
    for ax in axes:
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Noise Reduction (dB)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True)
        ax.legend([filtname for filtname in summary[dataType]])
    #plt.suptitle("Simulation in 2D")
    #plt.xlabel("Frequency (Hz)")
    #plt.ylabel("Regional Reduction")
    plotName = "reduction_by_frequency"
    if isinstance(selectPlots, int):
        plotName += "_" + str(selectPlots)
    outputPlot(plotMethod, expFolder, plotName)
    #plt.savefig(expFolder+"/redoverfreq.pdf", dpi=300, facecolor='w', edgecolor='w',
    #orientation='portrait', papertype=None, format="pdf",
    #transparent=False, bbox_inches=None, pad_inches=0.1)

def redbyfreqanddist (expFolder):
    summary, settings = openData(expFolder)
    freq = np.array(settings["NOISEFREQ"])
    dist = np.array(settings["SPEAKERDIM"])
    reduct = np.array(summary["avgRegionalReduction"]["MPC FF"])
    pred = np.array(summary["avgPointReduction"]["MPC FF"])
    
    #distances = [i for i in range(1,9,2)]
    numFreq = 4
    dist = np.array([float(d[0]) for d in dist])
    
    freq = freq.reshape(numFreq, -1)
    dist = dist.reshape(numFreq, -1)
    reduct = reduct.reshape(numFreq,-1)
    pred = pred.reshape(numFreq, -1)

    fig, axes = plt.subplots(1,2, figsize=[12,6])

    colors = ["b", "k", "r", "g"]

    for i in range(dist.shape[1]):
        axes[0].plot(freq[:,i], reduct[:,i], colors[i], linestyle="--", marker="x")
        axes[1].plot(freq[:,i], pred[:,i], colors[i], linestyle="--", marker="x")
        
    reduct = np.array(summary["avgRegionalReduction"]["KernelIP FF 8a"])
    pred = np.array(summary["avgPointReduction"]["KernelIP FF 8a"])
    reduct = reduct.reshape(numFreq,-1)
    pred = pred.reshape(numFreq, -1)
        
        
    for i in range(dist.shape[1]):
        axes[0].plot(freq[:,i], reduct[:,i], colors[i], linestyle="-", marker="x")
        axes[1].plot(freq[:,i], pred[:,i], colors[i], linestyle="-", marker="x")
    
    axes[0].set_title("Regional Reduction")
    axes[1].set_title("Point Redcution")
    #fig.suptitle("MPC with different speaker configurations")

    for ax in axes:
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Noise Reduction (dB)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True)
    axes[1].legend(["Speaker square size: "+str(d)+" m" for d in dist[0,:]])
    axes[0].legend(["Kernel IP: solid line", "MPC: dashed line"])
    
    plt.savefig(expFolder.joinpath("sdbynf_line.pdf"), dpi=300, facecolor='w', edgecolor='w',
    orientation='portrait', papertype=None, format="pdf",
    transparent=False, bbox_inches=None, pad_inches=0.1)








def sdbynf (expFolder):
    summary, settings = openData(expFolder)
    freq = np.array(settings["nf"])
    dist = np.array(settings["sd"])
    reduct = np.array(summary["avgRegionalReduction"]["MPC FF"])
    pred = np.array(summary["avgPointReduction"]["MPC FF"])
    
    
    freq = freq.reshape(-1,9)
    dist = dist.reshape(-1,9)
    reduct = reduct.reshape(-1,9)
    pred = pred.reshape(-1,9)
    
    
    fig= plt.figure()
    ax = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,2),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )
    
    vmax = np.max((np.max(reduct[0:5,:]), np.max(pred[0:5,:])))
    vmin = np.min((np.min(reduct[0:5,:]), np.min(pred[0:5,:])))
    p = ax[0].imshow(np.flip(reduct[0:5, :].T, axis=0), vmin=vmin, vmax=vmax)
    p2 = ax[1].imshow(np.flip(pred[0:5, :].T, axis=0), vmin=vmin, vmax=vmax)
    
    ax[0].set_title("Regional Reduction")
    ax[1].set_title("Point Reduction")
    #p3 = ax[2].imshow(np.flip((reduct[0:5,:]-pred[0:5, :]).T, axis=0))

    for a in ax:
        a.set_xticks(np.arange(5))
        a.set_xticklabels(freq[0:5,0])
        a.set_yticks(np.arange(9))
        a.set_yticklabels(np.flip(dist[0,:]))
        a.set_xlabel("Frequency (Hz)")
        a.set_ylabel("Speaker square size (m)")
    a.cax.colorbar(p)
    
    plt.savefig(expFolder.joinpath("sdbynf.pdf"), dpi=300, facecolor='w', edgecolor='w',
    orientation='portrait', papertype=None, format="pdf",
    transparent=False, bbox_inches=None, pad_inches=0.1)
    


def sdbynf_line (expFolder):
    summary, settings = openData(expFolder)
    freq = np.array(settings["nf"])
    dist = np.array(settings["sd"])
    reduct = np.array(summary["avgRegionalReduction"]["MPC FF"])
    pred = np.array(summary["avgPointReduction"]["MPC FF"])
    
    distances = [i for i in range(1,9,2)]
    
    freq = freq.reshape(-1,9)[0:5,distances]
    dist = dist.reshape(-1,9)[0:5, distances]
    reduct = reduct.reshape(-1,9)[0:5,distances]
    pred = pred.reshape(-1,9)[0:5,distances]

    fig, axes = plt.subplots(1,2, figsize=[12,6])

    for i in range(len(distances)):
        axes[0].plot(freq[:,i], reduct[:,i])
        axes[1].plot(freq[:,i], pred[:,i])
    
    axes[0].set_title("Regional Reduction")
    axes[1].set_title("Point Redcution")
    fig.suptitle("MPC with different speaker configurations")

    for ax in axes:
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Noise Reduction (dB)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True)
        ax.legend(["Speaker square size: "+str(d)+" m" for d in dist[0,:]])

    
    plt.savefig(expFolder.joinpath("sdbynf_line.pdf"), dpi=300, facecolor='w', edgecolor='w',
    orientation='portrait', papertype=None, format="pdf",
    transparent=False, bbox_inches=None, pad_inches=0.1)



def plotTwoEntries(entry1, entry2, summary, settings, expFolder):
    set1 = settings[entry1]
    set2 = settings[entry2]
    for xVal,yVal in zip(set1, set2):
        summary["avgRegionalReduction"]
        #plt.plot("o")


def plotSingleEntryMetrics(entry, summary, settings, expFolder, filtName="all"):
    paramVals = settings[entry]

    numMetrics = len(summary)
    fig, axes = plt.subplots(numMetrics,1, figsize=(10,4*numMetrics))
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
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(True)
    plt.legend()

    #plt.show()
    plt.savefig(expFolder.joinpath("metricsOver" + entry +".pdf"), dpi=300, facecolor='w', edgecolor='w',
    orientation='portrait', papertype=None, format="pdf",
    transparent=False, bbox_inches=None, pad_inches=0.1)

# def plotSingleEntryMetrics(entry, summary, settings, expFolder):
#     paramVals = settings[entry]

#     numMetrics = len(summary)-1
#     fig, axes = plt.subplots(numMetrics,1, figsize=(numMetrics*5,8))
#     fig.tight_layout(pad=2)

#     for ax, (metricName, metric) in zip(axes, summary.items()):
#         if metricName != "meta":
#             for algoName, result in metric.items():
#                 ax.plot(paramVals, result, "o", label=algoName)
#             if chooseLogScaling(paramVals):
#                 ax.set_xscale("log")
            
#             ax.set_xlabel(entry)
#             ax.set_ylabel(metricName)
#             ax.spines['right'].set_visible(False)
#             ax.spines['top'].set_visible(False)
#             ax.grid(True)
#     plt.legend()

#     #plt.show()
#     plt.savefig(expFolder.joinpath("metricsOver" + entry +".pdf"), dpi=300, facecolor='w', edgecolor='w',
#     orientation='portrait', papertype=None, format="pdf",
#     transparent=False, bbox_inches=None, pad_inches=0.1)

def getSingleEntryResults(entry, dataName, summary, settings):
    data = summary[dataName]
    paramVals = settings[entry]
    
def chooseLogScaling(values):
    values = np.array(values)
    valRatio = np.max(np.abs(values)) / np.min(np.abs(values))
    if valRatio >= 100:
        return True
    return False




if __name__ == "__main__":
    redbyfreqanddist("test_plots_2020_02_25_13_13")
    #sdbynf_line("test_plots_2020_02_14_20_27")