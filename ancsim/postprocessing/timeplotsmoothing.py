import os
import numpy as np
import scipy.signal as sig
import time

def downsample(signalToProcess, downsamplingfactor):
    pass


def processFolder(fldr="", params):
    print(os.getcwd()) 
  
    for f in os.listdir(fldr):
        fullpath = fldr+f
        if os.path.isfile(fullpath) and f.endswith(".tsv") and not f.endswith("processed.tsv"):
            newfilename = f[:-4] + "_processed.tsv"
            ar = np.genfromtxt(fullpath)
            
            ar[:,1] = smoothData(ar[:,1], params["smoothing"])
            ar[:,1] = downsample(ar[:,1], params["downsampling"])

            #outArray = np.vstack((np.arange(smoothedData.shape[0]), smoothedData))
            np.savetxt(fldr+newfilename, ar, fmt="%i %f")


def smoothData(signalToProcess, smoothLen, decibel=True):
    orig = signalToProcess
    if decibel:
        signalToProcess = 10**(signalToProcess / 10)

    signalToProcess = np.concatenate((np.ones(smoothLen-1), signalToProcess))
    smoothedArray = np.convolve(signalToProcess, np.ones(smoothLen)/smoothLen, mode="valid")

    if decibel:
        smoothedArray = 10*np.log10(smoothedArray)
    plt.plot(orig)
    plt.plot(smoothedArray)
    plt.show()

    return smoothedArray
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    params = {"smoothing" : 32,
              "downsampling" : 16}
    smoothFiles(fldr="reduction_100000/", params)
