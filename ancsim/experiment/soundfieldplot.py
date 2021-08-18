import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

import ancsim.experiment.multiexperimentutils as meu
import ancsim.saveloadsession as sess
import ancsim.experiment.plotscripts as ps
import ancsim.soundfield.roomimpulseresponse as rir
#from ancsim.simulatorsetup import setupSource
import ancsim.signal.sources
from ancsim.signal.filterclasses import FilterSum
import ancsim.utilities as util
import ancsim.array as ar


def generateSoundfieldForFolder(singleSetFolder, sessionFolder):
    for singleRunFolder in os.listdir(singleSetFolder):
        makeSoundfieldPlot(singleSetFolder + singleRunFolder, sessionFolder)


def makeSoundfieldPlot(singleRunFolder, sessionFolder, numPixels, normalize=True):
    config, pos, source, filterCoeffs, sourceRIR, speakerRIR = loadSession(
        singleRunFolder, sessionFolder, numPixels
    )
    pointsToSim = pos["target"]

    avgSoundfields = soundSim(
        config,
        pos["target"],
        source,
        filterCoeffs,
        sourceRIR["ref"],
        sourceRIR["target"],
        speakerRIR["target"],
    )

    zeroValue = np.mean(util.db2pow(avgSoundfields[0]))
    avgSoundfields = [util.pow2db(util.db2pow(sf) / zeroValue) for sf in avgSoundfields]
    maxVal = np.max([np.max(sf) for sf in avgSoundfields])
    minVal = np.min([np.min(sf) for sf in avgSoundfields])
    algoNames = ["Without ANC"] + [key for key in filterCoeffs.keys()]

    fig, axes = plt.subplots(1, len(filterCoeffs) + 1, figsize=(30, 7))
    fig.tight_layout(pad=2.5)

    #imAxisLen = int(np.sqrt(pointsToSim.shape[0]))
    pointsToSim, avgSoundfields = sortIntoGrid(pointsToSim, avgSoundfields)
    for i, ax in enumerate(axes):
        # clr = ax.pcolormesh(
        #     pointsToSim[:, 0].reshape(imAxisLen, imAxisLen),
        #     pointsToSim[:, 1].reshape(imAxisLen, imAxisLen),
        #     avgSoundfields[i].reshape(imAxisLen, imAxisLen),
        #     vmin=minVal,
        #     vmax=maxVal,
        #     cmap="pink",
        #     shading="nearest",
        # )
        clr = ax.pcolormesh(
            pointsToSim[..., 0],
            pointsToSim[..., 1],
            avgSoundfields[i],
            vmin=minVal,
            vmax=maxVal,
            cmap="pink",
            shading="nearest",
        )
        fig.colorbar(clr, ax=ax)

        ax.plot(pos["speaker"][:, 0], pos["speaker"][:, 1], "o")
        ax.plot(pos["source"][:, 0], pos["source"][:, 1], "o", color="m")
        ax.plot(pos["error"][:, 0], pos["error"][:, 1], "x", color="y")
        ax.axis("equal")
        ax.set_title(algoNames[i])
        ax.set_xlim(pointsToSim[:, 0].min(), pointsToSim[:, 0].max())
        ax.set_ylim(pointsToSim[:, 1].min(), pointsToSim[:, 1].max())
    ps.outputPlot("tikz", singleRunFolder, "soundfield")


def soundSim(
    config,
    pointsToSim,
    source,
    filterCoeffs,
    sourceToRef,
    sourceToPoints,
    speakerToPoints,
):
    minSamplesToAverage = 10000
    numTotPoints = pointsToSim.shape[0]
    refFiltLen = sourceToRef.ir.shape[-1]
    # blockSize = np.max([np.max(coefs.shape) for coefs in filterCoeffs])
    maxBlockSize = np.max(config["BLOCKSIZE"])
    startBuffer = maxBlockSize * int(
        np.ceil(
            (refFiltLen + sourceToPoints.ir.shape[-1] + maxBlockSize) / maxBlockSize
        )
    )
    samplesToAverage = maxBlockSize * int(np.ceil(minSamplesToAverage / maxBlockSize))
    totNumSamples = startBuffer + samplesToAverage

    # sourceToRefFilt = FilterSum(sourceToRef)
    sourceSig = source.getSamples(totNumSamples)

    refSig = sourceToRef.process(sourceSig)[:, refFiltLen:]
    speakerSigs = genSpeakerSignals(
        maxBlockSize, totNumSamples, refSig, filterCoeffs, refFiltLen
    )

    maxPointsInIter = 500
    avgSoundfield = []

    # Empty soundfield
    soundfield = np.zeros((numTotPoints))
    i = 0
    while i < numTotPoints:
        print(i)
        numPoints = np.min((maxPointsInIter, numTotPoints - i))
        sourceToPointsFilt = FilterSum(
            sourceToPoints.ir[:, i : i + numPoints, :]
        )
        pointNoise = sourceToPointsFilt.process(sourceSig)[:, startBuffer:]
        soundfield[i : i + numPoints] = 10 * np.log10(
            np.mean((pointNoise) ** 2, axis=-1)
        )
        i += numPoints
    avgSoundfield.append(soundfield)

    # Soundfield for each set of filtercoefficients
    for speakerSig in speakerSigs:
        soundfield = np.zeros((numTotPoints))
        i = 0
        while i < numTotPoints:
            print(i)
            numPoints = np.min((maxPointsInIter, numTotPoints - i))
            sourceToPointsFilt = FilterSum(
                sourceToPoints.ir[:, i : i + numPoints, :]
            )
            speakerToPointsFilt = FilterSum(
                speakerToPoints[:, i : i + numPoints, :]
            )
            pointSig = speakerToPointsFilt.process(speakerSig)[:, -samplesToAverage:]
            pointNoise = sourceToPointsFilt.process(sourceSig)[
                :, -samplesToAverage - maxBlockSize : -maxBlockSize
            ]
            soundfield[i : i + numPoints] = 10 * np.log10(
                np.mean((pointSig + pointNoise) ** 2, axis=-1)
            )
            i += numPoints
        avgSoundfield.append(soundfield)
    return avgSoundfield


def genSpeakerSignals(
    blockSize, totNumSamples, refSig, filterCoeffs, sourceToRefFiltLen
):
    outputs = []
    for ir in filterCoeffs.values():
        if ir.dtype == np.float64:
            filt = FilterSum(ir)
            outputs.append(filt.process(refSig[:, :-blockSize])[:, ir.shape[-1] :])
        elif ir.dtype == np.complex128:
            idx = blockSize
            numSamples = totNumSamples - sourceToRefFiltLen
            buf = np.zeros((ir.shape[1], numSamples))
            while idx + blockSize < numSamples:
                refBlock = refSig[:, idx - blockSize : idx + blockSize]
                refFreq = np.fft.fft(refBlock, axis=-1).T[:, :, None]
                Y = ir @ refFreq

                y = np.fft.ifft(Y, axis=0)
                assert np.mean(np.abs(np.imag(y))) < 0.00001
                y = np.squeeze(np.real(y)).T
                buf[:, idx : idx + blockSize] = y[:, blockSize:]
                # y[:,idx:idx+s.BLOCKSIZE] = y
                idx += blockSize
            outputs.append(buf[:, blockSize:-blockSize])
    return outputs


def loadSession(singleRunFolder, sessionFolder, numPixels):
    config = sess.loadConfig(singleRunFolder)
    config["TARGETPOINTSPLACEMENT"] = "image"
    config["NUMTARGET"] = numPixels
    # config["NUMEVALS"] = 128**2

    if config["LOADSESSION"]:
        pos, sourceRIR, speakerRIR = sess.loadSession(
            sessionFolder, singleRunFolder, config
        )
    else:
        raise NotImplementedError

    controlFilterPath = meu.getHighestNumberedFile(
        singleRunFolder, "controlFilter_", ".npz"
    )
    if controlFilterPath is None:
        raise ValueError
    filterCoeffs = np.load(controlFilterPath)
    #filterNames = [names for names in filterCoeffs.keys()]
    filterCoeffs = {key:val for key, val in filterCoeffs.items()}

    source = setupSource(config)

    return config, pos, source, filterCoeffs, sourceRIR, speakerRIR


def sortIntoGrid(pos, signals_to_sort, round_to_digits=4):
    """use other sort function, as this requires square grid. 
        This also differs in that it does not flip the y axis"""
    numPos = pos.shape[0]
    numEachSide = int(np.sqrt(numPos))
    assert numEachSide**2 == numPos
    xVal= np.unique(pos[:,0].round(round_to_digits))
    yVal = np.unique(pos[:,1].round(round_to_digits))

    sortIndices = np.zeros((numEachSide, numEachSide), dtype=int)

    for i, y in enumerate(yVal):
        rowIndices = np.where(np.abs(pos[:,1] - y) < 1e-4)[0]
        rowPermutation = np.argsort(pos[rowIndices,0])
        sortIndices[i,:] = rowIndices[rowPermutation]

    signals_to_sort = [sig[sortIndices] for sig in signals_to_sort]
    pos = pos[sortIndices,:]
    return pos, signals_to_sort
        










def compare_soundfields(pos, sig, algo_labels, row_labels, arrays_to_plot={}):
    """single pos array of shape (numPoints, spatialDim)
        sig and labels are lists of the same length, numAlgos
        each entry in sig is an array of shape (numAxes, numPoints), 
        so for example multiple frequencies can be plotted at the same time.
        
        The resulting plot is multiple axes, numAlgos rows high, and numAxes columns wide"""
    #num_rows, num_cols = sfp.get_num_pixels(pos)
    pos, sig = sort_for_imshow(pos, sig)
    if sig.ndim == 2:
        sig = sig[None,...]
    if sig.ndim == 3:
        sig = sig[None,...]

    numAlgos = sig.shape[0]
    numEachAlgo = sig.shape[1]
    extent = size_of_2d_plot([pos] + list(arrays_to_plot.values()))
    extent /= np.max(extent)
    scaling = 5

    fig, axes = plt.subplots(numAlgos, numEachAlgo, figsize=(numEachAlgo*extent[0]*scaling, numAlgos*extent[1]*scaling))
    fig.tight_layout(pad=2.5)

    


    for i, rows in enumerate(np.atleast_2d(axes)):
        for j, ax in enumerate(rows):
            sf_plot(ax, pos, sig[i,j,:,:], f"{algo_labels[i]}, {row_labels[j]}", arrays_to_plot)
    

def size_of_2d_plot(array_pos):
    """array_pos is a list/tuple of ndarrays of shape(any, spatialDim)
        describing the positions of all objects to be plotted. First axis 
        can be any value for the arrays, second axis must be 2 or more. Only
        the first two are considered. 
        
        returns np.array([x_size, y_size])"""
    array_pos = [ap[...,:2].reshape(-1,2) for ap in array_pos]
    all_pos = np.concatenate(array_pos, axis=0)
    extent = np.max(all_pos, axis=0) - np.min(all_pos, axis=0)  
    return extent[:2]
        

def sf_plot(ax, pos, sig, title="", arrays_to_plot = {}):
    """takes a single ax, pos and sig and creates a decent looking soundfield plot
        Assumes that pos and sig are already correctly sorted"""

    im = ax.imshow(sig, interpolation="none", extent=(pos[...,0].min(), pos[...,0].max(), pos[...,1].min(), pos[...,1].max()))
    
    # if isinstance(arrays_to_plot, ar.ArrayCollection):
    #     for array in arrays_to_plot:
    #         array.plot(ax)
    if isinstance(arrays_to_plot, dict):
        for arName, array in arrays_to_plot.items():
            ax.plot(array[:,0], array[:,1], "x", label=arName)
    else:
        raise ValueError

    ax.legend()
    ax.set_title(title)
    ps.setBasicPlotLook(ax)
    ax.axis("equal")
    plt.colorbar(im, ax=ax, orientation='vertical')
    #plt.colorbar(ax=ax)

def get_num_pixels(pos, pos_decimals=5):
    pos_cols = np.unique(pos[:,0].round(pos_decimals))
    pos_rows = np.unique(pos[:,1].round(pos_decimals))
    num_rows = len(pos_rows)
    num_cols = len(pos_cols)
    return num_rows, num_cols


def sort_for_imshow(pos, sig, pos_decimals=5):
    """ pos must be of shape (numPos, spatialDims) placed on a rectangular grid, 
        but can be in any order.
        sig can be a single signal or a list of signals of shape (numPos, signalDim), where each
        entry on first axis is the value for pos[0,:] """
    if pos.shape[1] == 3:
        assert np.allclose(pos[:,2], np.ones_like(pos[:,2])*pos[0,2])

    num_rows, num_cols = get_num_pixels(pos, pos_decimals)
    unique_x = np.unique(pos[:,0].round(pos_decimals))
    unique_y = np.unique(pos[:,1].round(pos_decimals))

    sortIndices = np.zeros((num_rows, num_cols), dtype=int)
    for i, y in enumerate(unique_y):
        rowIndices = np.where(np.abs(pos[:,1] - y) < 10**(-pos_decimals))[0]
        rowPermutation = np.argsort(pos[rowIndices,0])
        sortIndices[i,:] = rowIndices[rowPermutation]

    pos = pos[sortIndices,:]

    sig = np.moveaxis(np.atleast_3d(sig),1,2)
    dims = sig.shape[:2]
    sig_sorted = np.zeros((dims[0], dims[1], num_rows, num_cols))
    for i in range(dims[0]):
        for j in range(dims[1]):
            single_sig = sig[i,j,:]
            sig_sorted[i,j,:,:] = np.flip(single_sig[sortIndices],axis=0)
    sig_sorted = np.squeeze(sig_sorted)
    
    #sig = [np.flip(s[sortIndices],axis=0) for s in sig]
    
    return pos, sig_sorted