import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

import ancsim.experiment.multiexperimentutils as meu
import ancsim.saveloadsession as sess
from ancsim.experiment.plotscripts import outputPlot
import ancsim.soundfield.roomimpulseresponse as rir
from ancsim.simulatorsetup import setupSource
import ancsim.signal.sources
from ancsim.signal.filterclasses import FilterSum_IntBuffer
import ancsim.utilities as util


def generateSoundfieldForFolder(singleSetFolder, sessionFolder):
    for singleRunFolder in os.listdir(singleSetFolder):
        makeSoundfieldPlot(singleSetFolder + singleRunFolder, sessionFolder)


def makeSoundfieldPlot(singleRunFolder, sessionFolder, normalize=True):
    config, pos, source, filterCoeffs, sourceRIR, speakerRIR = loadSession(
        singleRunFolder, sessionFolder
    )
    pointsToSim = pos["target"]

    avgSoundfields = soundSim(
        config,
        pos["target"],
        source,
        filterCoeffs,
        sourceRIR[1],
        sourceRIR[3],
        speakerRIR["target"],
    )

    zeroValue = np.mean(util.db2pow(avgSoundfields[0]))
    avgSoundfields = [util.pow2db(util.db2pow(sf) / zeroValue) for sf in avgSoundfields]
    maxVal = np.max([np.max(sf) for sf in avgSoundfields]) - 10
    minVal = np.min([np.min(sf) for sf in avgSoundfields])

    fig, axes = plt.subplots(1, len(filterCoeffs) + 1, figsize=(30, 7))
    fig.tight_layout(pad=2.5)

    imAxisLen = int(np.sqrt(pointsToSim.shape[0]))
    for i, ax in enumerate(axes):
        clr = ax.pcolormesh(
            pointsToSim[:, 0].reshape(imAxisLen, imAxisLen),
            pointsToSim[:, 1].reshape(imAxisLen, imAxisLen),
            avgSoundfields[i].reshape(imAxisLen, imAxisLen),
            vmin=minVal,
            vmax=maxVal,
            cmap="pink",
        )
        fig.colorbar(clr, ax=ax)

        ax.plot(pos["speaker"][:, 0], pos["speaker"][:, 1], "o")
        ax.plot(pos["source"][:, 0], pos["source"][:, 1], "o", color="m")
        ax.plot(pos["error"][:, 0], pos["error"][:, 1], "x", color="y")
        ax.axis("equal")
        ax.set_xlim(pointsToSim[:, 0].min(), pointsToSim[:, 0].max())
        ax.set_ylim(pointsToSim[:, 1].min(), pointsToSim[:, 1].max())
    outputPlot("tikz", singleRunFolder, "soundfield")


def soundSim(
    config,
    pointsToSim,
    source,
    filterCoeffs,
    sourceToRef,
    sourceToPoints,
    speakerToPoints,
):
    minSamplesToAverage = 1000
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

    # sourceToRefFilt = FilterSum_IntBuffer(sourceToRef)
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
        sourceToPointsFilt = FilterSum_IntBuffer(
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
            sourceToPointsFilt = FilterSum_IntBuffer(
                sourceToPoints.ir[:, i : i + numPoints, :]
            )
            speakerToPointsFilt = FilterSum_IntBuffer(
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
    for ir in filterCoeffs:
        if ir.dtype == np.float64:
            filt = FilterSum_IntBuffer(ir)
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


def loadSession(singleRunFolder, sessionFolder):
    config = sess.loadConfig(singleRunFolder)
    config["TARGETPOINTSPLACEMENT"] = "image"
    config["NUMTARGET"] = 128 ** 2
    # config["NUMEVALS"] = 128**2

    if config["LOADSESSION"]:
        pos, sourceRIR, speakerRIR = sess.loadSession(
            config, sessionFolder, singleRunFolder
        )
    else:
        raise NotImplementedError

    controlFilterPath = meu.getHighestNumberedFile(
        singleRunFolder, "controlFilter_", ".npz"
    )
    if controlFilterPath is None:
        raise ValueError
    filterCoeffs = np.load(controlFilterPath)
    filterNames = [names for names in filterCoeffs.keys()]
    filterCoeffs = [val for names, val in filterCoeffs.items()]

    source = setupSource(config)

    return config, pos, source, filterCoeffs, sourceRIR, speakerRIR
