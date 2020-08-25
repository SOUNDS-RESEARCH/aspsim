import os
import sys
import json 
import numpy as np
import matplotlib.pyplot as plt

import ancsim.experiment.multiexperimentutils as meu
from ancsim.experiment.plotscripts import outputPlot
import ancsim.soundfield.roomimpulseresponse as rir
import ancsim.signal.sources
from ancsim.signal.filterclasses import FilterSum_IntBuffer


def generateSoundfieldForFolder(singleSetFolder):
    for singleRunFolder in os.listdir(singleSetFolder):
        makeSoundfieldPlot(singleSetFolder+singleRunFolder)


def makeSoundfieldPlot(singleRunFolder):
    failed, config, s, pos, pointsToSim, source, filterCoeffs, sourceToRef, sourceToPoints, speakerToPoints = loadSession(singleRunFolder)
    if failed:
        print("Session Load Failed")
        return
    
    avgSoundfields = soundSim(config, s, pos, pointsToSim, source, filterCoeffs, sourceToRef, sourceToPoints, speakerToPoints)

    maxVal = np.max([np.max(sf) for sf in avgSoundfields])
    minVal = np.min([np.min(sf) for sf in avgSoundfields])

    fig, axes = plt.subplots(1, len(filterCoeffs)+1, figsize = (30, 7))
    fig.tight_layout(pad=2.5)

    imAxisLen = int(np.sqrt(pointsToSim.shape[0]))
    for i, ax in enumerate(axes):
        clr = ax.pcolormesh(pointsToSim[:,0].reshape(imAxisLen, imAxisLen), 
                            pointsToSim[:,1].reshape(imAxisLen, imAxisLen), 
                        avgSoundfields[i].reshape(imAxisLen, imAxisLen), 
                        vmin=minVal, vmax=maxVal, cmap="magma")
        fig.colorbar(clr, ax=ax)

        ax.plot(pos.speaker[:,0], pos.speaker[:,1], "o")
        ax.plot(pos.source[:,0], pos.source[:,1], "o", color="m")
        ax.plot(pos.error[:,0], pos.error[:,1], "x", color="y")
        ax.axis("equal")
        ax.set_xlim(pointsToSim[:,0].min(), pointsToSim[:,0].max())
        ax.set_ylim(pointsToSim[:,1].min(), pointsToSim[:,1].max())
    outputPlot("tikz", singleRunFolder, "soundfield")

def soundSim(config, s, pos, pointsToSim, source, filterCoeffs, sourceToRef, sourceToPoints, speakerToPoints):
    numSamplesToAverage = 2000
    numPoints = sourceToPoints.shape[1]
    refFiltLen = sourceToRef.shape[-1]
    adaptiveFiltLen = filterCoeffs[0].shape[-1]
    totNumSamples = numSamplesToAverage + refFiltLen + s["BLOCKSIZE"] + sourceToPoints.shape[-1] + adaptiveFiltLen
    
    sourceToRefFilt = FilterSum_IntBuffer(sourceToRef)
    sourceSig = source.getSamples(totNumSamples)

    refSig = sourceToRefFilt.process(sourceSig)[:,refFiltLen:]
    speakerSigs = genSpeakerSignals(s, totNumSamples, refSig, filterCoeffs, sourceToRef)

    maxPointsInIter = 500
    avgSoundfield = []

    soundfield = np.zeros((numPoints))
    i = 0
    while i < numPoints:
        print(i)
        blockSize = np.min((maxPointsInIter, numPoints-i))
        sourceToPointsFilt = FilterSum_IntBuffer(sourceToPoints[:,i:i+blockSize,:])
        pointNoise = sourceToPointsFilt.process(sourceSig)[:,-numSamplesToAverage-s["BLOCKSIZE"]:-s["BLOCKSIZE"]]
        soundfield[i:i+blockSize] = 10*np.log10(np.mean((pointNoise)**2,axis=-1))  
        i += blockSize
    avgSoundfield.append(soundfield)

    for speakerSig in speakerSigs:
        soundfield = np.zeros((numPoints))
        i = 0
        while i < numPoints:
            print(i)
            blockSize = np.min((maxPointsInIter, numPoints-i))
            sourceToPointsFilt = FilterSum_IntBuffer(sourceToPoints[:,i:i+blockSize,:])
            speakerToPointsFilt = FilterSum_IntBuffer(speakerToPoints[:,i:i+blockSize,:])
            pointSig = speakerToPointsFilt.process(speakerSig)[:,-numSamplesToAverage:]
            pointNoise = sourceToPointsFilt.process(sourceSig)[:,-numSamplesToAverage-s["BLOCKSIZE"]:-s["BLOCKSIZE"]]
            soundfield[i:i+blockSize] = 10*np.log10(np.mean((pointSig + pointNoise)**2,axis=-1))
            i += blockSize
        avgSoundfield.append(soundfield)
    return avgSoundfield
    
def genSpeakerSignals(s, totNumSamples, refSig, filterCoeffs, sourceToRef):
    outputs = []
    for ir in filterCoeffs:
        if ir.dtype == np.float64:
            filt = FilterSum_IntBuffer(ir)
            outputs.append(filt.process(refSig[:,:-s["BLOCKSIZE"]])[:,ir.shape[-1]:])
        elif ir.dtype == np.complex128:
            idx = s["BLOCKSIZE"]
            numSamples = totNumSamples-sourceToRef.shape[-1]
            buf = np.zeros((ir.shape[1], numSamples))
            while idx+s["BLOCKSIZE"] < numSamples:
                refBlock = refSig[:,idx-s["BLOCKSIZE"]:idx+s["BLOCKSIZE"]]
                refFreq = np.fft.fft(refBlock,axis=-1).T[:,:,None]
                Y = ir @ refFreq
                
                y = np.fft.ifft(Y, axis=0)
                assert(np.mean(np.abs(np.imag(y))) < 0.00001)
                y = np.squeeze(np.real(y)).T
                buf[:,idx:idx+s["BLOCKSIZE"]] = y[:,s["BLOCKSIZE"]:]
                #y[:,idx:idx+s.BLOCKSIZE] = y
                idx += s["BLOCKSIZE"]
            outputs.append(buf[:,s["BLOCKSIZE"]:-s["BLOCKSIZE"]])
    return outputs


def loadSession(singleRunFolder):
    s = {}
    s["SAMPLERATE"] = 4000
    s["BLOCKSIZE"] = 512
    try:
        with open(singleRunFolder+"configfile.json") as f:
            config = json.load(f)
    except FileNotFoundError:
        return True,None,None,None,None,None,None,None,None,None
    
    filterCoeffsFile = meu.getHighestNumberedFile(singleRunFolder, "filterCoeffs_", ".npz")
    filterCoeffs = np.load(singleRunFolder+filterCoeffsFile)
    filterNames  = [names for names in filterCoeffs.keys()]
    filterCoeffs = [val for names,val in filterCoeffs.items()]
    
    pos = loadPositions(singleRunFolder+"pos.npz")
    irGenPos = loadPositions("savedsession/pos.npz")
    assert(posEqual(pos, irGenPos))
    pointsToSim = irGenPos.evals2
    
    loadedEvalsIr = np.load("savedsession/evals_ir_close.npz")
    sourceToPoints = loadedEvalsIr["sourceToEvals2"]
    speakerToPoints = loadedEvalsIr["speakerToEvals2"]

    if config["REFDIRECTLYOBTAINED"]:
        sourceToRef = np.ones((1,1,1))
    else:
        sourceToRef = rir.irSimulatedRoom3d(pos.source, pos.ref, config["ROOMSIZE"], config["ROOMCENTER"], sampleRate=s["SAMPLERATE"])

    if config["SOURCETYPE"] == "sine":
        source = sources.SineSource(50, config["NOISEFREQ"], s["SAMPLERATE"])
    elif config["SOURCETYPE"] == "noise":
        source = sources.BandlimitedNoiseSource(50, (config["NOISEFREQ"], config["NOISEFREQ"]+config["NOISEBANDWIDTH"]), s["SAMPLERATE"])
    else:
        raise NotImplementedError

    return False, config, s, pos, pointsToSim, source, filterCoeffs, sourceToRef, sourceToPoints, speakerToPoints


def loadPositions(filename):
    posDict = np.load(filename)
    class AllArrays: pass
    pos = AllArrays()
    pos.ref = posDict["ref"]
    pos.error = posDict["error"]
    pos.speaker = posDict["speaker"]
    pos.source = posDict["source"]
    pos.evals = posDict["evals"]
    pos.evals2 = posDict["evals2"]
    return pos

#Compares two positions sets, but not the evaluation sets (i.e. target, evals)
def posEqual(pos1, pos2):
    equal = (np.allclose(pos1.ref, pos2.ref) and 
            np.allclose(pos1.error, pos2.error) and
            np.allclose(pos1.speaker, pos2.speaker) and
            np.allclose(pos1.source, pos2.source))
    return equal


if __name__ == "__main__":
   makeSoundfieldPlot("multi_experiments/full_exp_2020_03_15_sinfix/figs_2020_04_14_23_46_1_200hz/")