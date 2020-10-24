import numpy as np
import ancsim.utilities as util

def calcBlockSizes(numSamples, idx, bufferSize, chunkSize):
    leftInBuffer = chunkSize+bufferSize-idx
    sampleCounter = 0
    blockSizes = []
    while sampleCounter < numSamples:
        bLen = np.min((numSamples-sampleCounter, leftInBuffer))
        blockSizes.append(bLen)
        sampleCounter += bLen
        leftInBuffer -= bLen 
        if leftInBuffer == 0:
            leftInBuffer = bufferSize
    return blockSizes

def getWhiteNoiseAtSNR(signal, dim, snr, identicalChannelPower=False):
    """generates Additive White Gaussian Noise 
        signal to calculate signal level. time dimension is last dimension
        dim is dimension of output AWGN
        snr, signal to noise ratio, in dB"""
    generatedNoise = np.random.standard_normal(dim)
    if identicalChannelPower:
        powerOfSignal = util.avgPower(signal)
    else:
        powerOfSignal = util.avgPower(signal, axis=-1)[:,None]
    generatedNoise *= np.sqrt(powerOfSignal) * (util.db2mag(-snr))
    return generatedNoise



def equivalentDelay(ir):
    """Defined according to the paper with theoretical bounds for
        FxLMS for single channel"""
    eqDelay = np.zeros((ir.shape[0], ir.shape[1]))
    
    for i in range(ir.shape[0]):
        for j in range(ir.shape[1]):
            thisIr = ir[i,j,:]
            eqDelay[i,j] = np.sum(np.arange(thisIr.shape[-1]) * 
                                (thisIr**2)) / np.sum(thisIr**2)
    return eqDelay
    
    