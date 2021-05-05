import numpy as np
import ancsim.utilities as util


def calcBlockSizes(numSamples, idx, bufferSize, chunkSize):
    leftInBuffer = chunkSize + bufferSize - idx
    sampleCounter = 0
    blockSizes = []
    while sampleCounter < numSamples:
        bLen = np.min((numSamples - sampleCounter, leftInBuffer))
        blockSizes.append(bLen)
        sampleCounter += bLen
        leftInBuffer -= bLen
        if leftInBuffer == 0:
            leftInBuffer = chunkSize
    return blockSizes


def findFirstIndexForBlock(earliestStartIndex, indexToEndAt, blockSize):
    """If processing in fixed size blocks, this function will give the index
        to start at if the processing should end at a specific index.
        Useful for preparation processing, where the exact startpoint isn't important
        but it is important to end at the correct place. """
    numSamples = indexToEndAt - earliestStartIndex
    numBlocks = numSamples // blockSize
    indexToStartAt = indexToEndAt - blockSize*numBlocks
    return indexToStartAt

def blockProcessUntilIndex(earliestStartIndex, indexToEndAt, blockSize):
    """Use as 
        for startIdx, endIdx in blockProcessUntilIndex(earliestStart, indexToEnd, blockSize):
            process(signal[...,startIdx:endIdx])
    
        If processing in fixed size blocks, this function will give the index
        to process for, if the processing should end at a specific index.
        Useful for preparation processing, where the exact startpoint isn't important
        but it is important to end at the correct place. """
    numSamples = indexToEndAt - earliestStartIndex
    numBlocks = numSamples // blockSize
    indexToStartAt = indexToEndAt - blockSize*numBlocks

    for i in range(numBlocks):
        yield indexToStartAt+i*blockSize, indexToStartAt+(i+1)*blockSize

def getWhiteNoiseAtSNR(rng, signal, dim, snr, identicalChannelPower=False):
    """generates Additive White Gaussian Noise
    signal to calculate signal level. time dimension is last dimension
    dim is dimension of output AWGN
    snr, signal to noise ratio, in dB"""
    generatedNoise = rng.standard_normal(dim)
    if identicalChannelPower:
        powerOfSignal = util.avgPower(signal)
    else:
        powerOfSignal = util.avgPower(signal, axis=-1)[:, None]
    generatedNoise *= np.sqrt(powerOfSignal) * (util.db2mag(-snr))
    return generatedNoise


def equivalentDelay(ir):
    """Defined according to the paper with theoretical bounds for
    FxLMS for single channel"""
    eqDelay = np.zeros((ir.shape[0], ir.shape[1]))

    for i in range(ir.shape[0]):
        for j in range(ir.shape[1]):
            thisIr = ir[i, j, :]
            eqDelay[i, j] = np.sum(
                np.arange(thisIr.shape[-1]) * (thisIr ** 2)
            ) / np.sum(thisIr ** 2)
    return eqDelay
