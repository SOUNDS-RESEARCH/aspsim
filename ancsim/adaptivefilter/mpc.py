import numpy as np

import ancsim.utilities as util
import ancsim.array as ar
from ancsim.signal.filterclasses import (
    Filter_IntBuffer,
    FilterSum_IntBuffer,
    FilterMD_IntBuffer,
    FilterSum_Freqdomain,
    FilterMD_Freqdomain
)
from ancsim.adaptivefilter.util import blockProcessUntilIndex, calcBlockSizes, getWhiteNoiseAtSNR
import ancsim.adaptivefilter.diagnostics as dia
import ancsim.signal.freqdomainfiltering as fdf
from ancsim.adaptivefilter.base import ActiveNoiseControlProcessor

class TDANCProcessor(ActiveNoiseControlProcessor):
    def __init__(self, config, arrays, blockSize, updateBlockSize, controlFiltLen):
        super().__init__(config, arrays, blockSize, updateBlockSize)
        self.filtLen = controlFiltLen
        self.controlFilter = FilterSum_IntBuffer(irLen=self.filtLen, numIn=self.numRef, numOut=self.numSpeaker)

        self.metadata["control filter length"] = controlFiltLen

    def genSpeakerSignals(self, numSamples):
        """"""
        self.sig["speaker"][:,self.idx:self.idx+numSamples] = \
            self.controlFilter.process(self.sig["ref"][:,self.idx-numSamples:self.idx])




class BlockFxLMS(TDANCProcessor):
    """Fully block based operation. Is equivalent to FastBlockFxLMS"""
    def __init__(self, config, arrays, blockSize, updateBlockSize, controlFiltLen, mu, beta, secPath):
        super().__init__(config, arrays, blockSize, updateBlockSize, controlFiltLen)
        self.name = "Block FxLMS"
        self.mu = mu
        self.beta = beta

        secPathIr = np.concatenate((np.zeros((secPath.shape[0], secPath.shape[1], self.blockSize)), secPath),axis=-1)
        self.secPathFilt = FilterMD_IntBuffer((self.numRef,), secPathIr)
        self.createNewBuffer("xf", (self.numRef, self.numSpeaker, self.numError))


        self.metadata["step size"] = self.mu
        self.metadata["beta"] = self.beta

    def prepare(self):
        self.sig["xf"][:, :, :, : self.idx] = np.transpose(
            self.secPathFilt.process(self.sig["ref"][:, :self.idx]), (2, 0, 1, 3)
        )

    def updateFilter(self):
        startIdx = self.updateIdx
        endIdx = self.updateIdx + self.updateBlockSize

        self.sig["xf"][:, :, :, startIdx : endIdx] = np.transpose(
            self.secPathFilt.process(self.sig["ref"][:, startIdx : endIdx]),
            (2, 0, 1, 3),
        )

        grad = np.zeros_like(self.controlFilter.ir)
        for n in range(startIdx, endIdx):
            Xf = np.flip(
                self.sig["xf"][:, :, :, n - self.filtLen + 1 : n + 1], axis=-1
            )
            grad += np.sum(Xf * self.sig["error"][None, None, :, n:n+1], axis=2)

        #print(np.sum(self.sig["xf"][:, :, :, self.updateIdx : self.idx] ** 2))
        norm = 1 / (
            np.sum(self.sig["xf"][:, :, :, startIdx : endIdx] ** 2)
            + self.beta
        )
        self.controlFilter.ir -= grad * self.mu * norm



class FastBlockFxLMS(TDANCProcessor):
    """Fully block based operation using overlap save and FFT. 
        updateBlockSize is equal to the control filter length"""
    def __init__(self, config, arrays, blockSize, controlFiltLen, mu, beta, secPath):
        super().__init__(config, arrays, blockSize, controlFiltLen, controlFiltLen)
        self.name = "Fast Block FxLMS"
        self.mu = mu
        self.beta = beta

        secPathIr = np.concatenate((np.zeros((secPath.shape[0], secPath.shape[1], self.blockSize)), 
                                    secPath,
                                    np.zeros((secPath.shape[0], secPath.shape[1], 2*self.filtLen-secPath.shape[-1]-self.blockSize))),
                                    axis=-1)
        secPathIr[...,self.filtLen:] = 0
        secPathTf = fdf.fftWithTranspose(secPathIr)

        self.secPathFilt = FilterMD_Freqdomain(dataDims=self.numRef, tf=secPathTf)
        self.createNewBuffer("xf", (self.numRef, self.numSpeaker, self.numError))

        self.metadata["step size"] = self.mu
        self.metadata["beta"] = self.beta

    def prepare(self):
        for startIdx, endIdx in blockProcessUntilIndex(0, self.idx, self.filtLen):
            self.sig["xf"][:, :, :, startIdx:endIdx] = np.transpose(
                self.secPathFilt.process(self.sig["ref"][:, startIdx:endIdx]), (2, 0, 1, 3)
            )

    def updateFilter(self):
        startIdx = self.updateIdx
        endIdx = self.updateIdx + self.updateBlockSize
        #assert self.idx - self.updateIdx == self.filtLen
        self.sig["xf"][:, :, :, startIdx : endIdx] = \
            np.moveaxis(self.secPathFilt.process(self.sig["ref"][:, startIdx : endIdx]), 2, 0)

        tdGrad = fdf.correlateSumTT(
            self.sig["error"][None, None, :, startIdx : endIdx],
            self.sig["xf"][:, :, :, endIdx - (2 * self.filtLen) : endIdx])

        norm = 1 / (
            np.sum(self.sig["xf"][:, :, :, startIdx : endIdx] ** 2)
            + self.beta
        )
        self.controlFilter.ir -= self.mu * norm * tdGrad
