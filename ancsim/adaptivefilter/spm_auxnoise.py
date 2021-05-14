import numpy as np
import scipy.signal.windows as win
from abc import abstractmethod
import itertools as it

import ancsim.adaptivefilter.mpc as mpc
from ancsim.signal.filterclasses import (
    FilterSum,
    FilterMD_IntBuffer,
    SinglePoleLowPass,
    FilterIndividualInputs,
    FilterSum_Freqdomain,
    FilterMD_Freqdomain,
    MovingAverage,
)
import ancsim.signal.sources as src
from ancsim.adaptivefilter.util import getWhiteNoiseAtSNR
from ancsim.adaptivefilter.conventional.lms import (
    NLMS,
    NLMS_FREQ,
    FastBlockNLMS,
    BlockNLMS,
)
import ancsim.signal.freqdomainfiltering as fdf
import ancsim.adaptivefilter.diagnostics as dia
import ancsim.utilities as util



class FastBlockFxLMSEriksson(mpc.TDANCProcessor):
    def __init__(self, config, arrays, blockSize, controlFiltLen, mu, beta, muSec, auxNoisePower):
        super().__init__(config, arrays, blockSize, controlFiltLen, controlFiltLen)
        self.mu = mu
        self.beta = beta
        self.name = "Fast Block OSPM Eriksson"
        self.muSec = muSec

        self.createNewBuffer("v", self.numSpeaker)
        self.createNewBuffer("f", self.numError)
        self.createNewBuffer("xf", (self.numRef, self.numSpeaker, self.numError))

        self.secPathFilt = FilterMD_Freqdomain(dataDims=self.numRef, filtDim=(self.numSpeaker, self.numError), irLen=self.filtLen)
        # self.secPathFiltExtraAux = FilterSum_Freqdomain(ir=np.copy(np.concatenate((
        #     self.secPathFilt.ir, np.zeros((*self.secPathFilt.ir.shape[:2], blockSize-self.secPathFilt.ir.shape[2]))),axis=-1)))

        self.auxNoiseSource = src.WhiteNoiseSource(numChannels=self.numSpeaker, power=auxNoisePower)

        self.secPathNLMS = FastBlockNLMS(self.filtLen, self.numSpeaker, self.numError, self.muSec, self.beta)

        true_secpath = np.pad(arrays.paths["speaker"]["error"], ((0,0),(0,0),(self.blockSize,0)), constant_values=0)
        true_secpath = fdf.fftWithTranspose(true_secpath, n=2*self.filtLen)
        self.diag.addNewDiagnostic("secpath_estimate", dia.StateNMSE(self.sim_info, "secPathFilt.tf", true_secpath, 128))

        #self.secPathFilt.tf[...] = np.zeros_like(self.secPathFilt.tf)
        
        self.metadata["mu"] = self.mu
        self.metadata["beta"] = self.beta
        self.metadata["muSec"] = muSec
        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power

    def genSpeakerSignals(self, numSamples):
        super().genSpeakerSignals(numSamples)
        v = self.auxNoiseSource.getSamples(numSamples)
        self.sig["v"][:, self.idx - numSamples : self.idx] = v
        self.sig["speaker"][:, self.idx : self.idx + numSamples] += v

    # def forwardPassImplement(self, numSamples):
    #     super().forwardPassImplement(numSamples)
    #     v = self.auxNoiseSource.getSamples(numSamples)
    #     self.buffers["v"][:, self.idx : self.idx + numSamples] = v
    #     self.y[:, self.idx : self.idx + numSamples] += v
        

    def updateFilter(self):
        self.updateSPM()
        startIdx = self.updateIdx
        endIdx = self.updateIdx + self.updateBlockSize

        self.sig["xf"][
            :, :, :, startIdx:endIdx
        ] = np.moveaxis(self.secPathFilt.process(self.sig["ref"][:, startIdx:endIdx]), 2, 0)

        tdGrad = fdf.correlateSumTT(
            self.sig["f"][None, None, :, startIdx:endIdx],
                self.sig["xf"][..., endIdx - (2 * self.filtLen) : endIdx])

        norm = 1 / (
            np.sum(self.sig["xf"][..., startIdx:endIdx] ** 2)
            + self.beta
        )

        self.controlFilter.ir -= self.mu * norm * tdGrad


    def updateSPM(self):
        startIdx = self.updateIdx
        endIdx = self.updateIdx + self.updateBlockSize

        vf = self.secPathNLMS.process(
            self.sig["v"][:, startIdx:endIdx]
        )

        self.sig["f"][:, startIdx:endIdx] = self.sig["error"][:, startIdx:endIdx] - vf
        
        self.secPathNLMS.update(self.sig["v"][:, endIdx-(2*self.filtLen):endIdx], 
                                self.sig["f"][:,endIdx-self.filtLen:endIdx])
 
        self.secPathFilt.tf[...] = np.moveaxis(self.secPathNLMS.filt.tf, 1,2)


# class FastBlockFxLMSSPMEriksson(FastBlockAuxNoise):
#     """Frequency domain equivalent algorithm to FxLMSSPMEriksson"""
#     def __init__(self, config, mu, beta, speakerRIR, blockSize, muSec, auxNoisePower):
#         super().__init__(config, mu, beta, speakerRIR, blockSize, muSec, auxNoisePower)
#         self.name = "Fast Block SPM Eriksson"

#         self.secPathNLMS = FastBlockNLMS(self.blockSize, self.numSpeaker, self.numError, self.muSec,self.beta)

#     def updateSPM(self):
#         vf = self.secPathNLMS.process(
#             self.buffers["v"][:, self.updateIdx : self.idx]
#         )

#         self.buffers["f"][:, self.updateIdx : self.idx] = (
#             self.e[:, self.updateIdx : self.idx] - vf
#         )

#         self.secPathNLMS.update(self.buffers["v"][:, self.idx-(2*self.blockSize):self.idx], 
#                                 self.buffers["f"][:,self.idx-self.blockSize:self.idx])
 
#         self.secPathEstimate.tf[...] = self.secPathNLMS.filt.tf
#         self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.tf)
#         self.diag.saveBlockData("secpath_select_freq", self.idx, self.secPathEstimate.tf)
#         self.diag.saveBlockData("secpath_phase_select_freq", self.idx, self.secPathEstimate.tf)
#         self.diag.saveBlockData("secpath_phase_weighted_select_freq", self.idx, self.secPathEstimate.tf)
#         self.diag.saveBufferData("recorded_ir", self.idx, np.real(fdf.ifftWithTranspose(self.secPathNLMS.filt.tf)[...,:self.blockSize]))

