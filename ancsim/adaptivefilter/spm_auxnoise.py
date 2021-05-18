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
import ancsim.soundfield.kernelinterpolation as ki


class FastBlockFxLMSAuxnoiseSPM(mpc.TDANCProcessor):
    def __init__(self, sim_info, arrays, blockSize, controlFiltLen, mu, beta, muSec, auxNoisePower):
        super().__init__(sim_info, arrays, blockSize, controlFiltLen, controlFiltLen)
        self.name = "Fast Block Auxnoise SPM"
        self.mu = mu
        self.beta = beta
        self.muSec = muSec

        self.createNewBuffer("v", self.numSpeaker)
        self.createNewBuffer("f", self.numError)
        self.createNewBuffer("xf", (self.numRef, self.numSpeaker, self.numError))

        self.auxNoiseSource = src.WhiteNoiseSource(numChannels=self.numSpeaker, power=auxNoisePower)

        self.secPathFilt = FilterMD_Freqdomain(dataDims=self.numRef, filtDim=(self.numSpeaker, self.numError), irLen=self.filtLen)

        true_secpath = np.pad(arrays.paths["speaker"]["error"], ((0,0),(0,0),(self.blockSize,0)), constant_values=0)
        true_secpath = fdf.fftWithTranspose(true_secpath, n=2*self.filtLen)
        self.diag.addNewDiagnostic("secpath_estimate", dia.StateNMSE(self.sim_info, "secPathFilt.tf", true_secpath, 128))

        self.metadata["mu"] = self.mu
        self.metadata["beta"] = self.beta
        self.metadata["muSec"] = muSec
        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power

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

    @abstractmethod
    def updateSPM(self):
        pass
        

class FastBlockFxLMSEriksson(FastBlockFxLMSAuxnoiseSPM):
    def __init__(self, sim_info, arrays, blockSize, controlFiltLen, mu, beta, muSec, auxNoisePower):
        super().__init__(sim_info, arrays, blockSize, controlFiltLen, mu, beta, muSec, auxNoisePower)
        self.name = "Fast Block FxLMS SPM Eriksson"

        self.secPathNLMS = FastBlockNLMS(self.filtLen, self.numSpeaker, self.numError, self.muSec, self.beta)

    def genSpeakerSignals(self, numSamples):
        super().genSpeakerSignals(numSamples)
        v = self.auxNoiseSource.getSamples(numSamples)
        self.sig["v"][:, self.idx - numSamples : self.idx] = v
        self.sig["speaker"][:, self.idx : self.idx + numSamples] += v

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


class FastBlockFxLMSKernelSPM(FastBlockFxLMSAuxnoiseSPM):
    """Uses reciprocal kernel interpolation to interpolated between loudspeaker placement
        To only need aux noise for a few speakers
        
        
        Kolla reshapes så att ordningnen på mics och speakers är rätt. Nu ger den kaos-resultat,
        men det skulle kunna vara bara att fel data interpoleras till fel kanal. """
    def __init__(self, sim_info, arrays, blockSize, controlFiltLen, mu, beta, muSec, auxNoisePower):
        super().__init__(sim_info, arrays, blockSize, controlFiltLen, mu, beta, muSec, auxNoisePower)
        self.name = "SPM Eriksson Reciprocal KI"
        self.errorPos = arrays["error"].pos
        self.speakerPos = arrays["speaker"].pos
        
        self.auxSpeakerIdx = np.array([0,2,4,6])#np.arange(self.numSpeaker)[0,2,4,6]
        self.kiSpeakerIdx = np.array([i for i in range(self.numSpeaker) if i not in self.auxSpeakerIdx])
        self.numKeep = len(self.auxSpeakerIdx)
        self.numKi = len(self.kiSpeakerIdx)
        
        self.atfkiFilt = ki.ATFKernelInterpolator(self.speakerPos[self.auxSpeakerIdx,:], 
                                                    self.speakerPos[self.kiSpeakerIdx,:],
                                self.errorPos, 1e-3, 155, controlFiltLen, 4096, 
                                self.sim_info.samplerate, self.sim_info.c, self.blockSize)
        
        trueSecPath = self.arrays.paths["speaker"]["error"]
        trueSecPath = np.pad(trueSecPath, 
                    ((0,0),(0,0),(self.blockSize, self.filtLen-trueSecPath.shape[-1]-self.blockSize)))
        testip = self.atfkiFilt.process(trueSecPath[self.auxSpeakerIdx,:,:])

        # kiTF = ki.getKRRParameters(ki.kernelReciprocal3d, 1e-3, 
        #                     (errorPos, speakerPos[self.kiSpeakerIdx,:]),
        #                     (errorPos, speakerPos[self.auxSpeakerIdx,:]),
        #                     4096,
        #                     self.samplerate, 
        #                     self.c)
        # self.kiFiltLen = 155
        # self.kiDly = self.kiFiltLen // 2
        # kiIR = np.transpose(fd.firFromFreqsWindow(kiTF, self.kiFiltLen), (1,0,2))
        # self.kiFilt = FilterSum_Freqdomain(ir=np.concatenate((kiIR, np.zeros(kiIR.shape[:-1]+(self.blockSize-kiIR.shape[-1],))),axis=-1))

        self.auxNoiseSource = src.WhiteNoiseSource(numChannels=self.numKeep, power=auxNoisePower)
        self.secPathNLMS = FastBlockNLMS(self.filtLen, self.numKeep, self.numError, self.muSec,self.beta)

    def genSpeakerSignals(self, numSamples):
        super().genSpeakerSignals(numSamples)
        v = self.auxNoiseSource.getSamples(numSamples)
        self.sig["v"][self.auxSpeakerIdx, self.idx - numSamples : self.idx] = v
        self.sig["speaker"][self.auxSpeakerIdx, self.idx : self.idx + numSamples] += v

    def updateSPM(self):
        startIdx = self.updateIdx
        endIdx = self.updateIdx + self.updateBlockSize

        vf = self.secPathNLMS.process(
            self.sig["v"][self.auxSpeakerIdx, startIdx:endIdx]
        )

        self.sig["f"][:, startIdx:endIdx] = self.sig["error"][:, startIdx:endIdx] - vf

        self.secPathNLMS.update(self.sig["v"][self.auxSpeakerIdx, endIdx-(2*self.filtLen):endIdx], 
                                self.sig["f"][:,endIdx-self.filtLen:endIdx])


        # ir = np.real(fdf.ifftWithTranspose(self.secPathNLMS.filt.tf)[...,:self.blockSize])
        # ir = np.transpose(ir, (1,0,2)).reshape(-1,self.blockSize)
        # self.kiFilt.buffer.fill(0)
        # ipIR = self.kiFilt.process(ir)
        # ipIR = ipIR.reshape((self.numSpeaker-self.numKeep, self.numError, self.blockSize))
        # ipIR = np.concatenate((ipIR[...,self.kiDly:], np.zeros(
        #             ipIR.shape[:-1] + (2*self.blockSize-ipIR.shape[-1]+self.kiDly,))),axis=-1)

        # ipTF = np.transpose(fdf.fftWithTranspose(ipIR), (0,2,1))
        currentIr = np.moveaxis(np.real(fdf.ifftWithTranspose(self.secPathNLMS.filt.tf)[...,:self.filtLen]), 0,1)
        kiIR = self.atfkiFilt.process(currentIr)
        kiTF = fdf.fftWithTranspose(kiIR, n=2*self.filtLen)
        self.secPathFilt.tf[:,self.auxSpeakerIdx,:] = np.moveaxis(self.secPathNLMS.filt.tf[...],1,2)
        self.secPathFilt.tf[:,self.kiSpeakerIdx,:] = kiTF #.reshape((2*self.filtLen,self.numKi,self.numError))

        
        # trueir = self.secPathFilt.ir[self.auxSpeakerIdx,...].reshape((-1,self.secPathFilt.ir.shape[-1]))
        # trueir = np.concatenate((trueir, np.zeros(
        #             trueir.shape[:-1] + (self.blockSize-trueir.shape[-1],))),axis=-1)
        # self.kiFilt.buffer.fill(0)
        # ipIR = self.kiFilt.process(ir)
        # ipIR = ipIR.reshape((self.numSpeaker-self.numKeep, self.numError, self.blockSize))
        # trueIpIR = self.secPathFilt.ir[self.kiSpeakerIdx,...]
        # import matplotlib.pyplot as plt


    # def forwardPassImplement(self, numSamples):
    #     super().forwardPassImplement(numSamples)

    #     v = self.auxNoiseSource.getSamples(numSamples)[self.auxSpeakerIdx,:]
    #     #for i in self.auxSpeakerIdx:
    #     self.buffers["v"][self.auxSpeakerIdx, self.idx : self.idx + numSamples] = v
    #     self.y[self.auxSpeakerIdx, self.idx : self.idx + numSamples] += v

    #     self.diag.saveBlockData("power_auxnoise", self.idx, np.mean(v**2))