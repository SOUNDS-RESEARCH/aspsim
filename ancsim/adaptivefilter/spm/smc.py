import numpy as np

import ancsim.signal.freqdomainfiltering as fdf
from ancsim.adaptivefilter.base import AdaptiveFilterFF
from ancsim.adaptivefilter.mpc import FastBlockFxLMS
from ancsim.adaptivefilter.conventional.lms import NLMS, FastBlockNLMS
from ancsim.adaptivefilter.conventional.rls import RLS
from ancsim.signal.filterclasses import FilterSum, FilterMD_IntBuffer, FilterMD_Freqdomain
from ancsim.adaptivefilter.util import getWhiteNoiseAtSNR
import ancsim.adaptivefilter.diagnostics as dia
from ancsim.utilities import measure


class FastBlockFxLMSSMC(FastBlockFxLMS):
    def __init__(self, config, mu, beta, speakerRIR, blockSize, muPrim, muSec):
        super().__init__(config, mu, beta, speakerRIR, blockSize)
        self.name = "Fast Block FxLMS SMC w/ NLMS"
        self.muPrim = muPrim
        self.muSec = muSec
        
        self.controlFilt.setIR (self.rng.standard_normal(size=(self.numRef, self.numSpeaker, blockSize)) * 1e-14)
        #ir = np.zeros((self.numRef, self.numSpeaker, blockSize))
        #ir[:,:,0] = 1e-8

        self.secPathNLMS = FastBlockNLMS(self.blockSize, self.numSpeaker, self.numError, self.muSec, self.beta)
        self.primPathNLMS = FastBlockNLMS(self.blockSize, self.numRef, self.numError, self.muPrim, self.beta)

        #self.secPathNLMS.filt.setIR(self.rng.standard_normal(size=(self.numSpeaker, self.numError, blockSize)) * 1e-14)
        #self.primPathNLMS.filt.setIR(self.rng.standard_normal(size=(self.numSpeaker, self.numError, blockSize)) * 1e-14)
        
        #self.secPathTrueMD = FilterMD_Freqdomain(tf=np.copy(self.secPathEstimate.tf),dataDims=self.numRef)
        # self.diag.addNewDiagnostic(
        #     "filtered_reference_est",
        #     dia.SignalEstimateNMSEBlock(
        #         self.sim_info.tot_samples,
        #         self.sim_info.sim_buffer,
        #         self.simChunkSize,
        #         smoothingLen=self.outputSmoothing,
        #         plotFrequency=self.plotFrequency,
        #     )
        # )

        self.diag.addNewDiagnostic(
            "secpath",
            dia.ConstantEstimateNMSE(
                self.secPathEstimate.tf,
                self.sim_info.tot_samples,
                self.sim_info.sim_buffer,
                self.simChunkSize,
                plotFrequency=self.plotFrequency,
            ),
        )

        self.diag.addNewDiagnostic(
            "secpath_select_freq",
            dia.ConstantEstimateNMSESelectedFrequencies(
                self.secPathEstimate.tf,
                100,
                500,
                self.samplerate,
                self.sim_info.tot_samples,
                self.sim_info.sim_buffer,
                self.simChunkSize,
                plotFrequency=self.plotFrequency,
            ),
        )

        self.diag.addNewDiagnostic(
            "secpath_phase_select_freq",
            dia.ConstantEstimatePhaseDifferenceSelectedFrequencies(
                self.secPathEstimate.tf,
                100,
                500,
                self.samplerate,
                self.sim_info.tot_samples,
                self.sim_info.sim_buffer,
                self.simChunkSize,
                plotFrequency=self.plotFrequency,
            ),
        )

        self.diag.addNewDiagnostic(
            "secpath_phase_weighted_select_freq",
            dia.ConstantEstimateWeightedPhaseDifference(
                self.secPathEstimate.tf,
                100,
                500,
                self.samplerate,
                self.sim_info.tot_samples,
                self.sim_info.sim_buffer,
                self.simChunkSize,
                plotFrequency=self.plotFrequency,
            ),
        )

        self.diag.addNewDiagnostic(
            "recorded_ir",
            dia.RecordIR(
                np.real(fdf.ifftWithTranspose(self.secPathEstimate.tf)[...,:self.blockSize]),
                self.sim_info.tot_samples, 
                self.sim_info.sim_buffer, 
                self.simChunkSize,
                (2,2),
                plotFrequency=self.plotFrequency,
            ),
        )

        # self.diag.addNewDiagnostic(
        #     "secpath_combinations",
        #     dia.ConstantEstimateNMSEAllCombinations(
        #         self.secPathEstimate.tf,
        #         self.sim_info.tot_samples,
        #         self.sim_info.sim_buffer,
        #         self.simChunkSize,
        #         plotFrequency=self.plotFrequency,
        #     ),
        # )

        self.secPathEstimate.tf[...] = np.zeros_like(self.secPathEstimate.tf)

        self.diag.addNewDiagnostic(
            "modelling_error",
            dia.SignalEstimateNMSEBlock(self.sim_info.tot_samples, self.sim_info.sim_buffer, self.simChunkSize, 
                                     plotFrequency=self.plotFrequency)
        )
        self.diag.addNewDiagnostic(
            "primary_error",
            dia.SignalEstimateNMSEBlock(self.sim_info.tot_samples, self.sim_info.sim_buffer, self.simChunkSize, 
                                    plotFrequency=self.plotFrequency)
        )
        self.diag.addNewDiagnostic(
            "secondary_error",
            dia.SignalEstimateNMSEBlock(self.sim_info.tot_samples, self.sim_info.sim_buffer, self.simChunkSize, 
                                    plotFrequency=self.plotFrequency)
        )

        self.metadata["muSec"] = self.muSec
        self.metadata["muPrim"] = self.muPrim

    def updateFilter(self):
        super().updateFilter()
        self.updateSPM()

# import matplotlib.pyplot as plt
# plt.plot(np.abs(np.fft.fft(trueXf[0,1,0,:],n=2048))/np.sum(np.abs(trueXf)**2))
# plt.plot(np.abs(np.fft.fft(self.buffers["xf"][0,1,0,self.idx-self.blockSize:self.idx],n=2048))/np.sum(np.abs(self.buffers["xf"][...,self.idx-self.blockSize:self.idx])**2))
# plt.show()

    def updateSPM(self):
        #trueXf = self.secPathTrueMD.process(self.x[:,self.idx-self.blockSize:self.idx])
        # self.diag.saveBlockData("filtered_reference_est", self.idx, 
        #         5*self.buffers["xf"][...,self.idx-self.blockSize:self.idx],
        #         trueXf)

        primNoiseEst = self.primPathNLMS.process(self.x[:,self.idx-self.blockSize:self.idx])
        secNoiseEst = self.secPathNLMS.process(self.y[:,self.idx-self.blockSize:self.idx])
        errorEst = primNoiseEst + secNoiseEst

        modellingError = self.e[:,self.idx-self.blockSize:self.idx] - errorEst

        self.primPathNLMS.update(self.x[:,self.idx-(2*self.blockSize):self.idx], modellingError)
        self.secPathNLMS.update(self.y[:,self.idx-(2*self.blockSize):self.idx], modellingError)

        self.secPathEstimate.tf[...] = self.secPathNLMS.filt.tf
        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_phase_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_phase_weighted_select_freq", self.idx, self.secPathEstimate.tf)
        #self.diag.saveBlockData("secpath_combinations", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("modelling_error", self.idx, errorEst, self.e[:,self.idx-self.blockSize:self.idx])
        self.diag.saveBlockData("primary_error", self.idx, primNoiseEst, self.buffers["primary"][:,self.idx-self.blockSize:self.idx])
        self.diag.saveBlockData("secondary_error", self.idx, secNoiseEst, self.buffers["yf"][:,self.idx-self.blockSize:self.idx])
        self.diag.saveBufferData("recorded_ir", self.idx, np.real(fdf.ifftWithTranspose(self.secPathNLMS.filt.tf)[...,:self.blockSize]))
        #self.spmIdx = self.idx

class FxLMSSMC(AdaptiveFilterFF):
    """Simultaneous Modeling and Control"""

    def __init__(
        self, config, mu, beta, speakerFilters, muPrim, muSec, secPathEstimateLen
    ):
        assert self.numRef == 1
        super().__init__(config, mu, beta, speakerFilters)
        self.name = "FxLMS SMC NLMS"
        self.sLen = secPathEstimateLen
        self.pLen = secPathEstimateLen
        self.muPrim = muPrim
        self.muSec = muSec
        self.spmIdx = self.updateIdx
        self.buffers["xf"] = np.zeros(
            (
                self.numRef,
                self.numSpeaker,
                self.numError,
                self.simChunkSize + s.sim_buffer,
            )
        )

        self.secPathEstimateMD = FilterMD_IntBuffer(
            dataDims=self.numRef,
            irLen=secPathEstimateLen,
            irDims=(self.numSpeaker, self.numError),
        )

        self.secPathEstimate = NLMS(
            irLen=secPathEstimateLen,
            numIn=self.numSpeaker,
            numOut=self.numError,
            stepSize=self.muSec,
            regularization=1e-4,
        )
        self.primPathEstimate = NLMS(
            irLen=secPathEstimateLen,
            numIn=self.numRef,
            numOut=self.numError,
            stepSize=self.muPrim,
            regularization=1e-3,
        )

        self.controlFilt.ir = np.random.normal(
            scale=0.00000001, size=self.controlFilt.ir.shape
        )

        self.diag.addNewDiagnostic(
            "secpath",
            dia.ConstantEstimateNMSE(
                speakerFilters["error"],
                self.sim_info.tot_samples,
                plotFrequency=self.plotFrequency,
            ),
        )
        self.diag.addNewDiagnostic(
            "modelling_error",
            dia.SignalEstimateNMSE(self.sim_info.tot_samples, plotFrequency=self.plotFrequency),
        )

    def resetBuffers(self):
        super().resetBuffers()
        self.spmIdx -= self.simChunkSize

    def updateFilter(self):
        self.buffers["xf"][:, :, :, self.updateIdx : self.idx] = np.transpose(
            self.secPathEstimateMD.process(self.x[:, self.updateIdx : self.idx]),
            (2, 0, 1, 3),
        )
        grad = np.zeros_like(self.controlFilt.ir)
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(
                self.buffers["xf"][:, :, :, n - self.filtLen + 1 : n + 1], axis=-1
            )
            grad += np.sum(Xf * self.e[None, None, :, n, None], axis=2) / (
                np.sum(Xf ** 2) + self.beta
            )

        self.controlFilt.ir -= grad * self.mu
        self.updateIdx = self.idx
        self.updateSPM()

    def updateSPM(self):
        dEstimate = self.primPathEstimate.process(self.x[:, self.spmIdx : self.idx])
        yfEstimate = self.secPathEstimate.process(self.y[:, self.spmIdx : self.idx])
        errorEstimate = dEstimate + yfEstimate
        modellingError = self.e[:, self.spmIdx : self.idx] - errorEstimate

        self.secPathEstimate.update(
            ref=self.y[:, self.spmIdx : self.idx], error=modellingError
        )
        self.primPathEstimate.update(
            ref=self.x[:, self.spmIdx : self.idx], error=modellingError
        )
        self.secPathEstimateMD.ir[:, :, :] = self.secPathEstimate.filt.ir[:, :, :]

        # print("Sec Path NMSE: ", np.sum((self.secPathFilt.ir - self.secPathEstimate.ir)**2) / np.sum(self.secPathFilt.ir**2))
        # print("Model    NMSE: ", np.sum(modelingError**2) / np.sum(self.e[:,self.spmIdx:self.idx]**2))

        self.diag.saveBlockData("secpath", self.spmIdx, self.secPathEstimate.filt.ir)
        self.diag.saveBlockData(
            "modelling_error",
            self.spmIdx,
            errorEstimate,
            self.e[:, self.spmIdx : self.idx],
        )

        self.spmIdx = self.idx


class FxLMSSMC_RLS(AdaptiveFilterFF):
    """Simultaneous Modeling and Control"""

    def __init__(
        self,
        config,
        mu,
        beta,
        speakerFilters,
        primForgetFactor,
        secForgetFactor,
        secPathEstimateLen,
    ):
        assert self.numRef == 1
        super().__init__(config, mu, beta, speakerFilters)
        self.name = "FxLMS SMC RLS"
        self.sLen = secPathEstimateLen
        self.pLen = secPathEstimateLen
        # self.muPrim = muPrim
        # self.muSec = muSec
        self.spmIdx = self.updateIdx
        self.buffers["xf"] = np.zeros(
            (
                self.numRef,
                self.numSpeaker,
                self.numError,
                self.simChunkSize + s.sim_buffer,
            )
        )

        self.secPathEstimateMD = FilterMD_IntBuffer(
            dataDims=self.numRef,
            irLen=secPathEstimateLen,
            irDims=(self.numSpeaker, self.numError),
        )

        self.secPathEstimate = RLS(
            irLen=secPathEstimateLen,
            numIn=self.numSpeaker,
            numOut=self.numError,
            forgettingFactor=primForgetFactor,
            signalPowerEst=0.01,
        )
        self.primPathEstimate = RLS(
            irLen=secPathEstimateLen,
            numIn=self.numRef,
            numOut=self.numError,
            forgettingFactor=secForgetFactor,
            signalPowerEst=0.01,
        )

        self.controlFilt.ir = np.random.normal(
            scale=1e-7, size=self.controlFilt.ir.shape
        )

        self.diag.addNewDiagnostic(
            "secpath",
            dia.ConstantEstimateNMSE(
                speakerFilters["error"],
                self.sim_info.tot_samples,
                plotFrequency=self.plotFrequency,
            ),
        )
        self.diag.addNewDiagnostic(
            "modelling_error",
            dia.SignalEstimateNMSE(self.sim_info.tot_samples, plotFrequency=self.plotFrequency),
        )

    def resetBuffers(self):
        super().resetBuffers()
        self.spmIdx -= self.simChunkSize

    def updateFilter(self):
        self.buffers["xf"][:, :, :, self.updateIdx : self.idx] = np.transpose(
            self.secPathEstimateMD.process(self.x[:, self.updateIdx : self.idx]),
            (2, 0, 1, 3),
        )
        grad = np.zeros_like(self.controlFilt.ir)
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(
                self.buffers["xf"][:, :, :, n - self.filtLen + 1 : n + 1], axis=-1
            )
            grad += np.sum(Xf * self.e[None, None, :, n, None], axis=2) / (
                np.sum(Xf ** 2) + self.beta
            )

        self.controlFilt.ir -= grad * self.mu
        self.updateIdx = self.idx
        self.updateSPM()
        # print("Est sum: ", np.sum(np.abs(self.secPathEstimate.ir)))
        # print("True sum", np.sum(np.abs(self.secPathFilt.ir)))

    def updateSPM(self):
        # dEst = self.primPathEstimate.process(self.x[:,self.spmIdx:self.idx])
        # yfEst = self.secPathEstimate.process(self.y[:,self.spmIdx:self.idx])
        # modelingError = self.e[:,self.spmIdx:self.idx] - dEst - yfEst

        dEstimate = self.primPathEstimate.process(self.x[:, self.spmIdx : self.idx])
        yfEstimate = self.secPathEstimate.process(self.y[:, self.spmIdx : self.idx])
        errorEstimate = dEstimate + yfEstimate

        self.secPathEstimate.update(
            ref=self.y[:, self.spmIdx : self.idx],
            desired=self.e[:, self.spmIdx : self.idx],
        )
        self.primPathEstimate.update(
            ref=self.x[:, self.spmIdx : self.idx],
            desired=self.e[:, self.spmIdx : self.idx],
        )
        self.secPathEstimateMD.ir[:, :, :] = self.secPathEstimate.filt.ir[:, :, :]

        # print("Sec Path NMSE: ", np.sum((self.secPathFilt.ir - self.secPathEstimate.ir)**2) / np.sum(self.secPathFilt.ir**2))
        # print("Model    NMSE: ", np.sum(modelingError**2) / np.sum(self.e[:,self.spmIdx:self.idx]**2))
        self.diag.saveBlockData("secpath", self.spmIdx, self.secPathEstimate.filt.ir)
        self.diag.saveBlockData(
            "modelling_error",
            self.spmIdx,
            errorEstimate,
            self.e[:, self.spmIdx : self.idx],
        )
        self.spmIdx = self.idx


class FxLMSSMC_old(AdaptiveFilterFF):
    """Simultaneous Modeling and Control"""

    def __init__(self, mu, beta, speakerFilters, muPrim, muSec, secPathEstimateLen):
        assert self.numRef == 1
        super().__init__(mu, beta, speakerFilters)
        self.name = "FxLMS SMC"
        self.sLen = secPathEstimateLen
        self.pLen = secPathEstimateLen
        self.muPrim = muPrim
        self.muSec = muSec
        self.spmIdx = self.updateIdx
        # self.controlFilt.ir[:,:,0] = 0.00001*np.ones((self.controlFilt.ir.shape[0], self.controlFilt.ir.shape[1]))
        self.buffers["xf"] = np.zeros(
            (
                self.numRef,
                self.numSpeaker,
                self.numError,
                self.simChunkSize + s.sim_buffer,
            )
        )

        self.secPathEstimateMD = FilterMD_IntBuffer(
            dataDims=self.numRef,
            irLen=secPathEstimateLen,
            irDims=(self.numSpeaker, self.numError),
        )
        initialSecPathEstimate = np.random.normal(
            scale=0.00001, size=self.secPathEstimateMD.ir.shape
        )
        # self.secPathEstimateMD.ir = initialSecPathEstimate
        self.secPathEstimate = FilterSum(
            irLen=secPathEstimateLen, numIn=self.numSpeaker, numOut=self.numError
        )
        # self.secPathEstimate.ir = initialSecPathEstimate

        self.primPathEstimate = FilterSum(
            irLen=secPathEstimateLen, numIn=self.numRef, numOut=self.numError
        )
        # self.primPathEstimate.ir = np.random.normal(scale=0.00001, size=self.primPathEstimate.ir.shape)
        self.controlFilt.ir = np.random.normal(
            scale=0.00000001, size=self.controlFilt.ir.shape
        )

    def prepare(self):
        pass

    def resetBuffers(self):
        super().resetBuffers()
        self.spmIdx -= self.simChunkSize

    def updateFilter(self):
        self.buffers["xf"][:, :, :, self.updateIdx : self.idx] = np.transpose(
            self.secPathEstimateMD.process(self.x[:, self.updateIdx : self.idx]),
            (2, 0, 1, 3),
        )
        grad = np.zeros_like(self.controlFilt.ir)
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(
                self.buffers["xf"][:, :, :, n - self.filtLen + 1 : n + 1], axis=-1
            )
            grad += np.sum(Xf * self.e[None, None, :, n, None], axis=2) / (
                np.sum(Xf ** 2) + self.beta
            )

        self.controlFilt.ir -= grad * self.mu
        self.updateIdx = self.idx
        self.updateSPM()
        # print("Est sum: ", np.sum(np.abs(self.secPathEstimate.ir)))
        # print("True sum", np.sum(np.abs(self.secPathFilt.ir)))

    def updateSPM(self):
        dEst = self.primPathEstimate.process(self.x[:, self.spmIdx : self.idx])
        yfEst = self.secPathEstimate.process(self.y[:, self.spmIdx : self.idx])
        modelingError = self.e[:, self.spmIdx : self.idx] - dEst - yfEst

        # yPower = np.mean(self.y[:, self.spmIdx-self.sLen:self.idx]**2)
        # xPower = np.mean(self.x[:, self.spmIdx-self.pLen:self.idx]**2)

        for i in range(self.idx - self.spmIdx):
            n = self.spmIdx + i
            X = np.flip(self.x[:, None, n - self.pLen : n], axis=-1)
            Y = np.flip(self.y[:, None, n - self.sLen : n], axis=-1)
            self.primPathEstimate.ir += (
                self.muPrim
                * modelingError[None, :, i : i + 1]
                * X
                / (np.sum(X ** 2) + 1e-3)
            )
            self.secPathEstimate.ir += (
                self.muSec
                * modelingError[None, :, i : i + 1]
                * Y
                / (np.sum(Y ** 2) + 1e-4)
            )
        self.secPathEstimateMD.ir[:, :, :] = self.secPathEstimate.ir[:, :, :]

        # print("Sec Path NMSE: ", np.sum((self.secPathFilt.ir - self.secPathEstimate.ir)**2) / np.sum(self.secPathFilt.ir**2))
        # print("Model    NMSE: ", np.sum(modelingError**2) / np.sum(self.e[:,self.spmIdx:self.idx]**2))
        self.spmIdx = self.idx


# import matplotlib.pyplot as plt
class FreqSMC(FastBlockFxLMS):
    def __init__(self, config, mu, beta, speakerRIR, muPrimary, muSecondary, blockSize):
        assert self.numRef == 1

        super().__init__(config, mu, beta, speakerRIR, blockSize)
        self.name = "Freq domain SMC"
        self.secUpdateIdx = self.updateIdx
        self.trueG = self.G
        self.G = np.zeros_like(
            self.G, dtype=np.complex128
        )  # + np.random.normal(scale=0.0000001, size=self.G.shape)
        self.H += np.random.normal(scale=0.00000001, size=self.H.shape)
        self.P = np.zeros(
            (2 * blockSize, self.numError, self.numRef), dtype=np.complex128
        )
        # self.P += np.random.normal(scale=0.0000001, size=self.P.shape)
        self.muPrim = muPrimary
        self.muSec = muSecondary

    def updateFilter(self):
        super().updateFilter()
        self.updateSecPath()

    def updateSecPath(self):
        Eest = (self.P + np.transpose(self.G, (0, 2, 1)) @ self.H) @ self.X
        v = self.E - Eest

        avgRefPow = np.mean(np.abs(self.X) ** 2, axis=0)
        # noPowFreqs = np.squeeze(np.abs(self.X / avgRefPow)**2 < 1e-3)

        pNorm = 1 / (1e-3 + avgRefPow)
        gNorm = 1 / (1e-3 + (np.sum(np.abs(self.Y) ** 2) / (2 * self.blockSize)))
        pGrad = self.muPrim * v @ self.X.conj() * pNorm
        # pGrad[noPowFreqs,:,:] = 0
        self.P += pGrad
        gGrad = self.muSec * self.Y.conj() @ np.transpose(v, (0, 2, 1)) * gNorm
        # gGrad[noPowFreqs,:,:] = 0
        self.G += gGrad
        # self.P[noPowFreqs,:,:] = 0
        # self.G[noPowFreqs,:,:] = 0

        print(
            "Sec Path NMSE: ",
            np.sum(np.abs(self.G - self.trueG) ** 2) / np.sum(np.abs(self.trueG) ** 2),
        )
        print(
            "Model    NMSE: ",
            np.sum(np.abs(self.E - Eest) ** 2) / np.sum(np.abs(self.E) ** 2),
        )
        # plt.plot(self.G[:,0,:])
        # plt.show()
