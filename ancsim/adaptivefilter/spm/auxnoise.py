import numpy as np
import scipy.signal.windows as win
from abc import abstractmethod
import itertools as it

from ancsim.adaptivefilter.base import AdaptiveFilterFF
from ancsim.adaptivefilter.mpc import FastBlockFxLMS
from ancsim.signal.filterclasses import (
    FilterSum,
    FilterMD_IntBuffer,
    SinglePoleLowPass,
    FilterIndividualInputs,
    FilterSum_Freqdomain,
    FilterMD_Freqdomain,
    MovingAverage,
)
from ancsim.signal.sources import WhiteNoiseSource, GoldSequenceSource
from ancsim.adaptivefilter.util import getWhiteNoiseAtSNR
from ancsim.signal.adaptivefilter import (
    NLMS,
    NLMS_FREQ,
    FastBlockNLMS,
    BlockNLMS,
)
import ancsim.signal.freqdomainfiltering as fdf
import ancsim.diagnostics.core as diacore
import ancsim.utilities as util



class FastBlockAuxNoise(FastBlockFxLMS):
    def __init__(self, config, mu, beta, speakerRIR, blockSize, muSec, auxNoisePower):
        super().__init__(config, mu, beta, speakerRIR, blockSize)
        self.name = "Fast Block Online SPM Auxnoise Baseclass"
        self.muSec = muSec

        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + self.sim_info.sim_buffer))
        self.buffers["f"] = np.zeros((self.numError, self.simChunkSize + self.sim_info.sim_buffer))

        self.secPathFiltExtraAux = FilterSum_Freqdomain(ir=np.copy(np.concatenate((
            self.secPathFilt.ir, np.zeros((*self.secPathFilt.ir.shape[:2], blockSize-self.secPathFilt.ir.shape[2]))),axis=-1)))
        self.secPathFiltExtraSpeak = FilterSum_Freqdomain(ir=np.copy(np.concatenate((
            self.secPathFilt.ir, np.zeros((*self.secPathFilt.ir.shape[:2], blockSize-self.secPathFilt.ir.shape[2]))),axis=-1)))

        self.auxNoiseSource = WhiteNoiseSource(
            power=auxNoisePower, num_channels=self.numSpeaker
        )


        self.secPathTrueMD = FilterMD_Freqdomain(tf=np.copy(self.secPathEstimate.tf), dataDims=self.numRef)
        # self.diag.add_diagnostic(
        #     "filtered_reference_est",
        #     diacore.SignalEstimateNMSEBlock(
        #         self.sim_info.tot_samples,
        #         self.sim_info.sim_buffer,
        #         self.simChunkSize,
        #         smoothingLen=self.outputSmoothing,
        #         plotFrequency=self.plotFrequency,
        #     )
        # )

        self.diag.add_diagnostic(
            "secpath",
            diacore.ConstantEstimateNMSE(
                self.secPathEstimate.tf,
                self.sim_info.tot_samples,
                self.sim_info.sim_buffer,
                self.simChunkSize,
                plotFrequency=self.plotFrequency,
            ),
        )

        self.diag.add_diagnostic(
            "secpath_select_freq",
            diacore.ConstantEstimateNMSESelectedFrequencies(
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

        self.diag.add_diagnostic(
            "secpath_phase_select_freq",
            diacore.ConstantEstimatePhaseDifferenceSelectedFrequencies(
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

        self.diag.add_diagnostic(
            "secpath_phase_weighted_select_freq",
            diacore.ConstantEstimateWeightedPhaseDifference(
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

        self.diag.add_diagnostic(
            "recorded_ir",
            diacore.RecordIR(
                np.real(fdf.ifftWithTranspose(self.secPathEstimate.tf)[...,:self.blockSize]),
                self.sim_info.tot_samples, 
                self.sim_info.sim_buffer, 
                self.simChunkSize,
                (2,2),
                plotFrequency=self.plotFrequency,
            ),
        )

        self.secPathEstimate.tf[...] = np.zeros_like(self.secPathEstimate.tf)

        self.diag.add_diagnostic(
            "power_auxnoise",
            diacore.RecordScalar(self.sim_info.tot_samples, self.sim_info.sim_buffer, self.simChunkSize,plotFrequency=self.plotFrequency)
        )
        # self.diag.add_diagnostic(
        #     "power_auxnoise_filtered",
        #     diacore.RecordScalar(self.sim_info.tot_samples, self.sim_info.sim_buffer, self.simChunkSize,plotFrequency=self.plotFrequency)
        # )
        self.diag.add_diagnostic(
            "power_loudspeaker",
            diacore.RecordScalar(self.sim_info.tot_samples, self.sim_info.sim_buffer, self.simChunkSize,plotFrequency=self.plotFrequency)
        )
        # self.diag.add_diagnostic(
        #     "power_loudspeaker_filtered",
        #     diacore.RecordScalar(self.sim_info.tot_samples, self.sim_info.sim_buffer, self.simChunkSize,plotFrequency=self.plotFrequency)
        # )
        self.diag.add_diagnostic(
            "power_error",
            diacore.RecordScalar(self.sim_info.tot_samples, self.sim_info.sim_buffer, self.simChunkSize,plotFrequency=self.plotFrequency)
        )
        

        self.metadata["muSec"] = muSec
        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power

    def forwardPassImplement(self, numSamples):
        super().forwardPassImplement(numSamples)
        self.diag.saveBlockData("power_loudspeaker", self.idx, np.mean(self.y[:,self.idx:self.idx+numSamples]**2))
        #self.diag.saveBlockData("power_loudspeaker_filtered", self.idx, 
        #        np.mean(self.secPathFiltExtraSpeak.process(
        #            self.y[:,self.idx:self.idx+numSamples])**2))

        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] += v

        self.diag.saveBlockData("power_auxnoise", self.idx, np.mean(v**2))
        #self.diag.saveBlockData("power_auxnoise_filtered", self.idx, np.mean(self.secPathFiltExtraAux.process(v)**2))
        

    def updateFilter(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False
        self.diag.saveBlockData("power_error", self.idx-self.blockSize, np.mean(self.e[:,self.idx-self.blockSize:self.idx]**2))

        self.updateSPM()

        self.buffers["xf"][
            :, :, :, self.updateIdx : self.idx
        ] = self.secPathEstimate.process(self.x[:, self.updateIdx : self.idx])

        tdGrad = fdf.correlateSumTT(
            self.buffers["f"][None, None, :, self.updateIdx : self.idx],
            np.transpose(
                self.buffers["xf"][..., self.idx - (2 * self.blockSize) : self.idx],
                (1, 2, 0, 3),
            ),
        )

        grad = fdf.fftWithTranspose(
            np.concatenate((tdGrad, np.zeros_like(tdGrad)), axis=-1)
        )
        norm = 1 / (
            np.sum(self.buffers["xf"][..., self.updateIdx : self.idx] ** 2)
            + self.beta
        )

        trueXf = self.secPathTrueMD.process(self.x[:,self.updateIdx:self.idx])
        # self.diag.saveBlockData("filtered_reference_est", self.idx, 
        #         self.buffers["xf"][...,self.updateIdx:self.idx],
        #         trueXf)

        self.controlFilt.tf -= self.mu * norm * grad
        self.updateIdx += self.blockSize
        self.updated = True


    @abstractmethod
    def updateSPM(self):
        pass




class FastBlockFxLMSSPMTuningless(FastBlockAuxNoise):
    """As we know the akf of the auxiliary noise, we should be able to get the
    wiener solution without inverting the estimated akf matrix, which is usually the problem"""

    def __init__(self, config, mu, beta, speakerRIR, blockSize, invForgetFactor, auxNoisePower):
        super().__init__(config, mu, beta, speakerRIR, blockSize, invForgetFactor, auxNoisePower)
        self.name = "Fast Block SPM Tuningless"

        self.secPathEstimateSPM = FilterSum_Freqdomain(
            numIn=self.numSpeaker, numOut=self.numError, irLen=blockSize
        )
        self.crossCorr = MovingAverage(
            1 - invForgetFactor,
            (2 * blockSize, self.numError, self.numSpeaker),
            dtype=np.complex128,
        )

        self.metadata["crossCorrForgetFactor"] = self.crossCorr.forgetFactor

    def updateSPM(self):
        vf = self.secPathEstimateSPM.process(
           self.buffers["v"][:, self.updateIdx : self.idx]
        )
        self.buffers["f"][:, self.updateIdx : self.idx] = (
           self.e[:, self.updateIdx : self.idx] - vf
        )

        crossCorr = fdf.correlateEuclidianTT(
            self.e[:, self.updateIdx : self.idx],
            self.buffers["v"][:, self.idx - 2 * self.blockSize : self.idx],
        )
        crossCorr = fdf.fftWithTranspose(
            np.concatenate((crossCorr, np.zeros_like(crossCorr)), axis=-1)
        )

        self.crossCorr.update(crossCorr / self.blockSize)
        self.secPathEstimateSPM.tf = self.crossCorr.state / self.auxNoiseSource.power
        self.secPathEstimate.tf[...] = self.secPathEstimateSPM.tf

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_phase_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_phase_weighted_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBufferData("recorded_ir", self.idx, np.real(fdf.ifftWithTranspose(self.secPathEstimateSPM.tf)[...,:self.blockSize]))


class FastBlockFxLMSSPMEriksson(FastBlockAuxNoise):
    """Frequency domain equivalent algorithm to FxLMSSPMEriksson"""

    def __init__(self, config, mu, beta, speakerRIR, blockSize, muSec, auxNoisePower):
        super().__init__(config, mu, beta, speakerRIR, blockSize, muSec, auxNoisePower)
        self.name = "Fast Block SPM Eriksson"

        self.secPathNLMS = FastBlockNLMS(self.blockSize, self.numSpeaker, self.numError, self.muSec,self.beta)

    def updateSPM(self):
        vf = self.secPathNLMS.process(
            self.buffers["v"][:, self.updateIdx : self.idx]
        )

        self.buffers["f"][:, self.updateIdx : self.idx] = (
            self.e[:, self.updateIdx : self.idx] - vf
        )

        self.secPathNLMS.update(self.buffers["v"][:, self.idx-(2*self.blockSize):self.idx], 
                                self.buffers["f"][:,self.idx-self.blockSize:self.idx])
 
        self.secPathEstimate.tf[...] = self.secPathNLMS.filt.tf
        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_phase_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_phase_weighted_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBufferData("recorded_ir", self.idx, np.real(fdf.ifftWithTranspose(self.secPathNLMS.filt.tf)[...,:self.blockSize]))


class FastBlockFxLMSSPMYuxue(FastBlockAuxNoise):
    """Implementation of proposed method from:
    Online secondary path modeling method
    with auxiliary noise power scheduling
    strategy for multi-channel adaptive
    active noise control system"""

    def __init__(
        self, config, mu, beta, speakerFilters, blockSize, minSecPathStepSize, auxNoisePower
    ):
        super().__init__(config, mu, beta, speakerFilters, blockSize, minSecPathStepSize, auxNoisePower)
        self.name = "Fast Block FxLMS SPM Yuxue"
        self.muSec = minSecPathStepSize
        self.auxNoisePower = auxNoisePower
        #self.spmLen = secPathEstimateLen

        # self.secPathEstimateSPM = FilterSum_Freqdomain(
        #     dataDims=self.numRef,
        #     irLen=secPathEstimateLen,
        #     irDims=(self.numSpeaker, self.numError),
        # )
        self.secPathEstimateSPM = FastBlockNLMS(
            numIn=self.numSpeaker,
            numOut=self.numError,
            blockSize=self.blockSize,
            stepSize=self.muSec
        )

        self.buffers["vEst"] = np.zeros(
            (self.numSpeaker, self.numError, self.simChunkSize + self.sim_info.sim_buffer)
        )

        self.buffers["eps"] = np.zeros(
            (self.numSpeaker, self.numError, self.simChunkSize + self.sim_info.sim_buffer)
        )

        self.Pf = MovingAverage(0.9, (self.numError,))
        self.Peps = MovingAverage(0.9, (self.numSpeaker, self.numError))
        self.Pv = MovingAverage(0.9, (self.numSpeaker,))

        self.diag.add_diagnostic(
            "mu_spm",
            diacore.RecordVectorSummary(
                self.sim_info.tot_samples,
                self.sim_info.sim_buffer,
                self.simChunkSize,
                plotFrequency=self.plotFrequency,
            ),
        )
        self.diag.add_diagnostic(
            "variable_auxnoise_power",
            diacore.RecordVector(
                self.numSpeaker, self.sim_info.tot_samples, self.sim_info.sim_buffer, self.simChunkSize, plotFrequency=self.plotFrequency
            ),
        )
        
        self.metadata["forgetFactorPf"] = self.Pf.forgetFactor
        self.metadata["forgetFactorPeps"] = self.Peps.forgetFactor
        self.metadata["forgetFactorPv"] = self.Pv.forgetFactor

    def calcEpsilon(self):
        self.buffers["eps"][:, :, self.updateIdx : self.idx] = self.e[
            None, :, self.updateIdx : self.idx
        ]
        for i, j in it.product(range(self.numSpeaker), range(self.numSpeaker)):
            if j != i:
                self.buffers["eps"][
                    j, :, self.updateIdx : self.idx
                ] -= self.buffers["vEst"][i, :, self.updateIdx : self.idx]

    # vf = self.secPathNLMS.process(
    #         self.buffers["v"][:, self.updateIdx : self.idx]
    #     )

    #     self.buffers["f"][:, self.updateIdx : self.idx] = (
    #         self.e[:, self.updateIdx : self.idx] - vf
    #     )

    #     self.secPathNLMS.update(self.buffers["v"][:, self.idx-(2*self.blockSize):self.idx], 
    #                             self.buffers["f"][:,self.idx-self.blockSize:self.idx])
 
    #     self.secPathEstimate.tf[...] = self.secPathNLMS.filt.tf
    #     self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.tf)


    def updateSPM(self):
        self.buffers["vEst"][
            :, :, self.updateIdx : self.idx
        ] = self.secPathEstimateSPM.processWithoutSum(
            self.buffers["v"][:, self.updateIdx : self.idx]
        )

        self.buffers["f"][:, self.updateIdx : self.idx] = self.e[:, self.updateIdx : self.idx] - np.sum(
            self.buffers["vEst"][:, :, self.updateIdx : self.idx], axis=0
        )
        self.calcEpsilon()

        self.Pf.update(np.mean(self.buffers["f"][:, self.updateIdx : self.idx] ** 2, axis=-1))
        self.Pv.update(
            np.mean(self.buffers["v"][:, self.updateIdx : self.idx] ** 2, axis=-1)
        )
        self.Peps.update(
            np.mean(self.buffers["eps"][:, :, self.updateIdx : self.idx] ** 2, axis=-1)
        )

        auxNoisePowerFactor = (np.sum(self.Pf.state) / np.sum(self.Peps.state, axis=-1))**2
        self.auxNoiseSource.setPower(self.auxNoisePower * auxNoisePowerFactor)
        self.secPathEstimateSPM.mu = (
            self.muSec * self.Pv.state[None, None,:] / self.Peps.state.T[None,:, :]
        )

        self.secPathEstimateSPM.update(self.buffers["v"][:, self.idx-(2*self.blockSize) : self.idx], 
                                        self.buffers["f"][:, self.updateIdx : self.idx])
        self.secPathEstimate.tf[...] = self.secPathEstimateSPM.filt.tf[...]

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimateSPM.filt.tf)
        self.diag.saveBlockData("secpath_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_phase_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_phase_weighted_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBufferData("recorded_ir", self.idx, np.real(fdf.ifftWithTranspose(self.secPathEstimateSPM.filt.tf)[...,:self.blockSize]))
        self.diag.saveBlockData("mu_spm", self.idx, self.secPathEstimateSPM.mu.ravel())

        self.diag.saveBlockData("variable_auxnoise_power", self.idx, auxNoisePowerFactor)







class FxLMSSPMErikssonBlock(AdaptiveFilterFF):
    """Using block based normalization for both ANC and SPM
    Corresponds to FreqAuxNoiseFxLMS"""

    def __init__(
        self, config, mu, beta, speakerRIR, muSPM, secPathEstimateLen, auxNoisePower=3
    ):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "FxLMS SPM Eriksson Block"
        self.muSPM = muSPM
        self.spmLen = secPathEstimateLen

        self.auxNoiseSource = WhiteNoiseSource(
            power=auxNoisePower, num_channels=self.numSpeaker
        )
        # self.auxNoiseSource = GoldSequenceSource(11, power=10, num_channels=self.numSpeaker)
        self.secPathEstimateMD = FilterMD_IntBuffer(
            dataDims=self.numRef,
            irLen=secPathEstimateLen,
            irDims=(self.numSpeaker, self.numError),
        )

        self.secPathEstimate = BlockNLMS(
            irLen=secPathEstimateLen,
            numIn=self.numSpeaker,
            numOut=self.numError,
            stepSize=muSPM,
        )

        self.buffers["xf"] = np.zeros(
            (
                self.numRef,
                self.numSpeaker,
                self.numError,
                self.simChunkSize + s.sim_buffer,
            )
        )
        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        self.buffers["vEst"] = np.zeros(
            (self.numError, self.simChunkSize + s.sim_buffer)
        )

        self.diag.add_diagnostic(
            "secpath",
            diacore.ConstantEstimateNMSE(
                speakerRIR["error"], self.sim_info.tot_samples, plotFrequency=self.plotFrequency
            ),
        )

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["muSPM"] = muSPM
        self.metadata["secPathEstimateLen"] = secPathEstimateLen
        self.metadata["auxNoisePower"] = auxNoisePower

    def updateSPM(self):
        self.buffers["vEst"][
            :, self.updateIdx : self.idx
        ] = self.secPathEstimate.process(
            self.buffers["v"][:, self.updateIdx : self.idx]
        )

        f = (
            self.e[:, self.updateIdx : self.idx]
            - self.buffers["vEst"][:, self.updateIdx : self.idx]
        )
        self.secPathEstimate.update(self.buffers["v"][:, self.updateIdx : self.idx], f)

        self.secPathEstimateMD.ir[...] = self.secPathEstimate.filt.ir

        self.diag.saveBlockData("secpath", self.updateIdx, self.secPathEstimate.filt.ir)

    def updateFilter(self):
        self.updateSPM()
        self.buffers["xf"][:, :, :, self.updateIdx : self.idx] = np.transpose(
            self.secPathEstimateMD.process(self.x[:, self.updateIdx : self.idx]),
            (2, 0, 1, 3),
        )
        grad = np.zeros_like(self.controlFilt.ir)
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(
                self.buffers["xf"][:, :, :, n - self.filtLen + 1 : n + 1], axis=-1
            )
            grad += np.sum(
                Xf * self.e[None, None, :, n, None], axis=2
            )  # / (np.sum(Xf**2) + self.beta)

        norm = 1 / (
            np.sum(self.buffers["xf"][:, :, :, self.updateIdx : self.idx] ** 2)
            + self.beta
        )
        self.controlFilt.ir -= grad * self.mu * norm

        self.updateIdx = self.idx

    def forwardPassImplement(self, numSamples):
        y = self.controlFilt.process(self.x[:, self.idx : self.idx + numSamples])
        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] = v + y


class FastBlockFxLMSSPMEriksson_old(FastBlockFxLMS):
    """Frequency domain equivalent algorithm to FxLMSSPMEriksson"""

    def __init__(self, config, mu, muSPM, speakerRIR, blockSize, auxNoisePower):
        self.beta = 1e-3
        super().__init__(config, mu, self.beta, speakerRIR, blockSize)
        self.name = "SPM Aux Noise Freq Domain"
        self.muSPM = muSPM
        self.diag.add_diagnostic(
            "secpath",
            diacore.ConstantEstimateNMSE(
                self.secPathEstimate.tf,
                self.sim_info.tot_samples,
                plotFrequency=self.plotFrequency,
            ),
        )
        # self.auxNoiseSource = GoldSequenceSource(11, power=10, num_channels=self.numSpeaker)
        self.auxNoiseSource = WhiteNoiseSource(power=auxNoisePower, num_channels=self.numSpeaker)

        self.secPathEstimate = FilterSum_Freqdomain(
            numIn=self.numSpeaker, numOut=self.numError, irLen=blockSize
        )
        self.secPathEstimateMD = FilterMD_Freqdomain(
            dataDims=self.numRef, tf=self.secPathEstimate.tf
        )

        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        self.buffers["f"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["muSPM"] = muSPM

    def forwardPassImplement(self, numSamples):
        super().forwardPassImplement(numSamples)

        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] += v

    def updateSPM(self):
        vf = self.secPathEstimate.process(
            self.buffers["v"][:, self.updateIdx : self.idx]
        )

        self.buffers["f"][:, self.updateIdx : self.idx] = (
            self.e[:, self.updateIdx : self.idx] - vf
        )

        grad = fdf.correlateEuclidianTT(
            self.buffers["f"][:, self.updateIdx : self.idx],
            self.buffers["v"][:, self.idx - (2 * self.blockSize) : self.idx],
        )

        grad = fdf.fftWithTranspose(
            np.concatenate((grad, np.zeros_like(grad)), axis=-1)
        )
        norm = 1 / (np.sum(self.buffers["v"][:, self.updateIdx : self.idx] ** 2) + self.beta)

        self.secPathEstimate.tf += self.muSPM * norm * grad
        self.secPathEstimateMD.tf[...] = self.secPathEstimate.tf

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.tf)

    def updateFilter(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False
        self.updateSPM()

        self.buffers["xf"][
            :, :, :, self.updateIdx : self.idx
        ] = self.secPathEstimateMD.process(self.x[:, self.updateIdx : self.idx])

        tdGrad = fdf.correlateSumTT(
            self.buffers["f"][None, None, :, self.updateIdx : self.idx],
            np.transpose(
                self.buffers["xf"][:, :, :, self.idx - (2 * self.blockSize) : self.idx],
                (1, 2, 0, 3),
            ),
        )

        grad = fdf.fftWithTranspose(
            np.concatenate((tdGrad, np.zeros_like(tdGrad)), axis=-1)
        )
        norm = 1 / (
            np.sum(self.buffers["xf"][:, :, :, self.updateIdx : self.idx] ** 2)
            + self.beta
        )
        # powerPerFreq = np.sum(np.abs(xfFreq)**2,axis=(1,2,3))
        # norm = 1 / (np.mean(powerPerFreq) + self.beta)

        self.controlFilt.tf -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True


class FxLMSSPMTuningless(AdaptiveFilterFF):
    """As we know the akf of the auxiliary noise, we should be able to get the
    wiener solution without inverting the estimated akf matrix, which is usually the problem.
    This is similar to the 'a tuning-less approach in secondary path
    modelling in active noise control systems'"""

    def __init__(self, config, mu, invForgetFactor, speakerRIR, secPathEstimateLen):
        self.beta = 1e-3
        super().__init__(config, mu, self.beta, speakerRIR)
        self.name = "SPM Aux Noise Wiener"
        self.secPathEstLen = secPathEstimateLen
        self.auxNoisePower = 3
        self.auxNoiseSource = WhiteNoiseSource(
            power=self.auxNoisePower, num_channels=self.numSpeaker
        )

        self.secPathEstimate = FilterSum(
            irLen=secPathEstimateLen, numIn=self.numSpeaker, numOut=self.numError
        )
        self.secPathEstimateMD = FilterMD_IntBuffer(
            dataDims=self.numRef,
            irLen=secPathEstimateLen,
            irDims=(self.numSpeaker, self.numError),
        )
        # self.crossCorr = SinglePoleLowPass(0.997, (self.numSpeaker, self.numError, secPathEstimateLen))
        self.crossCorr = MovingAverage(
            1 - invForgetFactor, (self.numSpeaker, self.numError, secPathEstimateLen)
        )

        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        self.buffers["xf"] = np.zeros(
            (
                self.numRef,
                self.numSpeaker,
                self.numError,
                self.simChunkSize + s.sim_buffer,
            )
        )

        self.diag.add_diagnostic(
            "secpath",
            diacore.ConstantEstimateNMSE(
                self.secPathFilt.ir, self.sim_info.tot_samples, plotFrequency=self.plotFrequency
            ),
        )

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["crossCorrForgetFactor"] = self.crossCorr.forgetFactor
        self.metadata["secPathEstimateLen"] = secPathEstimateLen

    def forwardPassImplement(self, numSamples):
        super().forwardPassImplement(numSamples)
        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] += v

    def updateSPM(self):
        cc = np.zeros_like(self.secPathEstimate.ir)
        for i in range(self.updateIdx, self.idx):
            cc += self.e[None, :, i : i + 1] * np.flip(
                self.buffers["v"][:, None, i - self.secPathEstLen + 1 : i + 1], axis=-1
            )
        cc /= self.idx - self.updateIdx

        self.crossCorr.update(cc)
        self.secPathEstimate.ir = self.crossCorr.state / self.auxNoisePower
        self.secPathEstimateMD.ir[:, :, :] = self.secPathEstimate.ir[:, :, :]

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.ir)

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

        self.updateSPM()
        self.updateIdx = self.idx
        self.updated = True






class FastBlockFxLMSSPMTuningless_old(FastBlockFxLMS):
    """As we know the akf of the auxiliary noise, we should be able to get the
    wiener solution without inverting the estimated akf matrix, which is usually the problem"""

    def __init__(self, config, mu, invForgetFactor, speakerRIR, blockSize, auxNoisePower):
        self.beta = 1e-3
        super().__init__(config, mu, self.beta, speakerRIR, blockSize)
        self.name = "SPM Aux Noise Freq Domain Wiener"
        self.auxNoiseSource = WhiteNoiseSource(
            power=self.auxNoisePower, num_channels=self.numSpeaker
        )

        self.G = np.transpose(self.G, (0, 2, 1))
        self.secPathEstimate = FilterSum_Freqdomain(
            numIn=self.numSpeaker, numOut=self.numError, irLen=blockSize
        )
        self.secPathEstimateMD = FilterMD_Freqdomain(
            dataDims=self.numRef, tf=np.zeros_like(self.secPathEstimate.tf)
        )
        self.crossCorr = MovingAverage(
            1 - invForgetFactor,
            (2 * blockSize, self.numError, self.numSpeaker),
            dtype=np.complex128,
        )

        self.buffers["v"] = np.zeros((self.numSpeaker, self.sim_info.sim_buffer + self.simChunkSize))
        self.buffers["f"] = np.zeros((self.numError, self.sim_info.sim_buffer + self.simChunkSize))

        self.diag.add_diagnostic(
            "secpath",
            diacore.ConstantEstimateNMSE(
                self.G, self.sim_info.tot_samples, plotFrequency=self.plotFrequency
            ),
        )

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["crossCorrForgetFactor"] = self.crossCorr.forgetFactor

    def forwardPassImplement(self, numSamples):
        super().forwardPassImplement(numSamples)
        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] += v

    def updateSPM(self):
        crossCorr = fdf.correlateEuclidianTT(
            self.e[:, self.updateIdx : self.idx],
            self.buffers["v"][:, self.idx - 2 * self.blockSize : self.idx],
        )
        crossCorr = fdf.fftWithTranspose(
            np.concatenate((crossCorr, np.zeros_like(crossCorr)), axis=-1)
        )

        self.crossCorr.update(crossCorr / self.blockSize)
        self.secPathEstimate.tf = self.crossCorr.state / self.auxNoiseSource.power
        self.secPathEstimateMD.tf[...] = self.secPathEstimate.tf

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimateMD.tf)

    def updateFilter(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False
        self.updateSPM()

        self.buffers["xf"][
            :, :, :, self.updateIdx : self.idx
        ] = self.secPathEstimateMD.process(self.x[:, self.updateIdx : self.idx])

        vf = self.secPathEstimate.process(
            self.buffers["v"][:, self.updateIdx : self.idx]
        )
        self.buffers["f"][:, self.updateIdx : self.idx] = (
            self.e[:, self.updateIdx : self.idx] - vf
        )

        tdGrad = fdf.correlateSumTT(
            self.buffers["f"][None, None, :, self.updateIdx : self.idx],
            np.transpose(
                self.buffers["xf"][:, :, :, self.idx - (2 * self.blockSize) : self.idx],
                (1, 2, 0, 3),
            ),
        )

        grad = fdf.fftWithTranspose(
            np.concatenate((tdGrad, np.zeros_like(tdGrad)), axis=-1)
        )
        norm = 1 / (
            np.sum(self.buffers["xf"][:, :, :, self.updateIdx : self.idx] ** 2)
            + self.beta
        )
        self.controlFilt.tf -= self.mu * norm * grad
        self.updateIdx += self.blockSize
        self.updated = True



class FxLMSSPMEriksson(AdaptiveFilterFF):
    """Using fully sample by sample normalization
    Will then not correspond exactly to frequency domain implementations"""

    def __init__(
        self, config, mu, beta, speakerRIR, muSPM, secPathEstimateLen, auxNoisePower=3
    ):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "FxLMS SPM Eriksson modern"
        self.muSPM = muSPM
        self.spmLen = secPathEstimateLen

        self.auxNoiseSource = WhiteNoiseSource(
            power=auxNoisePower, num_channels=self.numSpeaker
        )
        # self.auxNoiseSource = GoldSequenceSource(11, power=10, num_channels=self.numSpeaker)
        self.secPathEstimateMD = FilterMD_IntBuffer(
            dataDims=self.numRef,
            irLen=secPathEstimateLen,
            irDims=(self.numSpeaker, self.numError),
        )

        self.secPathEstimate = NLMS(
            irLen=secPathEstimateLen,
            numIn=self.numSpeaker,
            numOut=self.numError,
            stepSize=muSPM,
            channelIndependentNorm=False,
        )

        self.buffers["xf"] = np.zeros(
            (
                self.numRef,
                self.numSpeaker,
                self.numError,
                self.simChunkSize + s.sim_buffer,
            )
        )
        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        self.buffers["vEst"] = np.zeros(
            (self.numError, self.simChunkSize + s.sim_buffer)
        )

        self.diag.add_diagnostic(
            "secpath",
            diacore.ConstantEstimateNMSE(
                speakerRIR["error"], self.sim_info.tot_samples, plotFrequency=self.plotFrequency
            ),
        )

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["muSPM"] = muSPM
        self.metadata["secPathEstimateLen"] = secPathEstimateLen
        self.metadata["auxNoisePower"] = auxNoisePower

    def updateSPM(self):
        self.buffers["vEst"][
            :, self.updateIdx : self.idx
        ] = self.secPathEstimate.process(
            self.buffers["v"][:, self.updateIdx : self.idx]
        )

        f = (
            self.e[:, self.updateIdx : self.idx]
            - self.buffers["vEst"][:, self.updateIdx : self.idx]
        )
        self.secPathEstimate.update(self.buffers["v"][:, self.updateIdx : self.idx], f)

        self.secPathEstimateMD.ir[:, :, :] = self.secPathEstimate.filt.ir[:, :, :]

        self.diag.saveBlockData("secpath", self.updateIdx, self.secPathEstimate.filt.ir)

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
        self.updateSPM()
        self.updateIdx = self.idx

    def forwardPassImplement(self, numSamples):
        y = self.controlFilt.process(self.x[:, self.idx : self.idx + numSamples])
        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] = v + y


class FxLMSSPMEriksson_Gold(FxLMSSPMEriksson):
    def __init__(
        self, config, mu, beta, speakerRIR, muSPM, secPathEstimateLen, auxNoisePower=3
    ):
        super().__init__(
            config,
            mu,
            beta,
            speakerRIR,
            muSPM,
            secPathEstimateLen,
            auxNoisePower=auxNoisePower,
        )
        self.name = "FxLMS SPM Eriksson Gold Code"
        self.auxNoiseSource = GoldSequenceSource(
            11, power=auxNoisePower, num_channels=self.numSpeaker
        )


class FxLMS_SPM_Yuxue(AdaptiveFilterFF):
    """Implementation of proposed method from:
    Online secondary path modeling method
    with auxiliary noise power scheduling
    strategy for multi-channel adaptive
    active noise control system"""

    def __init__(
        self, config, mu, beta, speakerFilters, minSecPathStepSize, secPathEstimateLen
    ):
        super().__init__(config, mu, beta, speakerFilters)
        self.name = "FxLMS SPM Yuxue"
        self.muSecMin = minSecPathStepSize
        self.spmLen = secPathEstimateLen

        self.auxNoiseSource = GoldSequenceSource(
            11, power=1, num_channels=self.numSpeaker
        )

        self.secPathEstimateMD = FilterMD_IntBuffer(
            dataDims=self.numRef,
            irLen=secPathEstimateLen,
            irDims=(self.numSpeaker, self.numError),
        )
        self.secPathEstimate = NLMS(
            numIn=self.numSpeaker,
            numOut=self.numError,
            irLen=secPathEstimateLen,
            stepSize=self.muSecMin,
            filterType=FilterIndividualInputs,
        )

        # initialSecPathEstimate = np.random.normal(scale=0.0001, size=self.secPathEstimate.filt.ir.shape)
        # commonLength = np.min((secPathEstimateLen,speakerFilters["error"].shape[-1]))
        # initialSecPathEstimate[:,:,0:commonLength] = 0.5 * speakerFilters["error"][:,:,0:commonLength]

        # self.secPathEstimateMD.setIR(initialSecPathEstimate)
        # self.secPathEstimate.setIR(initialSecPathEstimate)

        self.buffers["xf"] = np.zeros(
            (
                self.numRef,
                self.numSpeaker,
                self.numError,
                self.simChunkSize + s.sim_buffer,
            )
        )
        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        self.buffers["vEst"] = np.zeros(
            (self.numSpeaker, self.numError, self.simChunkSize + s.sim_buffer)
        )

        self.buffers["eps"] = np.zeros(
            (self.numSpeaker, self.numError, self.simChunkSize + s.sim_buffer)
        )

        self.Pf = SinglePoleLowPass(0.9, (self.numError,))
        self.Peps = SinglePoleLowPass(0.9, (self.numSpeaker, self.numError))
        self.Pv = SinglePoleLowPass(0.9, (self.numSpeaker,))

        self.diag.add_diagnostic(
            "secpath",
            diacore.ConstantEstimateNMSE(
                speakerFilters["error"],
                self.sim_info.tot_samples,
                plotFrequency=self.plotFrequency,
            ),
        )
        self.diag.add_diagnostic(
            "mu_spm",
            diacore.RecordVector(
                self.numSpeaker * self.numError,
                self.sim_info.tot_samples,
                plotFrequency=self.plotFrequency,
            ),
        )
        self.diag.add_diagnostic(
            "aux_noise_power",
            diacore.RecordVector(
                self.numSpeaker, self.sim_info.tot_samples, plotFrequency=self.plotFrequency
            ),
        )

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["minSecPathStepSize"] = minSecPathStepSize
        self.metadata["secPathEstimateLen"] = secPathEstimateLen
        self.metadata["forgetFactorPf"] = self.Pf.forgetFactor
        self.metadata["forgetFactorPeps"] = self.Peps.forgetFactor
        self.metadata["forgetFactorPv"] = self.Pv.forgetFactor

    def calcEpsilon(self):
        self.buffers["eps"][:, :, self.updateIdx : self.idx] = self.e[
            None, :, self.updateIdx : self.idx
        ]
        for j in range(self.buffers["eps"].shape[0]):
            for i in range(self.buffers["eps"].shape[0]):
                if j != i:
                    self.buffers["eps"][
                        j, :, self.updateIdx : self.idx
                    ] -= self.buffers["vEst"][i, :, self.updateIdx : self.idx]

    def updateSPM(self):
        self.buffers["vEst"][
            :, :, self.updateIdx : self.idx
        ] = self.secPathEstimate.process(
            self.buffers["v"][:, self.updateIdx : self.idx]
        )

        f = self.e[:, self.updateIdx : self.idx] - np.sum(
            self.buffers["vEst"][:, :, self.updateIdx : self.idx], axis=0
        )
        self.calcEpsilon()

        self.Pf.update(np.mean(f ** 2, axis=-1))
        self.Pv.update(
            np.mean(self.buffers["v"][:, self.updateIdx : self.idx] ** 2, axis=-1)
        )
        self.Peps.update(
            np.mean(self.buffers["eps"][:, :, self.updateIdx : self.idx] ** 2, axis=-1)
        )

        newAuxNoisePower = np.sum(self.Pf.state) / np.sum(self.Peps.state, axis=-1)
        self.auxNoiseSource.setPower(newAuxNoisePower)
        self.secPathEstimate.mu = (
            self.muSecMin * self.Pv.state[:, None, None] / self.Peps.state[:, :, None]
        )

        self.secPathEstimate.update(self.buffers["v"][:, self.updateIdx : self.idx], f)
        self.secPathEstimateMD.ir[:, :, :] = self.secPathEstimate.filt.ir[:, :, :]

        self.diag.saveBlockData("secpath", self.updateIdx, self.secPathEstimate.filt.ir)
        self.diag.saveBlockData(
            "mu_spm", self.updateIdx, self.secPathEstimate.mu.ravel()
        )
        self.diag.saveBlockData("aux_noise_power", self.updateIdx, newAuxNoisePower)

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
        self.updateSPM()
        self.updateIdx = self.idx

    def forwardPassImplement(self, numSamples):
        y = self.controlFilt.process(self.x[:, self.idx : self.idx + numSamples])
        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:, self.idx : self.idx + numSamples] = v

        self.y[:, self.idx : self.idx + numSamples] = v + y


class FxLMSSPMZhang(AdaptiveFilterFF):
    """IS NOT FULLY FUNCTIONAL"""

    def __init__(self, config, mu, beta, speakerFilters, muSPM, secPathEstimateLen):
        super().__init__(config, mu, beta, speakerFilters)
        self.name = "FxLMS SPM Zhang"
        self.muSPM = muSPM
        self.spmLen = secPathEstimateLen
        self.auxNoiseLevel = -5

        # initialSecPathEstimate = np.random.normal(scale=0.01, size=speakerFilters["error"].shape) + 0.8 * speakerFilters["error"]

        self.secPathEstimateMD = FilterMD_IntBuffer(
            dataDims=self.numRef,
            irLen=secPathEstimateLen,
            irDims=(self.numSpeaker, self.numError),
        )
        # self.secPathEstimateMD.ir = initialSecPathEstimate
        self.secPathEstimate = FilterSum(
            irLen=secPathEstimateLen, numIn=self.numSpeaker, numOut=self.numError
        )
        # self.secPathEstimate.ir = initialSecPathEstimate

        self.distFiltLen = secPathEstimateLen
        self.disturbanceFilter = FilterSum(
            irLen=self.distFiltLen, numIn=self.numRef, numOut=self.numError
        )

        self.buffers["xf"] = np.zeros(
            (
                self.numRef,
                self.numSpeaker,
                self.numError,
                self.simChunkSize + s.sim_buffer,
            )
        )
        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        self.buffers["u"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))
        self.buffers["vEst"] = np.zeros(
            (self.numError, self.simChunkSize + s.sim_buffer)
        )

    def prepare(self):
        pass

    def updateDisturbanceFilter(self, g):
        grad = np.zeros_like(self.disturbanceFilter.ir)
        numSamples = self.idx - self.updateIdx
        for i in range(numSamples):
            grad += g[None, :, i : i + 1] * np.flip(
                self.x[
                    :, None, self.updateIdx + i - self.distFiltLen : self.updateIdx + i
                ],
                axis=-1,
            )

        self.disturbanceFilter.ir += self.muSPM * grad * 0

    def updateSPM(self):
        self.buffers["vEst"][
            :, self.updateIdx : self.idx
        ] = self.secPathEstimate.process(
            self.buffers["v"][:, self.updateIdx : self.idx]
        )

        g = (
            self.e[:, self.updateIdx : self.idx]
            - self.buffers["vEst"][:, self.updateIdx : self.idx]
        )
        self.updateDisturbanceFilter(g)

        self.buffers["u"][
            :, self.updateIdx : self.idx
        ] = self.disturbanceFilter.process(self.x[:, self.updateIdx : self.idx])
        f = g - self.buffers["u"][:, self.updateIdx : self.idx]

        grad = np.zeros_like(self.secPathEstimate.ir)
        numSamples = self.idx - self.updateIdx
        for i in range(numSamples):
            grad += (
                np.flip(
                    self.buffers["v"][
                        :,
                        None,
                        self.updateIdx + i - self.spmLen + 1 : self.updateIdx + i + 1,
                    ],
                    axis=-1,
                )
                * f[None, :, i : i + 1]
            )

        self.secPathEstimateMD.ir += self.muSPM * grad
        self.secPathEstimate.ir[:, :, :] = self.secPathEstimateMD.ir[:, :, :]

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
        self.updateSPM()
        self.updateIdx = self.idx

    def forwardPassImplement(self, numSamples):
        y = self.controlFilt.process(self.x[:, self.idx : self.idx + numSamples])

        v = getWhiteNoiseAtSNR(y, y.shape, self.auxNoiseLevel)
        self.buffers["v"][:, self.idx : self.idx + numSamples] = v

        self.y[:, self.idx : self.idx + numSamples] = v + y
