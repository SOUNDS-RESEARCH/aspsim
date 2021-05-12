import numpy as np
import scipy.signal.windows as win

from ancsim.adaptivefilter.base import AdaptiveFilterFF
from ancsim.adaptivefilter.mpc import FastBlockFxLMS
from ancsim.signal.filterclasses import (
    FilterSum_IntBuffer,
    FilterMD_IntBuffer,
    SinglePoleLowPass,
    MovingAverage,
)
from ancsim.signal.sources import WhiteNoiseSource, GoldSequenceSource
from ancsim.adaptivefilter.conventional.lms import NLMS, NLMS_FREQ, FastBlockNLMS
from ancsim.soundfield.kernelinterpolation import soundfieldInterpolation
from ancsim.adaptivefilter.util import getWhiteNoiseAtSNR
import ancsim.adaptivefilter.diagnostics as dia
import ancsim.utilities as util

# import matplotlib.pyplot as plt


class ConstrainedFastBlockFxLMS:
    pass  # Any class inheriting from this needs to be re-examined to use FastBlockFxLMS


class WienerAuxNoiseFreqFxLMS(ConstrainedFastBlockFxLMS):
    """As we know the akf of the auxiliary noise, we should be able to get the
    wiener solution without inverting the estimated akf matrix, which is usually the problem"""

    def __init__(self, config, mu, speakerRIR, blockSize):
        self.beta = 1e-3
        super().__init__(config, mu, self.beta, speakerRIR, blockSize)
        self.name = "SPM Aux Noise Freq Domain Wiener"
        self.auxNoisePower = 3
        self.auxNoiseSource = WhiteNoiseSource(
            power=self.auxNoisePower, numChannels=self.numSpeaker
        )

        self.secPathEstimate = np.zeros(
            (2 * blockSize, self.numError, self.numSpeaker), dtype=np.complex128
        )
        # self.secPathEstimateFD = np.zeros((2*blockSize, self.numError, self.numSpeaker), dtype=np.complex128)
        # self.crossCorr = SinglePoleLowPass(0.997, (self.numError,self.numSpeaker, blockSize))
        # self.crossCorr = MovingAverage(0.997, (self.numError, self.numSpeaker, blockSize))
        self.crossCorr = MovingAverage(
            0.997, (2 * blockSize, self.numError, self.numSpeaker), dtype=np.complex128
        )
        self.G = np.transpose(self.G, (0, 2, 1))
        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        # self.buffers["f"] = np.zeros((self.numError, s.sim_buffer+self.simChunkSize))

        self.diag.addNewDiagnostic(
            "secpath",
            dia.ConstantEstimateNMSE(
                self.G, self.sim_info.tot_samples, plotFrequency=self.plotFrequency
            ),
        )
        self.window = win.hamming(2 * self.blockSize, sym=False)[None, :]

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["crossCorrForgetFactor"] = self.crossCorr.forgetFactor

    def forwardPassImplement(self, numSamples):
        super().forwardPassImplement(numSamples)
        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] += v

    def updateSPM(self):
        # cc = np.zeros((self.numError,self.numSpeaker, self.blockSize))
        # for i in range(self.updateIdx, self.idx):
        #     cc += self.e[:,None,i:i+1] * np.flip(self.buffers["v"][None,:,i-self.blockSize+1:i+1], axis=-1)
        # cc /= self.blockSize
        # self.crossCorr.update(cc)
        # self.secPathEstimate = np.transpose(np.fft.fft(self.crossCorr.state,n=2*self.blockSize,axis=-1),(2,0,1)) / self.auxNoisePower
        eBlock = np.concatenate(
            (
                np.zeros_like(self.e[:, self.updateIdx : self.idx]),
                self.e[:, self.updateIdx : self.idx],
            ),
            axis=-1,
        )
        E = np.fft.fft(eBlock, axis=-1).T[:, :, None]
        V = np.fft.fft(
            self.buffers["v"][:, self.idx - 2 * self.blockSize : self.idx], axis=-1
        ).T[:, :, None]
        crossCorr = E @ np.transpose(V.conj(), (0, 2, 1))
        crossCorr = np.fft.ifft(crossCorr, axis=0)
        crossCorr[self.blockSize :, :, :] = 0
        crossCorr = np.fft.fft(crossCorr, axis=0)
        self.crossCorr.update(crossCorr / self.blockSize)
        self.secPathEstimate = self.crossCorr.state / self.auxNoisePower

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate)

    def updateFilter(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False

        self.updateSPM()

        V = np.fft.fft(
            self.buffers["v"][:, self.idx - (2 * self.blockSize) : self.idx], axis=-1
        ).T[:, :, None]
        Vf = self.secPathEstimate @ V
        vf = np.squeeze(np.fft.ifft(Vf, axis=0), axis=-1).T
        f = self.e[:, self.updateIdx : self.idx] - vf[:, self.blockSize :]

        self.F = np.fft.fft(
            np.concatenate((np.zeros((self.numError, self.blockSize)), f), axis=-1),
            axis=-1,
        ).T[:, :, None]

        grad = (
            np.transpose(self.secPathEstimate.conj(), (0, 2, 1))
            @ self.F
            @ np.transpose(self.X.conj(), (0, 2, 1))
        )

        tdGrad = np.fft.ifft(grad, axis=0)
        tdGrad[self.blockSize :, :, :] = 0
        grad = np.fft.fft(tdGrad, axis=0)

        powerPerFreq = np.sum(np.abs(self.secPathEstimate) ** 2, axis=(1, 2)) * np.sum(
            np.abs(self.X) ** 2, axis=(1, 2)
        )
        norm = 1 / (np.mean(powerPerFreq) + self.beta)
        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True

    def updateFilter_lazy(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False

        self.updateSPM()
        self.E = np.fft.fft(
            self.e[:, self.updateIdx - self.blockSize : self.idx], axis=-1
        ).T[:, :, None]
        self.V = np.fft.fft(
            self.buffers["v"][:, self.updateIdx - self.blockSize : self.idx], axis=-1
        ).T[:, :, None]
        self.F = self.E - self.secPathEstimate @ self.V
        grad = (
            np.transpose(self.secPathEstimate.conj(), (0, 2, 1))
            @ self.F
            @ np.transpose(self.X.conj(), (0, 2, 1))
        )

        # tdGrad = np.fft.ifft(grad, axis=0)
        # tdGrad[self.blockSize:,:] = 0
        # grad = np.fft.fft(tdGrad, axis=0)

        powerPerFreq = np.sum(np.abs(self.secPathEstimate) ** 2, axis=(1, 2)) * np.sum(
            np.abs(self.X) ** 2, axis=(1, 2)
        )
        norm = 1 / (np.mean(powerPerFreq) + self.beta)
        # norm = 1 / (powerPerFreq[:,None,None] + self.beta)
        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True


class KIWienerAuxNoiseFreqFxLMS(WienerAuxNoiseFreqFxLMS):
    """Exact same algorithm as WienerAuxNoiseFreqFxLMS, but solutions
    restricted to those in the RKHS"""

    def __init__(self, config, mu, speakerFilters, blockSize, errorPos):
        super().__init__(config, mu, speakerFilters, blockSize)
        kiRegParam = 1e-1
        self.kiParams = soundfieldInterpolation(
            errorPos, errorPos, 2 * blockSize, kiRegParam
        )
        self.name = "Kernel Interpolated Aux Noise Freq Domain Wiener"

        self.metadata["kernelRegularizationParameter"] = kiRegParam

    def updateSPM(self):

        cc = np.zeros((self.numError, self.numSpeaker, self.blockSize))
        for i in range(self.updateIdx, self.idx):
            cc += self.e[:, None, i : i + 1] * np.flip(
                self.buffers["v"][None, :, i - self.blockSize + 1 : i + 1], axis=-1
            )
        cc /= self.blockSize

        self.crossCorr.update(cc)

        self.secPathEstimate = (
            self.kiParams
            @ np.transpose(
                np.fft.fft(self.crossCorr.state, n=2 * self.blockSize, axis=-1),
                (2, 0, 1),
            )
            / self.auxNoisePower
        )

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate)


class KIFreqAuxNoiseFxLMS(ConstrainedFastBlockFxLMS):
    def __init__(self, config, mu, muSPM, speakerFilters, blockSize, errorPos):
        self.beta = 1e-3
        super().__init__(config, mu, self.beta, speakerFilters, blockSize)
        self.name = "SPM Aux Noise Freq Domain - Kernel Inteproaltion"
        self.muSPM = muSPM
        self.auxNoiseSource = GoldSequenceSource(
            11, power=1, numChannels=self.numSpeaker
        )
        # self.auxNoiseSource = WhiteNoiseSource(1, numChannels=self.numSpeaker)

        self.secPathEstimate = FastBlockNLMS(
            blockSize=blockSize,
            numIn=self.numSpeaker,
            numOut=self.numError,
            stepSize=muSPM,
            freqIndepNorm=False,
        )

        kiRegParam = 1e-1
        self.kiRestriction = soundfieldInterpolation(
            errorPos, errorPos, 2 * blockSize, kiRegParam
        )

        self.G = np.transpose(self.G, (0, 2, 1))
        # self.secPathEstimate = np.zeros((2*blockSize, self.numError, self.numSpeaker), dtype=np.complex128)
        # self.secPathEstimate.setIR(0.5*self.G)# + np.random.normal(scale=0.0001, size=self.G.shape))

        # self.buffers["xf"] = np.zeros((self.numRef, self.numSpeaker, self.numError, self.simChunkSize+s.sim_buffer))
        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        self.buffers["f"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))

        self.diag.addNewDiagnostic(
            "secpath",
            dia.ConstantEstimateNMSE(
                self.G, self.sim_info.tot_samples, plotFrequency=self.plotFrequency
            ),
        )

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["muSPM"] = muSPM
        self.metadata["kernelRegularizationParameter"] = kiRegParam

    def prepare(self):
        self.buffers["f"][:, : s.sim_buffer] = self.e[:, : s.sim_buffer]

    def forwardPassImplement(self, numSamples):
        super().forwardPassImplement(numSamples)

        # v = self.auxNoiseSource.getSamples(numSamples)
        v = getWhiteNoiseAtSNR(
            np.ones((5, 10)),
            (self.numSpeaker, numSamples),
            10,
            identicalChannelPower=True,
        )
        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] += v

    def updateSPM(self):
        vf = self.secPathEstimate.process(
            np.concatenate(
                (
                    np.zeros((self.numSpeaker, self.blockSize)),
                    self.buffers["v"][:, self.updateIdx : self.idx],
                ),
                axis=-1,
            )
        )

        self.buffers["f"][:, self.updateIdx : self.idx] = (
            self.e[:, self.updateIdx : self.idx] - vf
        )
        self.secPathEstimate.update(
            self.buffers["v"][:, self.updateIdx - self.blockSize : self.idx],
            self.buffers["f"][:, self.updateIdx : self.idx],
        )

        # self.secPathEstimate.ir = self.kiRestriction @ self.secPathEstimate.ir

    def updateFilter(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False

        self.updateSPM()
        self.F = np.fft.fft(
            np.concatenate(
                (
                    np.zeros((self.numError, self.blockSize)),
                    self.buffers["f"][:, self.updateIdx : self.idx],
                ),
                axis=-1,
            ),
            axis=-1,
        ).T[:, :, None]
        G = self.kiRestriction @ self.secPathEstimate.ir
        self.diag.saveBlockData("secpath", self.idx, G)
        grad = (
            np.transpose(G.conj(), (0, 2, 1))
            @ self.F
            @ np.transpose(self.X.conj(), (0, 2, 1))
        )

        tdGrad = np.fft.ifft(grad, axis=0)
        tdGrad[self.blockSize :, :] = 0
        grad = np.fft.fft(tdGrad, axis=0)

        powerPerFreq = np.sum(np.abs(G) ** 2, axis=(1, 2)) * np.sum(
            np.abs(self.X) ** 2, axis=(1, 2)
        )
        norm = 1 / (np.mean(powerPerFreq) + self.beta)
        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True


class KIPenalizedFreqAuxNoiseFxLMS(ConstrainedFastBlockFxLMS):
    def __init__(
        self,
        config,
        mu,
        muSPM,
        speakerFilters,
        blockSize,
        errorPos,
        lowFreqLim,
        highFreqLim,
    ):
        """lowFreqLim and highFreqLim is only for diagnostics. unit is in Hz"""
        self.beta = 1e-3
        super().__init__(config, mu, self.beta, speakerFilters, blockSize)
        self.name = "SPM Aux Noise Freq Domain - KI penalized"
        self.muSPM = muSPM
        self.eta = 100
        # self.auxNoiseSource = GoldSequenceSource(11, power=10, numChannels=self.numSpeaker)
        self.auxNoiseSource = WhiteNoiseSource(power=3, numChannels=self.numSpeaker)

        self.secPathEstimate = FastBlockNLMS(
            blockSize=blockSize,
            numIn=self.numSpeaker,
            numOut=self.numError,
            stepSize=muSPM,
            freqIndepNorm=False,
        )
        self.G = np.transpose(self.G, (0, 2, 1))

        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        self.buffers["f"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))

        kernelRegParam = 1e-4
        self.kiParams = self.constructInterpolation(errorPos, kernelRegParam)

        self.diag.addNewDiagnostic(
            "secpath",
            dia.ConstantEstimateNMSE(
                self.G, self.sim_info.tot_samples, plotFrequency=self.plotFrequency
            ),
        )
        self.diag.addNewDiagnostic(
            "secpathselectedfrequencies",
            dia.ConstantEstimateNMSESelectedFrequencies(
                self.G,
                lowFreqLim,
                highFreqLim,
                self.samplerate,
                self.sim_info.tot_samples,
                plotFrequency=self.plotFrequency,
            ),
        )

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["muSPM"] = muSPM
        self.metadata["kernelRegularizationParameter"] = kernelRegParam
        self.metadata["interpolationPenaltyParameter"] = self.eta

    def constructInterpolation(self, errorPos, kernelRegParam):
        kiParams = np.zeros(
            (2 * self.blockSize, self.numError, self.numError - 1), dtype=np.complex128
        )
        for m in range(self.numError):
            kiParams[:, m : m + 1, :] = soundfieldInterpolation(
                errorPos[m : m + 1, :],
                errorPos[np.arange(self.numError) != m, :],
                2 * self.blockSize,
                kernelRegParam,
            )
        return kiParams

    def interpolate(self):
        secPathIP = np.zeros_like(self.secPathEstimate.ir)
        for m in range(self.numError):
            secPathIP[:, m : m + 1, :] = (
                self.kiParams[:, m : m + 1, :]
                @ self.secPathEstimate.ir[:, np.arange(self.numError) != m, :]
            )
        return secPathIP

    def prepare(self):
        self.buffers["f"][:, : s.sim_buffer] = self.e[:, : s.sim_buffer]

    def forwardPassImplement(self, numSamples):
        super().forwardPassImplement(numSamples)

        v = self.auxNoiseSource.getSamples(numSamples)
        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] += v

    def updateSPM(self):
        vf = self.secPathEstimate.process(
            np.concatenate(
                (
                    np.zeros((self.numSpeaker, self.blockSize)),
                    self.buffers["v"][:, self.updateIdx : self.idx],
                ),
                axis=-1,
            )
        )

        interpolatedError = self.eta * (self.interpolate() - self.secPathEstimate.ir)

        self.buffers["f"][:, self.updateIdx : self.idx] = (
            self.e[:, self.updateIdx : self.idx] - vf
        )
        self.secPathEstimate.update(
            self.buffers["v"][:, self.updateIdx - self.blockSize : self.idx],
            self.buffers["f"][:, self.updateIdx : self.idx],
        )

        self.secPathEstimate.ir += self.mu * interpolatedError

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.ir)
        self.diag.saveBlockData(
            "secpathselectedfrequencies", self.idx, self.secPathEstimate.ir
        )

    def updateFilter(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False

        self.updateSPM()
        self.F = np.fft.fft(
            np.concatenate(
                (
                    np.zeros((self.numError, self.blockSize)),
                    self.buffers["f"][:, self.updateIdx : self.idx],
                ),
                axis=-1,
            ),
            axis=-1,
        ).T[:, :, None]

        grad = (
            np.transpose(self.secPathEstimate.ir.conj(), (0, 2, 1))
            @ self.F
            @ np.transpose(self.X.conj(), (0, 2, 1))
        )

        tdGrad = np.fft.ifft(grad, axis=0)
        tdGrad[self.blockSize :, :] = 0
        grad = np.fft.fft(tdGrad, axis=0)

        powerPerFreq = np.sum(
            np.abs(self.secPathEstimate.ir) ** 2, axis=(1, 2)
        ) * np.sum(np.abs(self.X) ** 2, axis=(1, 2))
        norm = 1 / (np.mean(powerPerFreq) + self.beta)
        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True





