import numpy as np

from ancsim.adaptivefilter.mpc import FastBlockFxLMS
from ancsim.adaptivefilter.ki import FastBlockKIFxLMS
from ancsim.adaptivefilter.spm.auxnoise import FastBlockFxLMSSPMEriksson, FastBlockAuxNoise, FastBlockFxLMSSPMYuxue
from ancsim.signal.sources import GoldSequenceSource, WhiteNoiseSource
from ancsim.signal.adaptivefilter import FastBlockNLMS, FastBlockWeightedNLMS
from ancsim.soundfield.kernelinterpolation import (
    soundfieldInterpolation,
    soundfieldInterpolationFIR,
)
import ancsim.diagnostics.core as diacore
import ancsim.diagnostics.diagnostics as dia
import ancsim.signal.freqdomainfiltering as fdf
import ancsim.soundfield.kernelinterpolation as ki
from ancsim.signal.filterclasses import (
    FilterSum_Freqdomain,
    FilterMD_Freqdomain,
    FilterMD_IntBuffer,
    FilterSum,
    MovingAverage
)



class FastBlockKIFxLMSSPMYuxue(FastBlockFxLMSSPMYuxue):
    def __init__(self, config, mu, beta, speakerRIR, blockSize, muSec, auxNoisePower, 
        errorPos,
        kiFiltLen,
        kernelReg,
        mcPointGen,
        ipVolume,
        mcNumPoints):
        super().__init__(config, mu, beta, speakerRIR, blockSize, muSec, auxNoisePower)
        
        self.name = "Fast Block KIFxLMS SPM Eriksson"
        assert kiFiltLen % 1 == 0
        kiFilt = ki.getAKernelTimeDomain3d(
            errorPos,
            kiFiltLen,
            kernelReg,
            mcPointGen,
            ipVolume,
            mcNumPoints,
            2048,
            self.samplerate,
            self.c,
        )
        assert kiFilt.shape[-1] <= blockSize

        self.kiDelay = kiFilt.shape[-1] // 2
        self.buffers["kixf"] = np.zeros(
            (
                self.numSpeaker,
                self.numRef,
                self.numError,
                self.simChunkSize + self.sim_info.sim_buffer,
            )
        )

        self.kiFilt = FilterSum_Freqdomain(
            tf=fdf.fftWithTranspose(kiFilt, n=2 * blockSize),
            dataDims=(self.numSpeaker, self.numRef),
        )

        self.metadata["kernelInterpolationDelay"] = int(self.kiDelay)
        self.metadata["kiFiltLength"] = int(kiFilt.shape[-1])
        self.metadata["mcPointGen"] = mcPointGen.__name__
        self.metadata["mcNumPoints"] = mcNumPoints
        self.metadata["ipVolume"] = ipVolume

    def xfNormalization(self):
        return 1 / (
            np.sum(self.buffers["xf"][:, :, :, self.updateIdx : self.idx] ** 2)
            + self.beta
        )

    def updateFilter(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False
        self.diag.saveBlockData("power_error", self.idx-self.blockSize, np.mean(self.e[:,self.idx-self.blockSize:self.idx]**2))
        self.updateSPM()

        self.buffers["xf"][
            :, :, :, self.updateIdx : self.idx
        ] = self.secPathEstimate.process(self.x[:, self.updateIdx : self.idx])

        self.buffers["kixf"][
            :, :, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
        ] = self.kiFilt.process(
            np.transpose(
                self.buffers["xf"][:, :, :, self.updateIdx : self.idx], (1, 2, 0, 3)
            )
        )

        tdGrad = fdf.correlateSumTT(
            self.e[
                None, None, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
            ],
            self.buffers["kixf"][:,:,:,
                self.idx
                - (2 * self.blockSize)
                - self.kiDelay : self.idx
                - self.kiDelay,
            ],
        )

        grad = fdf.fftWithTranspose(
            np.concatenate((tdGrad, np.zeros_like(tdGrad)), axis=-1)
        )
        norm = self.xfNormalization()

        self.controlFilt.tf -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True



class FastBlockKIFxLMSSPMYuxueErrorIP(FastBlockFxLMSSPMYuxue):
    def __init__(self, config, mu, beta, speakerRIR, blockSize, muSec, auxNoisePower, 
        errorPos,
        kiFiltLen,
        kernelReg,
        mcPointGen,
        ipVolume,
        mcNumPoints):
        super().__init__(config, mu, beta, speakerRIR, blockSize, muSec, auxNoisePower)
        
        self.name = "Fast Block KIFxLMS SPM Yuxue Error Interpolated"
        assert kiFiltLen % 1 == 0
        kiFilt = ki.getAKernelTimeDomain3d(
            errorPos,
            kiFiltLen,
            kernelReg,
            mcPointGen,
            ipVolume,
            mcNumPoints,
            2048,
            self.samplerate,
            self.c,
        )
        assert kiFilt.shape[-1] <= blockSize

        self.kiDelay = kiFilt.shape[-1] // 2

        self.buffers["kif"] = np.zeros((self.numError, self.simChunkSize + self.sim_info.sim_buffer))
        # self.buffers["kixf"] = np.zeros(
        #     (
        #         self.numSpeaker,
        #         self.numRef,
        #         self.numError,
        #         self.simChunkSize + self.sim_info.sim_buffer,
        #     )
        # )

        self.kiFilt = FilterSum_Freqdomain(
            tf=fdf.fftWithTranspose(kiFilt, n=2 * blockSize)
        )

        self.metadata["kernelInterpolationDelay"] = int(self.kiDelay)
        self.metadata["kiFiltLength"] = int(kiFilt.shape[-1])
        self.metadata["mcPointGen"] = mcPointGen.__name__
        self.metadata["mcNumPoints"] = mcNumPoints
        self.metadata["ipVolume"] = ipVolume

    def xfNormalization(self):
        return 1 / (
            np.sum(self.buffers["xf"][:, :, :, self.updateIdx-self.kiDelay : self.idx-self.kiDelay] ** 2)
            + self.beta
        )

    def updateFilter(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False
        self.diag.saveBlockData("power_error", self.idx-self.blockSize, np.mean(self.e[:,self.idx-self.blockSize:self.idx]**2))
        self.updateSPM()

        self.buffers["xf"][
            :, :, :, self.updateIdx : self.idx
        ] = self.secPathEstimate.process(self.x[:, self.updateIdx : self.idx])

        self.buffers["kif"][
            :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
        ] = self.kiFilt.process(self.buffers["f"][:, self.updateIdx : self.idx])

        tdGrad = fdf.correlateSumTT(
            self.buffers["kif"][
                None, None, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
            ],
            np.moveaxis(self.buffers["xf"][:,:,:,
                self.idx
                - (2 * self.blockSize)
                - self.kiDelay : self.idx
                - self.kiDelay,
            ],0,2),
        )

        grad = fdf.fftWithTranspose(
            np.concatenate((tdGrad, np.zeros_like(tdGrad)), axis=-1)
        )
        norm = self.xfNormalization()

        self.controlFilt.tf -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True



class FastBlockKIFxLMSSPMYuxueSameCost(FastBlockFxLMSSPMYuxue):
    """Uses the same interpoalted cost function for both ANC and SPM"""
    def __init__(self, config, mu, beta, speakerRIR, blockSize, muSec, auxNoisePower, 
        errorPos,
        kiFiltLen,
        kernelReg,
        mcPointGen,
        ipVolume,
        mcNumPoints):
        super().__init__(config, mu, beta, speakerRIR, blockSize, muSec, auxNoisePower)
        
        self.name = "Fast Block KIFxLMS SPM Yuxue Same Cost"
        assert kiFiltLen % 1 == 0
        kiFilt = ki.getAKernelTimeDomain3d(
            errorPos,
            kiFiltLen,
            kernelReg,
            mcPointGen,
            ipVolume,
            mcNumPoints,
            2048,
            self.samplerate,
            self.c,
        )
        assert kiFilt.shape[-1] <= blockSize

        self.kiDelay = kiFilt.shape[-1] // 2

        self.buffers["kif"] = np.zeros((self.numError, self.simChunkSize + self.sim_info.sim_buffer))

        self.kiFilt = FilterSum_Freqdomain(
            tf=fdf.fftWithTranspose(kiFilt, n=2 * blockSize)
        )

        self.metadata["kernelInterpolationDelay"] = int(self.kiDelay)
        self.metadata["kiFiltLength"] = int(kiFilt.shape[-1])
        self.metadata["mcPointGen"] = mcPointGen.__name__
        self.metadata["mcNumPoints"] = mcNumPoints
        self.metadata["ipVolume"] = ipVolume

    def xfNormalization(self):
        return 1 / (
            np.sum(self.buffers["xf"][:, :, :, self.updateIdx-self.kiDelay : self.idx-self.kiDelay] ** 2)
            + self.beta
        )

    def updateFilter(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False
        self.diag.saveBlockData("power_error", self.idx-self.blockSize, np.mean(self.e[:,self.idx-self.blockSize:self.idx]**2))
        self.updateSPM()

        self.buffers["xf"][
            :, :, :, self.updateIdx : self.idx
        ] = self.secPathEstimate.process(self.x[:, self.updateIdx : self.idx])

        tdGrad = fdf.correlateSumTT(
            self.buffers["kif"][
                None, None, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
            ],
            np.moveaxis(self.buffers["xf"][:,:,:,
                self.idx
                - (2 * self.blockSize)
                - self.kiDelay : self.idx
                - self.kiDelay,
            ],0,2),
        )

        grad = fdf.fftWithTranspose(
            np.concatenate((tdGrad, np.zeros_like(tdGrad)), axis=-1)
        )
        norm = self.xfNormalization()

        self.controlFilt.tf -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True

    
    def updateSPM(self):
        self.buffers["vEst"][
            :, :, self.updateIdx : self.idx
        ] = self.secPathEstimateSPM.processWithoutSum(
            self.buffers["v"][:, self.updateIdx : self.idx]
        )

        self.buffers["f"][:, self.updateIdx : self.idx] = self.e[:, self.updateIdx : self.idx] - np.sum(
            self.buffers["vEst"][:, :, self.updateIdx : self.idx], axis=0
        )
        self.buffers["kif"][
            :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
        ] = self.kiFilt.process(self.buffers["f"][:, self.updateIdx : self.idx])

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

        self.secPathEstimateSPM.update(self.buffers["v"][:, self.idx-(2*self.blockSize)-self.kiDelay : self.idx-self.kiDelay], 
                                        self.buffers["kif"][:, self.updateIdx-self.kiDelay : self.idx-self.kiDelay])
        self.secPathEstimate.tf[...] = self.secPathEstimateSPM.filt.tf[...]

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimateSPM.filt.tf)
        self.diag.saveBlockData("secpath_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_phase_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_phase_weighted_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBufferData("recorded_ir", self.idx, np.real(fdf.ifftWithTranspose(self.secPathEstimateSPM.filt.tf)[...,:self.blockSize]))
        self.diag.saveBlockData("mu_spm", self.idx, self.secPathEstimateSPM.mu.ravel())

        self.diag.saveBlockData("variable_auxnoise_power", self.idx, auxNoisePowerFactor)




class FastBlockKIFxLMSSPMEriksson_old(FastBlockFxLMSSPMEriksson):
    """"""

    def __init__(self, config, mu, muSPM, speakerFilters, blockSize, kiFilt):
        super().__init__(mu, muSPM, speakerFilters, blockSize)
        self.name = "Kernel IP Freq ANC with Eriksson SPM"
        assert kiFilt.shape[-1] <= blockSize
        assert kiFilt.shape[-1] % 1 == 0

        self.kiDelay = kiFilt.shape[-1] // 2
        self.buffers["kixf"] = np.zeros(
            (
                self.numSpeaker,
                self.numRef,
                self.numError,
                self.simChunkSize + s.sim_buffer,
            )
        )

        self.kiFilt = FilterSum_Freqdomain(
            tf=fdf.fftWithTranspose(kiFilt, n=2 * blockSize),
            dataDims=(self.numSpeaker, self.numRef),
        )

        self.metadata["kernelInterpolationDelay"] = int(self.kiDelay)
        self.metadata["kiFiltLength"] = int(kiFilt.shape[-1])

    def forwardPassImplement(self, numSamples):
        super().forwardPassImplement(numSamples)

    def updateFilter(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False
        super().updateSPM()

        self.buffers["xf"][
            :, :, :, self.updateIdx : self.idx
        ] = self.secPathEstimateMD.process(self.x[:, self.updateIdx : self.idx])

        self.buffers["kixf"][
            :, :, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
        ] = self.kiFilt.process(
            np.transpose(
                self.buffers["xf"][:, :, :, self.updateIdx : self.idx], (1, 2, 0, 3)
            )
        )

        tdGrad = fdf.correlateSumTT(
            self.buffers["f"][
                None, None, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
            ],
            self.buffers["kixf"][
                :,
                :,
                :,
                self.idx
                - (2 * self.blockSize)
                - self.kiDelay : self.idx
                - self.kiDelay,
            ],
        )

        grad = fdf.fftWithTranspose(
            np.concatenate((tdGrad, np.zeros_like(tdGrad)), axis=-1)
        )
        norm = 1 / (
            np.sum(
                self.buffers["kixf"][
                    :, :, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
                ]
                ** 2
            )
            + self.beta
        )

        self.controlFilt.tf -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True


class KernelSPM6(FastBlockFxLMSSPMEriksson):
    """Very similar to FastBlockKIFxLMSSPMEriksson. Applies the kernel interpolation
    filter to the spm update as well. Given by a straigtforward differentiation of
    the integral cost function int_omega|f(r)|^2dr where f = e - Ghat y_spm"""

    def __init__(
        self,
        config,
        mu,
        muSPM,
        speakerFilters,
        blockSize,
        errorPos,
        kiFiltLen,
        kernelReg,
        mcPointGen,
        ipVolume,
        mcNumPoints=100,
    ):
        super().__init__(config, mu, muSPM, speakerFilters, blockSize)
        self.name = "Fast Block FxLMS Eriksson SPM with weighted spm error"
        assert kiFiltLen <= blockSize
        assert kiFiltLen % 1 == 0
        kiFiltFir = getAKernelTimeDomain3d(
            errorPos,
            kiFiltLen,
            kernelReg,
            mcPointGen,
            ipVolume,
            mcNumPoints,
            2048,
            self.samplerate,
            self.c,
        )
        self.kiFilt = FilterSum_Freqdomain(
            tf=fdf.fftWithTranspose(kiFiltFir, n=2 * blockSize),
            dataDims=(self.numSpeaker, self.numRef),
        )
        self.kiFiltSecPathEst = FilterSum_Freqdomain(tf=self.kiFilt.tf)

        self.kiDelay = kiFiltFir.shape[-1] // 2
        self.buffers["kixf"] = np.zeros(
            (
                self.numSpeaker,
                self.numRef,
                self.numError,
                self.simChunkSize + s.sim_buffer,
            )
        )
        self.buffers["kif"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))

        self.metadata["kernelInterpolationDelay"] = int(self.kiDelay)
        self.metadata["kiFiltLength"] = int(kiFiltLen)
        self.metadata["mcPointGen"] = mcPointGen.__name__
        self.metadata["mcNumPoints"] = mcNumPoints
        self.metadata["ipVolume"] = ipVolume
        # self.metadata["normalization"] = self.normFunc.__name__

    def forwardPassImplement(self, numSamples):
        super().forwardPassImplement(numSamples)

    def updateSPM(self):
        vf = self.secPathEstimate.process(
            self.buffers["v"][:, self.updateIdx : self.idx]
        )

        self.buffers["f"][:, self.updateIdx : self.idx] = (
            self.e[:, self.updateIdx : self.idx] - vf
        )

        self.buffers["kif"][
            :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
        ] = self.kiFiltSecPathEst.process(
            self.buffers["f"][:, self.updateIdx : self.idx]
        )

        grad = fdf.correlateEuclidianTT(
            self.buffers["kif"][
                :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
            ],
            self.buffers["v"][
                :,
                self.idx
                - self.kiDelay
                - (2 * self.blockSize) : self.idx
                - self.kiDelay,
            ],
        )

        grad = fdf.fftWithTranspose(
            np.concatenate((grad, np.zeros_like(grad)), axis=-1)
        )
        norm = 1 / (
            np.sum(
                self.buffers["v"][
                    :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
                ]
                ** 2
            )
            + 1e-3
        )

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

        self.buffers["kixf"][
            :, :, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
        ] = self.kiFilt.process(
            np.transpose(
                self.buffers["xf"][:, :, :, self.updateIdx : self.idx], (1, 2, 0, 3)
            )
        )

        tdGrad = fdf.correlateSumTT(
            self.buffers["f"][
                None, None, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
            ],
            self.buffers["kixf"][
                :,
                :,
                :,
                self.idx
                - (2 * self.blockSize)
                - self.kiDelay : self.idx
                - self.kiDelay,
            ],
        )

        grad = fdf.fftWithTranspose(
            np.concatenate((tdGrad, np.zeros_like(tdGrad)), axis=-1)
        )
        norm = 1 / (
            np.sum(
                self.buffers["kixf"][
                    :, :, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
                ]
                ** 2
            )
            + self.beta
        )

        self.controlFilt.tf -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True


class ConstrainedFastBlockFxLMS:
    pass  # Any class inheriting from this needs to be re-examined to use FastBlockFxLMS


class FDKIFxLMSEriksson(ConstrainedFastBlockFxLMS):
    """DEPRECATED - DOES NOT HAVRE LINEAR CONVOLUTIONS"""

    def __init__(self, mu, beta, speakerFilters, blockSize, muSPM, kernFilt):
        super().__init__(mu, beta, speakerFilters, blockSize)
        self.name = "Kernel IP Freq ANC with Eriksson SPM"
        self.A = kernFilt

        self.auxNoiseSource = WhiteNoiseSource(power=3, num_channels=self.numSpeaker)
        self.secPathEstimate = FastBlockNLMS(
            blockSize=blockSize,
            numIn=self.numSpeaker,
            numOut=self.numError,
            stepSize=muSPM,
            freqIndepNorm=False,
        )

        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        self.buffers["f"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))

        self.G = np.transpose(self.G, (0, 2, 1))
        self.diag.add_diagnostic(
            "secpath",
            diacore.ConstantEstimateNMSE(
                self.G, self.sim_info.tot_samples, plotFrequency=self.plotFrequency
            ),
        )
        # self.combinedFilt = self.G.conj() @ self.A
        # self.normFactor = np.sqrt(np.sum(np.abs(self.G.conj() @ self.A @ np.transpose(self.G,(0,2,1))**2)))

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["muSPM"] = muSPM

    def prepare(self):
        self.buffers["f"][:, : s.sim_buffer] = self.e[:, : s.sim_buffer]

    def forwardPassImplement(self, numSamples):
        super().forwardPassImplement(numSamples)

        v = self.auxNoiseSource.get_samples(numSamples)
        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] += v

    # @util.measure("Update Kfreq")
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
            @ self.A
            @ self.F
            @ np.transpose(self.X.conj(), (0, 2, 1))
        )

        tdGrad = np.fft.ifft(grad, axis=0)
        tdGrad[self.blockSize :, :] = 0
        grad = np.fft.fft(tdGrad, axis=0)

        normFactor = np.sqrt(
            np.sum(
                np.abs(
                    np.transpose(self.secPathEstimate.ir.conj(), (0, 2, 1))
                    @ self.A
                    @ self.secPathEstimate.ir
                )
                ** 2,
                axis=(1, 2),
            )
        )
        # norm = 1 / ((normFactor *  \
        #            np.sum(np.abs(self.X)**2) / (2*self.blockSize)**2) + self.beta)

        powerPerFreq = normFactor * np.sum(np.abs(self.X) ** 2, axis=(1, 2))
        norm = 1 / (np.mean(powerPerFreq) + self.beta)

        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.update = True

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
        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.ir)


class FDKIFxLMSErikssonInterpolatedF(FDKIFxLMSEriksson):
    def __init__(self, config, mu, beta, speakerFilters, blockSize, muSPM, kernFilt):
        super().__init__(config, mu, beta, speakerFilters, blockSize, muSPM, kernFilt)
        self.name = "Kernel IP Freq ANC with Eriksson SPM, Weighted f"

        self.secPathEstimate = FastBlockWeightedNLMS(
            blockSize=blockSize,
            numIn=self.numSpeaker,
            numOut=self.numError,
            weightMatrix=kernFilt,
            stepSize=muSPM,
            freqIndepNorm=False,
        )
        self.diag.add_diagnostic(
            "weightedsecpath",
            diacore.ConstantEstimateNMSE(
                self.A @ self.G, self.sim_info.tot_samples, plotFrequency=self.plotFrequency
            ),
        )

    def updateSPM(self):
        super().updateSPM()
        self.diag.saveBlockData("weightedsecpath", self.idx, self.secPathEstimate.ir)


class KernelSPM2(ConstrainedFastBlockFxLMS):
    """DOES NOT HAVE LINEAR CONVOLUTIONS"""

    def __init__(self, config, mu, muSPM, speakerRIR, blockSize, errorPos):
        super().__init__(config, mu, 1e-3, speakerRIR, blockSize)
        self.name = "Freq ANC Kernel SPM 2"

        kernelReg = 1e-3
        self.kernelMat = soundfieldInterpolation(
            errorPos, errorPos, 2 * self.blockSize, kernelReg
        )

        self.auxNoiseSource = WhiteNoiseSource(power=3, num_channels=self.numSpeaker)
        self.secPathEstimate = FastBlockWeightedNLMS(
            blockSize=blockSize,
            numIn=self.numSpeaker,
            numOut=self.numError,
            stepSize=muSPM,
            weightMatrix=self.kernelMat,
            freqIndepNorm=False,
        )

        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        self.buffers["f"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))

        self.G = np.transpose(self.G, (0, 2, 1))
        self.diag.add_diagnostic(
            "secpath",
            diacore.ConstantEstimateNMSE(
                self.G, self.sim_info.tot_samples, plotFrequency=self.plotFrequency
            ),
        )
        # self.combinedFilt = self.G.conj() @ self.A
        # self.normFactor = np.sqrt(np.sum(np.abs(self.G.conj() @ self.A @ np.transpose(self.G,(0,2,1))**2)))

        self.metadata["auxNoiseSource"] = self.auxNoiseSource.__class__.__name__
        self.metadata["auxNoisePower"] = self.auxNoiseSource.power
        self.metadata["muSPM"] = muSPM
        self.metadata["kernelReg"] = kernelReg

    def prepare(self):
        self.buffers["f"][:, : s.sim_buffer] = self.e[:, : s.sim_buffer]

    def forwardPassImplement(self, numSamples):
        super().forwardPassImplement(numSamples)

        v = self.auxNoiseSource.get_samples(numSamples)
        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] += v

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

    def updateSPM(self):
        V = np.fft.fft(
            np.concatenate(
                (
                    np.zeros((self.numSpeaker, self.blockSize)),
                    self.buffers["v"][:, self.updateIdx : self.idx],
                ),
                axis=-1,
            ),
            axis=-1,
        ).T[:, :, None]
        Vf = self.kernelMat @ self.secPathEstimate.ir @ V
        vf = np.squeeze(np.fft.ifft(Vf, axis=0), axis=-1).T
        vf = np.real(vf[:, self.blockSize :])
        # vf = self.secPathEstimate.process(np.concatenate((np.zeros((self.numSpeaker,self.blockSize)),
        #                                                self.buffers["v"][:,self.updateIdx:self.idx]), axis=-1))

        self.buffers["f"][:, self.updateIdx : self.idx] = (
            self.e[:, self.updateIdx : self.idx] - vf
        )
        self.secPathEstimate.update(
            self.buffers["v"][:, self.updateIdx - self.blockSize : self.idx],
            self.buffers["f"][:, self.updateIdx : self.idx],
        )
        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.ir)


# class FxLMSKernelSPM1(ConstrainedFastBlockFxLMS):
#     def __init__(self, mu, beta, speakerFilters, muSPM, secPathEstimateLen):
#         super().__init__(mu, beta, speakerFilters)
#         self.name = "FxLMS SPM Eriksson"
#         self.muSPM = muSPM
#         self.spmLen = secPathEstimateLen
#         self.auxNoiseSource = GoldSequenceSource(11, num_channels=self.numSpeaker)

#         self.kernelParams = soundfieldInterpolation(errorPos, errorPos, 2*blockSize, 1e-2)

#         self.secPathEstimate = FilterMD_IntBuffer(dataDims=self.numRef, irLen=secPathEstimateLen, irDims=(self.numSpeaker, self.numError))
#         self.secPathEstimateSum = FilterSum(irLen=secPathEstimateLen, numIn=self.numSpeaker, numOut=self.numError)

#         initialSecPathEstimate = np.random.normal(scale=0.0001, size=self.secPathEstimate.ir.shape)
#         commonLength = np.min((secPathEstimateLen,speakerFilters["error"].shape[-1]))
#         initialSecPathEstimate[:,:,0:commonLength] = 0.5 * speakerFilters["error"][:,:,0:commonLength]

#         self.secPathEstimate.setIR(initialSecPathEstimate)
#         self.secPathEstimateSum.setIR(initialSecPathEstimate)

#         self.buffers["xf"] = np.zeros((self.numRef, self.numSpeaker, self.numError, self.simChunkSize+s.sim_buffer))
#         self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize+s.sim_buffer))
#         self.buffers["vEst"] = np.zeros((self.numError, self.simChunkSize+s.sim_buffer))

#         self.diag.add_diagnostic("secpath", diacore.ConstantEstimateNMSE(speakerFilters["error"], plotFrequency=self.plotFrequency))

#     def updateSPM(self):
#         self.buffers["vEst"][:,self.updateIdx:self.idx] = self.secPathEstimateSum.process(
#                                                             self.buffers["v"][:,self.updateIdx:self.idx])

#         f = self.e[:,self.updateIdx:self.idx] - self.buffers["vEst"][:,self.updateIdx:self.idx]

#         grad = np.zeros_like(self.secPathEstimateSum.ir)
#         numSamples = self.idx-self.updateIdx
#         for i in range(numSamples):
#             V = np.flip(self.buffers["v"][:,None,self.updateIdx+i-self.spmLen+1:self.updateIdx+i+1], axis=-1)
#             grad += V * f[None,:,i:i+1] / (np.sum(V**2) + 1e-4)

#         grad *= self.muSPM
#         self.secPathEstimate.ir += grad
#         self.secPathEstimateSum.ir[:,:,:] = self.secPathEstimate.ir[:,:,:]

#         self.diag.saveBlockData("secpath", self.updateIdx, self.secPathEstimate.ir)

#     def updateFilter(self):
#         self.buffers["xf"][:,:,:,self.updateIdx:self.idx] = np.transpose(self.secPathEstimate.process(
#                                                         self.x[:,self.updateIdx:self.idx]), (2,0,1,3))
#         grad = np.zeros_like(self.controlFilt.ir)
#         for n in range(self.updateIdx, self.idx):
#            Xf = np.flip(self.buffers["xf"][:,:,:,n-self.filtLen+1:n+1], axis=-1)
#            grad += np.sum(Xf * self.e[None, None,:,n,None],axis=2) / (np.sum(Xf**2) + self.beta)

#         self.controlFilt.ir -= grad * self.mu
#         #super().updateFilter()
#         self.updateSPM()
#         self.updateIdx = self.idx


#     def forwardPassImplement(self, numSamples):
#         y = self.controlFilt.process(self.x[:,self.idx:self.idx+numSamples])

#         v = self.auxNoiseSource.get_samples(numSamples)
#         self.buffers["v"][:,self.idx:self.idx+numSamples] = v

#         self.y[:,self.idx:self.idx+numSamples] = v + y


class KernelSPM3(FastBlockFxLMSSPMEriksson):
    """UPDATED VERSION FOR NEW FAST BLOCK FXLMS

    Attempts to update each sec path filter using data from all microphones.
        It uses the normal Freq domain Eriksson cost function, together with
        ||e - G_ip y_spm||^2 where G_ip is the sec path interpolated to microphone m
        from all positions except m (to avoid the inteprolation filter reducing to identity)."""

    def __init__(
        self, config, mu, muSPM, eta, speakerRIR, blockSize, errorPos, kiFiltLen
    ):
        super().__init__(config, mu, muSPM, speakerRIR, blockSize)
        self.name = "Freq ANC Kernel SPM 3"
        assert kiFiltLen % 2 == 1
        assert kiFiltLen <= blockSize
        self.eta = eta
        kernelReg = 1e-3
        self.kernelMat = self.constructInterpolation(errorPos, kernelReg, kiFiltLen)
        self.kiDelay = kiFiltLen // 2
        self.kiFreqSamples = np.max(1024, 4 * self.filtLen)

        self.nextBlockAuxNoise = self.auxNoiseSource.get_samples(blockSize)
        # self.nextBlockFilteredAuxNoise = np.zeros((self.numError, blockSize))

        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        self.buffers["vf"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))
        self.buffers["f"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))
        self.buffers["fip"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))
        self.buffers["vIP"] = np.zeros(
            (
                self.numSpeaker,
                self.numError,
                self.numError,
                self.simChunkSize + s.sim_buffer,
            )
        )

        self.metadata["kernelReg"] = kernelReg
        self.metadata["kernelCostWeight"] = self.eta
        self.metadata["kiFreqSamples"] = self.kiFreqSamples

    def constructInterpolation(self, errorPos, kernelRegParam, kiFiltLen):
        """Output is shape(numFreq, position interpolated to, positions interpolated from),
        which is (2*blocksize, M, M-1)"""
        kiParams = np.zeros(
            (2 * self.blockSize, self.numError, self.numError), dtype=np.complex128
        )
        kiFirAll = np.zeros((16, 16, 155))
        for m in range(self.numError):
            kiFIR = soundfieldInterpolationFIR(
                errorPos[m : m + 1, :],
                errorPos[np.arange(self.numError) != m, :],
                kiFiltLen,
                kernelRegParam,
                self.kiFreqSamples,
                self.spatialDims,
                self.samplerate,
                self.c,
            )

            kiFreq = fdf.fftWithTranspose(
                np.concatenate(
                    (
                        kiFIR,
                        np.zeros(
                            (kiFIR.shape[:-1] + (2 * self.blockSize - kiFiltLen,))
                        ),
                    ),
                    axis=-1,
                )
            )
            kiParams[:, m : m + 1, np.arange(self.numError) != m] = kiFreq

            kiFirAll[m : m + 1, np.arange(self.numError) != m, :] = kiFIR
        return kiParams

    def getVariableNumberOfFutureSamples():
        v = np.zeros((self.numSpeaker, numSamples))
        numOldSamples = np.min((numSamples, self.futureAuxNoise.shape[-1]))
        v[:, :numOldSamples] = self.futureAuxNoise[:, :numOldSamples]
        numNewSamples = numSamples - self.futureAuxNoise.shape[-1]
        if numNewSamples > 0:
            v[:, numOldSamples:] = self.auxNoiseSource.get_samples(numNewSamples)

        newToReplace = self.futureAuxNoise.shape[-1] - numOldSamples
        self.futureAuxNoise[:, :newToReplace] = self.futureAuxNoise[:, numOldSamples:]
        self.futureAuxNoise[:, newToReplace:] = self.auxNoiseSource.get_samples(
            newToReplace
        )

    def forwardPassImplement(self, numSamples):
        assert numSamples == self.blockSize
        self.y[:, self.idx : self.idx + self.blockSize] = self.controlFilt.process(
            self.x[:, self.idx : self.idx + self.blockSize]
        )

        v = self.nextBlockAuxNoise
        self.nextBlockAuxNoise[...] = self.auxNoiseSource.get_samples(numSamples)

        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] += v
        self.updated = False

    def interpolateFIR(kiFilt, fir):
        """kiFilt is (outDim, inDim, irLen)
        fir is (extraDim, inDim, irLen2)
        out is (extraDim, outDim, irLen2)"""
        assert kiFilt.shape[-1] % 2 == 1
        out = np.zeros((fir.shape[:-1] + (fir.shape[-1] + kiFilt.shape[-1] - 1,)))
        for i in range(kiFilt.shape[0]):  # outDim
            for j in range(kiFilt.shape[1]):  # inDim
                for k in range(fir.shape[0]):  # extraDim
                    out[k, i, :] += np.convolve(
                        kiFirAll[i, j, :], self.secPathFilt.ir[k, j, :], "full"
                    )
        return out[..., irLen // 2 : -irLen // 2 + 1]

    def getIPGrad(self):
        # Calculate error signal
        nextBlockVf = fdf.convolveSum(
            self.secPathEstimate.tf,
            np.concatenate(
                (
                    self.buffers["v"][:, self.updateIdx : self.idx],
                    self.nextBlockAuxNoise,
                ),
                axis=-1,
            ),
        )
        vfWithFutureValues = np.concatenate(
            (
                self.buffers["vf"][
                    :, self.idx - (2 * self.blockSize) + self.kiDelay : self.idx
                ],
                nextBlockVf[:, : self.kiDelay],
            ),
            axis=-1,
        )
        vWithFutureValues = np.concatenate(
            (
                self.buffers["v"][
                    :, self.idx - (2 * self.blockSize) + self.kiDelay : self.idx
                ],
                self.nextBlockAuxNoise[:, : self.kiDelay],
            ),
            axis=-1,
        )

        vfIP = fdf.convolveSum(self.kernelMat, vfWithFutureValues)

        self.buffers["fip"][:, self.updateIdx : self.idx] = (
            self.e[:, self.updateIdx : self.idx] - vfIP
        )

        # Calculate the interpolated reference
        self.buffers["vIP"][..., self.updateIdx : self.idx] = np.transpose(
            fdf.convolveEuclidianFT(self.kernelMat, vWithFutureValues), (2, 1, 0, 3)
        )

        # Calculate the correlation, ie. gradient
        grad = fdf.correlateSumTT(
            self.buffers["fip"][None, None, :, self.updateIdx : self.idx],
            self.buffers["vIP"][..., self.idx - (2 * self.blockSize) : self.idx],
        )
        grad = np.transpose(grad, (1, 0, 2))
        grad = fdf.fftWithTranspose(
            np.concatenate((grad, np.zeros_like(grad)), axis=-1)
        )

        norm = 1 / (
            np.sum(self.buffers["vIP"][..., self.updateIdx : self.idx] ** 2) + 1e-3
        )
        return grad * norm

    def getNormalGrad(self):
        self.buffers["vf"][:, self.updateIdx : self.idx] = self.secPathEstimate.process(
            self.buffers["v"][:, self.updateIdx : self.idx]
        )

        self.buffers["f"][:, self.updateIdx : self.idx] = (
            self.e[:, self.updateIdx : self.idx]
            - self.buffers["vf"][:, self.updateIdx : self.idx]
        )

        grad = fdf.correlateEuclidianTT(
            self.buffers["f"][:, self.updateIdx : self.idx],
            self.buffers["v"][:, self.idx - (2 * self.blockSize) : self.idx],
        )

        grad = fdf.fftWithTranspose(
            np.concatenate((grad, np.zeros_like(grad)), axis=-1)
        )
        norm = 1 / (np.sum(self.buffers["v"][:, self.updateIdx : self.idx] ** 2) + 1e-3)
        return grad * norm

    def updateSPM(self):
        grad = self.getNormalGrad()
        grad += self.eta * self.getIPGrad()

        self.secPathEstimate.tf += self.muSPM * grad
        self.secPathEstimateMD.tf[...] = self.secPathEstimate.tf

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.tf)


class KernelSPM4(FastBlockFxLMSSPMEriksson):
    """
    Attempts to update each sec path filter using data from all microphones.
    Uses the kernel including all microphones. Therefore only uses the
    interpolated cost function"""

    def __init__(self, config, mu, muSPM, speakerRIR, blockSize, errorPos, kiFiltLen):
        super().__init__(config, mu, muSPM, speakerRIR, blockSize)
        self.name = "Freq ANC Kernel SPM 4"
        assert kiFiltLen % 2 == 1
        assert kiFiltLen <= blockSize
        kernelReg = 1e-3
        self.kernelMat = self.constructInterpolation(errorPos, kernelReg, kiFiltLen)
        self.kiDelay = kiFiltLen // 2
        self.kiFreqSamples = np.max(1024, 4 * self.filtLen)

        self.nextBlockAuxNoise = self.auxNoiseSource.get_samples(blockSize)

        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        self.buffers["vf"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))
        self.buffers["f"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))
        self.buffers["fip"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))
        self.buffers["vIP"] = np.zeros(
            (
                self.numSpeaker,
                self.numError,
                self.numError,
                self.simChunkSize + s.sim_buffer,
            )
        )

        self.metadata["kernelReg"] = kernelReg
        self.metadata["kiFreqSamples"] = self.kiFreqSamples
        # self.metadata["kernelCostWeight"] = self.eta

    def constructInterpolation(self, errorPos, kernelRegParam, kiFiltLen):
        """Output is shape(numFreq, position interpolated to, positions interpolated from),
        which is (2*blocksize, M, M)"""
        kiParams = np.zeros(
            (2 * self.blockSize, self.numError, self.numError), dtype=np.complex128
        )

        kiFIR = soundfieldInterpolationFIR(
            errorPos,
            errorPos,
            kiFiltLen,
            kernelRegParam,
            self.kiFreqSamples,
            self.spatialDims,
            self.samplerate,
            self.c,
        )

        kiFreq = fdf.fftWithTranspose(
            np.concatenate(
                (
                    kiFIR,
                    np.zeros((kiFIR.shape[:-1] + (2 * self.blockSize - kiFiltLen,))),
                ),
                axis=-1,
            )
        )
        return kiFreq

    def forwardPassImplement(self, numSamples):
        assert numSamples == self.blockSize
        self.y[:, self.idx : self.idx + self.blockSize] = self.controlFilt.process(
            self.x[:, self.idx : self.idx + self.blockSize]
        )

        v = self.nextBlockAuxNoise
        self.nextBlockAuxNoise[...] = self.auxNoiseSource.get_samples(numSamples)

        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] += v
        self.updated = False

    def getIPGrad(self):
        # Calculate error signal
        nextBlockVf = fdf.convolveSum(
            self.secPathEstimate.tf,
            np.concatenate(
                (
                    self.buffers["v"][:, self.updateIdx : self.idx],
                    self.nextBlockAuxNoise,
                ),
                axis=-1,
            ),
        )
        vfWithFutureValues = np.concatenate(
            (
                self.buffers["vf"][
                    :, self.idx - (2 * self.blockSize) + self.kiDelay : self.idx
                ],
                nextBlockVf[:, : self.kiDelay],
            ),
            axis=-1,
        )
        vWithFutureValues = np.concatenate(
            (
                self.buffers["v"][
                    :, self.idx - (2 * self.blockSize) + self.kiDelay : self.idx
                ],
                self.nextBlockAuxNoise[:, : self.kiDelay],
            ),
            axis=-1,
        )

        vfIP = fdf.convolveSum(self.kernelMat, vfWithFutureValues)

        self.buffers["fip"][:, self.updateIdx : self.idx] = (
            self.e[:, self.updateIdx : self.idx] - vfIP
        )

        # Calculate the interpolated reference
        self.buffers["vIP"][..., self.updateIdx : self.idx] = np.transpose(
            fdf.convolveEuclidianFT(self.kernelMat, vWithFutureValues), (2, 1, 0, 3)
        )

        # Calculate the correlation, ie. gradient
        grad = fdf.correlateSumTT(
            self.buffers["fip"][None, None, :, self.updateIdx : self.idx],
            self.buffers["vIP"][..., self.idx - (2 * self.blockSize) : self.idx],
        )
        grad = np.transpose(grad, (1, 0, 2))
        grad = fdf.fftWithTranspose(
            np.concatenate((grad, np.zeros_like(grad)), axis=-1)
        )

        norm = 1 / (
            np.sum(self.buffers["vIP"][..., self.updateIdx : self.idx] ** 2) + 1e-3
        )
        return grad * norm

    def updateSPM(self):
        self.secPathEstimate.tf += self.muSPM * self.getIPGrad()
        self.secPathEstimateMD.tf[...] = self.secPathEstimate.tf

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.tf)


class KernelSPM5(FastBlockFxLMSSPMEriksson):
    """Closely related to version 3. Uses verified interpolation method in time domain"""

    def __init__(
        self, config, mu, muSPM, eta, speakerRIR, blockSize, errorPos, kiFiltLen
    ):
        super().__init__(config, mu, muSPM, speakerRIR, blockSize)
        self.name = "Freq ANC Kernel SPM 5"
        assert kiFiltLen % 2 == 1
        assert kiFiltLen <= blockSize
        self.eta = eta
        kernelReg = 1e-3
        self.kiFilt = FilterMD_IntBuffer(
            dataDims=self.numSpeaker,
            ir=self.constructInterpolation(errorPos, kernelReg, kiFiltLen),
        )

        self.secPathEstimate = FilterSum(
            numIn=self.numSpeaker, numOut=self.numError, irLen=blockSize
        )
        self.secPathEstimateIP = FilterSum(
            numIn=self.numSpeaker, numOut=self.numError, irLen=blockSize
        )
        self.secPathEstimateMD = FilterMD_IntBuffer(
            dataDims=self.numRef, ir=self.secPathEstimate.ir
        )
        self.kiDelay = kiFiltLen // 2
        self.kiFreqSamples = np.max(1024, 4 * self.filtLen)

        self.nextBlockAuxNoise = self.auxNoiseSource.get_samples(blockSize)
        # self.nextBlockFilteredAuxNoise = np.zeros((self.numError, blockSize))

        self.diag.add_diagnostic(
            "secpath",
            diacore.ConstantEstimateNMSE(
                self.secPathFilt.ir, self.sim_info.tot_samples, plotFrequency=self.plotFrequency
            ),
        )

        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        self.buffers["vf"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))
        self.buffers["f"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))
        self.buffers["fip"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))
        self.buffers["vIP"] = np.zeros(
            (
                self.numSpeaker,
                self.numError,
                self.numError,
                self.simChunkSize + s.sim_buffer,
            )
        )

        self.metadata["kernelReg"] = kernelReg
        self.metadata["kernelCostWeight"] = self.eta
        self.metadata["kiFreqSamples"] = self.kiFreqSamples

    def updateFilter(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False
        self.updateSPM()

        self.buffers["xf"][:, :, :, self.updateIdx : self.idx] = np.transpose(
            self.secPathEstimateMD.process(self.x[:, self.updateIdx : self.idx]),
            (1, 0, 2, 3),
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
        # powerPerFreq = np.sum(np.abs(xfFreq)**2,axis=(1,2,3))
        # norm = 1 / (np.mean(powerPerFreq) + self.beta)

        self.controlFilt.tf -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True

    def constructInterpolation(self, errorPos, kernelRegParam, kiFiltLen):
        """Output is shape(numFreq, position interpolated to, positions interpolated from),
        which is (2*blocksize, M, M-1)"""
        kiParams = np.zeros(
            (2 * self.blockSize, self.numError, self.numError), dtype=np.complex128
        )
        kiFirAll = np.zeros((self.numError, self.numError, kiFiltLen))
        for m in range(self.numError):
            kiFIR = soundfieldInterpolationFIR(
                errorPos[m : m + 1, :],
                errorPos[np.arange(self.numError) != m, :],
                kiFiltLen,
                kernelRegParam,
                self.kiFreqSamples,
                self.spatialDims,
                self.samplerate,
                self.c,
            )

            kiFreq = fdf.fftWithTranspose(
                np.concatenate(
                    (
                        kiFIR,
                        np.zeros(
                            (kiFIR.shape[:-1] + (2 * self.blockSize - kiFiltLen,))
                        ),
                    ),
                    axis=-1,
                )
            )
            kiParams[:, m : m + 1, np.arange(self.numError) != m] = kiFreq

            kiFirAll[m : m + 1, np.arange(self.numError) != m, :] = kiFIR
        return kiFirAll

    def forwardPassImplement(self, numSamples):
        assert numSamples == self.blockSize
        self.y[:, self.idx : self.idx + self.blockSize] = self.controlFilt.process(
            self.x[:, self.idx : self.idx + self.blockSize]
        )

        v = self.nextBlockAuxNoise
        self.nextBlockAuxNoise[...] = self.auxNoiseSource.get_samples(numSamples)

        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] += v
        self.updated = False

    def interpolateFIR(self, kiFilt, fir):
        """kiFilt is (outDim, inDim, irLen)
        fir is (extraDim, inDim, irLen2)
        out is (extraDim, outDim, irLen2)"""
        assert kiFilt.shape[-1] % 2 == 1
        out = np.zeros((fir.shape[:-1] + (fir.shape[-1] + kiFilt.shape[-1] - 1,)))
        for i in range(kiFilt.shape[0]):  # outDim
            for j in range(kiFilt.shape[1]):  # inDim
                for k in range(fir.shape[0]):  # extraDim
                    out[k, i, :] += np.convolve(kiFilt[i, j, :], fir[k, j, :], "full")
        return out[..., kiFilt.shape[-1] // 2 : -kiFilt.shape[-1] // 2 + 1]

    def getIPGrad(self):
        # Calculate error signal
        self.secPathEstimateIP.ir = self.interpolateFIR(
            self.kiFilt.ir, self.secPathEstimate.ir
        )
        vfIP = self.secPathEstimateIP.process(
            self.buffers["v"][..., self.updateIdx : self.idx]
        )

        self.buffers["fip"][:, self.updateIdx : self.idx] = (
            self.e[:, self.updateIdx : self.idx] - vfIP
        )

        vWithFutureValues = np.concatenate(
            (
                self.buffers["v"][
                    :, self.idx - (self.blockSize) + self.kiDelay : self.idx
                ],
                self.nextBlockAuxNoise[:, : self.kiDelay],
            ),
            axis=-1,
        )

        self.buffers["vIP"][..., self.updateIdx : self.idx] = np.transpose(
            self.kiFilt.process(vWithFutureValues), (2, 1, 0, 3)
        )

        # Calculate the correlation, ie. gradient
        grad = fdf.correlateSumTT(
            self.buffers["fip"][None, None, :, self.updateIdx : self.idx],
            self.buffers["vIP"][..., self.idx - (2 * self.blockSize) : self.idx],
        )

        norm = 1 / (
            np.sum(self.buffers["vIP"][..., self.updateIdx : self.idx] ** 2) + 1e-3
        )
        return grad * norm

    def getNormalGrad(self):
        self.buffers["vf"][:, self.updateIdx : self.idx] = self.secPathEstimate.process(
            self.buffers["v"][:, self.updateIdx : self.idx]
        )

        self.buffers["f"][:, self.updateIdx : self.idx] = (
            self.e[:, self.updateIdx : self.idx]
            - self.buffers["vf"][:, self.updateIdx : self.idx]
        )

        grad = fdf.correlateEuclidianTT(
            self.buffers["f"][:, self.updateIdx : self.idx],
            self.buffers["v"][:, self.idx - (2 * self.blockSize) : self.idx],
        )

        # grad = fdf.fftWithTranspose(np.concatenate(
        #         (grad, np.zeros_like(grad)),axis=-1),addEmptyDim=False)
        norm = 1 / (np.sum(self.buffers["v"][:, self.updateIdx : self.idx] ** 2) + 1e-3)
        grad = np.transpose(grad, (1, 0, 2))
        return grad * norm

    def updateSPM(self):
        grad = self.getNormalGrad()
        grad += self.eta * self.getIPGrad()

        self.secPathEstimate.ir += self.muSPM * grad
        self.secPathEstimateMD.ir[...] = self.secPathEstimate.ir

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.ir)


class FreqAuxNoiseFxLMS2:
    pass  # Any class inheriting from this needs to be re-examined to use a class with fully linear convolutions


class KernelSPM3_old(FreqAuxNoiseFxLMS2):
    """Attempts to update each sec path filter using data from all microphones.
    It uses the normal Freq domain Eriksson cost function, together with
    ||e - G_ip y_spm||^2 where G_ip is the sec path interpolated to microphone m
    from all positions except m (to avoid the inteprolation filter reducing to identity)."""

    def __init__(self, mu, muSPM, eta, speakerRIR, blockSize, errorPos):
        super().__init__(
            mu,
            muSPM,
            speakerRIR,
            blockSize,
        )
        self.name = "Freq ANC Kernel SPM 3"
        self.eta = eta
        kernelReg = 1e-3
        self.kernelMat = self.constructInterpolation(errorPos, kernelReg)

        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))
        self.buffers["f"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))
        self.buffers["fip"] = np.zeros((self.numError, self.simChunkSize + s.sim_buffer))
        self.buffers["VIP"] = np.zeros(
            (
                self.numSpeaker,
                self.numError,
                self.numError,
                self.simChunkSize + s.sim_buffer,
            )
        )

        self.metadata["kernelReg"] = kernelReg
        self.metadata["kernelCostWeight"] = self.eta

    def constructInterpolation(self, errorPos, kernelRegParam):
        """Output is shape(numFreq, position interpolated to, positions interpolated from),
        which is (2*blocksize, M, M-1)"""
        kiParams = np.zeros(
            (2 * self.blockSize, self.numError, self.numError), dtype=np.complex128
        )
        for m in range(self.numError):
            kiParams[
                :, m : m + 1, np.arange(self.numError) != m
            ] = soundfieldInterpolation(
                errorPos[m : m + 1, :],
                errorPos[np.arange(self.numError) != m, :],
                2 * self.blockSize,
                kernelRegParam,
            )
            # for m2 in range(self.numError):
            #     if m2 < m:
            #         kiParams[:,m:m+1,m2] = kip[:,:,m2]
            #     elif m2 > m:
            #         kiParams[:,m:m+1,m2]
        return kiParams

    def filterAuxNoise(self, secPath, auxNoise):
        auxNoise = np.fft.fft(auxNoise, axis=-1).T[:, :, None]
        output = secPath @ auxNoise
        output = np.squeeze(np.fft.ifft(output, axis=0), axis=-1).T
        output = output[:, self.blockSize :]
        return np.real(output)

    def interpolateAuxNoise(self):
        V = np.fft.fft(
            self.buffers["v"][:, self.idx - (2 * self.blockSize) : self.idx], axis=-1
        ).T[:, :, None]

        VIP = self.kernelMat[:, None, :, :] * V[:, :, :, None]
        VIP = np.transpose(np.fft.ifft(VIP, axis=0), (1, 2, 3, 0))
        self.buffers["VIP"][:, :, :, self.updateIdx : self.idx] = np.real(
            VIP[:, :, :, self.blockSize :]
        )

        VIP = np.transpose(
            np.fft.fft(
                self.buffers["VIP"][
                    :, :, :, self.idx - (2 * self.blockSize) : self.idx
                ],
                axis=-1,
            ),
            (3, 0, 1, 2),
        )
        return VIP

    def getIPGrad(self):
        Gip = self.kernelMat @ self.secPathEstimate.ir
        vfIP = self.filterAuxNoise(
            Gip, self.buffers["v"][:, self.idx - (2 * self.blockSize) : self.idx]
        )
        self.buffers["fip"][:, self.updateIdx : self.idx] = (
            self.e[:, self.updateIdx : self.idx] - vfIP
        )

        VIP = self.interpolateAuxNoise()
        Fip = np.concatenate(
            (
                np.zeros_like(self.buffers["fip"][:, self.updateIdx : self.idx]),
                self.buffers["fip"][:, self.updateIdx : self.idx],
            ),
            axis=-1,
        )
        Fip = np.fft.fft(Fip, axis=-1).T[:, :, None]

        error2 = np.zeros(
            (2 * self.blockSize, self.numError, self.numSpeaker), dtype=np.complex128
        )
        for m in range(self.numError):
            error2[:, m, :] = np.sum(
                Fip[:, None, :, 0] * VIP[:, :, :, m].conj(), axis=2
            )
            # error2[:,m,:]

        powerPerFreq = np.sum(np.abs(VIP) ** 2, axis=(1, 2, 3))
        VIPnorm = 1 / (np.mean(powerPerFreq) + self.beta)
        ipGrad = self.eta * error2 * VIPnorm
        return ipGrad

    def getNormalGrad_new(self):
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
        # self.secPathEstimate.update(self.buffers["v"][:,self.updateIdx-self.blockSize:self.idx],
        #                            self.buffers["f"][:,self.updateIdx:self.idx])

        # update
        paddedError = np.concatenate(
            (
                np.zeros_like(self.buffers["f"][:, self.updateIdx : self.idx]),
                self.buffers["f"][:, self.updateIdx : self.idx],
            ),
            axis=-1,
        )
        F = np.fft.fft(paddedError, axis=-1).T[:, :, None]
        V = np.fft.fft(
            self.buffers["v"][:, self.idx - (2 * self.blockSize) : self.idx], axis=-1
        ).T[:, :, None]

        gradient = F @ np.transpose(V.conj(), (0, 2, 1))
        # tdgrad = np.fft.ifft(gradient, axis=0)
        # tdgrad[self.blockSize:,:,:] = 0
        # gradient = np.fft.fft(tdgrad, axis=0)

        norm = 1 / (np.mean(np.transpose(V.conj(), (0, 2, 1)) @ V) + self.beta)
        return norm * gradient

    def getNormalGrad(self):
        # vf = self.filterAuxNoise(self.secPathEstimate.ir, self.buffers["v"][:,self.idx-(2*self.blockSize):self.idx])
        vf = self.secPathEstimate.process(
            self.buffers["v"][:, self.idx - (2 * self.blockSize) : self.idx]
        )
        self.buffers["f"][:, self.updateIdx : self.idx] = (
            self.e[:, self.updateIdx : self.idx] - vf
        )

        F = np.concatenate(
            (
                np.zeros_like(self.buffers["f"][:, self.updateIdx : self.idx]),
                self.buffers["f"][:, self.updateIdx : self.idx],
            ),
            axis=-1,
        )
        F = np.fft.fft(F, axis=-1).T[:, :, None]

        V = np.fft.fft(
            self.buffers["v"][:, self.idx - (2 * self.blockSize) : self.idx], axis=-1
        ).T[:, :, None]
        error1 = F @ np.transpose(V.conj(), (0, 2, 1))

        powerPerFreq = np.sum(np.abs(V) ** 2, axis=(1, 2))
        Vnorm = 1 / (np.mean(powerPerFreq) + self.beta)

        normalGrad = error1 * Vnorm
        return normalGrad

    def updateSPM(self):
        grad = self.getNormalGrad() + self.eta * self.getIPGrad()
        grad = np.fft.ifft(grad, axis=0)
        grad[self.blockSize :, :, :] = 0
        grad = np.fft.fft(grad, axis=0)

        self.secPathEstimate.ir += self.muSPM * grad

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.ir)





class FastBlockFxLMSSPMTuninglessArea(FastBlockAuxNoise):
    """As tuningless, but we interpolate over a small area around the microphone to get better estimates
        Tested, not better than FastBlockFxLMSSPMTuningless"""

    def __init__(self, config, mu, beta, speakerRIR, blockSize, invForgetFactor, auxNoisePower, errorPos):
        super().__init__(config, mu, beta, speakerRIR, blockSize, invForgetFactor, auxNoisePower)
        self.name = "Fast Block SPM Tuningless Area"

        self.secPathEstimateSPM = FilterSum_Freqdomain(
            numIn=self.numSpeaker, numOut=self.numError, irLen=blockSize
        )
        self.crossCorr = MovingAverage(
            1 - invForgetFactor,
            (2 * blockSize, self.numError, self.numSpeaker),
            dtype=np.complex128,
        )
        self.metadata["crossCorrForgetFactor"] = self.crossCorr.forgetFactor

        kiRegParam = 1e-3
        self.pointsPerError = 20
        kiLen = 155
        self.kiDly = 155//2
        errorInterpolPos = np.zeros((self.numError, self.pointsPerError, 3))
        errorInterpolPos = errorPos[:,None,:] + self.rng.uniform(low=-0.05,high=0.05,size=(1, self.pointsPerError,3))
        self.kiIR = [soundfieldInterpolationFIR(
            errorInterpolPos[:,i,:], errorPos, kiLen, kiRegParam, 2 * blockSize, 3, self.samplerate, self.c) for i in range(self.pointsPerError)]
        self.kiFilt = [FilterSum_Freqdomain(ir = np.concatenate((self.kiIR[i],np.zeros((
                self.kiIR[0].shape[0], self.kiIR[0].shape[1], self.blockSize-self.kiIR[0].shape[2]))),axis=-1)) for i in range(self.pointsPerError)]

    def updateSPM(self):
        vf = self.secPathEstimateSPM.process(
           self.buffers["v"][:, self.updateIdx : self.idx]
        )
        self.buffers["f"][:, self.updateIdx : self.idx] = (
           self.e[:, self.updateIdx : self.idx] - vf
        )

        crossCorr = np.zeros((self.numError, self.numSpeaker, self.blockSize))
        for i in range(self.pointsPerError):
            eIP = self.kiFilt[i].process(self.e[:, self.updateIdx : self.idx])

            crossCorr += fdf.correlateEuclidianTT(
                eIP,
                self.buffers["v"][:, self.idx - 2 * self.blockSize-self.kiDly : self.idx-self.kiDly],
            )
        crossCorr /= self.pointsPerError
        crossCorr = fdf.fftWithTranspose(
            np.concatenate((crossCorr, np.zeros_like(crossCorr)), axis=-1)
        )

        self.crossCorr.update(crossCorr / self.blockSize)
        self.secPathEstimateSPM.tf = self.crossCorr.state / self.auxNoiseSource.power
        self.secPathEstimate.tf[...] = self.secPathEstimateSPM.tf

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.tf)
        #self.diag.saveBlockData("secpath_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_phase_select_freq", self.idx, self.secPathEstimate.tf)
        #self.diag.saveBufferData("recorded_ir", self.idx, np.real(fdf.ifftWithTranspose(self.secPathEstimateSPM.tf)[...,:self.blockSize]))






class FastBlockFxLMSSMCKIRestricted(FastBlockFxLMS):
    def __init__(self, config, mu, beta, speakerRIR, blockSize, muPrim, muSec, errorPos):
        """Makes sure that hat{y}^f and hat{d} always follows the 
            spatial relationship between the microphones"""
        super().__init__(config, mu, beta, speakerRIR, blockSize)
        self.name = "Fast Block FxLMS SMC w/ NLMS KI Restricted"
        self.muPrim = muPrim
        self.muSec = muSec
        
        self.controlFilt.setIR (self.rng.standard_normal(size=(self.numRef, self.numSpeaker, blockSize)) * 1e-14)
        #ir = np.zeros((self.numRef, self.numSpeaker, blockSize))
        #ir[:,:,0] = 1e-8

        self.secPathNLMS = FastBlockNLMS(self.blockSize, self.numSpeaker, self.numError, self.muSec, self.beta)
        self.primPathNLMS = FastBlockNLMS(self.blockSize, self.numRef, self.numError, self.muPrim, self.beta)

       # self.buffers["yip"] = np.zeros((self.numError, self.numError, self.numSpeaker, self.simChunkSize+self.sim_info.sim_buffer))

        kiLen = 155
        self.kiFilt = self.constructInterpolation(errorPos, 1e-3, kiLen)

        self.kiDly = kiLen // 2
        self.eta = 0.1
        kiIR = np.concatenate((self.kiFilt, np.zeros((
            self.kiFilt.shape[0], self.kiFilt.shape[1], self.blockSize - self.kiFilt.shape[2]))),axis=-1)
        self.kiFilt = FilterSum_Freqdomain(ir=np.moveaxis(kiIR,0,1), dataDims=(self.numSpeaker,))
        #self.kiFiltMDsp = FilterMD_Freqdomain(ir=kiIR, dataDims=(self.numSpeaker,))

        #self.secPathNLMS.filt.setIR(self.rng.standard_normal(size=(self.numSpeaker, self.numError, blockSize)) * 1e-14)
        #self.primPathNLMS.filt.setIR(self.rng.standard_normal(size=(self.numSpeaker, self.numError, blockSize)) * 1e-14)
        
        #self.secPathTrueMD = FilterMD_Freqdomain(tf=np.copy(self.secPathEstimate.tf),dataDims=self.numRef)
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

        # self.diag.add_diagnostic(
        #     "recorded_ir",
        #     diacore.RecordIR(
        #         np.real(fdf.ifftWithTranspose(self.secPathEstimate.tf)[...,:self.blockSize]),
        #         self.sim_info.tot_samples, 
        #         self.sim_info.sim_buffer, 
        #         self.simChunkSize,
        #         (2,2),
        #         plotFrequency=self.plotFrequency,
        #     ),
        # )

        # self.diag.add_diagnostic(
        #     "secpath_combinations",
        #     diacore.ConstantEstimateNMSEAllCombinations(
        #         self.secPathEstimate.tf,
        #         self.sim_info.tot_samples,
        #         self.sim_info.sim_buffer,
        #         self.simChunkSize,
        #         plotFrequency=self.plotFrequency,
        #     ),
        # )

        self.secPathEstimate.tf[...] = np.zeros_like(self.secPathEstimate.tf)

        self.diag.add_diagnostic(
            "modelling_error",
            dia.SignalEstimateNMSEBlock(self.sim_info.tot_samples, self.sim_info.sim_buffer, self.simChunkSize, 
                                     plotFrequency=self.plotFrequency)
        )
        self.diag.add_diagnostic(
            "primary_error",
            dia.SignalEstimateNMSEBlock(self.sim_info.tot_samples, self.sim_info.sim_buffer, self.simChunkSize, 
                                    plotFrequency=self.plotFrequency)
        )
        self.diag.add_diagnostic(
            "secondary_error",
            dia.SignalEstimateNMSEBlock(self.sim_info.tot_samples, self.sim_info.sim_buffer, self.simChunkSize, 
                                    plotFrequency=self.plotFrequency)
        )

        self.metadata["muSec"] = self.muSec
        self.metadata["muPrim"] = self.muPrim

    def constructInterpolation(self, errorPos, kernelRegParam, kiFiltLen):
        """Output is shape(numFreq, position interpolated to, positions interpolated from),
        which is (2*blocksize, M, M-1)"""
        kiFirAll = np.zeros((self.numError, self.numError, kiFiltLen))
        for m in range(self.numError):
            kiFIR = soundfieldInterpolationFIR(
                errorPos[m : m + 1, :],
                errorPos[np.arange(self.numError) != m, :],
                kiFiltLen,
                kernelRegParam,
                4096,
                self.spatialDims,
                self.samplerate,
                self.c,
            )

            kiFirAll[m : m + 1, np.arange(self.numError) != m, :] = kiFIR
        return kiFirAll


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

        tdSecPath = np.real(np.moveaxis(fdf.ifftWithTranspose(self.secPathNLMS.filt.tf),0,1))
        self.kiFilt.buffer.fill(0)
        ipFilt = self.kiFilt.process(tdSecPath[...,:self.blockSize])
        ipFilt[...,:-self.kiDly] = ipFilt[...,self.kiDly:]
        ipFilt = np.concatenate((ipFilt, np.zeros_like(ipFilt)), axis=-1)

        ipError = fdf.fftWithTranspose((tdSecPath - ipFilt))
        self.secPathNLMS.filt.tf -= self.eta*np.moveaxis(ipError,1,2)

        # print("Mean diff, error, kierror")
        # print(np.mean((np.abs(tdSecPath - ipFilt))))
        # print(np.mean(np.abs(tdSecPath)))
        # print(np.mean(np.abs(ipFilt)))

        #self.buffers["yip"][...,self.idx-self.blockSize:self.idx] = self.kiFiltMDsp.process(self.y[:,self.idx-self.blockSize:self.idx])

        modellingError = self.e[:,self.idx-self.blockSize:self.idx] - errorEst
        # secGradTd = fdf.correlateSumTT(modellingError[None,None,:,:], 
        #                             np.moveaxis(self.buffers["yip"][...,self.idx-2*self.blockSize:self.idx],2,1))
        # secGrad = fdf.fftWithTranspose(np.concatenate((secGradTd, np.zeros_like(secGradTd)),axis=-1))
        # secNorm = 1 / (np.sum(self.buffers["yip"][...,self.idx-self.blockSize:self.idx]**2) + self.beta)

        self.primPathNLMS.update(self.x[:,self.idx-2*self.blockSize:self.idx], modellingError)
        self.secPathNLMS.update(self.y[:,self.idx-2*self.blockSize:self.idx], modellingError)
        #self.secPathNLMS.filt.tf += self.secPathNLMS.mu * secNorm * secGrad

        self.secPathEstimate.tf[...] = self.secPathNLMS.filt.tf
        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_select_freq", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("secpath_phase_weighted_select_freq", self.idx, self.secPathEstimate.tf)
        #self.diag.saveBlockData("secpath_combinations", self.idx, self.secPathEstimate.tf)
        self.diag.saveBlockData("modelling_error", self.idx, errorEst, self.e[:,self.idx-self.blockSize:self.idx])
        self.diag.saveBlockData("primary_error", self.idx, primNoiseEst, self.buffers["primary"][:,self.idx-self.blockSize:self.idx])
        self.diag.saveBlockData("secondary_error", self.idx, secNoiseEst, self.buffers["yf"][:,self.idx-self.blockSize:self.idx])
        #self.diag.saveBufferData("recorded_ir", self.idx, np.real(fdf.ifftWithTranspose(self.secPathNLMS.filt.tf)[...,:self.blockSize]))
        #self.spmIdx = self.idx