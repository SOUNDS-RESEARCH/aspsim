import numpy as np
from abc import ABC, abstractmethod

import ancsim.utilities as util
import ancsim.array as ar
from ancsim.signal.filterclasses import (
    Filter_IntBuffer,
    FilterSum,
    FilterMD_IntBuffer,
    FilterSum_Freqdomain,
    FilterMD_Freqdomain
)
import ancsim.signal.filterclasses as fc
from ancsim.adaptivefilter.util import blockProcessUntilIndex, calcBlockSizes
import ancsim.diagnostics.core as diacore
import ancsim.signal.freqdomainfiltering as fdf
import ancsim.signal.filterdesign as fd
import ancsim.processor as afbase
import ancsim.signal.kernelinterpolation as ki




class ActiveNoiseControlProcessor(afbase.AudioProcessor):
    def __init__(self, sim_info, arrays, blockSize, updateBlockSize, **kwargs):
        super().__init__(sim_info, arrays, blockSize, **kwargs)
        self.updateBlockSize = updateBlockSize
        self.numRef = self.arrays["ref"].num
        self.numError = self.arrays["error"].num
        self.numSpeaker = self.arrays["speaker"].num
        
        self.diag.add_diagnostic("noise_reduction", dia.SignalPowerRatio(self.sim_info, "error", "source~error"))
        if "target" in self.arrays:
            self.diag.add_diagnostic("spatial_noise_reduction", 
                dia.SignalPowerRatio(self.sim_info, "target", "source~target", 1024, False))

        self.updateIdx = self.sim_info.sim_buffer
        
        self.metadata["update block size"] = updateBlockSize

    def resetBuffer(self):
        super().resetBuffer()
        self.updateIdx -= self.sim_info.sim_chunk_size

    def process(self, numSamples):
        self.genSpeakerSignals(numSamples)
        if self.updateIdx <= self.idx - self.updateBlockSize:
            self.updateFilter()
            self.updateIdx += self.updateBlockSize

    @abstractmethod
    def genSpeakerSignals(self, numSamples):
        pass

    @abstractmethod
    def updateFilter(self):
        pass





class TDANCProcessor(ActiveNoiseControlProcessor):
    def __init__(self, sim_info, arrays, blockSize, updateBlockSize, controlFiltLen, **kwargs):
        super().__init__(sim_info, arrays, blockSize, updateBlockSize, **kwargs)
        self.filtLen = controlFiltLen
        #self.controlFilter = FilterSum(irLen=self.filtLen, numIn=self.numRef, numOut=self.numSpeaker)
        self.controlFilter = FilterSum(np.zeros((self.numRef, self.numSpeaker, self.filtLen)))

        self.metadata["control filter length"] = controlFiltLen

    def genSpeakerSignals(self, numSamples):
        """"""
        self.sig["speaker"][:,self.idx:self.idx+numSamples] = \
            self.controlFilter.process(self.sig["ref"][:,self.idx-numSamples:self.idx])


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


class BlockFxLMS(TDANCProcessor):
    """Fully block based operation. Is equivalent to FastBlockFxLMS"""
    def __init__(self, config, arrays, blockSize, updateBlockSize, controlFiltLen, mu, beta, secPath, **kwargs):
        super().__init__(config, arrays, blockSize, updateBlockSize, controlFiltLen, **kwargs)
        self.name = "Block FxLMS"
        self.mu = mu
        self.beta = beta

        secPathIr = np.concatenate((np.zeros((secPath.shape[0], secPath.shape[1], self.blockSize)), secPath),axis=-1)
        self.secPathFilt = fc.createFilter(ir=secPathIr, broadcastDim=self.numRef, sumOverInput=False)
        
        #FilterMD_IntBuffer((self.numRef,), secPathIr)
        self.createNewBuffer("xf", (self.numRef, self.numSpeaker, self.numError))


        self.metadata["step size"] = self.mu
        self.metadata["beta"] = self.beta

    def prepare(self):
        super().prepare()
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
    def __init__(self, config, arrays, blockSize, controlFiltLen, mu, beta, secPath, **kwargs):
        super().__init__(config, arrays, blockSize, controlFiltLen, controlFiltLen, **kwargs)
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
        super().prepare()
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
























def reduceIRLength(ir, tolerance, maxSamplesToReduce=np.inf):
    firstNonZeroIndex = 0
    if tolerance > 0:
        maxVal = np.abs(ir).max()
        for i in range(ir.shape[-1]):
            if np.abs(ir[..., i]).max() / maxVal > tolerance:
                firstNonZeroIndex = i
                break

    samplesToReduce = np.min((maxSamplesToReduce, firstNonZeroIndex))
    reducedIR = ir[..., samplesToReduce:]
    return reducedIR, samplesToReduce


class KIFxLMS(TDANCProcessor):
    """KIFxLMS for IWAENC/ICASSP paper.
    Coming from kernel derivation 10, or equivalently 12

    normalization options is 'xf', 'xfApprox', and 'kixf'"""
    def __init__(self, config, arrays, blockSize, updateBlockSize, controlFiltLen, 
                    mu, beta, secPath, errorPos, kiFiltLen, kernelReg, targetRegion, 
                    mcNumPoints, normalization="xf", delayRemovalTolerance=0, **kwargs):

        super().__init__(config, arrays, blockSize, updateBlockSize, controlFiltLen, **kwargs)
        self.name = "KI-FxLMS"
        self.mu = mu
        self.beta = beta

        secPathIr = np.concatenate((np.zeros((secPath.shape[0], secPath.shape[1], self.blockSize)), secPath),axis=-1)
        self.secPathFilt = FilterMD_IntBuffer((self.numRef,), secPathIr)

        freqKernelFilter = ki.getKernelWeightingFilter(
            ki.kernelHelmholtz3d,
            kernelReg,
            errorPos, 
            targetRegion,
            mcNumPoints,
            np.max((4*controlFiltLen, 2048)),
            self.samplerate,
            self.c
        )

        kiFilt,_ = fd.firFromFreqsWindow(freqKernelFilter, kiFiltLen)

        reducedSecPath, numSamplesRemoved = reduceIRLength(
            secPathIr,
            tolerance=delayRemovalTolerance,
            maxSamplesToReduce=kiFilt.shape[-1] // 2,
        )
        print("removed ", numSamplesRemoved, " samples")

        self.kiDelay = kiFilt.shape[-1] // 2 - numSamplesRemoved
        combinedFilt = np.zeros(
            (
                *reducedSecPath.shape[0:-1],
                reducedSecPath.shape[-1] + kiFilt.shape[-1] - 1,
            )
        )
        for i in range(self.numError):
            for j in range(self.numError):
                for k in range(reducedSecPath.shape[0]):
                    combinedFilt[k, j, :] += np.convolve(
                        reducedSecPath[k, i, :], kiFilt[i, j, :], "full"
                    )

        self.kixfFilt = FilterMD_IntBuffer(dataDims=self.numRef, ir=combinedFilt)
        self.createNewBuffer("kixf", (self.numRef, self.numSpeaker, self.numError))


        # CREATE NORMALIZATION FILTERS:
        if normalization == "xf":
            self.secPathEstimate = FilterMD_IntBuffer(
                dataDims=self.numRef, ir=secPathIr
            )
            self.createNewBuffer("xf", (self.numRef, self.numSpeaker, self.numError))
            self.normFunc = self.xfNormalization

        # elif normalization == "xfApprox":
        #     self.normFunc = self.xfApproxNormalization
        #     self.sig["xfnorm"] = np.zeros((1, self.simChunkSize + self.sim_info.sim_buffer))
        #     self.secPathNormFilt = Filter_IntBuffer(
        #         np.sum(self.secPathFilt.ir ** 2, axis=(0, 1))
        #     )
        elif normalization == "kixf":
            self.normFunc = self.kixfNormalization
        else:
            raise ValueError("Invalid normalization option")

        self.metadata["stepsize"] = self.mu
        self.metadata["beta"] = self.beta
        self.metadata["kernelInterpolationDelay"] = int(self.kiDelay)
        self.metadata["samplesRemovedFromSecPath"] = int(numSamplesRemoved)
        self.metadata["delayRemovalTolerance"] = delayRemovalTolerance
        self.metadata["kiFiltLength"] = kiFiltLen
        #self.metadata["mcPointGen"] = mcPointGen.__name__
        #self.metadata["volume of interpolated region"] = ipVolume
        self.metadata["mcNumPoints"] = mcNumPoints

    def kixfNormalization(self):
        return 1 / (
            np.sum(
                self.sig["kixf"][
                    :, :, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
                ]
                ** 2
            )
            + self.beta
        )

    def xfApproxNormalization(self):
        self.sig["xfnorm"][
            :, self.updateIdx : self.idx
        ] = self.secPathNormFilt.process(
            np.sum(self.sig["ref"][:, self.updateIdx : self.idx] ** 2, axis=0, keepdims=True)
        )

        return 1 / (
            np.sum(self.sig["xfnorm"][:, self.updateIdx : self.idx]) + self.beta
        )

    def xfNormalization(self):
        self.sig["xf"][
            ..., self.updateIdx : self.idx
        ] = np.moveaxis(self.secPathEstimate.process(self.sig["ref"][:, self.updateIdx : self.idx]), 2, 0)
        xfPower = np.sum(self.sig["xf"][:, :, :, self.updateIdx : self.idx] ** 2)
        # print("Algo with blocksize ", self.bs, "xf power: ", xfPower)
        return 1 / (xfPower + self.beta)

    def prepare(self):
        self.sig["kixf"][:, :, :, : self.idx] = np.transpose(
            self.kixfFilt.process(self.sig["ref"][:, : self.idx]), (2, 0, 1, 3)
        )
        self.sig["xf"][..., : self.idx] = np.moveaxis(self.secPathEstimate.process(
            self.sig["ref"][:, : self.idx]), 2, 0)

    def updateFilter(self):
        self.sig["kixf"][:, :, :, self.updateIdx : self.idx] = np.transpose(
            self.kixfFilt.process(self.sig["ref"][:, self.updateIdx : self.idx]), (2, 0, 1, 3)
        )

        grad = np.zeros_like(self.controlFilter.ir)
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(
                self.sig["kixf"][:, :, :, n - self.filtLen + 1 : n + 1], axis=-1
            )
            grad += np.sum(Xf * self.sig["error"][None, None, :, n - self.kiDelay, None], axis=2)

        norm = self.normFunc()
        self.controlFilter.ir -= grad * self.mu * norm
        #self.updateIdx = self.idx


class FastBlockKIFxLMS(TDANCProcessor):
    """kernelFilt is the impulse resposnse of kernel interpolation filter A in the time domain
    The filter is attained by monte carlo integration of the frequency domain filter from Ito-sans paper
    and then IFFT-> truncate -> window

    It is the frequency domain version of tdkernelderivation 10, or equivalently, 12"""

    def __init__(
        self,
        config, arrays, blockSize, controlFiltLen, 
        mu,
        beta,
        secPath,
        errorPos,
        kiFiltLen,
        kernelReg,
        roi,
        mcNumPoints=100,
        delayRemovalTolerance=0,
        **kwargs,
    ):
        super().__init__(config, arrays, blockSize, controlFiltLen, controlFiltLen, **kwargs)
        self.mu = mu
        self.beta = beta
        self.name = "Fast Block KI-FxLMS"

        self.createNewBuffer("xf", (self.numRef, self.numSpeaker, self.numError))
        self.createNewBuffer("kixf", (self.numRef, self.numSpeaker, self.numError))

        assert kiFiltLen % 1 == 0
        kiFilt = ki.getKernelWeightingFilter(
            ki.kernelHelmholtz3d,
            kernelReg,
            errorPos,
            roi,
            mcNumPoints,
            4096,
            self.samplerate,
            self.c,
        )
        kiFilt,_ = fd.firFromFreqsWindow(kiFilt, kiFiltLen)
        assert kiFilt.shape[-1] <= self.updateBlockSize

        secPath = np.concatenate((np.zeros((secPath.shape[0], secPath.shape[1], self.blockSize)), 
                                secPath), axis=-1)
        secPath, numSamplesRemoved = reduceIRLength(
            secPath,
            tolerance=delayRemovalTolerance,
            maxSamplesToReduce=kiFilt.shape[-1] // 2,
        )
        secPath = np.concatenate((secPath,
                                    np.zeros((secPath.shape[0], secPath.shape[1], 2*self.filtLen-secPath.shape[-1]))),
                                    axis=-1)
        secPath[...,self.filtLen:] = 0
        secPathTf = fdf.fftWithTranspose(secPath)
        self.secPathFilt = FilterMD_Freqdomain(dataDims=self.numRef, tf=secPathTf)

        # self.secPathEstimate = FilterMD_Freqdomain(
        #     dataDims=self.numRef,
        #     ir=np.concatenate(
        #         (
        #             reducedSecPath,
        #             np.zeros((reducedSecPath.shape[:-1] + (blockSize-reducedSecPath.shape[-1],))),
        #             #np.zeros((reducedSecPath.shape[:-1] + (numSamplesRemoved,))),
        #         ),
        #         axis=-1,
        #     ),
        # )

        self.kiDelay = kiFilt.shape[-1] // 2 - numSamplesRemoved
        
        self.kiFilt = FilterSum_Freqdomain(
            tf=fdf.fftWithTranspose(kiFilt, n=2*self.filtLen),
            dataDims=(self.numSpeaker, self.numRef),
        )

        self.normFunc = self.xfNormalization

        self.metadata["step size"] = self.mu
        self.metadata["beta"] = self.beta
        self.metadata["kernelInterpolationDelay"] = int(self.kiDelay)
        self.metadata["samplesRemovedFromSecPath"] = int(numSamplesRemoved)
        self.metadata["delayRemovalTolerance"] = delayRemovalTolerance
        self.metadata["kiFiltLength"] = int(kiFilt.shape[-1])
        self.metadata["roi"] = roi.__class__.__name__
        self.metadata["mcNumPoints"] = mcNumPoints
        self.metadata["roiVolume"] = roi.volume
        self.metadata["normalization"] = self.normFunc.__name__

    def kixfNormalization(self):
        return 1 / (
            np.sum(
                self.sig["kixf"][
                    :, :, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
                ]
                ** 2
            )
            + self.beta
        )

    def xfNormalization(self):
        return 1 / (
            np.sum(self.sig["xf"][:, :, :, self.updateIdx : self.idx] ** 2)
            + self.beta
        )

    def updateFilter(self):
        startIdx = self.updateIdx
        endIdx = self.updateIdx + self.updateBlockSize

        self.sig["xf"][:,:,:,startIdx:endIdx] = np.moveaxis(self.secPathFilt.process(self.sig["ref"][:,  startIdx:endIdx]), 2, 0)

        self.sig["kixf"][
            :, :, :, startIdx - self.kiDelay : endIdx - self.kiDelay
        ] = np.moveaxis(self.kiFilt.process(
            np.transpose(
                self.sig["xf"][:, :, :,  startIdx:endIdx], (1, 0, 2, 3)
            )), 1, 0)

        tdGrad = fdf.correlateSumTT(
            self.sig["error"][
                None, None, :, startIdx - self.kiDelay : endIdx - self.kiDelay
            ],
            self.sig["kixf"][:,:,:,
                endIdx
                - (2 * self.updateBlockSize)
                - self.kiDelay : endIdx
                - self.kiDelay,
            ],
        )
        norm = self.xfNormalization()
        self.controlFilter.ir -= self.mu * norm * tdGrad





class FastBlockDKIFxLMS(FastBlockFxLMS):
    """ Fast Block KIFxLMS using directional kernel"""
    def __init__(
        self,
        config,
        mu,
        beta,
        speakerRIR,
        blockSize,
        errorPos,
        kiFiltLen,
        kernelReg,
        angle,
        directionWeight,
        roi,
        mcNumPoints,
        delayRemovalTolerance=0,
        **kwargs, 
    ):
        super().__init__(config, mu, beta, speakerRIR, blockSize, **kwargs)
        self.name = "Fast Block Directional KIFxLMS"
        assert kiFiltLen % 1 == 0
        assert kiFiltLen <= blockSize
        kiFilt = ki.getKernelWeightingFilter(
            ki.kernelDirectional3d,
            kernelReg,
            errorPos,
            roi,
            mcNumPoints,
            4096,
            self.samplerate,
            self.c,
            angle,
            directionWeight
        )
        kiFilt,_ = fd.firFromFreqsWindow(kiFilt, kiFiltLen)
        

        reducedSecPath, numSamplesRemoved = reduceIRLength(
            np.transpose(self.secPathFilt.ir, (1, 0, 2)),
            tolerance=delayRemovalTolerance,
            maxSamplesToReduce=kiFilt.shape[-1] // 2,
        )
        self.secPathEstimate = FilterMD_Freqdomain(
            dataDims=self.numRef,
            ir=np.concatenate(
                (
                    reducedSecPath,
                    np.zeros((reducedSecPath.shape[:-1] + (blockSize-reducedSecPath.shape[-1],))),
                    #np.zeros((reducedSecPath.shape[:-1] + (numSamplesRemoved,))),
                ),
                axis=-1,
            ),
        )

        self.kiDelay = kiFilt.shape[-1] // 2 - numSamplesRemoved
        self.sig["kixf"] = np.zeros(
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

        self.normFunc = self.xfNormalization

        self.metadata["kernelInterpolationDelay"] = int(self.kiDelay)
        self.metadata["samplesRemovedFromSecPath"] = int(numSamplesRemoved)
        self.metadata["delayRemovalTolerance"] = delayRemovalTolerance
        self.metadata["kiFiltLength"] = int(kiFilt.shape[-1])
        self.metadata["roi"] = roi.__class__.__name__
        self.metadata["mcNumPoints"] = mcNumPoints
        self.metadata["ipVolume"] = roi.volume
        self.metadata["normalization"] = self.normFunc.__name__
        self.metadata["angle"] = angle
        self.metadata["directionalWeight"] = directionWeight

    def kixfNormalization(self):
        return 1 / (
            np.sum(
                self.sig["kixf"][
                    :, :, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
                ]
                ** 2
            )
            + self.beta
        )

    def xfNormalization(self):
        return 1 / (
            np.sum(self.sig["xf"][:, :, :, self.updateIdx : self.idx] ** 2)
            + self.beta
        )

    def updateFilter(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False

        self.sig["xf"][
            :, :, :, self.updateIdx : self.idx
        ] = self.secPathEstimate.process(self.sig["ref"][:, self.updateIdx : self.idx])

        self.sig["kixf"][
            :, :, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
        ] = self.kiFilt.process(
            np.transpose(
                self.sig["xf"][:, :, :, self.updateIdx : self.idx], (1, 2, 0, 3)
            )
        )

        tdGrad = fdf.correlateSumTT(
            self.e[
                None, None, :, self.updateIdx - self.kiDelay : self.idx - self.kiDelay
            ],
            self.sig["kixf"][
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
        norm = self.xfNormalization()
        # powerPerFreq = np.sum(np.abs(xfFreq)**2,axis=(1,2,3))
        # norm = 1 / (np.mean(powerPerFreq) + self.beta)

        self.controlFilt.tf -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True
