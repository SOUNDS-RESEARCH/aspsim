import numpy as np
import ancsim.utilities as util
from ancsim.adaptivefilter.base import AdaptiveFilterFF
from ancsim.adaptivefilter.mpc import FxLMS_FF
from ancsim.signal.filterclasses import FilterMD_IntBuffer, Filter_IntBuffer
from ancsim.signal.filterclasses import FilterSum


# Derivation 10 but with simple heuristic norm, using the reference filtered through secondary paths
# utilizes the silence in the beginning of the secondary paths to lessen the extra delays.
class KernelIP10_minimumdelay(FxLMS_FF):
    def __init__(self, config, mu, beta, speakerRIR, tdA, delayRemovalTolerance=1e-3):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "Kernel IP 10 min delay"
        self.A = np.transpose(tdA, (1, 2, 0))
        self.M = self.A.shape[-1] // 2
        self.combinedFilt = np.zeros(
            (
                self.secPathFilt.ir.shape[0],
                self.secPathFilt.ir.shape[1],
                self.secPathFilt.ir.shape[-1] + tdA.shape[0] - 1,
            )
        )
        for i in range(tdA.shape[-1]):
            for j in range(tdA.shape[-1]):
                for k in range(self.secPathFilt.ir.shape[0]):
                    self.combinedFilt[k, j, :] += np.convolve(
                        self.secPathFilt.ir[k, i, :], tdA[:, i, j], "full"
                    )

        if delayRemovalTolerance > 0:
            relLim = delayRemovalTolerance
            maxVal = np.abs(self.combinedFilt).max()
            for i in range(self.combinedFilt.shape[-1]):
                if np.abs(self.combinedFilt[:, :, i]).max() / maxVal > relLim:
                    firstNonZeroIndex = i
                    break

            samplesToShift = np.min((self.M, firstNonZeroIndex))
            self.M -= samplesToShift
            self.combinedFilt = self.combinedFilt[:, :, samplesToShift:]

        self.secPathXfFilt.setIR(self.combinedFilt)
        self.secPathNormFilt = Filter_IntBuffer(
            np.sum(self.secPathFilt.ir ** 2, axis=(0, 1))
        )
        self.buffers["xfnorm"] = np.zeros((1, self.simChunkSize + self.sim_info.sim_buffer))

    def prepare(self):
        self.buffers["xf"][:, :, :, 0 : self.idx] = np.transpose(
            self.secPathXfFilt.process(self.x[:, 0 : self.idx]), (2, 0, 1, 3)
        )

        self.buffers["xfnorm"][:, 0 : self.idx] = self.secPathNormFilt.process(
            np.sum(self.x[:, 0 : self.idx] ** 2, axis=0, keepdims=True)
        )

    # @util.measure("Update K10")
    def updateFilter(self):
        self.buffers["xf"][:, :, :, self.updateIdx : self.idx] = np.transpose(
            self.secPathXfFilt.process(self.x[:, self.updateIdx : self.idx]),
            (2, 0, 1, 3),
        )

        self.buffers["xfnorm"][
            :, self.updateIdx : self.idx
        ] = self.secPathNormFilt.process(
            np.sum(self.x[:, self.updateIdx : self.idx] ** 2, axis=0, keepdims=True)
        )

        grad = np.zeros_like(self.controlFilt.ir)
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(
                self.buffers["xf"][:, :, :, n - self.filtLen + 1 : n + 1], axis=-1
            )
            refPower = np.sum(
                self.buffers["xfnorm"][:, n - self.filtLen - self.M + 1 : n + 1]
            )
            grad += np.sum(Xf * self.e[None, None, :, n - self.M, None], axis=2) / (
                refPower + self.beta
            )

        self.controlFilt.ir -= grad * self.mu
        self.updateIdx = self.idx


class KernelIP10_interpolatederror(FxLMS_FF):
    def __init__(self, config, mu, beta, speakerFilters, tdA):
        super().__init__(config, mu, beta, speakerFilters)
        self.name = "Kernel IP 10 interpolated error"

        self.kernelFilt = FilterSum(np.transpose(tdA, (1, 2, 0)))
        self.M = self.kernelFilt.ir.shape[-1] // 2

        self.secPathXfFilt.setIR(speakerFilters["error"])
        self.secPathNormFilt = Filter_IntBuffer(
            np.sum(self.secPathFilt.ir ** 2, axis=(0, 1))
        )
        self.buffers["xfnorm"] = np.zeros((1, self.simChunkSize + self.sim_info.sim_buffer))

    def prepare(self):
        self.buffers["xf"][:, :, :, 0 : self.idx] = np.transpose(
            self.secPathXfFilt.process(self.x[:, 0 : self.idx]), (2, 0, 1, 3)
        )

        self.buffers["xfnorm"][:, 0 : self.idx] = self.secPathNormFilt.process(
            np.sum(self.x[:, 0 : self.idx] ** 2, axis=0, keepdims=True)
        )

    # @util.measure("Update K10")
    def updateFilter(self):
        self.buffers["xf"][:, :, :, self.updateIdx : self.idx] = np.transpose(
            self.secPathXfFilt.process(self.x[:, self.updateIdx : self.idx]),
            (2, 0, 1, 3),
        )

        self.buffers["xfnorm"][
            :, self.updateIdx : self.idx
        ] = self.secPathNormFilt.process(
            np.sum(self.x[:, self.updateIdx : self.idx] ** 2, axis=0, keepdims=True)
        )

        ipError = self.kernelFilt.process(self.e[:, self.updateIdx : self.idx])

        grad = np.zeros_like(self.controlFilt.ir)
        for i, n in enumerate(range(self.updateIdx, self.idx)):
            Xf = np.flip(
                self.buffers["xf"][
                    :, :, :, n - self.filtLen + 1 - self.M : n + 1 - self.M
                ],
                axis=-1,
            )
            refPower = np.sum(
                self.buffers["xfnorm"][:, n - self.filtLen - self.M + 1 : n + 1]
            )

            grad += np.sum(Xf * ipError[None, None, :, i, None], axis=2) / (
                refPower + self.beta
            )

        self.controlFilt.ir -= grad * self.mu
        self.updateIdx = self.idx


# Derivation 10 but with simple heuristic norm, using the reference filtered through secondary paths
class KernelIP10_xfnorm(AdaptiveFilterFF):
    def __init__(self, config, mu, beta, speakerRIR, tdA):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "Kernel IP 10 xfnorm"
        self.A = np.transpose(tdA, (1, 2, 0))
        # self.Afilt = FilterSum(self.A)
        self.combinedFilt = np.zeros(
            (
                secPathError.shape[0],
                secPathError.shape[1],
                secPathError.shape[-1] + tdA.shape[0] - 1,
            )
        )
        for i in range(tdA.shape[-1]):
            for j in range(tdA.shape[-1]):
                for k in range(secPathError.shape[0]):
                    self.combinedFilt[k, j, :] += np.convolve(
                        secPathError[k, i, :], tdA[:, i, j], "full"
                    )
        self.secPathXfFilt.setIR(self.combinedFilt)

        self.secPathNormFilt = Filter_IntBuffer(np.sum(secPathError ** 2, axis=(0, 1)))
        self.buffers["xfnorm"] = np.zeros((1, self.simChunkSize + self.sim_info.sim_buffer))

        # self.xfFilt = FilterMD_IntBuffer((self.numRef,), secPathError)
        # self.buffers["xfnormal"] = np.zeros((self.numRef, self.numSpeaker, self.numError, self.simChunkSize+self.sim_info.sim_buffer))

    def updateFilter(self):
        grad = np.zeros_like(self.H)
        M = self.A.shape[-1] // 2
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(self.xf[:, :, :, n - self.filtLen + 1 : n + 1], axis=-1)

            # refPower = np.sum(self.buffers["xfnormal"][:,:,:,n-self.filtLen-M+1:n-M+1]**2)
            # print("Old: ", refPower)
            refPower = np.sum(
                self.buffers["xfnorm"][:, n - self.filtLen - M + 1 : n + 1]
            )
            # print("New: ", refPower)
            grad += np.sum(Xf * self.e[None, None, :, n - M, None], axis=2) / (
                refPower + self.beta
            )
        self.H -= grad * self.mu
        self.updateIdx = self.idx

    def forwardPassImplement(self, numSamples, noiseAtError, noiseAtRef, errorMicNoise):
        for i in range(numSamples):
            n = self.idx + i
            self.x[:, n] = np.squeeze(noiseAtRef[:, i : i + 1])
            X = np.flip(self.x[:, n - self.filtLen + 1 : n + 1], axis=-1)
            self.y[:, n] = np.sum(X[:, None, :] * self.H, axis=(0, -1))

        yf = self.secPathErrorFilt.process(self.y[:, self.idx : self.idx + numSamples])
        self.e[:, self.idx : self.idx + numSamples] = noiseAtError + yf + errorMicNoise

        self.xf[:, :, :, self.idx : self.idx + numSamples] = np.transpose(
            self.secPathXfFilt.process(self.x[:, self.idx : self.idx + numSamples]),
            (2, 0, 1, 3),
        )

        self.buffers["xfnorm"][
            :, self.idx : self.idx + numSamples
        ] = self.secPathNormFilt.process(
            np.sum(
                self.x[:, self.idx : self.idx + numSamples] ** 2, axis=0, keepdims=True
            )
        )
        # self.buffers["xfnormal"][:,:,:,self.idx:self.idx+numSamples] = np.transpose(self.xfFilt.process(
        #                                                self.x[:,self.idx:self.idx+numSamples]), (2,0,1,3))


# Derivation 10 but with simple heuristic norm using the kernel filtered reference signal
class KernelIP10(AdaptiveFilterFF):
    def __init__(self, config, mu, beta, speakerRIR, tdA):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "Kernel IP 10"
        self.A = np.transpose(tdA, (1, 2, 0))
        # self.Afilt = FilterSum(self.A)
        self.combinedFilt = np.zeros(
            (
                secPathError.shape[0],
                secPathError.shape[1],
                secPathError.shape[-1] + tdA.shape[0] - 1,
            )
        )
        for i in range(tdA.shape[-1]):
            for j in range(tdA.shape[-1]):
                for k in range(secPathError.shape[0]):
                    self.combinedFilt[k, j, :] += np.convolve(
                        secPathError[k, i, :], tdA[:, i, j], "full"
                    )
        self.secPathXfFilt.setIR(self.combinedFilt)
        # self.secPathXfFilt = FilterMD_IntBuffer((self.numRef,), combinedFilt)

    # @util.measure("Kernel 10 update")
    def updateFilter(self):
        grad = np.zeros_like(self.H)
        # M = self.A.shape[-1]//2

        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(self.xf[:, :, :, n - self.filtLen + 1 : n + 1], axis=-1)

            grad += np.sum(
                Xf * self.e[None, None, :, n - self.A.shape[-1] // 2, None], axis=2
            ) / (np.sum(Xf ** 2) + self.beta)
        self.H -= grad * self.mu
        self.updateIdx = self.idx

    # @util.measure("Kernel 10 forward pass")
    def forwardPassImplement(self, numSamples, noiseAtError, noiseAtRef, errorMicNoise):
        for i in range(numSamples):
            n = self.idx + i
            self.x[:, n] = np.squeeze(noiseAtRef[:, i : i + 1])
            X = np.flip(self.x[:, n - self.filtLen + 1 : n + 1], axis=-1)
            self.y[:, n] = np.sum(X[:, None, :] * self.H, axis=(0, -1))

        yf = self.secPathErrorFilt.process(self.y[:, self.idx : self.idx + numSamples])
        self.e[:, self.idx : self.idx + numSamples] = noiseAtError + yf + errorMicNoise

        self.xf[:, :, :, self.idx : self.idx + numSamples] = np.transpose(
            self.secPathXfFilt.process(self.x[:, self.idx : self.idx + numSamples]),
            (2, 0, 1, 3),
        )


# Derivation 10
class KernelIP10_postErrorNorm(AdaptiveFilterFF):
    def __init__(self, config, mu, beta, speakerRIR, tdA):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "Kernel IP 10 posterior error normalization"
        self.A = np.transpose(tdA, (1, 2, 0))
        # self.Afilt = FilterSum(self.A)
        self.combinedIR = np.zeros(
            (
                secPathError.shape[0],
                secPathError.shape[1],
                secPathError.shape[-1] + tdA.shape[0] - 1,
            )
        )
        for i in range(tdA.shape[-1]):
            for j in range(tdA.shape[-1]):
                for k in range(secPathError.shape[0]):
                    self.combinedIR[k, j, :] += np.convolve(
                        secPathError[k, i, :], tdA[:, i, j], "full"
                    )

        self.xKernelFilt = FilterMD_IntBuffer((self.numRef,), ir=self.combinedIR)
        self.buffers["xKern"] = np.zeros(
            (
                self.numRef,
                self.numSpeaker,
                self.numError,
                self.simChunkSize + self.sim_info.sim_buffer,
            )
        )

    def updateFilter(self):
        grad = np.zeros_like(self.H)

        self.buffers["xKern"][:, :, :, self.updateIdx : self.idx] = np.transpose(
            self.xKernelFilt.process(self.x[:, self.updateIdx : self.idx]), (2, 0, 1, 3)
        )
        eOffset = self.A.shape[-1] // 2

        for n in range(self.updateIdx, self.idx):
            xKern = np.flip(
                self.buffers["xKern"][:, :, :, n - self.filtLen + 1 : n + 1], axis=-1
            )
            currentGrad = np.sum(
                xKern * self.e[None, None, :, n - eOffset, None], axis=2
            )

            Xf = np.flip(
                self.xf[:, :, :, n - eOffset - self.filtLen + 1 : n - eOffset + 1],
                axis=-1,
            )
            P = np.sum(Xf * currentGrad[:, :, None, :], axis=(0, 1, 3))
            norm = np.sum(self.e[:, n - eOffset] * P) / (np.sum(P ** 2) + self.beta)

            grad += currentGrad * norm
        self.H -= grad * self.mu
        self.updateIdx = self.idx

    def forwardPassImplement(self, numSamples, noiseAtError, noiseAtRef, errorMicNoise):
        for i in range(numSamples):
            n = self.idx + i
            self.x[:, n] = np.squeeze(noiseAtRef[:, i : i + 1])
            X = np.flip(self.x[:, n - self.filtLen + 1 : n + 1], axis=-1)
            self.y[:, n] = np.sum(X[:, None, :] * self.H, axis=(0, -1))

        yf = self.secPathErrorFilt.process(self.y[:, self.idx : self.idx + numSamples])
        self.e[:, self.idx : self.idx + numSamples] = noiseAtError + yf + errorMicNoise

        self.xf[:, :, :, self.idx : self.idx + numSamples] = np.transpose(
            self.secPathXfFilt.process(self.x[:, self.idx : self.idx + numSamples]),
            (2, 0, 1, 3),
        )


# Derivation 10
class KernelIP_tdSimple_newnormMoreConservative(AdaptiveFilterFF):
    def __init__(self, config, mu, beta, speakerRIR, tdA):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "Kernel IP Timedomain simple newnorm more conversavite"
        self.A = np.transpose(tdA, (1, 2, 0))
        # self.Afilt = FilterSum(self.A)
        self.combinedFilt = np.zeros(
            (
                secPathError.shape[0],
                secPathError.shape[1],
                secPathError.shape[-1] + tdA.shape[0] - 1,
            )
        )
        for i in range(tdA.shape[-1]):
            for j in range(tdA.shape[-1]):
                for k in range(secPathError.shape[0]):
                    self.combinedFilt[k, j, :] += np.convolve(
                        secPathError[k, i, :], tdA[:, i, j], "full"
                    )
        self.secPathXfFilt.setIR(self.combinedFilt)

    def updateFilter(self):
        grad = np.zeros_like(self.H)
        alpha = 0.3
        # norm = 0

        eOffset = self.A.shape[-1] // 2
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(self.xf[:, :, :, n - self.filtLen + 1 : n + 1], axis=-1)

            currentGrad = np.sum(Xf * self.e[None, None, :, n - eOffset, None], axis=2)
            conservativeGrad = 2 * currentGrad * np.sum(Xf ** 2, axis=2)

            norm = np.sum(currentGrad ** 2) / (
                np.sum(np.sum(currentGrad[:, :, None, :] * Xf, axis=(0, 1, 3)) ** 2)
                + self.beta
            )
            grad += ((1 - alpha) * currentGrad + alpha * conservativeGrad) * norm
        self.H -= grad * self.mu
        self.updateIdx = self.idx

    def forwardPassImplement(self, numSamples, noiseAtError, noiseAtRef, errorMicNoise):
        for i in range(numSamples):
            n = self.idx + i
            self.x[:, n] = np.squeeze(noiseAtRef[:, i : i + 1])
            X = np.flip(self.x[:, n - self.filtLen + 1 : n + 1], axis=-1)
            self.y[:, n] = np.sum(X[:, None, :] * self.H, axis=(0, -1))

        yf = self.secPathErrorFilt.process(self.y[:, self.idx : self.idx + numSamples])
        self.e[:, self.idx : self.idx + numSamples] = noiseAtError + yf + errorMicNoise

        self.xf[:, :, :, self.idx : self.idx + numSamples] = np.transpose(
            self.secPathXfFilt.process(self.x[:, self.idx : self.idx + numSamples]),
            (2, 0, 1, 3),
        )
