import numpy as np

from ancsim.adaptivefilter.mpc import FxLMS_FF
from ancsim.signal.filterclasses import Filter_IntBuffer


class Kernel12(FxLMS_FF):
    def __init__(self, config, mu, beta, speakerFilters, kernelFilt):
        super().__init__(config, mu, beta, speakerFilters)
        self.name = "Kernel 12"
        # self.kernelFilt = kernelFilt
        self.M = kernelFilt.shape[-1] // 2
        self.combinedFilt = np.zeros(
            (
                self.secPathFilt.ir.shape[0],
                self.secPathFilt.ir.shape[1],
                self.secPathFilt.ir.shape[-1] + kernelFilt.shape[-1] - 1,
            )
        )
        for i in range(kernelFilt.shape[0]):
            for j in range(kernelFilt.shape[0]):
                for k in range(self.secPathFilt.ir.shape[0]):
                    self.combinedFilt[k, j, :] += np.convolve(
                        self.secPathFilt.ir[k, i, :], kernelFilt[i, j, :], "full"
                    )

        relLim = 1e-3
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
        self.buffers["xfnorm"] = np.zeros((1, s.SIMBUFFER + self.simChunkSize))

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
