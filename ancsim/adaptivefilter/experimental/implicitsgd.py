import numpy as np

from ancsim.adaptivefilter.base import AdaptiveFilterFF
from ancsim.signal.filterclasses import FilterSum_IntBuffer


class ImplicitMPC(AdaptiveFilterFF):
    def __init__(self, config, mu, beta, speakerRIR):
        self.name = "Implicit Update FF MPC"
        super().__init__(config, mu, beta, speakerRIR)
        self.buffers["yfilt"] = np.zeros(
            (self.numError, self.simChunkSize + self.sim_info.sim_buffer)
        )

    def updateFilter(self):
        vecLen = self.numRef * self.numSpeaker * self.filtLen
        # grad = np.zeros_like(vecLen, self.numError)

        wVec = np.zeros((vecLen, 1))
        for u in range(self.numRef):
            for l in range(self.numSpeaker):
                for i in range(self.filtLen):
                    wVec[
                        u * (self.filtLen * self.numSpeaker) + l * self.filtLen + i, 0
                    ] = self.H[u, l, i]

        for n in range(self.updateIdx, self.idx):
            xf = np.zeros((vecLen, self.numError))
            for u in range(self.numRef):
                for l in range(self.numSpeaker):
                    for i in range(self.filtLen):
                        xf[
                            u * (self.filtLen * self.numSpeaker) + l * self.filtLen + i,
                            :,
                        ] = self.xf[u, l, :, n - i]

            mat = np.sum(xf[:, None, :] * xf[None, :, :], axis=-1)
            matInv = np.linalg.pinv(np.eye(vecLen) + self.mu * mat)
            factor2 = wVec - self.mu * np.sum(
                xf * (self.e[:, n : n + 1].T - self.buffers["yfilt"][:, n : n + 1].T),
                axis=-1,
                keepdims=True,
            )

            wVec = matInv @ factor2

        for u in range(self.numRef):
            for l in range(self.numSpeaker):
                for i in range(self.filtLen):
                    self.H[u, l, i] = wVec[
                        u * (self.filtLen * self.numSpeaker) + l * self.filtLen + i, :
                    ]
        self.updateIdx = self.idx

    def forwardPassImplement(self, numSamples, noiseAtError, noiseAtRef, errorMicNoise):
        for i in range(numSamples):
            n = self.idx + i
            self.x[:, n] = np.squeeze(noiseAtRef[:, i : i + 1])
            X = np.flip(self.x[:, n - self.filtLen + 1 : n + 1], axis=-1)
            self.y[:, n] = np.sum(X[:, None, :] * self.H, axis=(0, -1))

        yf = self.secPathErrorFilt.process(self.y[:, self.idx : self.idx + numSamples])
        self.buffers["yfilt"][:, self.idx : self.idx + numSamples] = yf
        self.e[:, self.idx : self.idx + numSamples] = noiseAtError + yf + errorMicNoise

        self.xf[:, :, :, self.idx : self.idx + numSamples] = np.transpose(
            self.secPathXfFilt.process(self.x[:, self.idx : self.idx + numSamples]),
            (2, 0, 1, 3),
        )
