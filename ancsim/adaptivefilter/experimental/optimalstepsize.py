import numpy as np

from ancsim.adaptivefilter.base import AdaptiveFilterFF
from ancsim.adaptivefilter.util import equivalentDelay


class MPC_FF_postErrorNorm(AdaptiveFilterFF):
    def __init__(self, config, mu, beta, speakerRIR):
        super().__init__(config, mu, beta, speakerRIR)
        self.name = "MPC FF posterior error minimization normalization"

    def updateFilter(self):
        grad = np.zeros_like(self.H)
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(self.xf[:, :, :, n - self.filtLen + 1 : n + 1], axis=-1)
            # grad += np.sum(Xf * self.e[None, None,:,n,None],axis=2)
            grad = np.sum(Xf * self.e[None, None, :, n, None], axis=2)

            norm = np.sum(grad ** 2) / (
                self.beta
                + np.sum(np.sum(grad[:, :, None, :] * Xf, axis=(0, 1, 3)) ** 2)
            )
            self.H -= grad * norm * self.mu
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


class MPC_FF_OptimalStepSize(AdaptiveFilterFF):
    def __init__(self, config, beta, speakerRIR):
        super().__init__(config, 1, beta, speakerRIR)
        self.name = "MPC FF Optimal Step Size"
        D = equivalentDelay(secPathError)
        if config["SOURCETYPE"] == "noise":
            Bw = config["NOISEBANDWIDTH"] / self.samplerate
        else:
            Bw = 1

        self.normFactor = np.sum(secPathError ** 2) * (
            self.H.shape[-1] + (2 / Bw) * np.max(D)
        )

    def updateFilter(self):
        grad = np.zeros_like(self.H)
        sigmax = (
            np.mean(self.x[:, self.updateIdx - self.filtLen + 1 : self.idx + 1] ** 2)
            + self.beta
        )

        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(self.xf[:, :, :, n - self.filtLen + 1 : n + 1], axis=-1)
            grad += np.sum(Xf * self.e[None, None, :, n, None], axis=2)
        self.H -= grad / (sigmax * self.normFactor)
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


class MPC_FF_OptimalStepSize2(AdaptiveFilterFF):
    def __init__(self, config, beta, speakerRIR, blockSize):
        super().__init__(config, 1, beta, speakerRIR)
        self.name = "MPC FF Optimal Step Size more intelligent sigmax estimation"
        D = equivalentDelay(secPathError)
        if config["SOURCETYPE"] == "noise":
            Bw = config["NOISEBANDWIDTH"] / self.samplerate
        else:
            Bw = 1

        self.sigmax = None
        self.forgetFactor = self.calcForgetFactor(blockSize / self.samplerate, 0.1)
        self.normFactor = np.sum(secPathError ** 2) * (
            self.H.shape[-1] + (2 / Bw) * np.max(D)
        )

    def calcForgetFactor(self, secPerUpdate, secToForget, forgetLim=0.1):
        return np.exp(np.log(forgetLim) * secPerUpdate / secToForget)

    def updateSigmax(self):
        if self.sigmax is None:
            self.sigmax = np.mean(
                self.x[:, self.updateIdx - self.filtLen + 1 : self.idx + 1] ** 2
            )
        else:
            self.sigmax *= self.forgetFactor
            self.sigmax += (1 - self.forgetFactor) * np.mean(
                self.x[:, self.updateIdx - self.filtLen + 1 : self.idx + 1] ** 2
            )

    def updateFilter(self):
        grad = np.zeros_like(self.H)
        self.updateSigmax()

        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(self.xf[:, :, :, n - self.filtLen + 1 : n + 1], axis=-1)
            grad += np.sum(Xf * self.e[None, None, :, n, None], axis=2)
        self.H -= grad / ((self.sigmax * self.normFactor) + self.beta)
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


class MPC_FF_OptimalStepSize3(AdaptiveFilterFF):
    def __init__(self, config, beta, speakerRIR, blockSize):
        super().__init__(config, 1, beta, speakerRIR)
        self.name = "MPC FF Optimal Step Size"
        D = equivalentDelay(secPathError)
        if config["SOURCETYPE"] == "noise":
            Bw = config["NOISEBANDWIDTH"] / self.samplerate
        else:
            Bw = 1

        self.sigmax = None
        self.forgetFactor = self.calcForgetFactor(blockSize / self.samplerate, 0.1)
        self.normFactor = np.sum(secPathError ** 2) * (
            self.H.shape[-1] + (2 / Bw) * np.max(D)
        )

    def calcForgetFactor(self, secPerUpdate, secToForget, forgetLim=0.1):
        return np.exp(np.log(forgetLim) * secPerUpdate / secToForget)

    def updateSigmax(self):
        if self.sigmax is None:
            self.sigmax = np.mean(
                self.x[:, self.updateIdx + 1 : self.idx + 1] ** 2, axis=(-1)
            )
        else:
            self.sigmax *= self.forgetFactor
            self.sigmax += (1 - self.forgetFactor) * np.mean(
                self.x[:, self.updateIdx + 1 : self.idx + 1] ** 2, axis=-1
            )

    def updateFilter(self):
        grad = np.zeros_like(self.H)
        self.updateSigmax()

        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(self.xf[:, :, :, n - self.filtLen + 1 : n + 1], axis=-1)
            grad += np.sum(Xf * self.e[None, None, :, n, None], axis=2)
        self.H -= grad / ((self.sigmax[:, None, None] * self.normFactor) + self.beta)
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
