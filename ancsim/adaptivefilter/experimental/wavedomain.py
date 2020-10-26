import numpy as np
import scipy.signal.windows as spwin

from ancsim.adaptivefilter.base import AdaptiveFilterFFComplex
import ancsim.soundfield.wavedomain as wd
import ansim.settings as s


class WaveDomainFF(AdaptiveFilterFFComplex):
    def __init__(self, mu, beta, speakerFilters, blockSize, errorMicPos, speakerPos):
        super().__init__(mu, beta, speakerFilters, blockSize)
        self.numHarm = (np.min((errorMicPos.shape[0], speakerPos.shape[0])) - 1) // 2
        self.blockSize = blockSize
        self.Te, self.TeInv = wd.waveTransformCircularMicArray(
            errorMicPos, self.numHarm
        )
        self.Tl, self.TlInv = wd.waveTransformSpeakerArray(
            speakerPos, errorMicPos, self.numHarm, 2 * blockSize
        )

        self.secPathEstimate = np.zeros(
            (2 * self.blockSize, self.numHarm, self.numHarm), dtype=np.complex128
        )
        self.H = np.zeros(
            (2 * self.blockSize, self.numHarm, self.numHarm), dtype=np.complex128
        )

        self.window = spwin.hamming(2 * self.blockSize, sym=False)[None, :]

    def getCMatrix(self, center, width):
        pass

    def updateFilter_ref(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False

        self.E = np.fft.fft(self.e[:, self.updateIdx : self.idx], axis=-1)
        grad = self.G.conj() @ self.E @ np.transpose(self.X.conj(), (0, 2, 1))

        # norm = 1 / (np.sum(np.abs(self.G)**2) * np.sum(np.abs(self.X)**2) + self.beta)
        norm = 1 / (
            (
                np.sum(np.abs(self.G) ** 2)
                * np.sum(np.abs(self.X) ** 2)
                / (2 * self.blockSize) ** 2
            )
            + self.beta
        )
        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True

    def updateFilter(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False
        self.E = np.fft.fft(
            self.e[:, self.updateIdx - self.blockSize : self.idx] * self.window, axis=-1
        )
        self.X = np.fft.fft(
            self.x[:, self.updateIdx - self.blockSize : self.idx] * self.window, axis=-1
        )

    def updateSPM(self):
        pass

    def forwardPassImplement(self, numSamples):
        assert numSamples == self.blockSize
        self.X = np.fft.fft(
            self.x[:, self.idx - self.blockSize : self.idx + self.blockSize]
            * self.window,
            axis=-1,
        ).T[:, :, None]
        self.Xw = self.Te

        self.Y = self.H @ self.X

        y = np.fft.ifft(self.Y, axis=0)
        if np.mean(np.abs(np.imag(y))) > 0.00001:
            print(
                "WARNING: Frequency domain adaptive filter truncates significant imaginary component"
            )

        y = np.squeeze(np.real(y), axis=-1).T
        y = y[:, self.blockSize :]
        self.y[:, self.idx : self.idx + self.blockSize] = y
        self.updated = False
