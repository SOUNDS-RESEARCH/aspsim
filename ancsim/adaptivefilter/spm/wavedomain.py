import ancsim.soundfield.wavedomain as wd

from ancsim.signal.sources import WhiteNoiseSource


class UnconstrainedFreqAuxNoiseFxLMS(ConstrainedFastBlockFxLMS):
    """The idea is to take a more straightforward block processing approach.
    It will not be equivalent to it time domain analogue, but will probably
    be more usable, if more complex frequency domain processing is needed

    Not finished"""

    def __init__(self, mu, muSPM, speakerFilters, blockSize):
        self.beta = 1e-3
        super().__init__(mu, self.beta, speakerFilters, blockSize)
        self.name = "SPM Aux Noise Freq Domain Unconstrained"
        self.muSPM = muSPM
        # self.auxNoiseSource = GoldSequenceSource(11, power=2, num_channels=self.numSpeaker)
        self.auxNoiseSource = WhiteNoiseSource(power=3, num_channels=self.numSpeaker)

        self.secPathEstimate = NLMS_FREQ(
            numFreq=2 * blockSize,
            numIn=self.numSpeaker,
            numOut=self.numError,
            stepSize=muSPM,
        )
        self.G = np.transpose(self.G, (0, 2, 1))
        self.buffers["v"] = np.zeros((self.numSpeaker, self.simChunkSize + s.sim_buffer))

        self.diag.add_diagnostic(
            "secpath",
            diacore.ConstantEstimateNMSE(
                self.G, self.sim_info.tot_samples, plotFrequency=self.plotFrequency
            ),
        )
        self.window = win.hamming(2 * self.blockSize, sym=False)[None, :]

        self.metaData

    def forwardPassImplement(self, numSamples):
        super().forwardPassImplement(numSamples)
        v = self.auxNoiseSource.get_samples(numSamples)
        self.buffers["v"][:, self.idx : self.idx + numSamples] = v
        self.y[:, self.idx : self.idx + numSamples] += v

    def updateSPM(self):
        self.E = np.fft.fft(
            self.e[:, self.updateIdx - self.blockSize : self.idx] * self.window, axis=-1
        ).T[:, :, None]

        self.V = np.fft.fft(
            self.buffers["v"][:, self.updateIdx - self.blockSize : self.idx]
            * self.window,
            axis=-1,
        ).T[:, :, None]

        self.F = (
            self.E - self.secPathEstimate.ir @ self.V
        )  # self.buffers["vEst"][:,self.updateIdx:self.idx]

        self.secPathEstimate.update(self.V, self.F)

        self.diag.saveBlockData("secpath", self.idx, self.secPathEstimate.ir)

    def updateFilter(self):
        assert self.idx - self.updateIdx == self.blockSize
        assert self.updated == False

        self.updateSPM()

        self.F = self.E - self.secPathEstimate.ir @ self.V
        grad = (
            np.transpose(self.secPathEstimate.ir.conj(), (0, 2, 1))
            @ self.F
            @ np.transpose(self.X.conj(), (0, 2, 1))
        )

        # tdGrad = np.fft.ifft(grad, axis=0)
        # tdGrad[self.blockSize:,:] = 0
        # grad = np.fft.fft(tdGrad, axis=0)

        powerPerFreq = np.sum(
            np.abs(self.secPathEstimate.ir) ** 2, axis=(1, 2)
        ) * np.sum(np.abs(self.X) ** 2, axis=(1, 2))
        norm = 1 / (np.mean(powerPerFreq) + self.beta)
        self.H -= self.mu * norm * grad

        self.updateIdx += self.blockSize
        self.updated = True
