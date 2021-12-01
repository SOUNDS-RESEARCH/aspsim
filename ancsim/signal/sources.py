import numpy as np
import numba as nb
import pathlib
import scipy.signal as spsig
import soundfile as sf
from abc import ABC, abstractmethod

import ancsim.utilities as util



class SourceArray:
    def __init__(self, sourceType, numSources, amplitude, *args):
        if isinstance(amplitude, (int, float)):
            amplitude = [amplitude for _ in range(numSources)]

        self.sources = [sourceType(amplitude[i], *args) for i in range(numSources)]

    def getSamples(self, numSamples):
        output = np.concatenate(
            [src.getSamples(numSamples) for src in self.sources], axis=0
        )
        return output


class Source(ABC):
    def __init__(self, num_channels, power, rng=None):
        self.num_channels = num_channels
        self.power = power

        if rng is None:
            self.rng = np.random.default_rng(1)
        else:
            self.rng = rng

        self.metadata = {}
        self.metadata["number of channels"] = self.num_channels
        self.metadata["power"] = self.power
    
    @abstractmethod
    def getSamples(self, numSamples):
        return np.zeros((self.num_channels, numSamples))

class SineSource(Source):
    def __init__(self, num_channels, power, freq, samplerate):
        super().__init__(num_channels, power)
        self.amplitude = np.sqrt(2 * self.power)
        # p = a^2 / 2
        # 2p = a^2
        # sqrt(2p) = a
        self.freq = freq
        self.samplerate = samplerate
        self.phase = self.rng.uniform(low=0, high=2 * np.pi, size=(self.num_channels,1))

        self.phasePerSample = 2 * np.pi * self.freq / self.samplerate

        self.metadata["frequency"] = self.freq

    def getSamples(self, numSamples):
        
        noise = (
            self.amplitude
            * np.cos(self.phasePerSample * np.arange(numSamples)[None,:] + self.phase)
        )
        self.phase = (self.phase + numSamples * self.phasePerSample) % (2 * np.pi)
        return noise

class MultiSineSource(Source):
    def __init__(self, num_channels, power, freq, samplerate):
        super().__init__(num_channels, power)
        if num_channels > 1:
            raise NotImplementedError
        if isinstance(power, (list, tuple, np.ndarray)):
            raise NotImplementedError
        if isinstance(freq, (list, tuple)):
            freq = np.array(freq)
        assert freq.ndim == 1
        self.freq = freq
        self.samplerate = samplerate
        self.numSines = len(freq)
        self.phase = self.rng.uniform(low=0, high=2 * np.pi, size=self.numSines)
        self.phasePerSample = 2 * np.pi * self.freq / self.samplerate
        self.amplitude = np.sqrt(2 * self.power / self.numSines)

    def getSamples(self, numSamples):
        noise = np.zeros((1, numSamples))
        for i in range(self.numSines):
            noise += (
                self.amplitude
                * np.cos(
                    np.arange(numSamples) * self.phasePerSample[i] + self.phase[i]
                )[None, :]
            )
            self.phase[i] = (self.phase[i] + numSamples * self.phasePerSample[i]) % (
                2 * np.pi
            )
        return noise



class WhiteNoiseSource(Source):
    def __init__(self, num_channels, power):
        super().__init__(num_channels, power)
        self.setPower(power)

    def getSamples(self, numSamples):
        return self.rng.normal(
            loc=0, scale=self.stdDev, size=(numSamples, self.num_channels)
        ).T
    
    def setPower(self, newPower):
        self.power = newPower
        self.stdDev = np.sqrt(newPower)

        if isinstance(self.power, np.ndarray):
            assert self.power.ndim == 1
            assert self.power.shape[0] == self.num_channels or \
                    self.power.shape[0] == 1


class BandlimitedNoiseSource(Source):
    def __init__(self, num_channels, power, freqLim, samplerate):
        super().__init__(num_channels, power)
        assert len(freqLim) == 2
        assert freqLim[0] < freqLim[1]
        #if num_channels > 1:
        #    raise NotImplementedError
        self.num_channels = num_channels

        wp = [freq for freq in freqLim]
        self.filtCoef = spsig.butter(16, wp, btype="bandpass", output="sos", fs=samplerate)
        testSig = spsig.sosfilt(self.filtCoef, self.rng.normal(size=(self.num_channels,10000)))
        self.testSigPow = np.mean(testSig ** 2)
        self.zi = spsig.sosfilt_zi(self.filtCoef)
        self.zi = np.tile(np.expand_dims(self.zi,1), (1,self.num_channels,1))

        self.setPower(power)
        
        self.metadata["frequency span"] = freqLim

    def getSamples(self, numSamples):
        noise = self.amplitude * self.rng.normal(size=(self.num_channels, numSamples))
        filtNoise, self.zi = spsig.sosfilt(self.filtCoef, noise, zi=self.zi, axis=-1)
        return filtNoise

    def setPower(self, newPower):
        self.power = newPower
        self.amplitude = np.sqrt(newPower / self.testSigPow)

        # if isinstance(self.power, np.ndarray):
        #     assert self.power.ndim == 1
        #     assert self.power.shape[0] == self.num_channels or \
        #             self.power.shape[0] == 1


class Counter(Source):
    def __init__(self, num_channels, start_number=0):
        super().__init__(num_channels, np.nan)
        if num_channels > 1:
            raise NotImplementedError
        self.current_number = start_number

    def getSamples(self, num_samples):
        values = np.arange(self.current_number, self.current_number+num_samples)
        self.current_number += num_samples
        return values




# class BandlimitedNoiseSource:
#     def __init__(self, amplitude, freqLim, samplerate):
#         assert len(freqLim) == 2
#         assert freqLim[0] < freqLim[1]
#         self.amplitude = amplitude
#         self.num_channels = 1
#         self.rng = np.random.default_rng(1)

#         wp = [freq for freq in freqLim]

#         self.filtCoef = spsig.butter(
#             16, wp, btype="bandpass", output="sos", fs=samplerate
#         )

#         testSig = spsig.sosfilt(self.filtCoef, self.rng.normal(size=10000))
#         testSigPow = np.mean(testSig ** 2)
#         self.normFactor = np.sqrt(0.5) / np.sqrt(testSigPow)

#         self.zi = spsig.sosfilt_zi(self.filtCoef)

#     def getSamples(self, numSamples):
#         noise = self.rng.normal(size=numSamples) * self.amplitude
#         filtNoise, self.zi = spsig.sosfilt(self.filtCoef, noise, zi=self.zi)
#         filtNoise *= self.normFactor
#         return filtNoise[None, :]


# class SineSource(Source):
#     def __init__(self, amplitude, freq, samplerate):
#         #super().__init__(num_channels, power)
#         self.amp = amplitude
#         self.freq = freq
#         self.samplerate = samplerate
#         rng = np.random.RandomState(1)
#         self.phase = rng.uniform(low=0, high=2 * np.pi)

#     def getSamples(self, numSamples):
#         phasePerSample = 2 * np.pi * self.freq / self.samplerate
#         noise = (
#             self.amp
#             * np.cos(np.arange(numSamples) * phasePerSample + self.phase)[None, :]
#         )
#         self.phase = (self.phase + numSamples * phasePerSample) % (2 * np.pi)
#         return noise





class LinearChirpSource:
    def __init__(self, amplitude, freqRange, samplesToSweep, samplerate):
        assert len(freqRange) == 2
        self.amp = amplitude
        self.samplerate = samplerate
        self.freqRange = freqRange

        self.samplesToSweep = samplesToSweep
        self.deltaFreq = (freqRange[1] - freqRange[0]) / samplesToSweep
        self.freq = freqRange[0]
        self.freqCounter = 0
        rng = np.random.default_rng(1)
        self.phase = rng.uniform(low=0, high=2 * np.pi)

    def nextPhase(self):
        self.phase += (self.freq / self.samplerate) + (
            self.deltaFreq / (2 * self.samplerate)
        )
        self.phase = self.phase % 1
        self.freq = self.freq + self.deltaFreq

    def getSamples(self, numSamples):
        noise = np.zeros((1, numSamples))
        samplesLeft = numSamples
        n = 0
        for blocks in range(1 + numSamples // self.samplesToSweep):
            blockSize = min(samplesLeft, self.samplesToSweep - self.freqCounter)
            for _ in range(blockSize):
                noise[0, n] = np.cos(2 * np.pi * self.phase)
                self.nextPhase()

                n += 1
                self.freqCounter += 1

                if self.samplesToSweep <= self.freqCounter:
                    self.deltaFreq = -self.deltaFreq
                    # if self.deltaFreq > 0:
                    #    self.freq = self.freqRange[0]
                    # else:
                    #    self.freq = self.freqRange[1]
                    self.freqCounter = 0
            samplesLeft -= blockSize
        noise *= self.amp
        return noise


class AudioFileSource:
    def __init__(self, amplitude, samplerate, filename, endTimestep, verbose=False):
        # filename = "noise_bathroom_fan.wav"
        # filename = "song_assemble.wav"
        self.amplitude = amplitude
        self.samplerate = samplerate
        self.audioSamples, srAudio = sf.read(filename)
        if self.audioSamples.ndim == 2:
            self.audioSamples = self.audioSamples[:, 0]
        self.currentSample = 0

        if srAudio != samplerate:
            assert srAudio % samplerate == 0
            assert srAudio > samplerate
            downsamplingFactor = srAudio // samplerate
            self.audioSamples = spsig.resample_poly(
                self.audioSamples, up=1, down=downsamplingFactor, axis=-1
            )
            # srAudio = srAudio // downsamplingFactor
            if verbose:
                print(
                    "AudioFileSource downsampled audio file from ",
                    srAudio,
                    " to ",
                    srAudio // downsamplingFactor,
                    " Hz",
                )

        # assert(srAudio == samplerate)
        assert endTimestep < self.audioSamples.shape[0]

    def getSamples(self, numSamples):
        sig = self.audioSamples[
            None, self.currentSample : self.currentSample + numSamples
        ]
        self.currentSample += numSamples
        return sig * self.amplitude


class MLS:
    """Generates a maximum length sequence"""
    def __init__(self, order, polynomial, state=None):
        self.order = order
        self.poly = polynomial
        self.state = None

    def getSamples(self, numSamples):
        seq, self.state = spsig.max_len_seq(
            self.order, state=self.state, length=numSamples, taps=self.poly
        )
        return seq * 2 - 1


class GoldSequenceSource:
    preferredSequences = {
        5: [[2], [1, 2, 3]],
        6: [[5], [1, 4, 5]],
        7: [[4], [4, 5, 6]],
        8: [[1, 2, 3, 6, 7], [1, 2, 7]],
        9: [[5], [3, 5, 6]],
        10: [[2, 5, 9], [3, 4, 6, 8, 9]],
        11: [[9], [3, 6, 9]],
    }

    def __init__(self, order, power=np.ones((1, 1)), num_channels=1):
        assert num_channels <= 2 ** order - 1
        self.num_channels = num_channels
        self.seqLength = 2 ** order - 1
        self.idx = 0
        self.setPower(power)

        if 5 <= order <= 11:
            mls1 = MLS(order, self.preferredSequences[order][0])
            mls2 = MLS(order, self.preferredSequences[order][1])
            seq1 = mls1.getSamples(2 ** order - 1)
            seq2 = mls2.getSamples(2 ** order - 1)
        elif order > 11:
            assert order % 2 == 1
            mls1 = MLS(order, None)
            seq1 = mls1.getSamples(2**order - 1)
            k = util.getSmallestCoprime(order)
            decimationFactor = int(2**k + 1)
            decimatedIndices = (np.arange(self.seqLength) * decimationFactor) % self.seqLength
            seq2 = np.copy(seq1[decimatedIndices])
            


        self.sequences = np.zeros((num_channels, self.seqLength))
        for i in range(num_channels):
            self.sequences[i, :] = np.negative(
                seq1 * np.roll(seq2, i)
            )  # XOR with integer shifts

    def getSamples(self, numSamples):
        outSignal = np.zeros((self.num_channels, numSamples))
        blockLengths = util.calcBlockSizes(numSamples, self.idx, self.seqLength)
        outIdx = 0
        for blockLen in blockLengths:
            outSignal[:, outIdx : outIdx + blockLen] = self.sequences[
                :, self.idx : self.idx + blockLen
            ]
            outIdx += blockLen
            self.idx = (self.idx + blockLen) % self.seqLength
        return outSignal * self.amplitude

    def setPower(self, newPower):
        if isinstance(newPower, (int, float)) or newPower.ndim == 0:
            self.power = np.ones((1,1)) * newPower
        elif newPower.ndim == 1:
            self.power = newPower[:, None]
        elif newPower.ndim == 2:
            assert newPower.shape[-1] == 1
            self.power = newPower
        else:
            raise ValueError
        self.amplitude = np.sqrt(self.power)
        

# class WhiteNoiseSource:
#     def __init__(self, power, num_channels=1):
#         self.num_channels = num_channels
#         self.setPower(power)
#         self.rng = np.random.default_rng(1)

#     def getSamples(self, numSamples):
#         return self.rng.normal(
#             loc=0, scale=self.stdDev, size=(numSamples, self.num_channels)
#         ).T
    
#     def setPower(self, newPower):
#         self.power = newPower
#         self.stdDev = np.sqrt(newPower)

#         if isinstance(self.power, np.ndarray):
#             assert self.power.ndim == 1
#             assert self.power.shape[0] == self.num_channels or \
#                     self.power.shape[0] == 1

# class WhiteNoiseSource_old:
#     def __init__(self, power, num_channels=1):
#         self.num_channels = num_channels
#         self.power = power
#         self.stdDev = np.sqrt(power)
#         self.rng = np.random.RandomState(1)

#     def getSamples(self, numSamples):
#         return self.rng.normal(
#             loc=0, scale=self.stdDev, size=(self.num_channels, numSamples)
#     )
    
#     def setPower(newPower):
#         self.power = newPower
#         self.stdDev = np.sqrt(newPower)



