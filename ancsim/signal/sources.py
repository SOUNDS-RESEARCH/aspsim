from enum import auto
import numpy as np
import scipy.signal as spsig
import scipy.linalg as splin
import soundfile as sf
from abc import ABC, abstractmethod

import ancsim.utilities as util
import ancsim.signal.correlation as cr
import ancsim.signal.filterclasses as fc

class Source(ABC):
    def __init__(self, num_channels, rng=None):
        self.num_channels = num_channels

        if rng is None:
            self.rng = np.random.default_rng(123456)
        else:
            self.rng = rng

        self.metadata = {}
        self.metadata["number of channels"] = self.num_channels
        #self.metadata["power"] = self.power
    
    @abstractmethod
    def get_samples(self, num_samples):
        return np.zeros((self.num_channels, num_samples))

class SourceArray:
    def __init__(self, sourceType, numSources, amplitude, *args):
        if isinstance(amplitude, (int, float)):
            amplitude = [amplitude for _ in range(numSources)]

        self.sources = [sourceType(amplitude[i], *args) for i in range(numSources)]

    def get_samples(self, num_samples):
        output = np.concatenate(
            [src.get_samples(num_samples) for src in self.sources], axis=0
        )
        return output






#SOURCES
class SineSource(Source):
    def __init__(self, num_channels, power, freq, samplerate, rng=None):
        super().__init__(num_channels, rng)
        self.power = power
        self.amplitude = np.sqrt(2 * self.power)
        # p = a^2 / 2
        # 2p = a^2
        # sqrt(2p) = a
        self.freq = freq
        self.samplerate = samplerate
        self.phase = self.rng.uniform(low=0, high=2 * np.pi, size=(self.num_channels,1))

        self.phasePerSample = 2 * np.pi * self.freq / self.samplerate

        if isinstance(self.power, (np.ndarray)):
            self.metadata["power"] = self.power.tolist()
        else:
            self.metadata["power"] = self.power
        self.metadata["frequency"] = self.freq

    def get_samples(self, num_samples):
        noise = (
            self.amplitude
            * np.cos(self.phasePerSample * np.arange(num_samples)[None,:] + self.phase)
        )
        self.phase = (self.phase + num_samples * self.phasePerSample) % (2 * np.pi)
        return noise

class MultiSineSource(Source):
    def __init__(self, num_channels, power, freq, samplerate, rng=None):
        super().__init__(num_channels, rng)
        if num_channels > 1:
            raise NotImplementedError
        if isinstance(power, (list, tuple, np.ndarray)):
            raise NotImplementedError
        if isinstance(freq, (list, tuple)):
            freq = np.array(freq)
        assert freq.ndim == 1
        self.power = power
        self.freq = freq
        self.samplerate = samplerate
        self.numSines = len(freq)
        self.phase = self.rng.uniform(low=0, high=2 * np.pi, size=self.numSines)
        self.phasePerSample = 2 * np.pi * self.freq / self.samplerate
        self.amplitude = np.sqrt(2 * self.power / self.numSines)

        if isinstance(self.power, (np.ndarray)):
            self.metadata["power"] = self.power.tolist()
        else:
            self.metadata["power"] = self.power
        self.metadata["frequency"] = self.freq

    def get_samples(self, num_samples):
        noise = np.zeros((1, num_samples))
        for i in range(self.numSines):
            noise += (
                self.amplitude
                * np.cos(
                    np.arange(num_samples) * self.phasePerSample[i] + self.phase[i]
                )[None, :]
            )
            self.phase[i] = (self.phase[i] + num_samples * self.phasePerSample[i]) % (
                2 * np.pi
            )
        return noise


class WhiteNoiseSource(Source):
    def __init__(self, num_channels, power, rng=None):
        super().__init__(num_channels, rng)
        self.set_power(power)

        if isinstance(self.power, (np.ndarray)):
            self.metadata["power"] = self.power.tolist()
        else:
            self.metadata["power"] = self.power

    def get_samples(self, num_samples):
        return self.rng.normal(
            loc=0, scale=self.stdDev, size=(num_samples, self.num_channels)
        ).T
    
    def set_power(self, newPower):
        self.power = newPower
        self.stdDev = np.sqrt(newPower)

        if isinstance(self.power, np.ndarray):
            assert self.power.ndim == 1
            assert self.power.shape[0] == self.num_channels or \
                    self.power.shape[0] == 1


class BandlimitedNoiseSource(Source):
    def __init__(self, num_channels, power, freqLim, samplerate, rng=None):
        super().__init__(num_channels, rng)
        assert len(freqLim) == 2
        assert freqLim[0] < freqLim[1]
        self.num_channels = num_channels

        wp = [freq for freq in freqLim]
        self.filtCoef = spsig.butter(16, wp, btype="bandpass", output="sos", fs=samplerate)

        testSig = spsig.sosfilt(self.filtCoef, self.rng.normal(size=(self.num_channels,10000)))
        self.testSigPow = np.mean(testSig ** 2)
        self.set_power(power)

        self.zi = spsig.sosfilt_zi(self.filtCoef)
        self.zi = np.tile(np.expand_dims(self.zi,1), (1,self.num_channels,1))

        if isinstance(self.power, (np.ndarray)):
            self.metadata["power"] = self.power.tolist()
        else:
            self.metadata["power"] = self.power
        self.metadata["frequency span"] = freqLim

    def get_samples(self, num_samples):
        noise = self.amplitude * self.rng.normal(size=(self.num_channels, num_samples))
        filtNoise, self.zi = spsig.sosfilt(self.filtCoef, noise, zi=self.zi, axis=-1)
        return filtNoise

    def set_power(self, newPower):
        self.power = newPower
        self.amplitude = np.sqrt(newPower / self.testSigPow)

        # if isinstance(self.power, np.ndarray):
        #     assert self.power.ndim == 1
        #     assert self.power.shape[0] == self.num_channels or \
        #             self.power.shape[0] == 1


class Counter(Source):
    """
    Counts from start_number and adds one per sample
        used primarily for debugging
    """
    def __init__(self, num_channels, start_number=0):
        super().__init__(num_channels)
        if num_channels > 1:
            raise NotImplementedError
        self.current_number = start_number

        self.metadata["start number"] = start_number 

    def get_samples(self, num_samples):
        values = np.arange(self.current_number, self.current_number+num_samples)
        self.current_number += num_samples
        return values


class PulseTrain(Source):
    def __init__(self, num_channels, amplitude, period_len, delay=0):
        super().__init__(num_channels)

        if isinstance(amplitude, (tuple, list, np.ndarray)):
            assert len(amplitude) == self.num_channels
            self.amplitude = np.array(amplitude)
        else:
            self.amplitude = np.ones(self.num_channels) * amplitude

        if isinstance(period_len, (tuple, list, np.ndarray)):
            assert len(period_len) == self.num_channels
            self.period_len = np.array(period_len)
        else:
            self.period_len = np.ones(self.num_channels)*period_len

        if isinstance(delay, (tuple ,list, np.ndarray)):
            assert len(delay) == self.num_channels
            self.delay = np.array(delay)
        else:
            self.delay = np.ones(self.num_channels) * delay

        self.idx = np.zeros(self.num_channels)
        self.idx -= self.delay

    def get_samples(self, num_samples):
        sig = np.zeros((self.num_channels, num_samples))

        for i in range(num_samples):
            add_pulse = np.mod(self.idx, self.period_len) == 0
            sig[add_pulse,i] = self.amplitude[add_pulse]
            self.idx += 1
        return sig





class AudioSource(Source):
    def __init__(self, audio, amp_factor = 1, end_mode = "repeat"):
        """
        audio is np.ndarray of audio samples to be played. 
                    Shape is (num_channels, num_samples)
        end_mode can be 'repeat' or 'zeros', which selects whether the 
                    sound should repeat indefinitely or just be 0 
                    after the supplied samples are over
        """
        assert isinstance(audio, np.ndarray)
        if audio.ndim == 1:
            audio = audio[None,:]
        elif audio.ndim != 2:
            raise ValueError
        super().__init__(audio.shape[0])

        self.audio = audio
        self.amp_factor = amp_factor
        self.end_mode = end_mode
        self.tot_samples = audio.shape[1]
        self.current_sample = 0

    def get_samples(self, num_samples):
        sig = np.zeros((self.num_channels, num_samples))
        if self.end_mode == "repeat":
            block_lengths = util.calcBlockSizes(num_samples, self.current_sample, self.tot_samples)
            i = 0
            for block_len in block_lengths:
                sig[:,i:i+block_len] = self.audio[:,self.current_sample:self.current_sample+block_len]
                self.current_sample = (self.current_sample + block_len) % self.tot_samples
                i += block_len
        elif self.end_mode == "zeros":
            raise NotImplementedError
        elif self.end_mode == "raise":
            if self.current_sample + num_samples >= self.tot_samples:
                raise StopIteration("End of audio signal")

            sig = self.audio[:,self.current_sample:self.current_sample+num_samples]
            self.current_sample += num_samples
            return sig
        else:
            raise ValueError("Invalid end mode")

        # try:
        #     sig = self.audio[
        #         :, self.current_sample : self.current_sample + num_samples
        #     ]
        # except IndexError:
        #     sig = np.zeros((self.num_channels, num_samples))
        #     num_end = self.tot_samples - self.current_sample
        #     sig[:,:num_end] = self.audio[:,self.current_sample:self.current_sample+num_end]
        #     if self.end_mode == "repeat":
        #         num_start = num_samples - num_end
        #         sig[:,num_end:] = self.audio[:,:num_start]
        #         self.current_sample = num_start
        #     elif self.end_mode == "zeros":
        #         pass
        #     else:
        #         raise ValueError("Invalid end mode")
                
        #self.current_sample += num_samples
        return sig * self.amp_factor

def load_audio_file(file_name, desired_sr = None, verbose=False):
    audio, audio_sr = sf.read(file_name)
    raise NotImplementedError("Check which axis is audio and which is channel. \
                                Make support for multiple channels (should be simple)")
    if audio.ndim == 2:
        audio = audio[:, 0]

    if desired_sr != audio_sr:
        assert audio_sr % desired_sr == 0
        assert audio_sr > desired_sr
        downsampling_factor = audio_sr // desired_sr
        audio = spsig.resample_poly(
            audio, up=1, down=downsampling_factor, axis=-1
        )
        # srAudio = srAudio // downsamplingFactor
        if verbose:
            print(f"AudioFileSource downsampled audio file from \
                    {audio_sr} to {audio_sr // downsampling_factor} Hz")

        return audio






def ar_coeffs_from_autocorr(autocorr):
    """
    returns the parameter vector a = [a_1, ..., a_p]
        that implements the AR system
        y(n) = a_1 y(n-1) + a_2 y(n-2) + ... + a_p y(n-p) + s(n)

        where s(n) is a white noise process with variance awgn_power

    """
    assert autocorr.ndim <= 2
    if autocorr.ndim == 2:
        assert autocorr.shape[0] == 1
        autocorr = autocorr[0,:]
    ar_coeffs = splin.solve_toeplitz(autocorr[:-1], autocorr[1:])
    noise_power = autocorr[0] - np.sum(autocorr[1:] * ar_coeffs)
    return ar_coeffs, noise_power

def ar_coeffs_to_transfer_function(ar_coeffs):
    """
    Gives the rational transfer function that will give rise to 
        the provided AR process, if a white noise is filtered
        through the returned filter.

    According to scipy.signal.lfilter which implements the system
        a_0 y(n) = b_0 x(n) + b_1 x(n-1) + ... + b_t x(n-t) 
                    -a_1 y(n-1) - a_2 y(n-2) - ... - a_p y(n-p)
    """

    denom_coeffs = np.concatenate(([1], -ar_coeffs))
    num_coeffs = np.array([1], dtype=float)
    return num_coeffs, denom_coeffs 

class AutocorrSource(Source):
    """

    autocorr is array of shape (num_channels, corr_len)
        with autocorr[:,0] being lag 0. 

    All channels are all independent. 
    Could be extended in the future to accept autocorr of shape 
        (num_channels, num_channels, corr_len) which could 
        include cross-correlations

    """
    def __init__(self, num_channels, autocorr, rng=None):
        super().__init__(num_channels, rng)
        assert autocorr.shape[0] == num_channels
        assert cr.is_autocorr_func(autocorr)
        if autocorr.ndim == 3:          #For channel inter-correlations
            raise NotImplementedError
        assert autocorr.ndim == 2

        self.autocorr = autocorr

        num_coeffs = []
        denom_coeffs = []
        self.noise_power = np.zeros(self.num_channels)
        for ch_idx in range(self.num_channels):
            ar_coeffs, self.noise_power[ch_idx] = ar_coeffs_from_autocorr(autocorr[ch_idx:ch_idx+1,:])
            nc, dc = ar_coeffs_to_transfer_function(ar_coeffs)
            num_coeffs.append(nc)
            denom_coeffs.append(dc)

        self.filt = fc.IIRFilter(num_coeffs, denom_coeffs)
        self.prepare()

    def prepare(self):
        for ch_idx in range(self.num_channels):
            self.filt.process(self.rng.normal(loc=0, scale=np.sqrt(self.noise_power[ch_idx]), size=(self.num_channels, self.filt.order[ch_idx])))
    
    def get_samples(self, num_samples):
        return self.filt.process(self.rng.normal(loc=0, scale=np.sqrt(self.noise_power)[:,None], size=(self.num_channels, num_samples)))












class MLS(Source):
    """Generates a maximum length sequence"""
    def __init__(self, num_channels, order, polynomial, state=None):
        super().__init__(num_channels, None)
        if num_channels > 1:
            raise NotImplementedError
        self.order = order
        self.poly = polynomial
        self.state = None

        self.metadata["order"] = self.order
        self.metadata["polynomial"] = self.poly
        self.metadata["start state"] = self.state

    def get_samples(self, num_samples):
        seq, self.state = spsig.max_len_seq(
            self.order, state=self.state, length=num_samples, taps=self.poly
        )
        return seq * 2 - 1


class GoldSequenceSource(Source):
    preferredSequences = {
        5: [[2], [1, 2, 3]],
        6: [[5], [1, 4, 5]],
        7: [[4], [4, 5, 6]],
        8: [[1, 2, 3, 6, 7], [1, 2, 7]],
        9: [[5], [3, 5, 6]],
        10: [[2, 5, 9], [3, 4, 6, 8, 9]],
        11: [[9], [3, 6, 9]],
    }

    def __init__(self, num_channels, power, order):
        super().__init__(num_channels)
        assert num_channels <= 2 ** order - 1
        self.num_channels = num_channels
        self.seqLength = 2 ** order - 1
        self.idx = 0
        self.set_power(power)
        
        if isinstance(self.power, (np.ndarray)):
            self.metadata["power"] = self.power.tolist()
        else:
            self.metadata["power"] = self.power
        self.metadata["order"] = order
        self.metadata["sequence length"] = self.seqLength

        if 5 <= order <= 11:
            mls1 = MLS(1, order, self.preferredSequences[order][0])
            mls2 = MLS(1, order, self.preferredSequences[order][1])
            seq1 = mls1.get_samples(2 ** order - 1)
            seq2 = mls2.get_samples(2 ** order - 1)
        elif order > 11:
            assert order % 2 == 1
            mls1 = MLS(1, order, None)
            seq1 = mls1.get_samples(2**order - 1)
            k = util.getSmallestCoprime(order)
            decimationFactor = int(2**k + 1)
            decimatedIndices = (np.arange(self.seqLength) * decimationFactor) % self.seqLength
            seq2 = np.copy(seq1[decimatedIndices])

        self.sequences = np.zeros((num_channels, self.seqLength))
        for i in range(num_channels):
            self.sequences[i, :] = np.negative(
                seq1 * np.roll(seq2, i)
            )  # XOR with integer shifts

    def get_samples(self, num_samples):
        outSignal = np.zeros((self.num_channels, num_samples))
        blockLengths = util.calcBlockSizes(num_samples, self.idx, self.seqLength)
        outIdx = 0
        for blockLen in blockLengths:
            outSignal[:, outIdx : outIdx + blockLen] = self.sequences[
                :, self.idx : self.idx + blockLen
            ]
            outIdx += blockLen
            self.idx = (self.idx + blockLen) % self.seqLength
        return outSignal * self.amplitude

    def set_power(self, newPower):
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




class LinearChirpSource(Source):
    def __init__(self, num_channels, amplitude, freqRange, samplesToSweep, samplerate, rng=None):
        raise NotImplementedError #change amplitude to power

        super.__init__(num_channels, rng)
        assert len(freqRange) == 2
        self.amp = amplitude
        self.samplerate = samplerate
        self.freqRange = freqRange

        self.samplesToSweep = samplesToSweep
        self.deltaFreq = (freqRange[1] - freqRange[0]) / samplesToSweep
        self.freq = freqRange[0]
        self.freqCounter = 0
        self.phase = self.rng.uniform(low=0, high=2 * np.pi)

    def nextPhase(self):
        self.phase += (self.freq / self.samplerate) + (
            self.deltaFreq / (2 * self.samplerate)
        )
        self.phase = self.phase % 1
        self.freq = self.freq + self.deltaFreq

    def get_samples(self, num_samples):
        noise = np.zeros((1, num_samples))
        samplesLeft = num_samples
        n = 0
        for blocks in range(1 + num_samples // self.samplesToSweep):
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