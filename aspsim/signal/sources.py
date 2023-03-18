import numpy as np
from abc import ABC, abstractmethod

import aspsim.utilities as util

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
    def __init__(self, source_type, num_sources, amplitude, *args):
        if isinstance(amplitude, (int, float)):
            amplitude = [amplitude for _ in range(num_sources)]

        self.sources = [source_type(amplitude[i], *args) for i in range(num_sources)]

    def get_samples(self, num_samples):
        output = np.concatenate(
            [src.get_samples(num_samples) for src in self.sources], axis=0
        )
        return output



class Sequence(Source):
    def __init__(self, audio, amp_factor = 1, end_mode = "repeat"):
        """ Will play the supplied sequence

        Parameters
        ----------
        audio is np.ndarray of audio samples to be played. 
                    Shape is (num_channels, num_samples)
        end_mode : can be any of {'repeat', 'zeros', 'raise'}
            with 'repeat' the sequence starts over again indefinitely
            with 'zeros' the sequence plays once and then only 0s 
            with 'raise' the source will raise an exception if get_samples() 
            is called after the sequence is finished
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
            block_lengths = util.calc_block_sizes(num_samples, self.current_sample, self.tot_samples)
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
        return sig * self.amp_factor


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

        self.phase_per_sample = 2 * np.pi * self.freq / self.samplerate

        if isinstance(self.power, (np.ndarray)):
            self.metadata["power"] = self.power.tolist()
        else:
            self.metadata["power"] = self.power
        self.metadata["frequency"] = self.freq

    def get_samples(self, num_samples):
        noise = (
            self.amplitude
            * np.cos(self.phase_per_sample * np.arange(num_samples)[None,:] + self.phase)
        )
        self.phase = (self.phase + num_samples * self.phase_per_sample) % (2 * np.pi)
        return noise


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
