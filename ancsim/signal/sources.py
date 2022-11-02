import numpy as np
from abc import ABC, abstractmethod

import ancsim.utilities as util

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



class Sequence(Source):
    def __init__(self, audio, amp_factor = 1, end_mode = "repeat"):
        """
            Will play the supplied sequence, either just once or repeat it indefinitely


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