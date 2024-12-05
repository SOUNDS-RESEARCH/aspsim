import numpy as np
from abc import ABC, abstractmethod

import aspcore.filter as fc

import aspsim.diagnostics.core as diacore
import aspsim.diagnostics.diagnostics as dia
from aspsim.simulator import Signals

class AudioProcessor(ABC):
    def __init__(self, sim_info, arrays, block_size, diagnostics={}, rng=None):
        self.sim_info = sim_info
        self.arrays = arrays
        self.block_size = block_size

        self.name = "Abstract Processor"
        self.metadata = {}
        self.sig = Signals(self.sim_info, self.arrays)

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def prepare(self):
        pass

    @abstractmethod
    def process(self, num_samples):
        """ microphone signals up to self.idx (excluding) are available. i.e. [:,choose_start_idx:self.idx] can be used
            To play a signal through controllable loudspeakers, add the values to self.sig['name-of-loudspeaker']
            for samples self.idx and forward, i.e. [:,self.idx:self.idx+self.block_size]. 
            Adding samples further ahead than that will likely cause a outOfBounds error. 
        """
        pass



class DebugProcessor(AudioProcessor):
    def __init__(self, sim_info, arrays, block_size, **kwargs):
        super().__init__(sim_info, arrays, block_size, **kwargs)
        self.name = "Debug Processor"
        self.processed_samples = 0
        self.manual_idx = 0
        self.filt = fc.create_filter(num_in=3, num_out=4, ir_len=5)

        self.mic = np.zeros((self.arrays["mic"].num, self.sim_info.tot_samples))
        self.ls = np.zeros((self.arrays["loudspeaker"].num, self.sim_info.tot_samples))
        

    def process(self, num_samples):
        self.sig["loudspeaker"][:,self.sig.idx:self.sig.idx+num_samples] = \
            self.sig["mic"][:,self.sig.idx-num_samples:self.sig.idx]

        for manual_i, i in zip(range(self.manual_idx,self.manual_idx+num_samples),range(self.sig.idx-num_samples, self.sig.idx)):
            if manual_i < self.sim_info.tot_samples:
                self.mic[:,manual_i] = self.sig["mic"][:,i]
                self.ls[:,manual_i] = self.sig["loudspeaker"][:,i]
                self.manual_idx += 1

        self.processed_samples += num_samples
        self.filt.ir += num_samples
















def calc_block_sizes_with_buffer(num_samples, idx, buffer_size, chunk_size):
    leftInBuffer = chunk_size + buffer_size - idx
    sampleCounter = 0
    block_sizes = []
    while sampleCounter < num_samples:
        bLen = np.min((num_samples - sampleCounter, leftInBuffer))
        block_sizes.append(bLen)
        sampleCounter += bLen
        leftInBuffer -= bLen
        if leftInBuffer == 0:
            leftInBuffer = chunk_size
    return block_sizes


def find_first_index_for_block(earliest_start_index, index_to_end_at, block_size):
    """If processing in fixed size blocks, this function will give the index
        to start at if the processing should end at a specific index.
        Useful for preparation processing, where the exact startpoint isn't important
        but it is important to end at the correct place. """
    num_samples = index_to_end_at - earliest_start_index
    num_blocks = num_samples // block_size
    index_to_start_at = index_to_end_at - block_size*num_blocks
    return index_to_start_at

def block_process_until_index(earliest_start_index, index_to_end_at, block_size):
    """Use as 
        for startIdx, endIdx in blockProcessUntilIndex(earliestStart, indexToEnd, block_size):
            process(signal[...,startIdx:endIdx])
    
        If processing in fixed size blocks, this function will give the index
        to process for, if the processing should end at a specific index.
        Useful for preparation processing, where the exact startpoint isn't important
        but it is important to end at the correct place. """
    num_samples = index_to_end_at - earliest_start_index
    num_blocks = num_samples // block_size
    index_to_start_at = index_to_end_at - block_size*num_blocks

    for i in range(num_blocks):
        yield index_to_start_at+i*block_size, index_to_start_at+(i+1)*block_size

