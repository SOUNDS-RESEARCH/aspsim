import numpy as np
from abc import ABC, abstractmethod

import aspcore.filterclasses as fc

import aspsim.diagnostics.core as diacore
import aspsim.diagnostics.diagnostics as dia
from aspsim.simulator import Signals

class AudioProcessor(ABC):
    def __init__(self, sim_info, arrays, block_size, diagnostics={}, rng=None):
        self.sim_info = sim_info
        self.arrays = arrays

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
        self.processed_samples += num_samples
        self.filt.ir += num_samples
        self.sig["loudspeaker"][:,self.sig.idx:self.sig.idx+num_samples] = \
            self.sig["mic"][:,self.sig.idx-num_samples:self.sig.idx]

        for manual_i, i in zip(range(self.manual_idx,self.manual_idx+num_samples),range(self.sig.idx-num_samples, self.sig.idx)):
            if manual_i < self.sim_info.tot_samples:
                self.mic[:,manual_i] = self.sig["mic"][:,i]
                self.ls[:,manual_i] = self.sig["loudspeaker"][:,i]
                self.manual_idx += 1














class PhaseCounter:
    """
    An index counter to keep track of non-overlapping continous phases
    
    Example:
    A processor needs the first 2000 samples for an initialization, 
    then must wait 5000 samples before beginning the real processing step.
    The class can then be used by providing
    phase_def = {
        'init' : 2000,
        'wait' : 5000,
        'process' : np.inf
    }
    and then checking if phase_counter.phase == 'init'
    or if phase_counter.current_phase_is('init'):
    
    The number is how many samples that each phase should be
    The first phase will start at sample 0.

    np.inf represents an infinite length
    This should naturally only be used for the last phase
    If all phases has finished, the phase will be None. 

    first_sample will be True on the first sample of each phase,
    allowing running one-time functions in each phase

    Extended implementation to block_size != 1 can be done later
    """
    def __init__(self, phase_lengths, verbose=False):
        assert isinstance(phase_lengths, dict)
        self.phase_lengths = phase_lengths
        self.verbose = verbose
        self.phase = None
        self.first_sample = True
        

        #phase_lengths = {name : length for name, length in self.phase_lengths.items() if length != 0}
        #phase_lengths = {name : length for name, length in self.phase_lengths.items()}
        
        #phase_idxs = [i for i in self.phase_lengths.values() if i != 0]
        self.phase_lengths = {name : i if i >= 0 else np.inf for name, i in self.phase_lengths.items()}
        #assert all([i != 0 for i in p_len])
        self.start_idxs = np.cumsum(list(self.phase_lengths.values())).tolist()
        self.start_idxs = [i if np.isinf(i) else int(i) for i in self.start_idxs]
        self.start_idxs.insert(0,0)

        self.phase_names = list(self.phase_lengths.keys())
        if self.start_idxs[-1] < np.inf:
            self.phase_names.append(None)
        else:
            self.start_idxs.pop()

        self.start_idxs = {phase_name:start_idx for phase_name, start_idx in zip(self.phase_names, self.start_idxs)}

        self._phase_names = [phase_name for phase_name, phase_len in self.phase_lengths.items() if phase_len > 0]
        self._start_idxs = [start_idx for start_idx, phase_len in zip(self.start_idxs.values(), self.phase_lengths.values()) if phase_len > 0]

        self.idx = 0
        self.next_phase()

    def next_phase(self):
        if self.verbose:
            print(f"Changed phase from {self.phase}")
            
        self.phase = self._phase_names.pop(0)
        self._start_idxs.pop(0)
        if len(self._start_idxs) == 0:
            self._start_idxs.append(np.inf)
        self.first_sample = True
        
        if self.verbose:
            print(f"to {self.phase}")

    def progress(self):
        self.idx += 1
        if self.idx >= self._start_idxs[0]:
            self.next_phase()
        else:
            self.first_sample = False

    def current_phase_is(self, phase_name):
        return self.phase == phase_name



class EventCounter:
    """
    An index counter to keep track of events that should 
    only happen every x samples

    event_def is a dictionary with all event
    each entry is 'event_name' : (frequency, offset)

    Example:
    event_counter = EventCounter({'event_1' : (256,0), 'event_2' : (1,0), 'event_3' : (1024,256)})
    event_2 will happen every sample, event_1 every 256 samples
    First at sample 256 all three events will happen simultaneouly. 

    To be used as:
    if 'event_name' in event_counter.event:
        do_thing()

    """
    def __init__(self, event_def):
        self.event_def = event_def
        self.event = []

        self.freq = {name : freq for name, (freq, offset) in event_def.items()}
        self.offset = {name : offset for name, (freq, offset) in event_def.items()}

        self.idx = 0

    def add_event(self, name, freq, offset):
        self.event_def[name] = (freq, offset)

    def check_events(self):
        self.event = []
        for name, (freq, offset) in self.event_def.items():
            if (self.idx - offset) % freq == 0:
                self.event.append(name)

    def progress(self):
        self.idx += 1 
        self.check_events()



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

