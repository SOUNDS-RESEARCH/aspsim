import numpy as np
from abc import ABC, abstractmethod

import aspcore.filterclasses as fc

import aspsim.diagnostics.core as diacore
import aspsim.diagnostics.diagnostics as dia



class FreeSourceHandler():
    def __init__(self, sim_info, arrays):
        self.sim_info = sim_info
        self.arrays = arrays
        self.glob_to_lcl_diff = 0

        self.sig = {}
        for src in self.arrays.free_sources():
            self.sig[src.name] = np.zeros((src.num, self.sim_info.sim_buffer+self.sim_info.sim_chunk_size))

    def glob_to_lcl_idx(self, glob_idx):
        return glob_idx + self.glob_to_lcl_diff

    def prepare(self):
        for src in self.arrays.free_sources():
            self.sig[src.name][:,:] = src.get_samples(self.sim_info.sim_buffer+self.sim_info.sim_chunk_size)
        self.glob_to_lcl_diff += self.sim_info.sim_buffer
        #self.idx = self.sim_info.sim_buffer

    def reset_buffers(self):
        for src_name, sig in self.sig.items():
            self.sig[src_name][:,:self.sim_info.sim_buffer] = sig[:,-self.sim_info.sim_buffer:]
            self.sig[src_name][:,self.sim_info.sim_buffer:] = self.arrays[src_name].get_samples(self.sim_info.sim_chunk_size)
            
        self.glob_to_lcl_diff -= self.sim_info.sim_chunk_size

    def copy_sig_to_proc(self, proc, glob_idx, block_size):
        lcl_idx = self.glob_to_lcl_idx(glob_idx)
        if lcl_idx >= self.sim_info.sim_buffer+self.sim_info.sim_chunk_size:
            self.reset_buffers()
            lcl_idx = self.glob_to_lcl_idx(glob_idx)

        for src in self.arrays.free_sources():
            proc.processor.sig[src.name][:,proc.processor.idx:proc.processor.idx+block_size] = \
                self.sig[src.name][:,lcl_idx-block_size:lcl_idx]



class ProcessorWrapper():
    def __init__(self, processor, arrays):
        self.processor = processor
        self.arrays = arrays
        self.sim_info = processor.sim_info
        self.block_size = processor.block_size

        self.path_filters = {}
        for src, mic, path in arrays.iter_paths():
            if src.name not in self.path_filters:
                self.path_filters[src.name] = {}
            self.path_filters[src.name][mic.name] = fc.create_filter(ir=path, sum_over_input=True, dynamic=(src.dynamic or mic.dynamic))

        for mic in self.arrays.mics():
            self.processor.create_buffer(mic.name, mic.num)
        for src in self.arrays.sources():
            self.processor.create_buffer(src.name, src.num)
        if self.sim_info.save_source_contributions:
            for src, mic in self.arrays.mic_src_combos():
                self.processor.create_buffer(src.name+"~"+mic.name, mic.num)
        #self.ctrlSources = list(arrays.of_type(ar.ArrayType.CTRLSOURCE))

    def prepare(self):
        #for src in self.arrays.free_sources():
        #    self.processor.sig[src.name][:,:self.sim_info.sim_buffer] = src.get_samples(self.sim_info.sim_buffer)
        for src, mic in self.arrays.mic_src_combos():
            propagated_signal = self.path_filters[src.name][mic.name].process(
                    self.processor.sig[src.name][:,:self.sim_info.sim_buffer])
            self.processor.sig[mic.name][:,:self.sim_info.sim_buffer] += propagated_signal
            if self.sim_info.save_source_contributions:
                self.processor.sig[src.name+"~"+mic.name][:,:self.sim_info.sim_buffer] = propagated_signal
        self.processor.idx = self.sim_info.sim_buffer
        self.processor.prepare()

    def reset(self):
        self.processor.reset()
    
    def reset_buffers(self, global_idx):
        for sig_name, sig in self.processor.sig.items():
            self.processor.sig[sig_name] = np.concatenate(
                (
                    sig[..., -self.sim_info.sim_buffer :],
                    np.zeros(sig.shape[:-1] + (self.sim_info.sim_chunk_size,)),
                ),
                axis=-1,
            )
        self.processor.idx -= self.sim_info.sim_chunk_size
        self.processor.reset_buffer()

    def propagate(self, global_idx):
        """ propagated audio from all sources.
            The mic_signals are calculated for the indices
            self.processor.idx to self.processor.idx+numSamples

            calls the processors process function so that it 
            can calculate new values for the controlled sources"""

        i = self.processor.idx

        #for src in self.arrays.free_sources():
        #    self.processor.sig[src.name][:,i:i+self.block_size] = src.get_samples(self.block_size)

        for src, mic in self.arrays.mic_src_combos():
            if src.dynamic or mic.dynamic:
                self.path_filters[src.name][mic.name].ir = self.arrays.paths[src.name][mic.name]
            propagated_signal = self.path_filters[src.name][mic.name].process(
                    self.processor.sig[src.name][...,i:i+self.block_size])
            self.processor.sig[mic.name][...,i:i+self.block_size] += propagated_signal
            if self.sim_info.save_source_contributions:
                self.processor.sig[src.name+"~"+mic.name][...,i:i+self.block_size] = propagated_signal
        self.processor.idx += self.block_size

        last_block = self.last_block_on_buffer()
        self.processor.diag.save_data(self.processor, self.processor.idx, global_idx, last_block)
        if last_block:
            self.reset_buffers(global_idx)

    def process(self, globalIdx):
        self.processor.process(self.block_size)
        #if self.processor.diag.shouldSaveData(globalIdx):
        
    def last_block_on_buffer(self):
        return self.processor.idx+self.block_size >= self.processor.sim_info.sim_chunk_size+self.processor.sim_info.sim_buffer
        #return self.processor.idx+2*self.block_size >= self.processor.sim_info.sim_chunk_size+self.processor.sim_info.sim_buffer
        
        

class AudioProcessor(ABC):
    def __init__(self, sim_info, arrays, block_size, diagnostics={}, rng=None):
        self.sim_info = sim_info
        self.arrays = arrays
        self.block_size = block_size

        self.name = "Abstract Processor"
        self.metadata = {"block size" : self.block_size}
        self.idx = 0
        self.sig = {}

        self.diag = diacore.DiagnosticHandler(self.sim_info, self.block_size) 
        for diag_name, diag in diagnostics.items():
            self.diag.add_diagnostic(diag_name, diag)

        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def prepare(self):
        self.diag.prepare()

    def reset_buffer(self):
        pass

    def get_simulator_info(self):
        pass

    def reset(self):
        """Should probably reset diagnostics to be properly reset
            Also, all user-defined signals and filters must be reset """
        for sig in self.sig.values():
            sig.fill(0)
        self.diag.reset()
        self.idx = self.sim_info.sim_buffer

    def create_buffer(self, name, dim):
        """dim is a tuple or int
        buffer is accessed as self.sig[name]
        and will have shape (dim[0], dim[1],...,dim[-1], simbuffer+simchunksize)"""
        if isinstance(dim, int):
            dim = (dim,)
        assert name not in self.sig
        self.sig[name] = np.zeros(dim + (self.sim_info.sim_buffer + self.sim_info.sim_chunk_size,))

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
        self.filt = fc.create_filter(num_in=3, num_out=4, ir_len=5)

        self.mic = np.zeros((self.arrays["mic"].num, self.sim_info.tot_samples))
        self.ls = np.zeros((self.arrays["loudspeaker"].num, self.sim_info.tot_samples))
        self.manual_idx = 0

    def process(self, num_samples):
        assert num_samples == self.block_size
        self.processed_samples += num_samples
        self.filt.ir += num_samples
        self.sig["loudspeaker"][:,self.idx:self.idx+self.block_size] = \
            self.sig["mic"][:,self.idx-self.block_size:self.idx]

        for manual_i, i in zip(range(self.manual_idx,self.manual_idx+num_samples),range(self.idx-num_samples, self.idx)):
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

# def generate_white_noise_at_snr(rng, signal, dim, snr, identical_channel_power=False):
#     """generates Additive White Gaussian Noise
#     signal to calculate signal level. time dimension is last dimension
#     dim is dimension of output AWGN
#     snr, signal to noise ratio, in dB"""
#     generated_noise = rng.standard_normal(dim)
#     if identical_channel_power:
#         power_of_signal = util.avgPower(signal)
#     else:
#         power_of_signal = util.avgPower(signal, axis=-1)[:, None]
#     generated_noise *= np.sqrt(power_of_signal) * (util.db2mag(-snr))
#     return generated_noise













class BlockLeastMeanSquares(AudioProcessor):
    def __init__(self, config, arrays, block_size, stepSize, beta, filtLen):
        super().__init__(config, arrays, block_size)
        self.name = "Least Mean Squares"

        self.numIn = self.arrays["input"].num
        self.numOut = self.arrays["desired"].num
        self.stepSize = stepSize
        self.beta = beta
        self.filtLen = filtLen

        self.create_buffer("error", self.numOut)
        self.create_buffer("estimate", self.numOut)

        self.controlFilter = fc.FilterSum(irLen=self.filtLen, numIn=self.numIn, numOut=self.numOut)


        self.diag.add_diagnostic("inputPower", dia.SignalPower("input", self.sim_info.tot_samples, self.outputSmoothing))
        self.diag.add_diagnostic("desiredPower", dia.SignalPower("desired", self.sim_info.tot_samples, self.outputSmoothing))
        self.diag.add_diagnostic("errorPower", dia.SignalPower("error", self.sim_info.tot_samples, self.outputSmoothing))
        

    def process(self, numSamples):
        self.sig["estimate"][:,self.idx-numSamples:self.idx] = self.controlFilter.process(
            self.sig["input"][:,self.idx-numSamples:self.idx])

        self.sig["error"][:,self.idx-numSamples:self.idx] = self.sig["desired"][:,self.idx-numSamples:self.idx] - \
                                                            self.sig["estimate"][:,self.idx-numSamples:self.idx]

        grad = np.zeros_like(self.controlFilter.ir)
        for n in range(numSamples):
            grad += np.flip(self.sig["input"][None,:,self.idx-self.filtLen-n:self.idx-n], axis=-1) * \
                            self.sig["error"][:,None,self.idx-n-1:self.idx-n]

        normalization = 1 / (np.sum(self.sig["input"][:,self.idx-self.filtLen:self.idx]**2) + self.beta)
        self.controlFilter.ir += self.stepSize * normalization * grad
                        


class LeastMeanSquares(AudioProcessor):
    """Conventional Least Mean Squares
        Accepts block inputs but processes sample by sample, giving better
        performance than the block based version for long blocks. 
    """
        
    def __init__(self, sim_info, arrays, block_size, stepSize, beta, filtLen):
        super().__init__(sim_info, arrays, block_size)
        self.name = "Least Mean Squares"

        self.input = "input"
        self.desired = "desired"
        self.error = "error"
        self.estimate = "estimate"

        self.numIn = self.arrays[self.input].num
        self.numOut = self.arrays[self.desired].num
        self.stepSize = stepSize
        self.beta = beta
        self.filtLen = filtLen

        self.create_buffer(self.error, self.numOut)
        self.create_buffer(self.estimate, self.numOut)

        self.controlFilter = fc.FilterSum(irLen=self.filtLen, numIn=self.numIn, numOut=self.numOut)


        self.diag.add_diagnostic("inputPower", dia.SignalPower(self.sim_info, self.input))
        self.diag.add_diagnostic("desiredPower", dia.SignalPower(self.sim_info, self.desired))
        self.diag.add_diagnostic("errorPower", dia.SignalPower(self.sim_info, self.error))
        self.diag.add_diagnostic("paramError", dia.StateNMSE(
                    self.sim_info,
                    "controlFilter.ir", 
                    np.pad(self.arrays.paths["source"][self.desired],((0,0),(0,0),(0,512)), mode="constant", constant_values=0), 
                    100))
        

    def process(self, numSamples):
        for i in range(self.idx-numSamples, self.idx):
            self.sig[self.estimate][:,i] = np.squeeze(self.controlFilter.process(self.sig[self.input][:,i:i+1]), axis=-1)

            self.sig[self.error][:,i] = self.sig[self.desired][:,i] - self.sig[self.estimate][:,i]

            grad = np.flip(self.sig[self.input][:,None,i+1-self.filtLen:i+1], axis=-1) * \
                            self.sig[self.error][None,:,i:i+1]

            normalization = 1 / (np.sum(self.sig[self.input][:,i+1-self.filtLen:i+1]**2) + self.beta)
            self.controlFilter.ir += self.stepSize * normalization * grad
