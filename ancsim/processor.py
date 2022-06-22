import numpy as np
from abc import ABC, abstractmethod

import ancsim.utilities as util
import ancsim.array as ar
import ancsim.signal.filterclasses as fc

import ancsim.diagnostics.core as diacore
import ancsim.diagnostics.diagnostics as dia
import ancsim.signal.freqdomainfiltering as fdf



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

        #for src in self.arrays.free_sources():
        #    self.processor.sig[src.name][:,:self.sim_info.sim_buffer] = src.get_samples(self.sim_info.sim_buffer)
# class Propagator():
#     def __init__(self, sim_info, arrays, source_names):
#         self.sim_info = sim_info
#         self.arrays = arrays
#         self.source_names = source_names

#     def prepare(self, sigs):
#         for src_name in self.source_names():
#             for mic in self.arrays.mics():
#                 propagated_signal = self.path_filters[src.name][mic.name].process(
#                         sigs[src_name][:,:self.sim_info.sim_buffer])
#                 self.processor.sig[mic.name][:,:self.sim_info.sim_buffer] += propagated_signal
#                 if self.sim_info.save_source_contributions:
#                     self.processor.sig[src_name+"~"+mic.name][:,:self.sim_info.sim_buffer] = propagated_signal
#         self.processor.idx = self.sim_info.sim_buffer
#         self.processor.prepare()



class ProcessorWrapper():
    def __init__(self, processor, arrays):
        self.processor = processor
        self.arrays = arrays
        self.sim_info = processor.sim_info
        self.blockSize = processor.blockSize

        self.path_filters = {}
        for src, mic, path in arrays.iter_paths():
            if src.name not in self.path_filters:
                self.path_filters[src.name] = {}
            self.path_filters[src.name][mic.name] = fc.createFilter(ir=path, sumOverInput=True, dynamic=(src.dynamic or mic.dynamic))

        for mic in self.arrays.mics():
            self.processor.createNewBuffer(mic.name, mic.num)
        for src in self.arrays.sources():
            self.processor.createNewBuffer(src.name, src.num)
        if self.sim_info.save_source_contributions:
            for src, mic in self.arrays.mic_src_combos():
                self.processor.createNewBuffer(src.name+"~"+mic.name, mic.num)
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
    
    def resetBuffers(self, globalIdx):
        for sigName, sig in self.processor.sig.items():
            self.processor.sig[sigName] = np.concatenate(
                (
                    sig[..., -self.sim_info.sim_buffer :],
                    np.zeros(sig.shape[:-1] + (self.sim_info.sim_chunk_size,)),
                ),
                axis=-1,
            )
        self.processor.idx -= self.sim_info.sim_chunk_size
        self.processor.resetBuffer()

    def propagate(self, globalIdx):
        """ propagated audio from all sources.
            The mic_signals are calculated for the indices
            self.processor.idx to self.processor.idx+numSamples

            calls the processors process function so that it 
            can calculate new values for the controlled sources"""

        i = self.processor.idx

        #for src in self.arrays.free_sources():
        #    self.processor.sig[src.name][:,i:i+self.blockSize] = src.get_samples(self.blockSize)

        for src, mic in self.arrays.mic_src_combos():
            if src.dynamic or mic.dynamic:
                self.path_filters[src.name][mic.name].ir = self.arrays.paths[src.name][mic.name]
            propagated_signal = self.path_filters[src.name][mic.name].process(
                    self.processor.sig[src.name][:,i:i+self.blockSize])
            self.processor.sig[mic.name][:,i:i+self.blockSize] += propagated_signal
            if self.sim_info.save_source_contributions:
                self.processor.sig[src.name+"~"+mic.name][:,i:i+self.blockSize] = propagated_signal
        self.processor.idx += self.blockSize

        last_block = self.last_block_on_buffer()
        self.processor.diag.saveData(self.processor, self.processor.idx, globalIdx, last_block)
        if last_block:
            self.resetBuffers(globalIdx)

    def process(self, globalIdx):
        self.processor.process(self.blockSize)
        #if self.processor.diag.shouldSaveData(globalIdx):
        
    def last_block_on_buffer(self):
        return self.processor.idx+self.blockSize >= self.processor.sim_info.sim_chunk_size+self.processor.sim_info.sim_buffer
        #return self.processor.idx+2*self.blockSize >= self.processor.sim_info.sim_chunk_size+self.processor.sim_info.sim_buffer
        
        

class AudioProcessor(ABC):
    def __init__(self, sim_info, arrays, blockSize, diagnostics={}):
        self.sim_info = sim_info
        self.blockSize = blockSize

        self.name = "Abstract Processor"
        self.arrays = arrays
        self.diag = diacore.DiagnosticHandler(self.sim_info, self.blockSize)
        self.rng = np.random.default_rng(1)
        self.metadata = {"block size" : self.blockSize}
        self.idx = 0

        self.sig = {}
        
        for diagName, diag in diagnostics.items():
            self.diag.add_diagnostic(diagName, diag)

    def prepare(self):
        self.diag.prepare()

    def resetBuffer(self):
        pass

    def getSimulatorInfo(self):
        pass

    def reset(self):
        """Should probably reset diagnostics to be properly reset
            Also, all user-defined signals and filters must be reset """
        for sig in self.sig.values():
            sig.fill(0)
        self.diag.reset()
        self.idx = self.sim_info.sim_buffer

    def createNewBuffer(self, name, dim):
        """dim is a tuple or int
        buffer is accessed as self.sig[name]
        and will have shape (dim[0], dim[1],...,dim[-1], simbuffer+simchunksize)"""
        if isinstance(dim, int):
            dim = (dim,)
        assert name not in self.sig
        self.sig[name] = np.zeros(dim + (self.sim_info.sim_buffer + self.sim_info.sim_chunk_size,))

    @abstractmethod
    def process(self, numSamples):
        """ microphone signals up to self.idx (excluding) are available. i.e. [:,choose_start_idx:self.idx] can be used
            To play a signal through controllable loudspeakers, add the values to self.sig['name-of-loudspeaker']
            for samples self.idx and forward, i.e. [:,self.idx:self.idx+self.blockSize]. 
            Adding samples further ahead than that will likely cause a outOfBounds error. 
        """
        pass



class DebugProcessor(AudioProcessor):
    def __init__(self, sim_info, arrays, blockSize, **kwargs):
        super().__init__(sim_info, arrays, blockSize, **kwargs)
        self.name = "Debug Processor"
        self.processed_samples = 0
        self.filt = fc.createFilter(numIn=3, numOut=4, irLen=5)

    def process(self, numSamples):
        assert numSamples == self.blockSize
        self.processed_samples += numSamples
        self.filt.ir += numSamples
        self.sig["loudspeaker"][:,self.idx:self.idx+self.blockSize] = \
            self.sig["mic"][:,self.idx-self.blockSize:self.idx]



class VolumeControl(AudioProcessor):
    def __init__(self, config, arrays, blockSize, volumeFactor, **kwargs):
        super().__init__(config, arrays, blockSize, **kwargs)
        self.name = "Volume Control"
        self.volumeFactor = volumeFactor

        self.diag.add_diagnostic("inputPower", dia.SignalPower("input", self.sim_info.tot_samples, self.outputSmoothing))
        self.diag.add_diagnostic("outputPower", dia.SignalPower("output", self.sim_info.tot_samples, self.outputSmoothing))

        self.metadata["volume factor"] = self.volumeFactor

    def process(self, numSamples):
        self.sig["output"][:,self.idx:self.idx+numSamples] = self.volumeFactor * \
            self.sig["input"][:,self.idx-numSamples:self.idx]


















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

    Extended implementation to blocksize != 1 can be done later
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



def calcBlockSizes(numSamples, idx, bufferSize, chunkSize):
    leftInBuffer = chunkSize + bufferSize - idx
    sampleCounter = 0
    blockSizes = []
    while sampleCounter < numSamples:
        bLen = np.min((numSamples - sampleCounter, leftInBuffer))
        blockSizes.append(bLen)
        sampleCounter += bLen
        leftInBuffer -= bLen
        if leftInBuffer == 0:
            leftInBuffer = chunkSize
    return blockSizes


def findFirstIndexForBlock(earliestStartIndex, indexToEndAt, blockSize):
    """If processing in fixed size blocks, this function will give the index
        to start at if the processing should end at a specific index.
        Useful for preparation processing, where the exact startpoint isn't important
        but it is important to end at the correct place. """
    numSamples = indexToEndAt - earliestStartIndex
    numBlocks = numSamples // blockSize
    indexToStartAt = indexToEndAt - blockSize*numBlocks
    return indexToStartAt

def blockProcessUntilIndex(earliestStartIndex, indexToEndAt, blockSize):
    """Use as 
        for startIdx, endIdx in blockProcessUntilIndex(earliestStart, indexToEnd, blockSize):
            process(signal[...,startIdx:endIdx])
    
        If processing in fixed size blocks, this function will give the index
        to process for, if the processing should end at a specific index.
        Useful for preparation processing, where the exact startpoint isn't important
        but it is important to end at the correct place. """
    numSamples = indexToEndAt - earliestStartIndex
    numBlocks = numSamples // blockSize
    indexToStartAt = indexToEndAt - blockSize*numBlocks

    for i in range(numBlocks):
        yield indexToStartAt+i*blockSize, indexToStartAt+(i+1)*blockSize

def getWhiteNoiseAtSNR(rng, signal, dim, snr, identicalChannelPower=False):
    """generates Additive White Gaussian Noise
    signal to calculate signal level. time dimension is last dimension
    dim is dimension of output AWGN
    snr, signal to noise ratio, in dB"""
    generatedNoise = rng.standard_normal(dim)
    if identicalChannelPower:
        powerOfSignal = util.avgPower(signal)
    else:
        powerOfSignal = util.avgPower(signal, axis=-1)[:, None]
    generatedNoise *= np.sqrt(powerOfSignal) * (util.db2mag(-snr))
    return generatedNoise


