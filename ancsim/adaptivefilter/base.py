import numpy as np
from abc import ABC, abstractmethod

import ancsim.utilities as util
import ancsim.array as ar
import ancsim.signal.filterclasses as fc

from ancsim.adaptivefilter.util import blockProcessUntilIndex, calcBlockSizes, getWhiteNoiseAtSNR
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
        assert isinstance(dim, tuple)
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



class ActiveNoiseControlProcessor(AudioProcessor):
    def __init__(self, sim_info, arrays, blockSize, updateBlockSize, **kwargs):
        super().__init__(sim_info, arrays, blockSize, **kwargs)
        self.updateBlockSize = updateBlockSize
        self.numRef = self.arrays["ref"].num
        self.numError = self.arrays["error"].num
        self.numSpeaker = self.arrays["speaker"].num
        
        self.diag.add_diagnostic("noise_reduction", dia.SignalPowerRatio(self.sim_info, "error", "source~error"))
        if "target" in self.arrays:
            self.diag.add_diagnostic("spatial_noise_reduction", 
                dia.SignalPowerRatio(self.sim_info, "target", "source~target", 1024, False))

        self.updateIdx = self.sim_info.sim_buffer
        
        self.metadata["update block size"] = updateBlockSize

    def resetBuffer(self):
        super().resetBuffer()
        self.updateIdx -= self.sim_info.sim_chunk_size

    def process(self, numSamples):
        self.genSpeakerSignals(numSamples)
        if self.updateIdx <= self.idx - self.updateBlockSize:
            self.updateFilter()
            self.updateIdx += self.updateBlockSize

    @abstractmethod
    def genSpeakerSignals(self, numSamples):
        pass

    @abstractmethod
    def updateFilter(self):
        pass


