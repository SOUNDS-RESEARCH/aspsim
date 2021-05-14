import numpy as np
from abc import ABC, abstractmethod

import ancsim.utilities as util
import ancsim.array as ar
from ancsim.signal.filterclasses import (
    Filter_IntBuffer,
    FilterSum,
    FilterMD_IntBuffer,
    FilterSum_Freqdomain,
    FilterMD_Freqdomain
)
from ancsim.adaptivefilter.util import blockProcessUntilIndex, calcBlockSizes, getWhiteNoiseAtSNR
import ancsim.adaptivefilter.diagnostics as dia
import ancsim.signal.freqdomainfiltering as fdf




class ProcessorWrapper():
    def __init__(self, processor, arrays):
        self.processor = processor
        self.arrays = arrays
        self.blockSize = processor.blockSize

        self.path_filters = {}
        for src, mic, path in arrays.iter_paths():
            if src.name not in self.path_filters:
                self.path_filters[src.name] = {}
            self.path_filters[src.name][mic.name] = FilterSum(path)

        for mic in self.arrays.mics():
            self.processor.createNewBuffer(mic.name, mic.num)
        for src in self.arrays.sources():
            self.processor.createNewBuffer(src.name, src.num)
        for src, mic in self.arrays.mic_src_combos():
            self.processor.createNewBuffer(src.name+"~"+mic.name, mic.num)
        #self.ctrlSources = list(arrays.of_type(ar.ArrayType.CTRLSOURCE))

    def prepare(self):
        for src in self.arrays.free_sources():
            self.processor.sig[src.name][:,:self.processor.sim_info.sim_buffer] = src.getSamples(self.processor.sim_info.sim_buffer)
        for src, mic in self.arrays.mic_src_combos():
            propagated_signal = self.path_filters[src.name][mic.name].process(
                    self.processor.sig[src.name][:,:self.processor.sim_info.sim_buffer])
            self.processor.sig[src.name+"~"+mic.name][:,:self.processor.sim_info.sim_buffer] = propagated_signal
            self.processor.sig[mic.name][:,:self.processor.sim_info.sim_buffer] += propagated_signal
        self.processor.idx = self.processor.sim_info.sim_buffer
        self.processor.prepare()

    def reset(self):
        self.processor.reset()
    
    def resetBuffers(self, globalIdx):
        for sigName, sig in self.processor.sig.items():
            self.processor.sig[sigName] = np.concatenate(
                (
                    sig[..., -self.processor.sim_info.sim_buffer :],
                    np.zeros(sig.shape[:-1] + (self.processor.sim_info.sim_chunk_size,)),
                ),
                axis=-1,
            )
        self.processor.idx -= self.processor.sim_info.sim_chunk_size
        self.processor.resetBuffer()

    def propagate(self, globalIdx):
        """ propagated audio from  all sources.
            The mic_signals are calculated for the indices
            self.processor.idx to self.processor.idx+numSamples

            calls the processors process function so that it 
            can calculate new values for the controlled sources"""

        i = self.processor.idx

        for src in self.arrays.free_sources():
            self.processor.sig[src.name][:,i:i+self.blockSize] = src.getSamples(self.blockSize)

        for src, mic in self.arrays.mic_src_combos():
                propagated_signal = self.path_filters[src.name][mic.name].process(
                        self.processor.sig[src.name][:,i:i+self.blockSize])
                self.processor.sig[src.name+"~"+mic.name][:,i:i+self.blockSize] = propagated_signal
                self.processor.sig[mic.name][:,i:i+self.blockSize] += propagated_signal
        self.processor.idx += self.blockSize

    def process(self, globalIdx):
        self.processor.process(self.blockSize)
        #if self.processor.diag.shouldSaveData(globalIdx):
        self.processor.diag.saveData(self.processor, self.processor.sig, self.processor.idx, globalIdx)
        if self.processor.idx+2*self.blockSize >= self.processor.sim_info.sim_chunk_size+self.processor.sim_info.sim_buffer:
            self.resetBuffers(globalIdx)

        
        

class AudioProcessor(ABC):
    def __init__(self, sim_info, arrays, blockSize):
        self.sim_info = sim_info
        # self.sim_info.sim_buffer = config["sim_buffer"]
        # self.sim_info.sim_chunk_size = config["sim_chunk_size"]
        # self.sim_info.tot_samples = config["tot_samples"]
        # self.samplerate = config["samplerate"]

        # self.spatialDims = config["spatial_dims"]
        # self.c = config["c"]
        # self.micSNR = config["mic_snr"]

        # self.saveRawData = config["save_raw_data"]
        # self.outputSmoothing = config["output_smoothing"]
        # self.plotFrequency = config["plot_frequency"]

        self.blockSize = blockSize

        self.name = "Abstract Processor"
        self.arrays = arrays
        self.diag = dia.DiagnosticHandler(self.sim_info, self.blockSize)
        self.rng = np.random.default_rng(1)
        self.metadata = {"block size" : self.blockSize}
        self.idx = 0

        self.sig = {}

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
        """"""
        pass


class VolumeControl(AudioProcessor):
    def __init__(self, config, arrays, blockSize, volumeFactor):
        super().__init__(config, arrays, blockSize)
        self.name = "Volume Control"
        self.volumeFactor = volumeFactor

        self.diag.addNewDiagnostic("inputPower", dia.SignalPower("input", self.sim_info.tot_samples, self.outputSmoothing))
        self.diag.addNewDiagnostic("outputPower", dia.SignalPower("output", self.sim_info.tot_samples, self.outputSmoothing))

        self.metadata["volume factor"] = self.volumeFactor

    def process(self, numSamples):
        self.sig["output"][:,self.idx:self.idx+numSamples] = self.volumeFactor * \
            self.sig["input"][:,self.idx-numSamples:self.idx]



class ActiveNoiseControlProcessor(AudioProcessor):
    def __init__(self, sim_info, arrays, blockSize, updateBlockSize):
        super().__init__(sim_info, arrays, blockSize)
        self.updateBlockSize = updateBlockSize
        self.numRef = self.arrays["ref"].num
        self.numError = self.arrays["error"].num
        self.numSpeaker = self.arrays["speaker"].num
        
        self.diag.addNewDiagnostic("noise_reduction", dia.NoiseReduction(self.sim_info, "error", "source~error"))
        if "target" in self.arrays:
            self.diag.addNewDiagnostic("spatial_noise_reduction", 
                dia.NoiseReduction(self.sim_info, "target", "source~target", 1024, False))

        self.updateIdx = self.sim_info.sim_buffer
        
        self.metadata["update block size"] = updateBlockSize

    def resetBuffer(self):
        super().resetBuffer()
        self.updateIdx -= self.sim_info.sim_chunk_size

    def process(self, numSamples):
        #self.diag.saveBlockData()
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





# class AdaptiveFilterFF(ActiveNoiseControlProcessor):
#     def __init__(self, config, mu, beta, speakerRIR):
#         self.name = "Adaptive Filter Timedomain Baseclass"
#         self.numRef = config["NUMREF"]
#         self.numError = config["NUMERROR"]
#         self.numSpeaker = config["NUMSPEAKER"]
#         self.numTarget = config["NUMTARGET"]
#         self.filtLen = config["FILTLENGTH"]
#         self.micSNR = config["mic_snr"]
#         self.saveRawData = config["save_raw_data"]
#         self.outputSmoothing = config["output_smoothing"]
#         self.plotFrequency = config["plot_frequency"]
#         self.c = config["c"]
#         self.samplerate = config["samplerate"]
#         self.spatialDims = config["spatial_dims"]
#         self.sim_info.tot_samples = config["tot_samples"]
#         self.sim_info.sim_chunk_size = config["sim_chunk_size"]
#         self.sim_info.sim_buffer = config["sim_buffer"]
#         self.mu = mu
#         self.beta = beta

#         self.rng = np.random.RandomState(5)

#         self.controlFilt = FilterSum(
#             np.zeros((self.numRef, self.numSpeaker, self.filtLen))
#         )

#         self.buffers = {}

#         self.y = np.zeros((self.numSpeaker, self.sim_info.sim_chunk_size + self.sim_info.sim_buffer))
#         self.x = np.zeros((self.numRef, self.sim_info.sim_chunk_size + self.sim_info.sim_buffer))
#         self.e = np.zeros((self.numError, self.sim_info.sim_chunk_size + self.sim_info.sim_buffer))

#         self.secPathFilt = FilterSum(speakerRIR["error"])

#         self.diag = DiagnosticHandler(self.sim_info.sim_buffer, self.sim_info.sim_chunk_size)
#         self.diag.addNewDiagnostic(
#             "reduction_microphones",
#             NoiseReductionExternalSignals(
#                 self.numError,
#                 self.sim_info.tot_samples,
#                 self.sim_info.sim_buffer,
#                 self.sim_info.sim_chunk_size,
#                 smoothingLen=self.outputSmoothing,
#                 saveRawData=self.saveRawData,
#                 plotFrequency=self.plotFrequency,
#             ),
#         )
#         self.diag.addNewDiagnostic(
#             "reduction_regional",
#             NoiseReduction(
#                 self.numTarget,
#                 speakerRIR["target"],
#                 self.sim_info.tot_samples,
#                 self.sim_info.sim_buffer,
#                 self.sim_info.sim_chunk_size,
#                 smoothingLen=self.outputSmoothing,
#                 saveRawData=self.saveRawData,
#                 plotFrequency=self.plotFrequency,
#             ),
#         )
#         # self.diag.addNewDiagnostic("soundfield_target",
#         #                             SoundfieldImage(self.numEvals, speakerRIR["evals"],
#         #                             self.sim_info.sim_buffer, self.sim_info.sim_chunk_size,
#         #                             beginAtBuffer=self.genSoundfieldAtChunk,
#         #                             plotFrequency=self.plotFrequency))

#         self.idx = self.sim_info.sim_buffer
#         self.updateIdx = self.sim_info.sim_buffer

#         self.metadata = {"mu": mu, "beta": beta, "controlFiltLen": self.filtLen}

#     def prepare(self):
#         """Will be called once, after buffers are filled, right before regular operations
#         Nothing required for forward pass can be set up here, but any initialization
#         needed for the filter update is fine. Calculating filtered references with reference
#         signals from before time 0 is a good use."""
#         pass

#     def forwardPassImplement(self, numSamples):
#         """Must insert self.y[:,self.idx:self.idx:numSamples] with new values.
#         self.x[:,self.idx:self.idx:numSamples] are new reference signal values
#         last valid index for self.e is self.idx-1"""
#         self.y[:, self.idx : self.idx + numSamples] = self.controlFilt.process(
#             self.x[:, self.idx : self.idx + numSamples]
#         )

#     def outputRawData(self):
#         return self.controlFilt.ir

#     @abstractmethod
#     def updateFilter(self):
#         pass

#     def resetBuffers(self):
#         self.diag.saveBufferData("reduction_microphones", self.idx, self.e)
#         self.diag.saveBufferData("reduction_regional", self.idx, self.y)
#         # self.diag.saveBufferData("soundfield_target", self.idx, self.y)
#         self.diag.resetBuffers(self.idx)

#         self.y = np.concatenate(
#             (
#                 self.y[:, -self.sim_info.sim_buffer :],
#                 np.zeros((self.y.shape[0], self.sim_info.sim_chunk_size)),
#             ),
#             axis=-1,
#         )
#         self.x = np.concatenate(
#             (
#                 self.x[:, -self.sim_info.sim_buffer :],
#                 np.zeros((self.x.shape[0], self.sim_info.sim_chunk_size)),
#             ),
#             axis=-1,
#         )
#         self.e = np.concatenate(
#             (
#                 self.e[:, -self.sim_info.sim_buffer :],
#                 np.zeros((self.e.shape[0], self.sim_info.sim_chunk_size)),
#             ),
#             axis=-1,
#         )

#         for bufName, buf in self.buffers.items():
#             self.buffers[bufName] = np.concatenate(
#                 (
#                     buf[..., -self.sim_info.sim_buffer :],
#                     np.zeros(buf.shape[:-1] + (self.sim_info.sim_chunk_size,)),
#                 ),
#                 axis=-1,
#             )

#         self.idx -= self.sim_info.sim_chunk_size
#         self.updateIdx -= self.sim_info.sim_chunk_size

#     def forwardPass(self, numSamples, noises):
#         blockSizes = calcBlockSizes(
#             numSamples, self.idx, self.sim_info.sim_buffer, self.sim_info.sim_chunk_size
#         )
#         errorMicNoise = getWhiteNoiseAtSNR(
#             self.rng, noises["error"], (self.numError, numSamples), self.micSNR
#         )
#         refMicNoise = getWhiteNoiseAtSNR(
#             self.rng, noises["ref"], (self.numRef, numSamples), self.micSNR
#         )

#         numComputed = 0
#         for blockSize in blockSizes:
#             self.x[:, self.idx : self.idx + blockSize] = (
#                 noises["ref"][:, numComputed : numComputed + blockSize]
#                 + refMicNoise[:, numComputed : numComputed + blockSize]
#             )
#             self.forwardPassImplement(blockSize)

#             self.diag.saveBlockData(
#                 "reduction_microphones",
#                 self.idx,
#                 noises["error"][:, numComputed : numComputed + blockSize],
#             )
#             self.diag.saveBlockData(
#                 "reduction_regional",
#                 self.idx,
#                 noises["target"][:, numComputed : numComputed + blockSize],
#             )
#             # self.diag.saveBlockData("soundfield_target", self.idx, noiseAtEvals[:,numComputed:numComputed+blockSize])

#             self.e[:, self.idx : self.idx + blockSize] = (
#                 noises["error"][:, numComputed : numComputed + blockSize]
#                 + errorMicNoise[:, numComputed : numComputed + blockSize]
#                 + self.secPathFilt.process(self.y[:, self.idx : self.idx + blockSize])
#             )

#             assert self.idx + blockSize <= self.sim_info.sim_buffer + self.sim_info.sim_chunk_size
#             if self.idx + blockSize >= (self.sim_info.sim_chunk_size + self.sim_info.sim_buffer):
#                 self.resetBuffers()
#             self.idx += blockSize
#             numComputed += blockSize


# class AdaptiveFilterFFComplex(ActiveNoiseControlProcessor):
#     def __init__(self, config, mu, beta, speakerRIR, blockSize):
#         self.name = "Adaptive Filter Feedforward Complex"

#         self.numRef = config["NUMREF"]
#         self.numError = config["NUMERROR"]
#         self.numSpeaker = config["NUMSPEAKER"]
#         self.numTarget = config["NUMTARGET"]
#         self.micSNR = config["mic_snr"]
#         self.saveRawData = config["save_raw_data"]
#         self.outputSmoothing = config["output_smoothing"]
#         self.plotFrequency = config["plot_frequency"]
#         self.c = config["c"]
#         self.samplerate = config["samplerate"]
#         self.spatialDims = config["spatial_dims"]
#         self.sim_info.tot_samples = config["tot_samples"]
#         self.sim_info.sim_chunk_size = config["sim_chunk_size"]
#         self.sim_info.sim_buffer = config["sim_buffer"]
#         self.mu = mu
#         self.beta = beta
#         self.blockSize = blockSize

#         self.rng = np.random.RandomState(5)

#         self.H = np.zeros(
#             (2 * blockSize, self.numSpeaker, self.numRef), dtype=np.complex128
#         )
#         self.controlFilt = FilterSum_Freqdomain(
#             numIn=self.numRef, numOut=self.numSpeaker, irLen=blockSize
#         )
#         self.buffers = {}

#         self.x = np.zeros((self.numRef, self.sim_info.sim_chunk_size + self.sim_info.sim_buffer))
#         self.e = np.zeros((self.numError, self.sim_info.sim_chunk_size + self.sim_info.sim_buffer))
#         self.y = np.zeros((self.numSpeaker, self.sim_info.sim_chunk_size + self.sim_info.sim_buffer))
#         self.buffers["primary"] = np.zeros((self.numError, self.sim_info.sim_chunk_size + self.sim_info.sim_buffer))
#         self.buffers["yf"] = np.zeros((self.numError, self.sim_info.sim_chunk_size + self.sim_info.sim_buffer))

#         self.G = np.transpose(
#             np.fft.fft(speakerRIR["error"], n=2 * blockSize, axis=-1), (2, 0, 1)
#         )

#         self.secPathFilt = FilterSum(speakerRIR["error"])

#         self.diag = DiagnosticHandler(self.sim_info.sim_buffer, self.sim_info.sim_chunk_size)
#         self.diag.addNewDiagnostic(
#             "reduction_microphones",
#             NoiseReductionExternalSignals(
#                 self.numError,
#                 self.sim_info.tot_samples,
#                 self.sim_info.sim_buffer,
#                 self.sim_info.sim_chunk_size,
#                 smoothingLen=self.outputSmoothing,
#                 saveRawData=self.saveRawData,
#                 plotFrequency=self.plotFrequency,
#             ),
#         )
#         if "target" in speakerRIR:
#             self.diag.addNewDiagnostic(
#                 "reduction_regional",
#                 NoiseReduction(
#                     self.numTarget,
#                     speakerRIR["target"],
#                     self.sim_info.tot_samples,
#                     self.sim_info.sim_buffer,
#                     self.sim_info.sim_chunk_size,
#                     smoothingLen=self.outputSmoothing,
#                     saveRawData=self.saveRawData,
#                     plotFrequency=self.plotFrequency,
#                 ),
#             )

        
#         self.diag.addNewDiagnostic(
#             "power_primary_noise",
#                 RecordScalar(self.sim_info.tot_samples, self.sim_info.sim_buffer, self.sim_info.sim_chunk_size,plotFrequency=self.plotFrequency)
#         )
#         # self.diag.addNewDiagnostic("soundfield_target",
#         #                             SoundfieldImage(self.numEvals, speakerRIR["evals"],
#         #                             self.sim_info.sim_buffer, self.sim_info.sim_chunk_size,
#         #                             beginAtBuffer=self.genSoundfieldAtChunk,
#         #                             plotFrequency=self.plotFrequency))
#         self.idx = self.sim_info.sim_buffer
#         self.updateIdx = self.sim_info.sim_buffer

#         self.metadata = {"mu": mu, "beta": beta, "controlFiltNumFreq": 2 * blockSize}

#     def outputRawData(self):
#         """Can be extended to include other filters and quantities"""
#         return self.controlFilt.tf

#     def prepare(self):
#         pass

#     @abstractmethod
#     def forwardPassImplement(self):
#         pass

#     @abstractmethod
#     def updateFilter(self):
#         pass

#     def resetBuffers(self):
#         self.diag.saveBufferData("reduction_microphones", self.idx, self.e)
#         if self.diag.hasDiagnostic("reduction_regional"):
#             self.diag.saveBufferData("reduction_regional", self.idx, self.y)
#         # self.diag.saveBufferData("soundfield_target", self.idx, self.y)
#         self.diag.resetBuffers(self.idx)

#         for bufName, buf in self.buffers.items():
#             self.buffers[bufName] = np.concatenate(
#                 (
#                     buf[..., -self.sim_info.sim_buffer :],
#                     np.zeros(buf.shape[:-1] + (self.sim_info.sim_chunk_size,)),
#                 ),
#                 axis=-1,
#             )

#         self.x = np.concatenate(
#             (
#                 self.x[:, -self.sim_info.sim_buffer :],
#                 np.zeros((self.x.shape[0], self.sim_info.sim_chunk_size)),
#             ),
#             axis=-1,
#         )
#         self.e = np.concatenate(
#             (
#                 self.e[:, -self.sim_info.sim_buffer :],
#                 np.zeros((self.e.shape[0], self.sim_info.sim_chunk_size)),
#             ),
#             axis=-1,
#         )

#         self.idx -= self.sim_info.sim_chunk_size
#         self.updateIdx -= self.sim_info.sim_chunk_size

#     def forwardPass(self, numSamples, noises):
#         assert numSamples == self.blockSize

#         if self.idx + numSamples >= (self.sim_info.sim_chunk_size + self.sim_info.sim_buffer):
#             self.resetBuffers()

#         errorMicNoise = getWhiteNoiseAtSNR(
#             self.rng, noises["error"], (self.numError, numSamples), self.micSNR
#         )
#         refMicNoise = getWhiteNoiseAtSNR(
#             self.rng, noises["ref"], (self.numRef, numSamples), self.micSNR
#         )

#         self.x[:, self.idx : self.idx + numSamples] = noises["ref"] + refMicNoise
#         self.forwardPassImplement(numSamples)

#         self.buffers["primary"][:,self.idx:self.idx+numSamples] = noises["error"]
#         self.diag.saveBlockData("reduction_microphones", self.idx, noises["error"])
#         if self.diag.hasDiagnostic("reduction_regional"):
#             self.diag.saveBlockData("reduction_regional", self.idx, noises["target"])

    
#         self.diag.saveBlockData("power_primary_noise", self.idx, np.mean(noises["error"]**2))

#         self.buffers["yf"][:,self.idx:self.idx+numSamples] = self.secPathFilt.process(self.y[:, self.idx : self.idx + numSamples])

#         self.e[:, self.idx : self.idx + numSamples] = (
#             noises["error"]
#             + errorMicNoise
#             + self.buffers["yf"][:,self.idx:self.idx+numSamples]
#             )
        

#         self.idx += numSamples
