import numpy as np
from abc import ABC, abstractmethod

import ancsim.utilities as util
import ancsim.array as ar
from ancsim.signal.filterclasses import (
    Filter_IntBuffer,
    FilterSum_IntBuffer,
    FilterMD_IntBuffer,
    FilterSum_Freqdomain,
)
from ancsim.adaptivefilter.util import calcBlockSizes, getWhiteNoiseAtSNR
import ancsim.adaptivefilter.diagnostics as dia




class ProcessorWrapper():
    def __init__(self, processor, arrays, blockSize):
        self.processor = processor
        self.arrays = arrays
        self.blockSize = blockSize

        self.path_filters = {}
        for src, mic, path in arrays.iter_paths():
            if src.name not in self.path_filters:
                self.path_filters[src.name] = {}
            self.path_filters[src.name][mic.name] = FilterSum_IntBuffer(path)

        for mic in self.arrays.mics():
            self.processor.createNewBuffer(mic.name, mic.num)
        for src in self.arrays.sources():
            self.processor.createNewBuffer(src.name, src.num)
        for src, mic in self.arrays.mic_src_combos():
            self.processor.createNewBuffer(src.name+"~"+mic.name, mic.num)
        #self.ctrlSources = list(arrays.of_type(ar.ArrayType.CTRLSOURCE))

    def prepare(self):
        for src in self.arrays.free_sources():
            self.processor.sig[src.name][:,:self.processor.simBuffer] = src.getSamples(self.processor.simBuffer)
        for src, mic in self.arrays.mic_src_combos():
            propagated_signal = self.path_filters[src.name][mic.name].process(
                    self.processor.sig[src.name][:,:self.processor.simBuffer])
            self.processor.sig[src.name+"~"+mic.name][:,:self.processor.simBuffer] = propagated_signal
            self.processor.sig[mic.name][:,:self.processor.simBuffer] += propagated_signal
        self.processor.idx = self.processor.simBuffer
        self.processor.prepare()

    def reset(self):
        self.processor.reset()
    
    def resetBuffers(self, globalIdx):
        self.processor.diag.saveData(self.processor.sig, self.processor.idx, globalIdx)
        for sigName, sig in self.processor.sig.items():
            self.processor.sig[sigName] = np.concatenate(
                (
                    sig[..., -self.processor.simBuffer :],
                    np.zeros(sig.shape[:-1] + (self.processor.simChunkSize,)),
                ),
                axis=-1,
            )
        self.processor.idx -= self.processor.simChunkSize
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

        if self.processor.idx+self.blockSize >= self.processor.simChunkSize+self.processor.simBuffer:
            self.resetBuffers(globalIdx)

    def process(self):
        self.processor.process(self.blockSize)
        

# class ProcessorWrapper_onlyctrlsources():
#     def __init__(self, processor, arrays, blockSize):
#         #self.config = config
#         self.processor = processor
#         self.arrays = arrays
#         self.blockSize = blockSize

#         self.path_filters = {}
#         for src, mic, path in arrays.iter_paths():
#             if src.name not in self.path_filters:
#                 self.path_filters[src.name] = {}
#             self.path_filters[src.name][mic.name] = FilterSum_IntBuffer(path)
            
#         #self.ctrlSources = list(arrays.of_type(ar.ArrayType.CTRLSOURCE))

#     def prepare(self):
#         self.processor.prepare()

#     def reset(self):
#         self.processor.reset()
    
#     def resetBuffers(self):
#         #self.processor.diag.resetBuffers(self.processor.idx)
#         for sigName, sig in self.processor.sig.items():
#             self.processor.sig[sigName] = np.concatenate(
#                 (
#                     sig[..., -self.processor.simBuffer :],
#                     np.zeros(sig.shape[:-1] + (self.processor.simChunkSize,)),
#                 ),
#                 axis=-1,
#             )

#         self.processor.idx -= self.processor.simChunkSize

#     def propagate(self, freeSourceSig):
#         """ propagated audio from the controlled sources.
#             inputSignals is the audio from the free sources for samples idx:idx+numSamples
#             The mic_signals are calculated for the indices
#             self.processor.idx to self.processor.idx+numSamples

#             calls the processors process function so that it 
#             can calculate new values for the controlled sources"""

#         if self.processor.idx+self.blockSize >= self.processor.simChunkSize+self.processor.simBuffer:
#             self.resetBuffers()

#         i = self.processor.idx

#         for mic in self.arrays.mics():
#             self.processor.sig[mic.name][:,i:i+self.blockSize] = freeSourceSig[mic.name][:,i:i+self.blockSize]

#             for src in self.arrays.of_type(ar.ArrayType.CTRLSOURCE):
#                 propagated_signal = self.path_filters[src.name][mic.name].process(
#                         self.processor.sig[src.name][:,i:i+self.blockSize])
#                 self.processor.sig[mic.name+"~"+src.name][:,i:i+self.blockSize] = propagated_signal
#                 self.processor.sig[mic.name][:,i:i+self.blockSize] += propagated_signal

#     def process(self):
#         self.processor.process(self.blockSize)
#         self.processor.idx += self.blockSize

        

# class FreeSourcePropagator():
#     # make this work similar to processorwrapper.
#     # The functionality should be the same, only that free sources are
#     # not dependent on outside information to generate its data. 
#     # So this class should be able to directly output free source signals after 
#     # they have been propagated to the microphpone positions. 

#     def __init__(self):
#         pass
    
#     def resetBuffers(self):
#         for sigName in self.sig.keys():
#             self.sig[sigName][:,:self.simBuffer] = self.sig[sigName][:,-self.simBuffer:]
#             self.sig[sigName][:,self.simBuffer:] = 0
#         self.idx -= self.simChunkSize

#         assert 0 <= self.idx <= self.simBuffer

#     def prepare(self, config, arrays):
#         self.sigLen = config["SIMCHUNKSIZE"] + config["SIMBUFFER"]
#         self.simChunkSize = config["SIMCHUNKSIZE"]
#         self.simBuffer = config["SIMBUFFER"]
#         self.idx = 0

#         self.sig = {}
#         self.source = {}
#         self.filt = {}
#         for src, mic, path in arrays.iter_paths_from(ar.ArrayType.FREESOURCE):
#             if src.name not in self.filt:
#                 self.filt[src.name] = {}
#             if src.name not in self.sig:
#                 self.sig[src.name] = np.zeros((src.num, self.sigLen))

#             if mic.name not in self.sig:
#                 self.sig[mic.name] = np.zeros((mic.num, self.sigLen))

#             if src.name not in self.source:
#                 self.source[src.name] = src.source

#             self.filt[src.name][mic.name] = FilterSum_IntBuffer(path)
        

#     def process(self, numSamples):
#         if self.idx+numSamples > self.sigLen:
#             self.resetBuffers()

#         for srcName, srcFilters in self.filt.items():
#             srcSamples = self.source[srcName].getSamples(numSamples)
#             for micName, filt in srcFilters.items():
#                 micSamples = filt.process(srcSamples)
#                 self.sig[micName][:,self.idx:self.idx+numSamples] = micSamples
#                 self.sig[srcName][:,self.idx:self.idx+numSamples] = srcSamples

#         self.idx += numSamples



class AudioProcessor(ABC):
    def __init__(self, config, arrays):
        self.simBuffer = config["SIMBUFFER"]
        self.simChunkSize = config["SIMCHUNKSIZE"]
        self.endTimeStep = config["ENDTIMESTEP"]
        self.samplerate = config["SAMPLERATE"]

        self.spatialDims = config["SPATIALDIMS"]
        self.c = config["C"]
        self.micSNR = config["MICSNR"]

        self.saveRawData = config["SAVERAWDATA"]
        self.outputSmoothing = config["OUTPUTSMOOTHING"]
        self.plotFrequency = config["PLOTFREQUENCY"]
        self.genSoundfieldAtChunk = config["GENSOUNDFIELDATCHUNK"]

        self.name = "Abstract Processor"
        self.arrays = arrays
        self.diag = dia.DiagnosticHandler(self.simBuffer, self.simChunkSize)
        self.rng = np.random.default_rng(1)
        self.metadata = {}
        self.idx = 0

        self.sig = {}

    def prepare(self):
        pass

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
        self.idx = self.simBuffer

    def createNewBuffer(self, name, dim):
        """dim is a tuple with the
        buffer is accessed as self.sig[name]
        and will have shape (dim[0], dim[1],...,dim[-1], simbuffer+simchunksize)"""
        if isinstance(dim, int):
            dim = (dim,)
        assert name not in self.sig
        self.sig[name] = np.zeros(dim + (self.simBuffer + self.simChunkSize,))

    @abstractmethod
    def process(self, numSamples):
        """"""
        pass


class VolumeControl(AudioProcessor):
    def __init__(self, config, volumeFactor):
        super().__init__(config)
        self.name = "Volume Control"
        self.volumeFactor = volumeFactor

        self.metadata["volume factor"] = self.volumeFactor

    def process(self, numSamples):
        pass
            


class ActiveNoiseControlProcessor(AudioProcessor):
    def __init__(self, config, arrays):
        super().__init__(config, arrays)
        self.numRef = self.arrays["ref"].num
        self.numError = self.arrays["error"].num
        self.numSpeaker = self.arrays["speaker"].num
        
        self.diag.addNewDiagnostic("noise_reduction", dia.NoiseReduction("error", "source~error", self.endTimeStep, 128, False))

        #self.createNewBuffer("ref", self.numRef)
        #self.createNewBuffer("error", self.numError)
        #self.createNewBuffer("speaker", self.numSpeaker)

        self.updateIdx = self.simBuffer

    def resetBuffer(self):
        super().resetBuffer()
        self.updateIdx -= self.simChunkSize

    def process(self, numSamples):
        #self.diag.saveBlockData()
        self.genSpeakerSignals(numSamples)
        if self.updateIdx < self.idx:
            self.updateFilter()
            self.updateIdx = self.idx

    @abstractmethod
    def genSpeakerSignals(self, numSamples):
        pass

    @abstractmethod
    def updateFilter(self):
        pass


class TDANCProcessor(ActiveNoiseControlProcessor):
    def __init__(self, config, arrays, controlFiltLen):
        super().__init__(config, arrays)
        self.filtLen = controlFiltLen
        self.controlFilter = FilterSum_IntBuffer(irLen=self.filtLen, numIn=self.numRef, numOut=self.numSpeaker)

    def genSpeakerSignals(self, numSamples):
        """Borde bytas till att nygenererade speaker samples är från self.idx:self.idx+numSamples, men
           att man bara har tillgång till mic-samples fram till self.idx."""
        self.sig["speaker"][:,self.idx:self.idx+numSamples] = \
            self.controlFilter.process(self.sig["ref"][:,self.idx-numSamples:self.idx])


class FxLMS_block(TDANCProcessor):
    """Fully block based operation. Is equivalent to FastBlockFxLMS"""
    def __init__(self, config, arrays, controlFiltLen, mu, beta, secPath):
        super().__init__(config, arrays, controlFiltLen)
        self.name = "FxLMS Feedforward"
        self.mu = mu
        self.beta = beta

        self.secPathFilt = FilterMD_IntBuffer((self.numRef,), secPath)
        self.createNewBuffer("xf", (self.numRef, self.numSpeaker, self.numError))

    def prepare(self):
        self.sig["xf"][:, :, :, : self.idx] = np.transpose(
            self.secPathFilt.process(self.sig["ref"][:, : self.idx]), (2, 0, 1, 3)
        )

    def updateFilter(self):
        self.sig["xf"][:, :, :, self.updateIdx : self.idx] = np.transpose(
            self.secPathFilt.process(self.sig["ref"][:, self.updateIdx : self.idx]),
            (2, 0, 1, 3),
        )

        grad = np.zeros_like(self.controlFilter.ir)
        for n in range(self.updateIdx, self.idx):
            Xf = np.flip(
                self.sig["xf"][:, :, :, n - self.filtLen + 1 : n + 1], axis=-1
            )
            grad += np.sum(Xf * self.sig["error"][None, None, :, n:n+1], axis=2)

        norm = 1 / (
            np.sum(self.sig["xf"][:, :, :, self.updateIdx : self.idx] ** 2)
            + self.beta
        )
        self.controlFilter.ir -= grad * self.mu * norm





class AdaptiveFilterFF(ActiveNoiseControlProcessor):
    def __init__(self, config, mu, beta, speakerRIR):
        self.name = "Adaptive Filter Timedomain Baseclass"
        self.numRef = config["NUMREF"]
        self.numError = config["NUMERROR"]
        self.numSpeaker = config["NUMSPEAKER"]
        self.numTarget = config["NUMTARGET"]
        self.filtLen = config["FILTLENGTH"]
        self.micSNR = config["MICSNR"]
        self.saveRawData = config["SAVERAWDATA"]
        self.outputSmoothing = config["OUTPUTSMOOTHING"]
        self.plotFrequency = config["PLOTFREQUENCY"]
        self.genSoundfieldAtChunk = config["GENSOUNDFIELDATCHUNK"]
        self.c = config["C"]
        self.samplerate = config["SAMPLERATE"]
        self.spatialDims = config["SPATIALDIMS"]
        self.endTimeStep = config["ENDTIMESTEP"]
        self.simChunkSize = config["SIMCHUNKSIZE"]
        self.simBuffer = config["SIMBUFFER"]
        self.mu = mu
        self.beta = beta

        self.rng = np.random.RandomState(5)

        self.controlFilt = FilterSum_IntBuffer(
            np.zeros((self.numRef, self.numSpeaker, self.filtLen))
        )

        self.buffers = {}

        self.y = np.zeros((self.numSpeaker, self.simChunkSize + self.simBuffer))
        self.x = np.zeros((self.numRef, self.simChunkSize + self.simBuffer))
        self.e = np.zeros((self.numError, self.simChunkSize + self.simBuffer))

        self.secPathFilt = FilterSum_IntBuffer(speakerRIR["error"])

        self.diag = DiagnosticHandler(self.simBuffer, self.simChunkSize)
        self.diag.addNewDiagnostic(
            "reduction_microphones",
            NoiseReductionExternalSignals(
                self.numError,
                self.endTimeStep,
                self.simBuffer,
                self.simChunkSize,
                smoothingLen=self.outputSmoothing,
                saveRawData=self.saveRawData,
                plotFrequency=self.plotFrequency,
            ),
        )
        self.diag.addNewDiagnostic(
            "reduction_regional",
            NoiseReduction(
                self.numTarget,
                speakerRIR["target"],
                self.endTimeStep,
                self.simBuffer,
                self.simChunkSize,
                smoothingLen=self.outputSmoothing,
                saveRawData=self.saveRawData,
                plotFrequency=self.plotFrequency,
            ),
        )
        # self.diag.addNewDiagnostic("soundfield_target",
        #                             SoundfieldImage(self.numEvals, speakerRIR["evals"],
        #                             self.simBuffer, self.simChunkSize,
        #                             beginAtBuffer=self.genSoundfieldAtChunk,
        #                             plotFrequency=self.plotFrequency))

        self.idx = self.simBuffer
        self.updateIdx = self.simBuffer

        self.metadata = {"mu": mu, "beta": beta, "controlFiltLen": self.filtLen}

    def prepare(self):
        """Will be called once, after buffers are filled, right before regular operations
        Nothing required for forward pass can be set up here, but any initialization
        needed for the filter update is fine. Calculating filtered references with reference
        signals from before time 0 is a good use."""
        pass

    def forwardPassImplement(self, numSamples):
        """Must insert self.y[:,self.idx:self.idx:numSamples] with new values.
        self.x[:,self.idx:self.idx:numSamples] are new reference signal values
        last valid index for self.e is self.idx-1"""
        self.y[:, self.idx : self.idx + numSamples] = self.controlFilt.process(
            self.x[:, self.idx : self.idx + numSamples]
        )

    def outputRawData(self):
        return self.controlFilt.ir

    @abstractmethod
    def updateFilter(self):
        pass

    def resetBuffers(self):
        self.diag.saveBufferData("reduction_microphones", self.idx, self.e)
        self.diag.saveBufferData("reduction_regional", self.idx, self.y)
        # self.diag.saveBufferData("soundfield_target", self.idx, self.y)
        self.diag.resetBuffers(self.idx)

        self.y = np.concatenate(
            (
                self.y[:, -self.simBuffer :],
                np.zeros((self.y.shape[0], self.simChunkSize)),
            ),
            axis=-1,
        )
        self.x = np.concatenate(
            (
                self.x[:, -self.simBuffer :],
                np.zeros((self.x.shape[0], self.simChunkSize)),
            ),
            axis=-1,
        )
        self.e = np.concatenate(
            (
                self.e[:, -self.simBuffer :],
                np.zeros((self.e.shape[0], self.simChunkSize)),
            ),
            axis=-1,
        )

        for bufName, buf in self.buffers.items():
            self.buffers[bufName] = np.concatenate(
                (
                    buf[..., -self.simBuffer :],
                    np.zeros(buf.shape[:-1] + (self.simChunkSize,)),
                ),
                axis=-1,
            )

        self.idx -= self.simChunkSize
        self.updateIdx -= self.simChunkSize

    def forwardPass(self, numSamples, noises):
        blockSizes = calcBlockSizes(
            numSamples, self.idx, self.simBuffer, self.simChunkSize
        )
        errorMicNoise = getWhiteNoiseAtSNR(
            self.rng, noises["error"], (self.numError, numSamples), self.micSNR
        )
        refMicNoise = getWhiteNoiseAtSNR(
            self.rng, noises["ref"], (self.numRef, numSamples), self.micSNR
        )

        numComputed = 0
        for blockSize in blockSizes:
            self.x[:, self.idx : self.idx + blockSize] = (
                noises["ref"][:, numComputed : numComputed + blockSize]
                + refMicNoise[:, numComputed : numComputed + blockSize]
            )
            self.forwardPassImplement(blockSize)

            self.diag.saveBlockData(
                "reduction_microphones",
                self.idx,
                noises["error"][:, numComputed : numComputed + blockSize],
            )
            self.diag.saveBlockData(
                "reduction_regional",
                self.idx,
                noises["target"][:, numComputed : numComputed + blockSize],
            )
            # self.diag.saveBlockData("soundfield_target", self.idx, noiseAtEvals[:,numComputed:numComputed+blockSize])

            self.e[:, self.idx : self.idx + blockSize] = (
                noises["error"][:, numComputed : numComputed + blockSize]
                + errorMicNoise[:, numComputed : numComputed + blockSize]
                + self.secPathFilt.process(self.y[:, self.idx : self.idx + blockSize])
            )

            assert self.idx + blockSize <= self.simBuffer + self.simChunkSize
            if self.idx + blockSize >= (self.simChunkSize + self.simBuffer):
                self.resetBuffers()
            self.idx += blockSize
            numComputed += blockSize


class AdaptiveFilterFFComplex(ActiveNoiseControlProcessor):
    def __init__(self, config, mu, beta, speakerRIR, blockSize):
        self.name = "Adaptive Filter Feedforward Complex"

        self.numRef = config["NUMREF"]
        self.numError = config["NUMERROR"]
        self.numSpeaker = config["NUMSPEAKER"]
        self.numTarget = config["NUMTARGET"]
        self.micSNR = config["MICSNR"]
        self.saveRawData = config["SAVERAWDATA"]
        self.outputSmoothing = config["OUTPUTSMOOTHING"]
        self.plotFrequency = config["PLOTFREQUENCY"]
        self.genSoundfieldAtChunk = config["GENSOUNDFIELDATCHUNK"]
        self.c = config["C"]
        self.samplerate = config["SAMPLERATE"]
        self.spatialDims = config["SPATIALDIMS"]
        self.endTimeStep = config["ENDTIMESTEP"]
        self.simChunkSize = config["SIMCHUNKSIZE"]
        self.simBuffer = config["SIMBUFFER"]
        self.mu = mu
        self.beta = beta
        self.blockSize = blockSize

        self.rng = np.random.RandomState(5)

        self.H = np.zeros(
            (2 * blockSize, self.numSpeaker, self.numRef), dtype=np.complex128
        )
        self.controlFilt = FilterSum_Freqdomain(
            numIn=self.numRef, numOut=self.numSpeaker, irLen=blockSize
        )
        self.buffers = {}

        self.x = np.zeros((self.numRef, self.simChunkSize + self.simBuffer))
        self.e = np.zeros((self.numError, self.simChunkSize + self.simBuffer))
        self.y = np.zeros((self.numSpeaker, self.simChunkSize + self.simBuffer))
        self.buffers["primary"] = np.zeros((self.numError, self.simChunkSize + self.simBuffer))
        self.buffers["yf"] = np.zeros((self.numError, self.simChunkSize + self.simBuffer))

        self.G = np.transpose(
            np.fft.fft(speakerRIR["error"], n=2 * blockSize, axis=-1), (2, 0, 1)
        )

        self.secPathFilt = FilterSum_IntBuffer(speakerRIR["error"])

        self.diag = DiagnosticHandler(self.simBuffer, self.simChunkSize)
        self.diag.addNewDiagnostic(
            "reduction_microphones",
            NoiseReductionExternalSignals(
                self.numError,
                self.endTimeStep,
                self.simBuffer,
                self.simChunkSize,
                smoothingLen=self.outputSmoothing,
                saveRawData=self.saveRawData,
                plotFrequency=self.plotFrequency,
            ),
        )
        if "target" in speakerRIR:
            self.diag.addNewDiagnostic(
                "reduction_regional",
                NoiseReduction(
                    self.numTarget,
                    speakerRIR["target"],
                    self.endTimeStep,
                    self.simBuffer,
                    self.simChunkSize,
                    smoothingLen=self.outputSmoothing,
                    saveRawData=self.saveRawData,
                    plotFrequency=self.plotFrequency,
                ),
            )

        
        self.diag.addNewDiagnostic(
            "power_primary_noise",
                RecordScalar(self.endTimeStep, self.simBuffer, self.simChunkSize,plotFrequency=self.plotFrequency)
        )
        # self.diag.addNewDiagnostic("soundfield_target",
        #                             SoundfieldImage(self.numEvals, speakerRIR["evals"],
        #                             self.simBuffer, self.simChunkSize,
        #                             beginAtBuffer=self.genSoundfieldAtChunk,
        #                             plotFrequency=self.plotFrequency))
        self.idx = self.simBuffer
        self.updateIdx = self.simBuffer

        self.metadata = {"mu": mu, "beta": beta, "controlFiltNumFreq": 2 * blockSize}

    def outputRawData(self):
        """Can be extended to include other filters and quantities"""
        return self.controlFilt.tf

    def prepare(self):
        pass

    @abstractmethod
    def forwardPassImplement(self):
        pass

    @abstractmethod
    def updateFilter(self):
        pass

    def resetBuffers(self):
        self.diag.saveBufferData("reduction_microphones", self.idx, self.e)
        if self.diag.hasDiagnostic("reduction_regional"):
            self.diag.saveBufferData("reduction_regional", self.idx, self.y)
        # self.diag.saveBufferData("soundfield_target", self.idx, self.y)
        self.diag.resetBuffers(self.idx)

        for bufName, buf in self.buffers.items():
            self.buffers[bufName] = np.concatenate(
                (
                    buf[..., -self.simBuffer :],
                    np.zeros(buf.shape[:-1] + (self.simChunkSize,)),
                ),
                axis=-1,
            )

        self.x = np.concatenate(
            (
                self.x[:, -self.simBuffer :],
                np.zeros((self.x.shape[0], self.simChunkSize)),
            ),
            axis=-1,
        )
        self.e = np.concatenate(
            (
                self.e[:, -self.simBuffer :],
                np.zeros((self.e.shape[0], self.simChunkSize)),
            ),
            axis=-1,
        )

        self.idx -= self.simChunkSize
        self.updateIdx -= self.simChunkSize

    def forwardPass(self, numSamples, noises):
        assert numSamples == self.blockSize

        if self.idx + numSamples >= (self.simChunkSize + self.simBuffer):
            self.resetBuffers()

        errorMicNoise = getWhiteNoiseAtSNR(
            self.rng, noises["error"], (self.numError, numSamples), self.micSNR
        )
        refMicNoise = getWhiteNoiseAtSNR(
            self.rng, noises["ref"], (self.numRef, numSamples), self.micSNR
        )

        self.x[:, self.idx : self.idx + numSamples] = noises["ref"] + refMicNoise
        self.forwardPassImplement(numSamples)

        self.buffers["primary"][:,self.idx:self.idx+numSamples] = noises["error"]
        self.diag.saveBlockData("reduction_microphones", self.idx, noises["error"])
        if self.diag.hasDiagnostic("reduction_regional"):
            self.diag.saveBlockData("reduction_regional", self.idx, noises["target"])

    
        self.diag.saveBlockData("power_primary_noise", self.idx, np.mean(noises["error"]**2))

        self.buffers["yf"][:,self.idx:self.idx+numSamples] = self.secPathFilt.process(self.y[:, self.idx : self.idx + numSamples])

        self.e[:, self.idx : self.idx + numSamples] = (
            noises["error"]
            + errorMicNoise
            + self.buffers["yf"][:,self.idx:self.idx+numSamples]
            )
        

        self.idx += numSamples
