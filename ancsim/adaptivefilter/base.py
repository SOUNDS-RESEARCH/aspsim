import numpy as np
from abc import ABC, abstractmethod

import ancsim.utilities as util
from ancsim.signal.filterclasses import (
    Filter_IntBuffer,
    FilterSum_IntBuffer,
    FilterMD_IntBuffer,
    FilterSum_Freqdomain,
)
from ancsim.adaptivefilter.util import calcBlockSizes, getWhiteNoiseAtSNR
from ancsim.adaptivefilter.diagnostics import (
    DiagnosticHandler,
    NoiseReductionExternalSignals,
)
from ancsim.adaptivefilter.diagnostics import (
    NoiseReduction,
    SoundfieldImage,
    ConstantEstimateNMSE,
    RecordScalar,
)


class AudioProcessor(ABC):
    def __init__(self, config):
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

        self.buf = {}

        self.diag = DiagnosticHandler(self.simBuffer, self.simChunkSize)

        self.idx = self.simBuffer

        self.metadata = {}

    def reset(self):
        """Should probably reset diagnostics to be properly reset
            Also, all user-defined signals and filters must be reset """
        for bufName, buf in self.buf.items():
            buf.fill(0)
        self.idx = self.simBuffer

    def createNewBuffer(name, dim):
        """dim is a tuple with the
        buffer is accessed as self.buf[name]
        and will have shape (dim[0], dim[1],...,dim[-1], simbuffer+simchunksize)"""
        if isinstance(dim, int):
            dim = (dim,)
        assert name not in self.buf
        self.buf[name] = np.zeros(dim + (self.simBuffer + self.simChunkSize,))

    @abstractmethod
    def process(self, numSamples, inputSignals):
        pass

    def resetBuffers(self):
        self.diag.resetBuffers(self.idx)

        for bufName, buf in self.buf.items():
            self.buffers[bufName] = np.concatenate(
                (
                    buf[..., -s.SIMBUFFER :],
                    np.zeros(buf.shape[:-1] + (self.simChunkSize,)),
                ),
                axis=-1,
            )

        self.idx -= self.simChunkSize


class ActiveNoiseControlProcessor(AudioProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.numRef = config["NUMREF"]
        self.numError = config["NUMERROR"]
        self.numSpeaker = config["NUMSPEAKER"]
        self.numTarget = config["NUMTARGET"]
        self.filtLen = config["FILTLENGTH"]

        self.createNewBuffer("ref", self.numRef)
        self.createNewBuffer("error", self.numError)
        self.createNewBuffer("speaker", self.numSpeaker)
        self.createNewBuffer("target", self.numTarget)

        self.updateIdx = self.simBuffer

    def process(self, numSamples, noises):
        self.forwardPass(numSamples, noises)
        self.updateFilter()

    @abstractmethod
    def forwardPass(self):
        pass

    @abstractmethod
    def updateFilter(self):
        pass

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
