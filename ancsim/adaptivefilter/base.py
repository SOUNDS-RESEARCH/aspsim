import numpy as np
from abc import ABC, abstractmethod

import ancsim.settings as s
import ancsim.utilities as util
from ancsim.signal.filterclasses import Filter_IntBuffer, FilterSum_IntBuffer, FilterMD_IntBuffer, FilterSum_Freqdomain
from ancsim.adaptivefilter.util import calcBlockSizes, getWhiteNoiseAtSNR
from ancsim.adaptivefilter.diagnostics import DiagnosticHandler, NoiseReductionExternalSignals
from ancsim.adaptivefilter.diagnostics import NoiseReduction, SoundfieldImage, ConstantEstimateNMSE


class AdaptiveFilterFF(ABC):
    def __init__(self, config, mu, beta, speakerRIR):
        self.name = "Adaptive Filter Timedomain Baseclass"
        self.filtLen = config["FILTLENGTH"]
        self.errorMicSNR = config["ERRORMICSNR"]
        self.refMicSNR = config["REFMICSNR"]
        self.mu = mu
        self.beta = beta

        self.controlFilt = FilterSum_IntBuffer(np.zeros((s.NUMREF,s.NUMSPEAKER,self.filtLen)))
        

        self.buffers = {}

        self.y = np.zeros((s.NUMSPEAKER,s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.x = np.zeros((s.NUMREF,s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.e = np.zeros((s.NUMERROR,s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.secPathFilt = FilterSum_IntBuffer(speakerRIR["error"])

        self.diag = DiagnosticHandler()
        self.diag.addNewDiagnostic("reduction_microphones", 
                                    NoiseReductionExternalSignals(s.NUMERROR, 
                                    plotFrequency=s.PLOTFREQUENCY))
        self.diag.addNewDiagnostic("reduction_regional", 
                                    NoiseReduction(s.NUMTARGET, speakerRIR["target"], 
                                    plotFrequency=s.PLOTFREQUENCY))
        self.diag.addNewDiagnostic("soundfield_target", 
                                    SoundfieldImage(s.NUMEVALS, speakerRIR["evals"], 
                                    beginAtBuffer=s.GENSOUNDFIELDATCHUNK,
                                    plotFrequency=s.PLOTFREQUENCY))

        self.idx = s.SIMBUFFER
        self.updateIdx = s.SIMBUFFER

        self.metadata = {"mu" : mu,
                         "beta" : beta,
                         "controlFiltLen" : self.filtLen}

    def prepare(self):
        """Will be called once, after buffers are filled, right before regular operations
            Nothing required for forward pass can be set up here, but any initialization
            needed for the filter update is fine. Calculating filtered references with reference
            signals from before time 0 is a good use. """
        pass

    def forwardPassImplement(self, numSamples):
        """Must insert self.y[:,self.idx:self.idx:numSamples] with new values. 
            self.x[:,self.idx:self.idx:numSamples] are new reference signal values
            last valid index for self.e is self.idx-1"""
        self.y[:,self.idx:self.idx+numSamples] = self.controlFilt.process(self.x[:,self.idx:self.idx+numSamples])

    def outputRawData(self):
        return self.controlFilt.ir

    @abstractmethod
    def updateFilter(self):
        pass

    def resetBuffers(self):
        self.diag.saveBufferData("reduction_microphones", self.idx, self.e)
        self.diag.saveBufferData("reduction_regional", self.idx, self.y)
        self.diag.saveBufferData("soundfield_target", self.idx, self.y)
        self.diag.resetBuffers(self.idx)

        self.y = np.concatenate((self.y[:,-s.SIMBUFFER:], np.zeros((self.y.shape[0], s.SIMCHUNKSIZE))) ,axis=-1)
        self.x = np.concatenate((self.x[:,-s.SIMBUFFER:], np.zeros((self.x.shape[0], s.SIMCHUNKSIZE))) ,axis=-1)
        self.e = np.concatenate((self.e[:,-s.SIMBUFFER:], np.zeros((self.e.shape[0], s.SIMCHUNKSIZE))) ,axis=-1)

        for bufName, buf in self.buffers.items():
            self.buffers[bufName] = np.concatenate((buf[...,-s.SIMBUFFER:], np.zeros(buf.shape[:-1] + (s.SIMCHUNKSIZE,))), axis=-1)

        self.idx -= s.SIMCHUNKSIZE
        self.updateIdx -= s.SIMCHUNKSIZE

    def forwardPass(self, numSamples, noiseAtError, noiseAtRef, noiseAtTarget, noiseAtEvals):
        blockSizes = calcBlockSizes(numSamples, self.idx)
        errorMicNoise = getWhiteNoiseAtSNR(noiseAtError, (s.NUMERROR, numSamples), self.errorMicSNR)
        refMicNoise = getWhiteNoiseAtSNR(noiseAtRef, (s.NUMREF, numSamples), self.refMicSNR)

        numComputed = 0
        for blockSize in blockSizes:
            self.x[:,self.idx:self.idx+blockSize] = noiseAtRef[:,numComputed:numComputed+blockSize] + \
                                                    refMicNoise[:,numComputed:numComputed+blockSize]
            self.forwardPassImplement(blockSize)

            self.diag.saveBlockData("reduction_microphones", self.idx, noiseAtError[:,numComputed:numComputed+blockSize])
            self.diag.saveBlockData("reduction_regional", self.idx, noiseAtTarget[:,numComputed:numComputed+blockSize])
            self.diag.saveBlockData("soundfield_target", self.idx, noiseAtEvals[:,numComputed:numComputed+blockSize])
            
            self.e[:,self.idx:self.idx+blockSize] = noiseAtError[:,numComputed:numComputed+blockSize] + \
                                                    errorMicNoise[:,numComputed:numComputed+blockSize] + \
                                                    self.secPathFilt.process(self.y[:,self.idx:self.idx+blockSize])

            assert(self.idx+blockSize <= s.SIMBUFFER+s.SIMCHUNKSIZE)
            if self.idx + blockSize >= (s.SIMCHUNKSIZE + s.SIMBUFFER):
                self.resetBuffers()
            self.idx += blockSize
            numComputed += blockSize


class AdaptiveFilterFFComplex(ABC):
    def __init__(self, config, mu, beta, speakerRIR, blockSize):
        self.name = "Adaptive Filter Feedforward Complex"
        self.H = np.zeros((2*blockSize, s.NUMSPEAKER, s.NUMREF), dtype=np.complex128)
        self.controlFilt = FilterSum_Freqdomain(numIn=s.NUMREF, numOut=s.NUMSPEAKER, irLen=blockSize)
        self.mu = mu
        self.beta = beta
        self.blockSize = blockSize

        self.buffers = {}

        self.x = np.zeros((s.NUMREF,s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.e = np.zeros((s.NUMERROR,s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.y = np.zeros((s.NUMSPEAKER,s.SIMCHUNKSIZE+s.SIMBUFFER))

        # self.G = np.transpose(np.fft.fft(speakerRIR["error"],n=2*blockSize,axis=-1), 
        #                       [speakerRIR["error"].ndim-1,] + 
        #                       [i for i in range(speakerRIR["error"].ndim-1)])
        self.G = np.transpose(np.fft.fft(speakerRIR["error"],n=2*blockSize,axis=-1),(2,0,1))
        

        self.secPathFilt = FilterSum_IntBuffer(speakerRIR["error"])

        self.diag = DiagnosticHandler()
        self.diag.addNewDiagnostic("reduction_microphones", 
                                    NoiseReductionExternalSignals(s.NUMERROR, 
                                    plotFrequency=s.PLOTFREQUENCY))
        self.diag.addNewDiagnostic("reduction_regional", 
                                    NoiseReduction(s.NUMTARGET, speakerRIR["target"], 
                                    plotFrequency=s.PLOTFREQUENCY))
        self.diag.addNewDiagnostic("soundfield_target", 
                                    SoundfieldImage(s.NUMEVALS, speakerRIR["evals"], 
                                    beginAtBuffer=s.GENSOUNDFIELDATCHUNK,
                                    plotFrequency=s.PLOTFREQUENCY))
        self.idx = s.SIMBUFFER
        self.updateIdx = s.SIMBUFFER

        self.metadata = {"mu" : mu,
                        "beta" : beta,
                        "controlFiltNumFreq" : 2*blockSize}
                        
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
        self.diag.saveBufferData("reduction_regional", self.idx, self.y)
        self.diag.saveBufferData("soundfield_target", self.idx, self.y)
        self.diag.resetBuffers(self.idx)
        #self.diag.saveDiagnostics(self.idx, e=self.e, y=self.y)

        for bufName, buf in self.buffers.items():
            self.buffers[bufName] = np.concatenate((buf[...,-s.SIMBUFFER:], np.zeros(buf.shape[:-1] + (s.SIMCHUNKSIZE,))), axis=-1)


        self.x = np.concatenate((self.x[:,-s.SIMBUFFER:], np.zeros((self.x.shape[0], s.SIMCHUNKSIZE))) ,axis=-1)
        self.e = np.concatenate((self.e[:,-s.SIMBUFFER:], np.zeros((self.e.shape[0], s.SIMCHUNKSIZE))) ,axis=-1)

        self.idx -= s.SIMCHUNKSIZE
        self.updateIdx -= s.SIMCHUNKSIZE

    def forwardPass(self, numSamples, noiseAtError, noiseAtRef, noiseAtTarget, noiseAtEvals):
        assert (numSamples == self.blockSize)

        if self.idx+numSamples >= (s.SIMCHUNKSIZE + s.SIMBUFFER):
            self.resetBuffers()

        errorMicNoise = getWhiteNoiseAtSNR(noiseAtError, (s.NUMERROR, numSamples), s.ERRORMICNOISE)
        refMicNoise = getWhiteNoiseAtSNR(noiseAtRef, (s.NUMREF, numSamples), s.REFMICNOISE)

        self.x[:,self.idx:self.idx+numSamples] = noiseAtRef + refMicNoise
        #self.diag.saveToBuffers(self.idx, noiseAtError, noiseAtTarget, noiseAtEvals, noiseAtEvals2)

        self.forwardPassImplement(numSamples)

        self.diag.saveBlockData("reduction_microphones", self.idx, noiseAtError)
        self.diag.saveBlockData("reduction_regional", self.idx, noiseAtTarget)
        self.diag.saveBlockData("soundfield_target", self.idx, noiseAtEvals)

        self.e[:,self.idx:self.idx+numSamples] = noiseAtError + errorMicNoise + \
                                                self.secPathFilt.process(self.y[:,self.idx:self.idx+numSamples])

        self.idx += numSamples
        