import numpy as np
from copy import deepcopy
from enum import Enum
from abc import ABC, abstractmethod
from functools import partial
import itertools as it

import ancsim.utilities as util
import ancsim.experiment.plotscripts as psc
from ancsim.signal.filterclasses import (
    Filter_IntBuffer,
    FilterSum_IntBuffer,
    FilterMD_IntBuffer,
)
import ancsim.adaptivefilter.diagnosticplots as dplot
import ancsim.adaptivefilter.diagnosticsummary as dsum


class DIAGNOSTICTYPE(Enum):
    perSimBuffer = 1
    perSample = 2
    perBlock = 3


class OUTPUTFUNCTION(Enum):
    functionoftime = 1
    soundfield = 2
    savenpz = 3


class SUMMARYFUNCTION(Enum):
    none = 0
    lastValue = 1
    meanNearTimeIdx = 2


class PlotDiagnosticDispatcher:
    def __init__(self, filters, outputFormat):
        self.outputFormat = outputFormat
        self.funcInfo = self.getUniqueFuncInfo(filters)
        self.checkDiagnosticDefinitions(filters)

    def checkDiagnosticDefinitions(self, filters):
        """Check that all diagnostics with the same identifier
        has the same functionality defined"""
        for diagName, info in self.funcInfo.items():
            for filt in filters:
                if filt.diag.hasDiagnostic(diagName):
                    assert info == filt.diag.diagnostics[diagName].info

    def plotThisBuffer(self, bufferIdx, funcInfo):
        return bufferIdx >= funcInfo.beginAtBuffer and (
            bufferIdx % funcInfo.plotFrequency == 0 or bufferIdx - 1 == 0
        )

    def dispatch(self, filters, timeIdx, bufferIdx, folder):
        for diagName, funcInfo in self.funcInfo.items():
            if self.plotThisBuffer(bufferIdx, funcInfo):
                diagSet = self.getDiagnosticSet(diagName, filters)
                self.dispatchPlot(funcInfo, diagName, diagSet, timeIdx, folder)
                self.dispatchSummary(funcInfo, diagName, diagSet, timeIdx, folder)

    def dispatchPlot(self, funcInfo, diagName, diagSet, timeIdx, folder):

        if not isinstance(funcInfo.outputType, (list, tuple)):
            funcInfo.outputType = [funcInfo.outputType]
        for i, outType in enumerate(funcInfo.outputType):
            if len(funcInfo.outputType) == 1:
                outputs = {key: diag.getOutput() for key, diag in diagSet.items()}
            elif len(funcInfo.outputType) > 1:
                outputs = {key: diag.getOutput()[i] for key, diag in diagSet.items()}
            metadata = diagSet[list(diagSet.keys())[0]].metadata
            outType(diagName, outputs, metadata, timeIdx, folder, self.outputFormat)

        # if funcInfo.outputType == OUTPUTFUNCTION.functionoftime:
        #     dplot.functionOfTimePlot(diagName, diagSet, timeIdx, folder, self.outputFormat)
        # elif funcInfo.outputType == OUTPUTFUNCTION.soundfield:
        #     dplot.soundfieldPlot(diagName, diagSet, timeIdx, folder, self.outputFormat)
        # else:
        #     raise NotImplementedError

    def dispatchSummary(self, funcInfo, diagName, diagSet, timeIdx, folder):
        if not isinstance(funcInfo.summaryType, (list, tuple)):
            funcInfo.summaryType = [funcInfo.summaryType]

        for i, sumType in enumerate(funcInfo.summaryType):
            if len(funcInfo.summaryType) == 1:
                outputs = {key: diag.getOutput() for key, diag in diagSet.items()}
            elif len(funcInfo.summaryType) > 1:
                outputs = {key: diag.getOutput()[i] for key, diag in diagSet.items()}

            if sumType == SUMMARYFUNCTION.none:
                return
            elif sumType == SUMMARYFUNCTION.lastValue:
                summaryValues = dsum.lastValue(outputs, timeIdx)
            elif sumType == SUMMARYFUNCTION.meanNearTimeIdx:
                summaryValues = dsum.meanNearTimeIdx(outputs, timeIdx)
            else:
                raise NotImplementedError
            dsum.addToSummary(diagName, summaryValues, timeIdx, folder)

    def getDiagnosticSet(self, diagnosticName, filters):
        diagnostic = {}
        for filt in filters:
            if filt.diag.hasDiagnostic(diagnosticName):
                diagnostic[filt.name] = filt.diag.diagnostics[diagnosticName]
        return diagnostic

    def getUniqueFuncInfo(self, filters):
        funcInfo = {}
        for filt in filters:
            for key in filt.diag.diagnostics.keys():
                if key not in funcInfo:
                    funcInfo[key] = filt.diag.getFuncInfo(key)
        return funcInfo


class DiagnosticHandler:
    def __init__(self, simBuffer, simChunkSize):
        self.simBuffer = simBuffer
        self.simChunkSize = simChunkSize

        self.diagnostics = {}
        self.perSimBufferDiagnostics = []
        self.perBlockDiagnostics = []
        self.perSampleDiagnostics = []

        self.bufferIdx = 0
        self.diagnosticSamplesLeft = 0

    def getFuncInfo(self, name):
        return self.diagnostics[name].info

    def hasDiagnostic(self, name):
        return name in self.diagnostics

    def addNewDiagnostic(self, name, diagnostic):
        if diagnostic.info.type == DIAGNOSTICTYPE.perSimBuffer:
            self.perSimBufferDiagnostics.append(name)
        elif diagnostic.info.type == DIAGNOSTICTYPE.perBlock:
            self.perBlockDiagnostics.append(name)
        elif diagnostic.info.type == DIAGNOSTICTYPE.perSample:
            self.perSampleDiagnostics.append(name)
        else:
            raise ValueError
        self.diagnostics[name] = diagnostic

    def saveBlockData(self, name, idx, *args):
        self.diagnostics[name].saveBlockData(idx, *args)

    def saveBufferData(self, name, idx, *args):
        if name in self.perBlockDiagnostics or name in self.perSampleDiagnostics:
            startIdx = self.simBuffer - self.diagnosticSamplesLeft
            numLeftOver = self.simBuffer + self.simChunkSize - idx
            saveStartIdx = (
                self.bufferIdx * self.simChunkSize - self.diagnosticSamplesLeft
            )
            saveEndIdx = (self.bufferIdx + 1) * self.simChunkSize - numLeftOver

            self.diagnostics[name].saveDiagnostics(
                startIdx, idx, saveStartIdx, saveEndIdx, *args
            )
        elif name in self.perSimBufferDiagnostics:
            self.diagnostics[name].saveDiagnostics(self.bufferIdx, *args)

    def resetBuffers(self, idx):
        """Must be called after simulation buffer resets.
        Calls resetBuffer() for all owned diagnostic objects, and increments counters"""
        for key, val in self.diagnostics.items():
            val.resetBuffers()
        numLeftOver = self.simBuffer + self.simChunkSize - idx
        self.diagnosticSamplesLeft = numLeftOver
        self.bufferIdx += 1


class DiagnosticFunctionalityInfo:
    def __init__(
        self,
        diagnosticType,
        outputType,
        summaryType=SUMMARYFUNCTION.none,
        beginAtBuffer=0,
        plotFrequency=1,
    ):
        self.type = diagnosticType  # Use the appropriate Enum
        self.outputType = outputType  # Use the appropriate Enum
        self.summaryType = summaryType  # Use the appropriate Enum
        self.beginAtBuffer = beginAtBuffer
        self.plotFrequency = plotFrequency  # Unit is number of simulator-buffers

    def __eq__(self, other):
        return (
            self.type == other.type
            and self.outputType == other.outputType
            and self.summaryType == other.summaryType
            and self.beginAtBuffer == other.beginAtBuffer
            and self.plotFrequency == other.plotFrequency
        )


class SoundfieldImage:
    def __init__(
        self,
        numPoints,
        acousticPath,
        simBuffer,
        simChunkSize,
        beginAtBuffer=0,
        plotFrequency=1,
    ):
        self.simBuffer = simBuffer
        self.simChunkSize = simChunkSize
        self.info = DiagnosticFunctionalityInfo(
            DIAGNOSTICTYPE.perSimBuffer,
            dplot.soundfieldPlot,
            SUMMARYFUNCTION.none,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
        )
        self.samplesForSF = np.min((2048, self.simBuffer, self.simChunkSize))
        self.totalNoise = np.zeros((numPoints, self.samplesForSF))
        self.primaryNoise = np.zeros((numPoints, self.simChunkSize + self.simBuffer))
        self.pathFilter = FilterSum_IntBuffer(acousticPath)

    def getOutput(self):
        return self.totalNoise

    def resetBuffers(self):
        pass

    def saveDiagnostics(self, chunkIdx, y):
        if (
            chunkIdx >= self.info.beginAtBuffer - 1
            and (chunkIdx + 1) % self.info.plotFrequency == 0
        ):
            startIdx = self.simBuffer
            yFiltered = self.pathFilter.process(
                y[
                    :,
                    startIdx - self.pathFilter.irLen + 1 : startIdx + self.samplesForSF,
                ]
            )
            self.totalNoise[:, :] = (
                self.primaryNoise[:, startIdx : startIdx + self.samplesForSF]
                + yFiltered[:, -self.samplesForSF :]
            )

    def saveBlockData(self, idx, newPrimaryNoise):
        numSamples = newPrimaryNoise.shape[-1]
        self.primaryNoise[:, idx : idx + numSamples] = newPrimaryNoise


class RecordSpectrum:
    def __init__(
        self,
        numPoints,
        acousticPath,
        simBuffer,
        simChunkSize,
        beginAtBuffer=0,
        plotFrequency=1,
    ):
        self.simBuffer = simBuffer
        self.simChunkSize = simChunkSize
        self.info = DiagnosticFunctionalityInfo(
            DIAGNOSTICTYPE.perSimBuffer,
            dplot.soundfieldPlot,
            SUMMARYFUNCTION.none,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
        )
        self.samplesForSF = np.min((2048, self.simBuffer, self.simChunkSize))
        self.totalNoise = np.zeros((numPoints, self.samplesForSF))
        self.primaryNoise = np.zeros((numPoints, self.simChunkSize + self.simBuffer))
        self.pathFilter = FilterSum_IntBuffer(acousticPath)

    def getOutput(self):
        return self.totalNoise

    def resetBuffers(self):
        pass

    def saveDiagnostics(self, chunkIdx, y):
        if (
            chunkIdx >= self.info.beginAtBuffer - 1
            and (chunkIdx + 1) % self.info.plotFrequency == 0
        ):
            startIdx = self.simBuffer
            yFiltered = self.pathFilter.process(
                y[
                    :,
                    startIdx - self.pathFilter.irLen + 1 : startIdx + self.samplesForSF,
                ]
            )
            self.totalNoise[:, :] = (
                self.primaryNoise[:, startIdx : startIdx + self.samplesForSF]
                + yFiltered[:, -self.samplesForSF :]
            )

    def saveBlockData(self, idx, newPrimaryNoise):
        numSamples = newPrimaryNoise.shape[-1]
        self.primaryNoise[:, idx : idx + numSamples] = newPrimaryNoise


class RecordIR():
    def __init__(
        self,
        trueValue,
        endTimeStep,
        simBuffer,
        simChunkSize,
        maxNumIR, #tuple
        beginAtBuffer=0,
        plotFrequency=1,
    ):
        self.selectIR = [slice(None,numIr) for numIr in maxNumIR]
        self.selectIR.append(slice(None))
        self.selectIR = tuple(self.selectIR)
        self.trueValue = trueValue[self.selectIR].reshape((-1, trueValue.shape[-1]))
        self.endTimeStep = endTimeStep
        self.simBuffer = simBuffer
        self.simChunkSize = simChunkSize

        self.latestIR = np.zeros_like(self.trueValue)

        self.metadata = {"xlabel" : "Samples", "ylabel" : "Amplitude", "label" : ["True Value", "Estimate"]}

        self.info = DiagnosticFunctionalityInfo(
                DIAGNOSTICTYPE.perSimBuffer,
                dplot.plotIR,
                SUMMARYFUNCTION.none,
                beginAtBuffer=beginAtBuffer,
                plotFrequency=plotFrequency,
            )

    def getOutput(self):
        return [self.trueValue, self.latestIR]

    def saveDiagnostics(self, bufferIdx, ir):
        self.latestIR[...] = ir[self.selectIR].reshape((-1, ir.shape[-1]))

    def resetBuffers(self):
        pass

    def saveBlockData(self, idx):
        pass






class PerBlockDiagnostic(ABC):
    def __init__(
        self,
        endTimeStep,
        simBuffer,
        simChunkSize,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        self.simBuffer = simBuffer
        self.simChunkSize = simChunkSize
        self.info = DiagnosticFunctionalityInfo(
            DIAGNOSTICTYPE.perBlock,
            dplot.functionOfTimePlot,
            SUMMARYFUNCTION.meanNearTimeIdx,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
        )

        self.dataBuffer = np.full((endTimeStep), np.nan)
        self.bufferIdx = 0

        self.metadata = {"title": "", "xlabel": "Samples", "ylabel": ""}
        for key, value in kwargs.items():
            self.metadata[key] = value

    def getOutput(self):
        return self.dataBuffer

    def saveDiagnostics(self, startIdx, endIdx, saveStartIdx, saveEndIdx):
        pass

    def resetBuffers(self):
        self.bufferIdx += 1

    @abstractmethod
    def saveBlockData(self, idx):
        pass


class ConstantEstimateNMSE(PerBlockDiagnostic):
    def __init__(
        self,
        trueValue,
        endTimeStep,
        simBuffer,
        simChunkSize,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        super().__init__(
            endTimeStep,
            simBuffer,
            simChunkSize,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
            **kwargs
        )
        self.trueValue = np.copy(trueValue)
        self.trueValuePower = np.sum(np.abs(trueValue) ** 2)

        self.metadata["title"] = "Constant Estimate NMSE"
        self.metadata["ylabel"] = "NMSE (dB)"

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx * self.simChunkSize + (idx - self.simBuffer)
        self.dataBuffer[totIdx] = 10 * np.log10(1e-30 + 
            np.sum(np.abs(self.trueValue - currentEstimate) ** 2) / self.trueValuePower
        )

class ConstantEstimateNMSEAllCombinations(PerBlockDiagnostic):
    def __init__(
        self,
        trueValue,
        endTimeStep,
        simBuffer,
        simChunkSize,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        super().__init__(
            endTimeStep,
            simBuffer,
            simChunkSize,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
            **kwargs
        )
        self.trueValue = np.copy(trueValue)
        self.trueValuePower = np.sum(np.abs(trueValue) ** 2)


        assert self.trueValue.ndim == 3
        numChannels = np.math.factorial(self.trueValue.shape[1]) * \
                        np.math.factorial(self.trueValue.shape[2])
        self.dataBuffer = np.full((numChannels, endTimeStep), np.nan)

        
        

        self.metadata["title"] = "Constant Estimate NMSE"
        self.metadata["ylabel"] = "NMSE (dB)"

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx * self.simChunkSize + (idx - self.simBuffer)

        for i,(ax1,ax2) in enumerate(it.product(it.permutations(range(self.trueValue.shape[1])), 
                                                it.permutations(range(self.trueValue.shape[2])))):
            crntEst = currentEstimate[:,ax1,:]
            self.dataBuffer[i,totIdx] = 10 * np.log10(1e-30 + 
                np.sum(np.abs(self.trueValue - crntEst[:,:,ax2]) ** 2) / self.trueValuePower
            )

class ConstantEstimateNMSESelectedFrequencies(PerBlockDiagnostic):
    def __init__(
        self,
        trueValue,
        lowFreq,
        highFreq,
        sampleRate,
        endTimeStep,
        simBuffer,
        simChunkSize,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        super().__init__(
            endTimeStep,
            simBuffer,
            simChunkSize,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
            **kwargs
        )
        numFreqBins = trueValue.shape[0]
        self.lowBin = int(numFreqBins * lowFreq / sampleRate)+1
        self.highBin = int(numFreqBins * highFreq / sampleRate)

        self.trueValue = np.copy(trueValue[self.lowBin : self.highBin, ...])
        self.trueValuePower = np.sum(np.abs(self.trueValue) ** 2)

        self.metadata["title"] = "Constant Estimate NMSE Selected Frequencies"
        self.metadata["ylabel"] = "NMSE (dB)"
        self.metadata["low frequency"] = lowFreq
        self.metadata["high frequency"] = highFreq

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx * self.simChunkSize + (idx - self.simBuffer)
        self.dataBuffer[totIdx] = 10 * np.log10(
            np.sum(
                np.abs(
                    self.trueValue - currentEstimate[self.lowBin : self.highBin, ...]
                )
                ** 2
            )
            / self.trueValuePower
        )



class ConstantEstimatePhaseDifferenceSelectedFrequencies(PerBlockDiagnostic):
    def __init__(
        self,
        trueValue,
        lowFreq,
        highFreq,
        sampleRate,
        endTimeStep,
        simBuffer,
        simChunkSize,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        super().__init__(
            endTimeStep,
            simBuffer,
            simChunkSize,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
            **kwargs
        )

        self.dataBuffer = np.full((endTimeStep), np.nan)
        numFreqBins = trueValue.shape[0]
        self.lowBin = int(numFreqBins * lowFreq / sampleRate)+1
        self.highBin = int(numFreqBins * highFreq / sampleRate)
        self.truePhase = np.copy(np.angle(trueValue[self.lowBin:self.highBin,...]))

        self.metadata["title"] = "Constant Estimate Phase"
        self.metadata["ylabel"] = "Average Difference (rad)"
        #self.metadata["label_suffix_channel"] = ["mean", "max"]

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx * self.simChunkSize + (idx - self.simBuffer)
        phaseDiff = np.abs(self.truePhase - np.angle(currentEstimate[self.lowBin:self.highBin,...]))
        phaseDiff = phaseDiff % np.pi
        #pha = np.min((np.abs(2*np.pi-phaseDiff),np.abs(phaseDiff), np.abs(2*np.pi+phaseDiff)))
        self.dataBuffer[totIdx] = np.mean(phaseDiff)
        #self.dataBuffer[1, totIdx] = np.max(phaseDiff)



class ConstantEstimateWeightedPhaseDifference(PerBlockDiagnostic):
    def __init__(
        self,
        trueValue,
        lowFreq,
        highFreq,
        sampleRate,
        endTimeStep,
        simBuffer,
        simChunkSize,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        super().__init__(
            endTimeStep,
            simBuffer,
            simChunkSize,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
            **kwargs
        )

        self.dataBuffer = np.full((endTimeStep), np.nan)
        numFreqBins = trueValue.shape[0]
        self.lowBin = int(numFreqBins * lowFreq / sampleRate)+1
        self.highBin = int(numFreqBins * highFreq / sampleRate)
        self.truePhase = np.copy(np.angle(trueValue[self.lowBin:self.highBin,...]))

        trueTotal = np.sum(np.abs(trueValue[self.lowBin:self.highBin,...]), axis=0, keepdims=True)
        self.weight = np.abs(trueValue[self.lowBin:self.highBin,...]) / trueTotal

        self.metadata["title"] = "Constant Estimate Phase"
        self.metadata["ylabel"] = "Average Difference (rad)"
        #self.metadata["label_suffix_channel"] = ["mean", "max"]

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx * self.simChunkSize + (idx - self.simBuffer)
        phaseDiff = np.abs(self.truePhase - np.angle(currentEstimate[self.lowBin:self.highBin,...]))
        phaseDiff = phaseDiff % np.pi
        #phaseDiff *= self.weight
        #pha = np.min((np.abs(2*np.pi-phaseDiff),np.abs(phaseDiff), np.abs(2*np.pi+phaseDiff)))
        self.dataBuffer[totIdx] = np.mean(np.sum(self.weight * phaseDiff,axis=0))
        #self.dataBuffer[1, totIdx] = np.max(phaseDiff)

class ConstantEstimateAmpDifference(PerBlockDiagnostic):
    def __init__(
        self,
        trueValue,
        endTimeStep,
        simBuffer,
        simChunkSize,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        super().__init__(
            endTimeStep,
            simBuffer,
            simChunkSize,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
            **kwargs
        )

        self.trueAmplitude = np.copy(np.abs(trueValue))
        self.trueAmpSum = np.sum(self.trueAmplitude)

        self.metadata["title"] = "Constant Estimate Amplitude"
        self.metadata["ylabel"] = "Normalized Sum Difference (dB)"

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx * self.simChunkSize + (idx - self.simBuffer)
        self.dataBuffer[totIdx] = 20 * np.log10(
            np.sum(np.abs(self.trueAmplitude - np.abs(currentEstimate)))
            / self.trueAmpSum
        )


class ConstantEstimatePhaseDifference(PerBlockDiagnostic):
    def __init__(
        self,
        trueValue,
        endTimeStep,
        simBuffer,
        simChunkSize,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        super().__init__(
            endTimeStep,
            simBuffer,
            simChunkSize,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
            **kwargs
        )

        self.truePhase = np.copy(np.angle(trueValue))

        self.metadata["title"] = "Constant Estimate Phase"
        self.metadata["ylabel"] = "Average Difference (rad)"

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx * self.simChunkSize + (idx - self.simBuffer)
        self.dataBuffer[totIdx] = np.mean(
            np.abs(self.truePhase - np.angle(currentEstimate))
        )


class SignalEstimateNMSE:
    def __init__(
        self,
        endTimeStep,
        simBuffer,
        simChunkSize,
        smoothingLen=1,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        """Assumes the signal to be estimated is of the form (numChannels, numSamples)"""
        self.simBuffer = simBuffer
        self.simChunkSize = simChunkSize
        self.info = DiagnosticFunctionalityInfo(
            DIAGNOSTICTYPE.perSample,
            dplot.functionOfTimePlot,
            SUMMARYFUNCTION.meanNearTimeIdx,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
        )

        self.recordedValues = np.full((endTimeStep), np.nan)
        self.bufferIdx = 0
        self.metadata = {
            "title": "Signal Estimate NMSE",
            "xlabel": "Samples",
            "ylabel": "NMSE (dB)",
        }
        # self.errorSmoother = Filter_IntBuffer(ir=np.ones((smoothingLen)))
        self.trueValuePowerSmoother = Filter_IntBuffer(
            ir=np.ones((smoothingLen)) / smoothingLen
        )

    def getOutput(self):
        return self.recordedValues

    def saveDiagnostics(self, startIdx, endIdx, saveStartIdx, saveEndIdx):
        pass

    def resetBuffers(self):
        self.bufferIdx += 1

    def saveBlockData(self, idx, estimate, trueValue):
        numSamples = estimate.shape[-1]
        
        totIdx = self.bufferIdx * self.simChunkSize + (idx - self.simBuffer)

        nmse = np.sum(
            np.abs(estimate - trueValue) ** 2, axis=tuple([i for i in range(estimate.ndim-1)])
        ) / self.trueValuePowerSmoother.process(
            np.sum(np.abs(trueValue) ** 2, axis=tuple([i for i in range(estimate.ndim-1)]))[None,:]
        )

        self.recordedValues[totIdx : totIdx + numSamples] = 10 * np.log10(nmse)


class SignalEstimateNMSEBlock:
    def __init__(
        self,
        endTimeStep,
        simBuffer,
        simChunkSize,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        """Assumes the signal to be estimated is of the form (numChannels, numSamples)"""
        self.simBuffer = simBuffer
        self.simChunkSize = simChunkSize
        self.info = DiagnosticFunctionalityInfo(
            DIAGNOSTICTYPE.perSample,
            dplot.functionOfTimePlot,
            SUMMARYFUNCTION.meanNearTimeIdx,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
        )

        self.recordedValues = np.full((endTimeStep), np.nan)
        self.bufferIdx = 0
        self.metadata = {
            "title": "Signal Estimate NMSE",
            "xlabel": "Samples",
            "ylabel": "NMSE (dB)",
        }
        # self.errorSmoother = Filter_IntBuffer(ir=np.ones((smoothingLen)))

    def getOutput(self):
        return self.recordedValues

    def saveDiagnostics(self, startIdx, endIdx, saveStartIdx, saveEndIdx):
        pass

    def resetBuffers(self):
        self.bufferIdx += 1

    def saveBlockData(self, idx, estimate, trueValue):
        numSamples = estimate.shape[-1]
        
        totIdx = self.bufferIdx * self.simChunkSize + (idx - self.simBuffer)

        nmse = np.sum(np.abs(estimate - trueValue) ** 2) / np.sum(np.abs(trueValue)**2)

        self.recordedValues[totIdx] = 10 * np.log10(1e-30+nmse)

#plt.plot(np.sum(np.abs(1*self.buffers["xf"][...,self.idx-self.blockSize:self.idx] - trueXf)**2,axis=(0,1,2)) / np.sum(np.abs(trueXf)**2))


class RecordScalar:
    def __init__(
        self,
        endTimeStep,
        simBuffer,
        simChunkSize,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        self.simBuffer = simBuffer
        self.simChunkSize = simChunkSize
        self.info = DiagnosticFunctionalityInfo(
            DIAGNOSTICTYPE.perBlock,
            dplot.functionOfTimePlot,
            SUMMARYFUNCTION.meanNearTimeIdx,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
        )

        self.recordedValues = np.full((endTimeStep), np.nan)
        self.bufferIdx = 0
        self.metadata = {"title": "Value of ", "xlabel": "Samples", "ylabel": "value"}

    def getOutput(self):
        return self.recordedValues

    def resetBuffers(self):
        self.bufferIdx += 1

    def saveBlockData(self, idx, value):
        totIdx = self.bufferIdx * self.simChunkSize + (idx - self.simBuffer)
        self.recordedValues[totIdx] = value


class RecordVector:
    def __init__(
        self,
        vectorLength,
        endTimeStep,
        simBuffer,
        simChunkSize,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        self.simBuffer = simBuffer
        self.simChunkSize = simChunkSize
        self.info = DiagnosticFunctionalityInfo(
            DIAGNOSTICTYPE.perBlock,
            dplot.functionOfTimePlot,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
        )

        self.recordedValues = np.full((vectorLength, endTimeStep), np.nan)
        self.bufferIdx = 0
        self.metadata = {"title": "Value of ", "xlabel": "Samples", "ylabel": "value"}

    def getOutput(self):
        return self.recordedValues

    def resetBuffers(self):
        self.bufferIdx += 1

    def saveBlockData(self, idx, value):
        totIdx = self.bufferIdx * self.simChunkSize + (idx - self.simBuffer)
        self.recordedValues[:, totIdx] = value


class RecordVectorSummary:
    def __init__(
        self,
        endTimeStep,
        simBuffer,
        simChunkSize,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        self.simBuffer = simBuffer
        self.simChunkSize = simChunkSize
        self.info = DiagnosticFunctionalityInfo(
            DIAGNOSTICTYPE.perBlock,
            dplot.functionOfTimePlot,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
        )

        self.recordedValues = np.full((3, endTimeStep), np.nan)
        #self.recordedMax = np.full((1, endTimeStep), np.nan)
        #self.recordedMin = np.full((1, endTimeStep), np.nan)
        #self.recordedMean = np.full((1, endTimeStep), np.nan)
        self.bufferIdx = 0
        self.metadata = {"title": "Value of ", "xlabel": "Samples", "ylabel": "value", "label_suffix_channel" : ["max", "mean", "min"]}

    def getOutput(self):
        return self.recordedValues

    def resetBuffers(self):
        self.bufferIdx += 1

    def saveBlockData(self, idx, value):
        totIdx = self.bufferIdx * self.simChunkSize + (idx - self.simBuffer)
        self.recordedValues[0, totIdx] = np.max(value)
        self.recordedValues[1, totIdx] = np.mean(value)
        self.recordedValues[2, totIdx] = np.min(value)


class NoiseReduction:
    def __init__(
        self,
        numPoints,
        acousticPath,
        endTimeStep,
        simBuffer,
        simChunkSize,
        smoothingLen=1,
        saveRawData=True,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        self.simBuffer = simBuffer
        self.simChunkSize = simChunkSize
        self.saveRawData = saveRawData
        if saveRawData:
            outputFunc = [dplot.functionOfTimePlot, dplot.savenpz]
            summaryFunc = [SUMMARYFUNCTION.meanNearTimeIdx, SUMMARYFUNCTION.none]
        else:
            outputFunc = dplot.functionOfTimePlot
            summaryFunc = SUMMARYFUNCTION.meanNearTimeIdx

        self.info = DiagnosticFunctionalityInfo(
            DIAGNOSTICTYPE.perSample,
            outputFunc,
            summaryFunc,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
        )
        self.pathFilter = FilterSum_IntBuffer(acousticPath)

        self.totalNoise = np.zeros((numPoints, self.simChunkSize + self.simBuffer))
        self.primaryNoise = np.zeros((numPoints, self.simChunkSize + self.simBuffer))

        self.totalNoisePower = np.zeros(endTimeStep)
        self.primaryNoisePower = np.zeros(endTimeStep)

        self.totalNoiseSmoother = Filter_IntBuffer(ir=np.ones((smoothingLen)), numIn=1)
        self.primaryNoiseSmoother = Filter_IntBuffer(
            ir=np.ones((smoothingLen)), numIn=1
        )

        self.noiseReduction = np.full((endTimeStep), np.nan)
        self.metadata = {
            "title": "Noise Reduction",
            "xlabel": "Samples",
            "ylabel": "Reduction (dB)",
        }
        for key, value in kwargs.items():
            self.metadata[key] = value

    def getOutput(self):
        if self.saveRawData:
            return [
                self.noiseReduction,
                {
                    "Total Noise Power": self.totalNoisePower,
                    "Primary Noise Power": self.primaryNoisePower,
                },
            ]
        else:
            return self.noiseReduction

    def saveDiagnostics(self, startIdx, endIdx, saveStartIdx, saveEndIdx, y):
        secondaryNoise = self.pathFilter.process(y[:, startIdx:endIdx])
        self.totalNoise[:, startIdx:endIdx] = (
            self.primaryNoise[:, startIdx:endIdx] + secondaryNoise
        )

        totalNoisePower = np.mean(
            self.totalNoise[:, startIdx:endIdx] ** 2, axis=0, keepdims=True
        )
        primaryNoisePower = np.mean(
            self.primaryNoise[:, startIdx:endIdx] ** 2, axis=0, keepdims=True
        )

        self.totalNoisePower[saveStartIdx:saveEndIdx] = totalNoisePower
        self.primaryNoisePower[saveStartIdx:saveEndIdx] = primaryNoisePower

        smoothTotalNoisePower = self.totalNoiseSmoother.process(totalNoisePower)
        smoothPrimaryNoisePower = self.primaryNoiseSmoother.process(primaryNoisePower)

        self.noiseReduction[saveStartIdx:saveEndIdx] = util.pow2db(
            smoothTotalNoisePower / smoothPrimaryNoisePower
        )

    def resetBuffers(self):
        self.totalNoise = np.concatenate(
            (
                self.totalNoise[:, -self.simBuffer :],
                np.zeros((self.totalNoise.shape[0], self.simChunkSize)),
            ),
            axis=-1,
        )
        self.primaryNoise = np.concatenate(
            (
                self.primaryNoise[:, -self.simBuffer :],
                np.zeros((self.primaryNoise.shape[0], self.simChunkSize)),
            ),
            axis=-1,
        )

    # def saveToBuffers(self, newPrimaryNoise, idx, numSamples):
    #     self.primaryNoise[:,idx:idx+numSamples] = newPrimaryNoise
    def saveBlockData(self, idx, newPrimaryNoise):
        numSamples = newPrimaryNoise.shape[-1]
        self.primaryNoise[:, idx : idx + numSamples] = newPrimaryNoise


class NoiseReductionExternalSignals:
    def __init__(
        self,
        numPoints,
        endTimeStep,
        simBuffer,
        simChunkSize,
        smoothingLen=1,
        saveRawData=True,
        beginAtBuffer=0,
        plotFrequency=1,
        **kwargs
    ):
        self.simBuffer = simBuffer
        self.simChunkSize = simChunkSize
        self.saveRawData = saveRawData
        if saveRawData:
            outputFunc = [dplot.functionOfTimePlot, dplot.savenpz]
            summaryFunc = [SUMMARYFUNCTION.meanNearTimeIdx, SUMMARYFUNCTION.none]
        else:
            outputFunc = dplot.functionOfTimePlot
            summaryFunc = SUMMARYFUNCTION.meanNearTimeIdx

        self.info = DiagnosticFunctionalityInfo(
            DIAGNOSTICTYPE.perSample,
            outputFunc,
            summaryFunc,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
        )
        self.primaryNoise = np.zeros((numPoints, self.simChunkSize + self.simBuffer))
        self.totalNoisePower = np.zeros(endTimeStep)
        self.primaryNoisePower = np.zeros(endTimeStep)
        self.totalNoiseSmoother = Filter_IntBuffer(ir=np.ones((smoothingLen)), numIn=1)
        self.primaryNoiseSmoother = Filter_IntBuffer(
            ir=np.ones((smoothingLen)), numIn=1
        )
        self.noiseReduction = np.full((endTimeStep), np.nan)

        self.metadata = {
            "title": "Noise Reduction",
            "xlabel": "Samples",
            "ylabel": "Reduction (dB)",
        }
        for key, value in kwargs.items():
            self.metadata[key] = value

    # def getOutput(self):
    #     return self.noiseReduction
    def getOutput(self):
        if self.saveRawData:
            return [
                self.noiseReduction,
                {
                    "Total Noise Power": self.totalNoisePower,
                    "Primary Noise Power": self.primaryNoisePower,
                },
            ]
        else:
            return self.noiseReduction

    def saveDiagnostics(self, startIdx, endIdx, saveStartIdx, saveEndIdx, e):
        totalNoisePower = np.mean(e[:, startIdx:endIdx] ** 2, axis=0, keepdims=True)
        primaryNoisePower = np.mean(
            self.primaryNoise[:, startIdx:endIdx] ** 2, axis=0, keepdims=True
        )

        totalNoisePowerSmooth = self.totalNoiseSmoother.process(totalNoisePower)
        primaryNoisePowerSmooth = self.primaryNoiseSmoother.process(primaryNoisePower)

        self.noiseReduction[saveStartIdx:saveEndIdx] = util.pow2db(
            totalNoisePowerSmooth / primaryNoisePowerSmooth
        )
        self.totalNoisePower[saveStartIdx:saveEndIdx] = totalNoisePower
        self.primaryNoisePower[saveStartIdx:saveEndIdx] = primaryNoisePower

    def resetBuffers(self):
        self.primaryNoise = np.concatenate(
            (
                self.primaryNoise[:, -self.simBuffer :],
                np.zeros((self.primaryNoise.shape[0], self.simChunkSize)),
            ),
            axis=-1,
        )

    def saveBlockData(self, idx, newPrimaryNoise):
        numSamples = newPrimaryNoise.shape[-1]
        self.primaryNoise[:, idx : idx + numSamples] = newPrimaryNoise


class saveTotalPower:
    def __init__(self, endTimeStep, acousticPath=None):
        self.info = DiagnosticFunctionalityInfo(
            DIAGNOSTICTYPE.perSample,
            OUTPUTFUNCTION.savenpz,
            SUMMARYFUNCTION.none,
            beginAtBuffer=beginAtBuffer,
            plotFrequency=plotFrequency,
        )
        self.noisePower = np.zeros(endTimeStep)
        if acousticPath is not None:
            self.pathFilter = FilterSum_IntBuffer(acousticPath)
        else:
            self.pathFilter = FilterSum_IntBuffer(np.ones((1, 1, 1)))

    def getOutput(self):
        return self.noisePower

    def saveDiagnostics(self, startIdx, endIdx, saveStartIdx, saveEndIdx, e):
        totalNoisePower = np.mean(e[:, startIdx:endIdx] ** 2, axis=0, keepdims=True)

        self.primaryNoisePower[saveStartIdx:saveEndIdx] = primaryNoisePower

    def resetBuffers(self):
        pass

    def saveBlockData(self, idx, newPrimaryNoise):
        numSamples = newPrimaryNoise.shape[-1]
        self.primaryNoise[:, idx : idx + numSamples] = newPrimaryNoise


class RecordAudio:
    def __init__(self, acousticPath, samplerate, maxLength, endTimeStep, simBuffer,
                 simChunkSize, beginAtBuffer=0, plotFrequency=1):
        self.pathFilter = FilterSum(ir=acousticPath)
        self.simBuffer = simBuffer
        self.simChunkSize = simChunkSize
        self.chunkIdx = 0

        self.withANC = np.full((self.pathFilter.numOut, endTimeStep), np.nan)
        self.withoutANC = np.full((self.pathFilter.numOut, endTimeStep), np.nan)

        self.primaryNoise = np.zeros((self.pathFilter.numOut, self.simBuffer+self.simChunkSize))

        self.info = DiagnosticFunctionalityInfo(
            dplot.createAudioFiles,
            beginAtBuffer,
            plotFrequency
        )
        self.metadata = {"samplerate" : samplerate,
                        "maxlength" : maxLength}

    def getOutput(self):
        return {"with_anc" : self.withANC, 
                "without_anc" : self.withoutANC}
    
    def saveDiagnostics(self, startIdx, endIdx, saveStartIdx, saveEndIdx, y):
        secondaryNoise = self.pathFilter.process(y[:, startIdx:endIdx])
        self.withANC[:, saveStartIdx:saveEndIdx] = self.primaryNoise[:,startIdx:endIdx] + secondaryNoise
        self.withoutANC[:,saveStartIdx:saveEndIdx] = self.primaryNoise[:,startIdx:endIdx]

    def saveBlockData(self, idx, newPrimaryNoise):
        self.primaryNoise[:,idx:idx+newPrimaryNoise.shape[-1]] = newPrimaryNoise

    def resetBuffers(self):
        self.primaryNoise = np.concatenate(
            (
                self.primaryNoise[:, -self.simBuffer:],
                np.zeros((self.primaryNoise.shape[0], self.simChunkSize)),
            ),
            axis=-1,
        )