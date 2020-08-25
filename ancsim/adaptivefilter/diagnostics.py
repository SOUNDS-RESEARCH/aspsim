import numpy as np
from copy import deepcopy
from enum import Enum
from abc import ABC, abstractmethod

import ancsim.utilities as util
import ancsim.settings as s
import ancsim.experiment.plotscripts as psc
from ancsim.signal.filterclasses import Filter_IntBuffer, FilterSum_IntBuffer, FilterMD_IntBuffer
from ancsim.experiment.diagnosticplots import functionOfTimePlot, soundfieldPlot
import ancsim.experiment.diagnosticsummary as dsum

class DIAGNOSTICTYPE (Enum):
    perSimBuffer = 1
    perSample = 2
    perBlock = 3

class OUTPUTFUNCTION (Enum):
    functionoftime = 1
    soundfield = 2
    
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
                    assert(info == filt.diag.diagnostics[diagName].info)
                    
    def plotThisBuffer(self, bufferIdx, funcInfo):
        return bufferIdx >= funcInfo.beginAtBuffer and (
            bufferIdx % funcInfo.plotFrequency == 0 or bufferIdx-1 == 0)

    def dispatch(self, filters, timeIdx, bufferIdx, folder):
        for diagName, funcInfo in self.funcInfo.items():
            if self.plotThisBuffer(bufferIdx, funcInfo):
                diagSet = self.getDiagnosticSet(diagName, filters)
                self.dispatchPlot(funcInfo, diagName, diagSet, timeIdx, folder)
                self.dispatchSummary(funcInfo, diagName, diagSet, timeIdx, folder)
                
    def dispatchPlot(self, funcInfo, diagName, diagSet, timeIdx, folder):
        if funcInfo.outputType == OUTPUTFUNCTION.functionoftime:
            functionOfTimePlot(diagName, diagSet, timeIdx, folder, self.outputFormat)
        elif funcInfo.outputType == OUTPUTFUNCTION.soundfield:
            soundfieldPlot(diagName, diagSet, timeIdx, folder, self.outputFormat)
        else:
            raise NotImplementedError
    
    def dispatchSummary(self, funcInfo, diagName, diagSet, timeIdx, folder):
        if funcInfo.summaryType == SUMMARYFUNCTION.none:
            return
        elif funcInfo.summaryType == SUMMARYFUNCTION.lastValue:
            summaryValues = dsum.lastValue(diagSet, timeIdx)
        elif funcInfo.summaryType == SUMMARYFUNCTION.meanNearTimeIdx:
            summaryValues = dsum.meanNearTimeIdx(diagSet, timeIdx)
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
    def __init__(self):
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
            startIdx = s.SIMBUFFER - self.diagnosticSamplesLeft
            numLeftOver = s.SIMBUFFER+s.SIMCHUNKSIZE - idx
            saveStartIdx = self.bufferIdx*s.SIMCHUNKSIZE - self.diagnosticSamplesLeft
            saveEndIdx = (self.bufferIdx+1)*s.SIMCHUNKSIZE - numLeftOver

            self.diagnostics[name].saveDiagnostics(startIdx, idx, saveStartIdx, saveEndIdx, *args)
        elif name in self.perSimBufferDiagnostics:
            self.diagnostics[name].saveDiagnostics(self.bufferIdx, *args)

    def resetBuffers(self, idx):
        """Must be called after simulation buffer resets. 
            Calls resetBuffer() for all owned diagnostic objects, and increments counters"""
        for key, val in self.diagnostics.items():
            val.resetBuffers()
        numLeftOver = s.SIMBUFFER+s.SIMCHUNKSIZE - idx
        self.diagnosticSamplesLeft = numLeftOver
        self.bufferIdx += 1


class DiagnosticFunctionalityInfo():
    def __init__(self, diagnosticType, outputType, 
                 summaryType=SUMMARYFUNCTION.none, 
                 beginAtBuffer=0, plotFrequency=1):
        self.type = diagnosticType           #Use the appropriate Enum
        self.outputType = outputType         #Use the appropriate Enum
        self.summaryType = summaryType       #Use the appropriate Enum
        self.beginAtBuffer = beginAtBuffer
        self.plotFrequency = plotFrequency #Unit is number of simulator-buffers
        

    def __eq__(self, other):
        return (self.type == other.type and
                self.outputType == other.outputType and
                self.summaryType == other.summaryType and
                self.beginAtBuffer == other.beginAtBuffer and
                self.plotFrequency == other.plotFrequency)
    

class SoundfieldImage():
    def __init__(self, numPoints, acousticPath, beginAtBuffer=0, plotFrequency=1):
        self.info = DiagnosticFunctionalityInfo(DIAGNOSTICTYPE.perSimBuffer,
                                                OUTPUTFUNCTION.soundfield,
                                                SUMMARYFUNCTION.none,
                                                beginAtBuffer=beginAtBuffer,
                                                plotFrequency=plotFrequency)
        self.samplesForSF = np.min((2048, s.SIMBUFFER, s.SIMCHUNKSIZE))
        self.totalNoise = np.zeros((numPoints, self.samplesForSF))
        self.primaryNoise = np.zeros((numPoints, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.pathFilter = FilterSum_IntBuffer(acousticPath)

    def getOutput(self):
        return self.totalNoise
        
    def resetBuffers(self):
        pass

    def saveDiagnostics(self, chunkIdx, y):
        if chunkIdx >= self.info.beginAtBuffer-1 and \
            (chunkIdx+1) % self.info.plotFrequency == 0:
            startIdx = s.SIMBUFFER
            yFiltered = self.pathFilter.process(y[:,startIdx-self.pathFilter.irLen+1:
                                                    startIdx+self.samplesForSF])
            self.totalNoise[:,:] = self.primaryNoise[:,startIdx:startIdx+self.samplesForSF] + \
                                    yFiltered[:,-self.samplesForSF:]

    def saveBlockData(self, idx, newPrimaryNoise):
        numSamples = newPrimaryNoise.shape[-1]
        self.primaryNoise[:,idx:idx+numSamples] = newPrimaryNoise


class RecordSpectrum():
    def __init__(self, numPoints, acousticPath, beginAtBuffer=0, plotFrequency=1):
        self.info = DiagnosticFunctionalityInfo(DIAGNOSTICTYPE.perSimBuffer,
                                                OUTPUTFUNCTION.soundfield,
                                                SUMMARYFUNCTION.none,
                                                beginAtBuffer=beginAtBuffer,
                                                plotFrequency=plotFrequency)
        self.samplesForSF = np.min((2048, s.SIMBUFFER, s.SIMCHUNKSIZE))
        self.totalNoise = np.zeros((numPoints, self.samplesForSF))
        self.primaryNoise = np.zeros((numPoints, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.pathFilter = FilterSum_IntBuffer(acousticPath)

    def getOutput(self):
        return self.totalNoise
        
    def resetBuffers(self):
        pass

    def saveDiagnostics(self, chunkIdx, y):
        if chunkIdx >= self.info.beginAtBuffer-1 and \
            (chunkIdx+1) % self.info.plotFrequency == 0:
            startIdx = s.SIMBUFFER
            yFiltered = self.pathFilter.process(y[:,startIdx-self.pathFilter.irLen+1:
                                                    startIdx+self.samplesForSF])
            self.totalNoise[:,:] = self.primaryNoise[:,startIdx:startIdx+self.samplesForSF] + \
                                    yFiltered[:,-self.samplesForSF:]

    def saveBlockData(self, idx, newPrimaryNoise):
        numSamples = newPrimaryNoise.shape[-1]
        self.primaryNoise[:,idx:idx+numSamples] = newPrimaryNoise

class PerBlockDiagnostic(ABC):
    def __init__(self, beginAtBuffer=0, plotFrequency=1, **kwargs):
        self.info = DiagnosticFunctionalityInfo(DIAGNOSTICTYPE.perBlock, 
                                                OUTPUTFUNCTION.functionoftime,
                                                SUMMARYFUNCTION.meanNearTimeIdx,
                                                beginAtBuffer=beginAtBuffer,
                                                plotFrequency=plotFrequency)

        self.dataBuffer = np.full((s.ENDTIMESTEP), np.nan)
        self.bufferIdx = 0

        self.metadata = {"title": "",
                        "xlabel" : "Samples",
                        "ylabel" : ""}
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
    def __init__(self, trueValue, beginAtBuffer=0, plotFrequency=1, **kwargs):
        super().__init__(beginAtBuffer=beginAtBuffer, plotFrequency=plotFrequency, **kwargs)
        self.trueValue = trueValue
        self.trueValuePower = np.sum(np.abs(trueValue)**2)

        self.metadata["title"] = "Constant Estimate NMSE"
        self.metadata["ylabel"] =  "NMSE (dB)"

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx*s.SIMCHUNKSIZE+(idx-s.SIMBUFFER)
        self.dataBuffer[totIdx] = 10*np.log10(np.sum(np.abs(self.trueValue - currentEstimate)**2) / self.trueValuePower)

class ConstantEstimateNMSESelectedFrequencies(PerBlockDiagnostic):
    def __init__(self, trueValue, lowFreq, highFreq, sampleRate, beginAtBuffer=0, plotFrequency=1, **kwargs):
        super().__init__(beginAtBuffer=beginAtBuffer, plotFrequency=plotFrequency, **kwargs)
        numFreqBins = trueValue.shape[0]
        self.lowBin = int(numFreqBins * lowFreq / sampleRate)
        self.highBin = int(numFreqBins * highFreq / sampleRate)

        self.trueValue = trueValue[self.lowBin:self.highBin,:,:]
        self.trueValuePower = np.sum(np.abs(self.trueValue)**2)

        self.metadata["title"] = "Constant Estimate NMSE Selected Frequencies"
        self.metadata["ylabel"] =  "NMSE (dB)"
        self.metadata["low frequency"] = lowFreq
        self.metadata["high frequency"] = highFreq

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx*s.SIMCHUNKSIZE+(idx-s.SIMBUFFER)
        self.dataBuffer[totIdx] = 10*np.log10(np.sum(np.abs(self.trueValue - \
                    currentEstimate[self.lowBin:self.highBin,:,:])**2) / self.trueValuePower)

class ConstantEstimateAmpDifference(PerBlockDiagnostic):
    def __init__(self, trueValue, beginAtBuffer=0, plotFrequency=1, **kwargs):
        super().__init__(beginAtBuffer=beginAtBuffer, plotFrequency=plotFrequency, **kwargs)

        self.trueAmplitude = np.abs(trueValue)
        self.trueAmpSum = np.sum(self.trueAmplitude)

        self.metadata["title"] =  "Constant Estimate Amplitude"
        self.metadata["ylabel"] = "Normalized Sum Difference (dB)"

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx*s.SIMCHUNKSIZE+(idx-s.SIMBUFFER)
        self.dataBuffer[totIdx] = 20*np.log10(np.sum(np.abs(self.trueAmplitude - \
                    np.abs(currentEstimate))) / self.trueAmpSum)

class ConstantEstimatePhaseDifference(PerBlockDiagnostic):
    def __init__(self, trueValue, beginAtBuffer=0, plotFrequency=1, **kwargs):
        super().__init__(beginAtBuffer=beginAtBuffer, plotFrequency=plotFrequency, **kwargs)

        self.truePhase = np.angle(trueValue)

        self.metadata["title"] =  "Constant Estimate Phase"
        self.metadata["ylabel"] = "Average Difference (rad)"

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx*s.SIMCHUNKSIZE+(idx-s.SIMBUFFER)
        self.dataBuffer[totIdx] = np.mean(np.abs(self.truePhase - np.angle(currentEstimate)))


class SignalEstimateNMSE():
    def __init__(self, smoothingLen=s.OUTPUTSMOOTHING, beginAtBuffer=0, plotFrequency=1, **kwargs):
        """Assumes the signal to be estimated is of the form (numChannels, numSamples)"""
        self.info = DiagnosticFunctionalityInfo(DIAGNOSTICTYPE.perSample,
                                                OUTPUTFUNCTION.functionoftime,
                                                SUMMARYFUNCTION.meanNearTimeIdx,
                                                beginAtBuffer=beginAtBuffer,
                                                plotFrequency=plotFrequency)

        self.recordedValues = np.full((s.ENDTIMESTEP), np.nan)
        self.bufferIdx = 0
        self.metadata = {"title": "Signal Estimate NMSE",
                        "xlabel" : "Samples",
                        "ylabel" : "NMSE (dB)"}
        #self.errorSmoother = Filter_IntBuffer(ir=np.ones((smoothingLen)))
        self.trueValuePowerSmoother = Filter_IntBuffer(ir=np.ones((smoothingLen))/smoothingLen)

    def getOutput(self):
        return self.recordedValues

    def saveDiagnostics(self, startIdx, endIdx, saveStartIdx, saveEndIdx):
        pass

    def resetBuffers(self):
        self.bufferIdx += 1

    def saveBlockData(self, idx, estimate, trueValue):
        numSamples = estimate.shape[-1]
        totIdx = self.bufferIdx*s.SIMCHUNKSIZE+(idx-s.SIMBUFFER)

        nmse = np.sum(np.abs(estimate - trueValue)**2, axis=0) / \
                self.trueValuePowerSmoother.process(np.sum(np.abs(trueValue)**2,axis=0, keepdims=True))
        
        self.recordedValues[totIdx:totIdx+numSamples] = 10*np.log10(nmse)


class RecordScalar():
    def __init__(self, beginAtBuffer=0, plotFrequency=1, **kwargs):
        self.info = DiagnosticFunctionalityInfo(DIAGNOSTICTYPE.perBlock,
                                                OUTPUTFUNCTION.functionoftime,
                                                SUMMARYFUNCTION.meanNearTimeIdx,
                                                beginAtBuffer=beginAtBuffer,
                                                plotFrequency=plotFrequency)

        self.recordedValues = np.full((s.ENDTIMESTEP), np.nan)
        self.bufferIdx = 0
        self.metadata = {"title": "Value of ",
                        "xlabel" : "Samples",
                        "ylabel" : "value"}

    def getOutput(self):
        return self.recordedValues

    def resetBuffers(self):
        self.bufferIdx += 1

    def saveBlockData(self, idx, value):
        totIdx = self.bufferIdx*s.SIMCHUNKSIZE+(idx-s.SIMBUFFER)
        self.recordedValues[totIdx] = value

class RecordVector():
    def __init__(self, vectorLength, beginAtBuffer=0, plotFrequency=1, **kwargs):
        self.info = DiagnosticFunctionalityInfo(DIAGNOSTICTYPE.perBlock,
                                                OUTPUTFUNCTION.functionoftime,
                                                beginAtBuffer=beginAtBuffer,
                                                plotFrequency=plotFrequency)

        self.recordedValues = np.full((vectorLength, s.ENDTIMESTEP), np.nan)
        self.bufferIdx = 0
        self.metadata = {"title": "Value of ",
                        "xlabel" : "Samples",
                        "ylabel" : "value"}

    def getOutput(self):
        return self.recordedValues

    def resetBuffers(self):
        self.bufferIdx += 1

    def saveBlockData(self, idx, value):
        totIdx = self.bufferIdx*s.SIMCHUNKSIZE+(idx-s.SIMBUFFER)
        self.recordedValues[:,totIdx] = value


class NoiseReduction():
    def __init__(self, numPoints, acousticPath, beginAtBuffer=0, plotFrequency=1, **kwargs):
        self.info = DiagnosticFunctionalityInfo(DIAGNOSTICTYPE.perSample,
                                                OUTPUTFUNCTION.functionoftime,
                                                SUMMARYFUNCTION.meanNearTimeIdx,
                                                beginAtBuffer=beginAtBuffer,
                                                plotFrequency=plotFrequency)
        self.pathFilter = FilterSum_IntBuffer(acousticPath)
        
        self.totalNoise = np.zeros((numPoints,s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.primaryNoise = np.zeros((numPoints, s.SIMCHUNKSIZE+s.SIMBUFFER))

        self.totalNoisePower = np.zeros(s.ENDTIMESTEP)
        self.primaryNoisePower = np.zeros(s.ENDTIMESTEP)

        self.totalNoiseSmoother = Filter_IntBuffer(ir=np.ones((s.OUTPUTSMOOTHING)),numIn=1)
        self.primaryNoiseSmoother = Filter_IntBuffer(ir=np.ones((s.OUTPUTSMOOTHING)),numIn=1)

        self.noiseReduction = np.full((s.ENDTIMESTEP), np.nan)
        self.metadata = {"title": "Noise Reduction",
                        "xlabel" : "Samples",
                        "ylabel" : "Reduction (dB)"}
        for key, value in kwargs.items():
            self.metadata[key] = value
    
    def getOutput(self):
        return self.noiseReduction

    def saveDiagnostics(self, startIdx, endIdx, saveStartIdx, saveEndIdx, y):
        secondaryNoise = self.pathFilter.process(y[:,startIdx:endIdx])
        self.totalNoise[:,startIdx:endIdx] = self.primaryNoise[:,startIdx:endIdx] + secondaryNoise

        totalNoisePower = np.mean(self.totalNoise[:,startIdx:endIdx]**2, axis=0,keepdims=True)
        totalNoisePower = self.totalNoiseSmoother.process(totalNoisePower)
        primaryNoisePower = np.mean(self.primaryNoise[:,startIdx:endIdx]**2, axis=0,keepdims=True)
        primaryNoisePower = self.primaryNoiseSmoother.process(primaryNoisePower)
        
        self.noiseReduction[saveStartIdx:saveEndIdx] = util.pow2db(totalNoisePower / primaryNoisePower)
        self.totalNoisePower[saveStartIdx:saveEndIdx] = totalNoisePower
        self.primaryNoisePower[saveStartIdx:saveEndIdx] = primaryNoisePower
    
    def resetBuffers(self):
        self.totalNoise = np.concatenate((self.totalNoise[:,-s.SIMBUFFER:],np.zeros((self.totalNoise.shape[0], s.SIMCHUNKSIZE))) ,axis=-1)
        self.primaryNoise = np.concatenate((self.primaryNoise[:,-s.SIMBUFFER:], 
                                    np.zeros((self.primaryNoise.shape[0], s.SIMCHUNKSIZE))), axis=-1)

    # def saveToBuffers(self, newPrimaryNoise, idx, numSamples):
    #     self.primaryNoise[:,idx:idx+numSamples] = newPrimaryNoise
    def saveBlockData(self, idx, newPrimaryNoise):
        numSamples = newPrimaryNoise.shape[-1]
        self.primaryNoise[:,idx:idx+numSamples] = newPrimaryNoise


class NoiseReductionExternalSignals():
    def __init__(self, numPoints, beginAtBuffer=0, plotFrequency=1, **kwargs):
        self.info = DiagnosticFunctionalityInfo(DIAGNOSTICTYPE.perSample,
                                                OUTPUTFUNCTION.functionoftime,
                                                SUMMARYFUNCTION.meanNearTimeIdx,
                                                beginAtBuffer=beginAtBuffer,
                                                plotFrequency=plotFrequency)
        self.primaryNoise = np.zeros((numPoints, s.SIMCHUNKSIZE+s.SIMBUFFER))
        self.totalNoisePower = np.zeros(s.ENDTIMESTEP)
        self.primaryNoisePower = np.zeros(s.ENDTIMESTEP)
        self.totalNoiseSmoother = Filter_IntBuffer(ir=np.ones((s.OUTPUTSMOOTHING)),numIn=1)
        self.primaryNoiseSmoother = Filter_IntBuffer(ir=np.ones((s.OUTPUTSMOOTHING)),numIn=1)
        self.noiseReduction = np.full((s.ENDTIMESTEP), np.nan)

        self.metadata = {"title": "Noise Reduction",
                         "xlabel" : "Samples",
                         "ylabel" : "Reduction (dB)"}
        for key, value in kwargs.items():
            self.metadata[key] = value

    def getOutput(self):
        return self.noiseReduction

    def saveDiagnostics(self, startIdx, endIdx, saveStartIdx, saveEndIdx, e):
        totalNoisePower = np.mean(e[:,startIdx:endIdx]**2, axis=0, keepdims=True)
        totalNoisePower = self.totalNoiseSmoother.process(totalNoisePower)
        primaryNoisePower = np.mean(self.primaryNoise[:,startIdx:endIdx]**2, axis=0,keepdims=True)
        primaryNoisePower = self.primaryNoiseSmoother.process(primaryNoisePower)
        
        self.noiseReduction[saveStartIdx:saveEndIdx] = util.pow2db(totalNoisePower / primaryNoisePower)
        self.totalNoisePower[saveStartIdx:saveEndIdx] = totalNoisePower
        self.primaryNoisePower[saveStartIdx:saveEndIdx] = primaryNoisePower

    def resetBuffers(self):
        self.primaryNoise = np.concatenate((self.primaryNoise[:,-s.SIMBUFFER:], 
                                    np.zeros((self.primaryNoise.shape[0], s.SIMCHUNKSIZE))), axis=-1)

    def saveBlockData(self, idx, newPrimaryNoise):
        numSamples = newPrimaryNoise.shape[-1]
        self.primaryNoise[:,idx:idx+numSamples] = newPrimaryNoise




