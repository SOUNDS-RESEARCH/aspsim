import numpy as np
from copy import deepcopy
from enum import Enum
from abc import ABC, abstractmethod
from functools import partial
import itertools as it
from operator import attrgetter

import ancsim.utilities as util
import ancsim.experiment.plotscripts as psc
from ancsim.signal.filterclasses import (
    Filter_IntBuffer,
    FilterSum,
    FilterMD_IntBuffer,
)
import ancsim.adaptivefilter.diagnosticplots as dplot
import ancsim.adaptivefilter.diagnosticsummary as dsum


# class DIAGNOSTICTYPE(Enum):
#     perSimBuffer = 1
#     perSample = 2
#     perBlock = 3

# class OUTPUTFUNCTION(Enum):
#     functionoftime = 1
#     soundfield = 2
#     savenpz = 3

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
                if diagName in filt.diag:
                    assert info == filt.diag.diagnostics[diagName].info

    def plotThisBuffer(self, bufferIdx, funcInfo):
        return bufferIdx >= funcInfo.plot_begin and (
            (bufferIdx+1) % funcInfo.plot_frequency == 0 or bufferIdx == 0
        )

    def dispatch(self, filters, timeIdx, bufferIdx, folder):
        for diagName, funcInfo in self.funcInfo.items():
            if self.plotThisBuffer(bufferIdx, funcInfo):
                diagSet = self.getDiagnosticSet(diagName, filters)
                self.dispatchPlot(funcInfo, diagName, diagSet, timeIdx, folder)
                self.dispatchSummary(funcInfo, diagName, diagSet, timeIdx, folder)

    def dispatchPlot(self, funcInfo, diagName, diagSet, timeIdx, folder):
        if not isinstance(funcInfo.outputFunction, (list, tuple)):
            funcInfo.outputFunction = [funcInfo.outputFunction]

        for i, outType in enumerate(funcInfo.outputFunction):
            if len(funcInfo.outputFunction) == 1:
                outputs = {key: diag.getOutput() for key, diag in diagSet.items()}
            elif len(funcInfo.outputFunction) > 1:
                outputs = {key: diag.getOutput()[i] for key, diag in diagSet.items()}
            plot_data = diagSet[list(diagSet.keys())[0]].plot_data
            outType(diagName, outputs, plot_data, timeIdx, folder, self.outputFormat)

        # if funcInfo.outputFunction == OUTPUTFUNCTION.functionoftime:
        #     dplot.functionOfTimePlot(diagName, diagSet, timeIdx, folder, self.outputFormat)
        # elif funcInfo.outputFunction == OUTPUTFUNCTION.soundfield:
        #     dplot.soundfieldPlot(diagName, diagSet, timeIdx, folder, self.outputFormat)
        # else:
        #     raise NotImplementedError

    def dispatchSummary(self, funcInfo, diagName, diagSet, timeIdx, folder):
        if not isinstance(funcInfo.summaryFunction, (list, tuple)):
            funcInfo.summaryFunction = [funcInfo.summaryFunction]

        for i, sumType in enumerate(funcInfo.summaryFunction):
            if len(funcInfo.summaryFunction) == 1:
                outputs = {key: diag.getOutput() for key, diag in diagSet.items()}
            elif len(funcInfo.summaryFunction) > 1:
                outputs = {key: diag.getOutput()[i] for key, diag in diagSet.items()}

            if sumType is not None:
                summaryValues = sumType(outputs, timeIdx)
                dsum.addToSummary(diagName, summaryValues, timeIdx, folder)
            # if sumType == SUMMARYFUNCTION.none:
            #     return
            # elif sumType == SUMMARYFUNCTION.lastValue:
            #     summaryValues = dsum.lastValue(outputs, timeIdx)
            # elif sumType == SUMMARYFUNCTION.meanNearTimeIdx:
            #     summaryValues = dsum.meanNearTimeIdx(outputs, timeIdx)
            # else:
            #     raise NotImplementedError
            # dsum.addToSummary(diagName, summaryValues, timeIdx, folder)

    def getDiagnosticSet(self, diagnosticName, processors):
        diagnostic = {}
        for proc in processors:
            if diagnosticName in proc.diag:
                diagnostic[proc.name] = proc.diag.diagnostics[diagnosticName]
        return diagnostic

    def getUniqueFuncInfo(self, filters):
        funcInfo = {}
        for filt in filters:
            for key in filt.diag.diagnostics.keys():
                if key not in funcInfo:
                    funcInfo[key] = filt.diag.getFuncInfo(key)
        return funcInfo


class DiagnosticHandler:
    def __init__(self, sim_info, block_size):
        self.sim_info = sim_info
        self.block_size = block_size
        # self.sim_buffer = sim_buffer
        # self.sim_chunk_size = sim_chunk_size

        self.diagnostics = {}
        self.lastGlobalSaveIdx = {}
        self.lastSaveIdx = {}
        self.samplesUntilSave = {}

        #self.lastIdx = self.sim_buffer
        #self.lastGlobalIdx = 0
        self.diagnosticSamplesLeft = 0

    def prepare(self):
        for diagName, diag in self.diagnostics.items():
            self.lastGlobalSaveIdx[diagName] = 0
            self.lastSaveIdx[diagName] = self.sim_info.sim_buffer
            self.samplesUntilSave[diagName] = self.sim_info.sim_buffer
            diag.prepare()
        #for diagName, diag in self.diagnostics.items():
        #    diag.save_frequency

    def __contains__(self, diagName):
        return diagName in self.diagnostics

    def reset(self):
        for diag in self.diagnostics.values():
            diag.reset()

    def getFuncInfo(self, name):
        return self.diagnostics[name].info

    def addNewDiagnostic(self, name, diagnostic):
        # if diagnostic.info.type == DIAGNOSTICTYPE.perSimBuffer:
        #     self.perSimBufferDiagnostics.append(name)
        # elif diagnostic.info.type == DIAGNOSTICTYPE.perBlock:
        #     self.perBlockDiagnostics.append(name)
        # elif diagnostic.info.type == DIAGNOSTICTYPE.perSample:
        #     self.perSampleDiagnostics.append(name)
        # else:
        #     raise ValueError
        self.diagnostics[name] = diagnostic

    def saveData(self, processor, sig, idx, globalIdx):
        for diagName, diag in self.diagnostics.items():
            if globalIdx >= self.lastGlobalSaveIdx[diagName] + self.samplesUntilSave[diagName]:
                
                if idx < self.lastSaveIdx[diagName]:
                    self.lastSaveIdx[diagName] -= self.sim_info.sim_chunk_size
                numSamples = idx - self.lastSaveIdx[diagName]

                diag.saveData(processor, sig, self.lastSaveIdx[diagName], idx, self.lastGlobalSaveIdx[diagName], globalIdx)

                self.samplesUntilSave[diagName] += diag.save_frequency - numSamples
                self.lastSaveIdx[diagName] += numSamples
                self.lastGlobalSaveIdx[diagName] += numSamples
                

    def saveData_old(self, sig, idx, globalIdx):
        if idx-self.lastIdx < globalIdx-self.lastGlobalIdx:
            self.lastIdx -= self.sim_info.sim_chunk_size

        for diagName, diag in self.diagnostics.items():
            diag.saveData(sig, self.lastIdx, idx, self.lastGlobalIdx, globalIdx)

        self.lastGlobalIdx = globalIdx
        self.lastIdx = idx



    # def saveBlockData(self, name, idx, *args):
    #     self.diagnostics[name].saveBlockData(idx, *args)

    # def saveBufferData(self, name, idx, *args):
    #     if name in self.perBlockDiagnostics or name in self.perSampleDiagnostics:
    #         startIdx = self.sim_buffer - self.diagnosticSamplesLeft
    #         numLeftOver = self.sim_buffer + self.sim_chunk_size - idx
    #         saveStartIdx = (
    #             self.bufferIdx * self.sim_chunk_size - self.diagnosticSamplesLeft
    #         )
    #         saveEndIdx = (self.bufferIdx + 1) * self.sim_chunk_size - numLeftOver

    #         self.diagnostics[name].saveDiagnostics(
    #             startIdx, idx, saveStartIdx, saveEndIdx, *args
    #         )
    #     elif name in self.perSimBufferDiagnostics:
    #         self.diagnostics[name].saveDiagnostics(self.bufferIdx, *args)

    # def resetBuffers(self, idx):
    #     """Must be called after simulation buffer resets.
    #     Calls resetBuffer() for all owned diagnostic objects, and increments counters"""
    #     for key, val in self.diagnostics.items():
    #         val.resetBuffers()
    #     numLeftOver = self.sim_buffer + self.sim_chunk_size - idx
    #     self.diagnosticSamplesLeft = numLeftOver
    #     self.bufferIdx += 1

class DiagnosticInfo:
    def __init__(
        self,
        outputFunction,
        summaryFunction=None,
        plot_begin=0,
        plot_frequency=1,
    ):
        self.outputFunction = outputFunction 
        self.summaryFunction = summaryFunction 
        self.plot_begin = plot_begin
        self.plot_frequency = plot_frequency  # Unit is number of simulator-buffers

    def __eq__(self, other):
        return (self.outputFunction == other.outputFunction
            and self.summaryFunction == other.summaryFunction
            and self.plot_begin == other.plot_begin
            and self.plot_frequency == other.plot_frequency
        )
        
class DiagnosticOverTime(ABC):
    def __init__(self, sim_info,
                        blockSize=None,
                        smoothing_len=None,
                        save_frequency=None, 
                        save_raw_data=None, 
                        plot_begin=None, 
                        plot_frequency=None):

        self.sim_info = sim_info
        self.blockSize = blockSize
        self.smoothing_len = smoothing_len
        self.save_frequency = save_frequency
        self.save_raw_data = save_raw_data
        self.plot_begin = plot_begin
        self.plot_frequency = plot_frequency

        if self.smoothing_len is None:
            self.smoothing_len = sim_info.output_smoothing
        if self.save_raw_data is None:
            self.save_raw_data = sim_info.save_raw_data
        if self.plot_begin is None:
            self.plot_begin = sim_info.plot_begin
        if self.plot_frequency is None:
            self.plot_frequency = sim_info.plot_frequency
        if self.save_frequency is None:
            if self.blockSize is None:
                raise ValueError
            #hacky solution, works if simbuffer is large enough 
            self.save_frequency = 1024#max(min(self.sim_info.sim_buffer, self.sim_info.sim_chunk_size) - self.blockSize, self.blockSize)
            #assert self.save_frequency > 0

        outputFunc = [dplot.functionOfTimePlot]
        if save_raw_data:
            outputFunc.append(dplot.savenpz)

        self.info = DiagnosticInfo(
            outputFunc,
            dsum.meanNearTimeIdx,
            self.plot_begin,
            self.plot_frequency,
        )

        self.plot_data = {
            "title" : "",
            "xlabel" : "Samples",
            "ylabel" : "",
        }
        
    def prepare(self):
        pass
        #self.blockSize = blockSize
        
    @abstractmethod
    def saveData(self, processor, sig, startIdx, endIdx, saveStartIdx, saveEndIdx):
        pass

    @abstractmethod
    def getOutput(self):
        pass


class SignalDiagnostic(DiagnosticOverTime):
    def __init__(self, sim_info, **kwargs):
        super().__init__(sim_info, **kwargs)


class StateDiagnostic(DiagnosticOverTime):
    def __init__(self, sim_info, property_name, save_frequency, **kwargs):
        super().__init__(sim_info, save_frequency=save_frequency, **kwargs)
        self.property_name = property_name
        self.get_property = attrgetter(property_name)

    @abstractmethod
    def saveData(self, processor, sig, _unusedIdx1, _unusedIdx2, _unusedIdx3, globalIdx):
        prop = self.get_property(processor)


# class FunctionOfTimeDiagnostic():
#     def __init__(self, save_frequency, smoothing_len, save_raw_data, plot_begin, plot_frequency):
#         self.save_frequency = save_frequency
#         self.smoothing_len = smoothing_len
#         self.save_raw_data = save_raw_data

#         outputFunc = [dplot.functionOfTimePlot]
#         if save_raw_data:
#             outputFunc.append(dplot.savenpz)

#         self.info = DiagnosticInfo(
#             outputFunc,
#             dsum.meanNearTimeIdx,
#             plot_begin,
#             plot_frequency,
#         )

#         self.plot_data = {
#             "title" : "",
#             "xlabel" : "Samples",
#             "ylabel" : "",
#         }

#     @abstractmethod
#     def saveData(self, processor, sig, startIdx, endIdx, saveStartIdx, saveEndIdx):
#         pass

#     @abstractmethod
#     def getOutput(self):
#        pass

class SignalPowerRatio(SignalDiagnostic):
    def __init__(
        self,
        sim_info, 
        numeratorName,
        denominatorName,
        **kwargs
    ):
        super().__init__(sim_info, **kwargs)
        self.numeratorName = numeratorName
        self.denominatorName = denominatorName
        
        self.numPower = np.zeros(self.sim_info.tot_samples)
        self.denomPower = np.zeros(self.sim_info.tot_samples)
        self.numSmoother = Filter_IntBuffer(ir=np.ones((self.smoothing_len))/self.smoothing_len, numIn=1)
        self.denomSmoother = Filter_IntBuffer(ir=np.ones((self.smoothing_len))/self.smoothing_len, numIn=1)
        self.powerRatio = np.full((self.sim_info.tot_samples), np.nan)

        self.plot_data["title"] = "Power Ratio: " + self.numeratorName + " / " + self.denominatorName
        self.plot_data["ylabel"] = "Ratio (dB)"
        

    def saveData(self, processor, sig, startIdx, endIdx, saveStartIdx, saveEndIdx):
        numPower = np.mean(sig[self.numeratorName][..., startIdx:endIdx] ** 2, axis=0, keepdims=True)
        denomPower = np.mean(
            sig[self.denominatorName][..., startIdx:endIdx] ** 2, axis=0, keepdims=True
        )

        numPowerSmooth = self.numSmoother.process(numPower)
        denomPowerSmooth = self.denomSmoother.process(denomPower)

        self.powerRatio[saveStartIdx:saveEndIdx] = util.pow2db(
            numPowerSmooth / denomPowerSmooth
        )
        self.numPower[saveStartIdx:saveEndIdx] = numPower
        self.denomPower[saveStartIdx:saveEndIdx] = denomPower

    def getOutput(self):
        if self.save_raw_data:
            return [
                self.powerRatio,
                {
                    "Numerator Power": self.numPower,
                    "Denominator Power": self.denomPower,
                },
            ]
        else:
            return self.powerRatio

class StateNMSE(StateDiagnostic):
    def __init__(self,
                sim_info, 
                property_name,
                true_value, 
                save_frequency,
                **kwargs
            ):
        super().__init__(sim_info, property_name, save_frequency, **kwargs)
        self.true_value = true_value
        self.true_power = np.mean(np.abs(self.true_value)**2)
        self.nmse = np.full(self.sim_info.tot_samples, np.nan)

        self.plot_data["ylabel"] = "NMSE (dB)"
        self.plot_data["title"] = "NMSE"

    def saveData(self, processor, _sig, _unusedIdx1, _unusedIdx2, _unusedIdx3, globalIdx):
        self.nmse[globalIdx] = util.pow2db((np.mean(np.abs(self.true_value - self.get_property(processor))**2) / self.true_power))

    def getOutput(self):
        return self.nmse


class SignalPower(SignalDiagnostic):
    def __init__(self, 
                sim_info, 
                signal_name,
                **kwargs):
        super().__init__(sim_info, **kwargs)
        self.signal_name = signal_name
        self.signal_power = np.full(self.sim_info.tot_samples, np.nan)
        self.signal_smoother = Filter_IntBuffer(ir=np.ones((self.smoothing_len))/self.smoothing_len, numIn=1)

        self.plot_data["title"] = "Signal power"
        self.plot_data["ylabel"] = "Power (dB)"

        if self.save_raw_data:
            raise NotImplementedError
    
    def saveData(self, processor, sig, startIdx, endIdx, saveStartIdx, saveEndIdx):
        #print(startIdx, endIdx)
        signal_power = np.mean(sig[self.signal_name][:, startIdx:endIdx] ** 2, axis=0, keepdims=True)
        self.signal_power[saveStartIdx:saveEndIdx] = util.pow2db(self.signal_smoother.process(signal_power))

    def getOutput(self):
        return self.signal_power



# class NMSE(FunctionOfTimeDiagnostic):
#     def __init__(self, 
#                 signal_name, 
#                 tot_samples,
#                 smoothing_len=1,
#                 plot_begin=0,
#                 plot_frequency=1):
#         super.__init__(smoothing_len, False, plot_begin, plot_frequency)
#         self.signal_name = signal_name
#         self.signal_power = np.full(tot_samples, np.nan)
#         self.signal_smoother = Filter_IntBuffer(ir=np.ones((smoothing_len))/smoothing_len, numIn=1)

#         self.plot_data["title"] = "Power"
#         self.plot_data["ylabel"] = "Power (dB)"
    
#     def saveData(self, sig, startIdx, endIdx, saveStartIdx, saveEndIdx):
#         signal_power = np.mean(sig[self.signal_name][:, startIdx:endIdx] ** 2, axis=0, keepdims=True)
#         self.signal_power[saveStartIdx:saveEndIdx] = util.pow2db(self.signal_smoother.process(signal_power))

#     def getOutput(self):
#         return self.signal_power


# class NoiseReduction:
#     def __init__(
#         self,
#         numPoints,
#         acousticPath,
#         tot_samples,
#         sim_buffer,
#         sim_chunk_size,
#         smoothing_len=1,
#         save_raw_data=True,
#         plot_begin=0,
#         plot_frequency=1,
#         **kwargs
#     ):
#         self.sim_buffer = sim_buffer
#         self.sim_chunk_size = sim_chunk_size
#         self.save_raw_data = save_raw_data
#         if save_raw_data:
#             outputFunc = [dplot.functionOfTimePlot, dplot.savenpz]
#             summaryFunc = [SUMMARYFUNCTION.meanNearTimeIdx, SUMMARYFUNCTION.none]
#         else:
#             outputFunc = dplot.functionOfTimePlot
#             summaryFunc = SUMMARYFUNCTION.meanNearTimeIdx

#         self.info = DiagnosticInfo(
#             DIAGNOSTICTYPE.perSample,
#             outputFunc,
#             summaryFunc,
#             plot_begin=plot_begin,
#             plot_frequency=plot_frequency,
#         )
#         self.pathFilter = FilterSum(acousticPath)

#         self.totalNoise = np.zeros((numPoints, self.sim_chunk_size + self.sim_buffer))
#         self.primaryNoise = np.zeros((numPoints, self.sim_chunk_size + self.sim_buffer))

#         self.totalNoisePower = np.zeros(tot_samples)
#         self.primaryNoisePower = np.zeros(tot_samples)

#         self.totalNoiseSmoother = Filter_IntBuffer(ir=np.ones((smoothing_len)), numIn=1)
#         self.primaryNoiseSmoother = Filter_IntBuffer(
#             ir=np.ones((smoothing_len)), numIn=1
#         )

#         self.noiseReduction = np.full((tot_samples), np.nan)
#         self.metadata = {
#             "title": "Noise Reduction",
#             "xlabel": "Samples",
#             "ylabel": "Reduction (dB)",
#         }
#         for key, value in kwargs.items():
#             self.metadata[key] = value

#     def getOutput(self):
#         if self.save_raw_data:
#             return [
#                 self.noiseReduction,
#                 {
#                     "Total Noise Power": self.totalNoisePower,
#                     "Primary Noise Power": self.primaryNoisePower,
#                 },
#             ]
#         else:
#             return self.noiseReduction

#     def saveDiagnostics(self, startIdx, endIdx, saveStartIdx, saveEndIdx, y):
#         secondaryNoise = self.pathFilter.process(y[:, startIdx:endIdx])
#         self.totalNoise[:, startIdx:endIdx] = (
#             self.primaryNoise[:, startIdx:endIdx] + secondaryNoise
#         )

#         totalNoisePower = np.mean(
#             self.totalNoise[:, startIdx:endIdx] ** 2, axis=0, keepdims=True
#         )
#         primaryNoisePower = np.mean(
#             self.primaryNoise[:, startIdx:endIdx] ** 2, axis=0, keepdims=True
#         )

#         self.totalNoisePower[saveStartIdx:saveEndIdx] = totalNoisePower
#         self.primaryNoisePower[saveStartIdx:saveEndIdx] = primaryNoisePower

#         smoothTotalNoisePower = self.totalNoiseSmoother.process(totalNoisePower)
#         smoothPrimaryNoisePower = self.primaryNoiseSmoother.process(primaryNoisePower)

#         self.noiseReduction[saveStartIdx:saveEndIdx] = util.pow2db(
#             smoothTotalNoisePower / smoothPrimaryNoisePower
#         )

#     def resetBuffers(self):
#         self.totalNoise = np.concatenate(
#             (
#                 self.totalNoise[:, -self.sim_buffer :],
#                 np.zeros((self.totalNoise.shape[0], self.sim_chunk_size)),
#             ),
#             axis=-1,
#         )
#         self.primaryNoise = np.concatenate(
#             (
#                 self.primaryNoise[:, -self.sim_buffer :],
#                 np.zeros((self.primaryNoise.shape[0], self.sim_chunk_size)),
#             ),
#             axis=-1,
#         )

#     # def saveToBuffers(self, newPrimaryNoise, idx, numSamples):
#     #     self.primaryNoise[:,idx:idx+numSamples] = newPrimaryNoise
#     def saveBlockData(self, idx, newPrimaryNoise):
#         numSamples = newPrimaryNoise.shape[-1]
#         self.primaryNoise[:, idx : idx + numSamples] = newPrimaryNoise

# class NoiseReductionExternalSignals:
#     def __init__(
#         self,
#         numPoints,
#         tot_samples,
#         sim_buffer,
#         sim_chunk_size,
#         smoothing_len=1,
#         save_raw_data=True,
#         plot_begin=0,
#         plot_frequency=1,
#         **kwargs
#     ):
#         self.sim_buffer = sim_buffer
#         self.sim_chunk_size = sim_chunk_size
#         self.save_raw_data = save_raw_data
#         if save_raw_data:
#             outputFunc = [dplot.functionOfTimePlot, dplot.savenpz]
#             summaryFunc = [SUMMARYFUNCTION.meanNearTimeIdx, SUMMARYFUNCTION.none]
#         else:
#             outputFunc = dplot.functionOfTimePlot
#             summaryFunc = SUMMARYFUNCTION.meanNearTimeIdx

#         self.info = DiagnosticInfo(
#             DIAGNOSTICTYPE.perSample,
#             outputFunc,
#             summaryFunc,
#             plot_begin=plot_begin,
#             plot_frequency=plot_frequency,
#         )
#         self.primaryNoise = np.zeros((numPoints, self.sim_chunk_size + self.sim_buffer))
#         self.totalNoisePower = np.zeros(tot_samples)
#         self.primaryNoisePower = np.zeros(tot_samples)
#         self.totalNoiseSmoother = Filter_IntBuffer(ir=np.ones((smoothing_len)), numIn=1)
#         self.primaryNoiseSmoother = Filter_IntBuffer(
#             ir=np.ones((smoothing_len)), numIn=1
#         )
#         self.noiseReduction = np.full((tot_samples), np.nan)

#         self.metadata = {
#             "title": "Noise Reduction",
#             "xlabel": "Samples",
#             "ylabel": "Reduction (dB)",
#         }
#         for key, value in kwargs.items():
#             self.metadata[key] = value

#     # def getOutput(self):
#     #     return self.noiseReduction
#     def getOutput(self):
#         if self.save_raw_data:
#             return [
#                 self.noiseReduction,
#                 {
#                     "Total Noise Power": self.totalNoisePower,
#                     "Primary Noise Power": self.primaryNoisePower,
#                 },
#             ]
#         else:
#             return self.noiseReduction

#     def saveDiagnostics(self, startIdx, endIdx, saveStartIdx, saveEndIdx, e):
#         totalNoisePower = np.mean(e[:, startIdx:endIdx] ** 2, axis=0, keepdims=True)
#         primaryNoisePower = np.mean(
#             self.primaryNoise[:, startIdx:endIdx] ** 2, axis=0, keepdims=True
#         )

#         totalNoisePowerSmooth = self.totalNoiseSmoother.process(totalNoisePower)
#         primaryNoisePowerSmooth = self.primaryNoiseSmoother.process(primaryNoisePower)

#         self.noiseReduction[saveStartIdx:saveEndIdx] = util.pow2db(
#             totalNoisePowerSmooth / primaryNoisePowerSmooth
#         )
#         self.totalNoisePower[saveStartIdx:saveEndIdx] = totalNoisePower
#         self.primaryNoisePower[saveStartIdx:saveEndIdx] = primaryNoisePower

#     def resetBuffers(self):
#         self.primaryNoise = np.concatenate(
#             (
#                 self.primaryNoise[:, -self.sim_buffer :],
#                 np.zeros((self.primaryNoise.shape[0], self.sim_chunk_size)),
#             ),
#             axis=-1,
#         )

#     def saveBlockData(self, idx, newPrimaryNoise):
#         numSamples = newPrimaryNoise.shape[-1]
#         self.primaryNoise[:, idx : idx + numSamples] = newPrimaryNoise




















class SoundfieldImage:
    def __init__(
        self,
        numPoints,
        acousticPath,
        sim_buffer,
        sim_chunk_size,
        plot_begin=0,
        plot_frequency=1,
    ):
        self.sim_buffer = sim_buffer
        self.sim_chunk_size = sim_chunk_size
        self.info = DiagnosticInfo(
            DIAGNOSTICTYPE.perSimBuffer,
            dplot.soundfieldPlot,
            None,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
        )
        self.samplesForSF = np.min((2048, self.sim_buffer, self.sim_chunk_size))
        self.totalNoise = np.zeros((numPoints, self.samplesForSF))
        self.primaryNoise = np.zeros((numPoints, self.sim_chunk_size + self.sim_buffer))
        self.pathFilter = FilterSum(acousticPath)

    def getOutput(self):
        return self.totalNoise

    def resetBuffers(self):
        pass

    def saveDiagnostics(self, chunkIdx, y):
        if (
            chunkIdx >= self.info.plot_begin - 1
            and (chunkIdx + 1) % self.info.plot_frequency == 0
        ):
            startIdx = self.sim_buffer
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
        sim_buffer,
        sim_chunk_size,
        plot_begin=0,
        plot_frequency=1,
    ):
        self.sim_buffer = sim_buffer
        self.sim_chunk_size = sim_chunk_size
        self.info = DiagnosticInfo(
            DIAGNOSTICTYPE.perSimBuffer,
            dplot.soundfieldPlot,
            None,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
        )
        self.samplesForSF = np.min((2048, self.sim_buffer, self.sim_chunk_size))
        self.totalNoise = np.zeros((numPoints, self.samplesForSF))
        self.primaryNoise = np.zeros((numPoints, self.sim_chunk_size + self.sim_buffer))
        self.pathFilter = FilterSum(acousticPath)

    def getOutput(self):
        return self.totalNoise

    def resetBuffers(self):
        pass

    def saveDiagnostics(self, chunkIdx, y):
        if (
            chunkIdx >= self.info.plot_begin - 1
            and (chunkIdx + 1) % self.info.plot_frequency == 0
        ):
            startIdx = self.sim_buffer
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
        true_value,
        tot_samples,
        sim_buffer,
        sim_chunk_size,
        maxNumIR, #tuple
        plot_begin=0,
        plot_frequency=1,
    ):
        self.selectIR = [slice(None,numIr) for numIr in maxNumIR]
        self.selectIR.append(slice(None))
        self.selectIR = tuple(self.selectIR)
        self.true_value = true_value[self.selectIR].reshape((-1, true_value.shape[-1]))
        self.tot_samples = tot_samples
        self.sim_buffer = sim_buffer
        self.sim_chunk_size = sim_chunk_size

        self.latestIR = np.zeros_like(self.true_value)

        self.metadata = {"xlabel" : "Samples", "ylabel" : "Amplitude", "label" : ["True Value", "Estimate"]}

        self.info = DiagnosticInfo(
                DIAGNOSTICTYPE.perSimBuffer,
                dplot.plotIR,
                None,
                plot_begin=plot_begin,
                plot_frequency=plot_frequency,
            )

    def getOutput(self):
        return [self.true_value, self.latestIR]

    def saveDiagnostics(self, bufferIdx, ir):
        self.latestIR[...] = ir[self.selectIR].reshape((-1, ir.shape[-1]))

    def resetBuffers(self):
        pass

    def saveBlockData(self, idx):
        pass






class PerBlockDiagnostic(ABC):
    def __init__(
        self,
        tot_samples,
        sim_buffer,
        sim_chunk_size,
        plot_begin=0,
        plot_frequency=1,
        **kwargs
    ):
        self.sim_buffer = sim_buffer
        self.sim_chunk_size = sim_chunk_size
        self.info = DiagnosticInfo(
            DIAGNOSTICTYPE.perBlock,
            dplot.functionOfTimePlot,
            dsum.meanNearTimeIdx,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
        )

        self.dataBuffer = np.full((tot_samples), np.nan)
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
        true_value,
        tot_samples,
        sim_buffer,
        sim_chunk_size,
        plot_begin=0,
        plot_frequency=1,
        **kwargs
    ):
        super().__init__(
            tot_samples,
            sim_buffer,
            sim_chunk_size,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
            **kwargs
        )
        self.true_value = np.copy(true_value)
        self.true_valuePower = np.sum(np.abs(true_value) ** 2)

        self.metadata["title"] = "Constant Estimate NMSE"
        self.metadata["ylabel"] = "NMSE (dB)"

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx * self.sim_chunk_size + (idx - self.sim_buffer)
        self.dataBuffer[totIdx] = 10 * np.log10(1e-30 + 
            np.sum(np.abs(self.true_value - currentEstimate) ** 2) / self.true_valuePower
        )

class ConstantEstimateNMSEAllCombinations(PerBlockDiagnostic):
    def __init__(
        self,
        true_value,
        tot_samples,
        sim_buffer,
        sim_chunk_size,
        plot_begin=0,
        plot_frequency=1,
        **kwargs
    ):
        super().__init__(
            tot_samples,
            sim_buffer,
            sim_chunk_size,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
            **kwargs
        )
        self.true_value = np.copy(true_value)
        self.true_valuePower = np.sum(np.abs(true_value) ** 2)


        assert self.true_value.ndim == 3
        numChannels = np.math.factorial(self.true_value.shape[1]) * \
                        np.math.factorial(self.true_value.shape[2])
        self.dataBuffer = np.full((numChannels, tot_samples), np.nan)

        
        

        self.metadata["title"] = "Constant Estimate NMSE"
        self.metadata["ylabel"] = "NMSE (dB)"

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx * self.sim_chunk_size + (idx - self.sim_buffer)

        for i,(ax1,ax2) in enumerate(it.product(it.permutations(range(self.true_value.shape[1])), 
                                                it.permutations(range(self.true_value.shape[2])))):
            crntEst = currentEstimate[:,ax1,:]
            self.dataBuffer[i,totIdx] = 10 * np.log10(1e-30 + 
                np.sum(np.abs(self.true_value - crntEst[:,:,ax2]) ** 2) / self.true_valuePower
            )

class ConstantEstimateNMSESelectedFrequencies(PerBlockDiagnostic):
    def __init__(
        self,
        true_value,
        lowFreq,
        highFreq,
        sampleRate,
        tot_samples,
        sim_buffer,
        sim_chunk_size,
        plot_begin=0,
        plot_frequency=1,
        **kwargs
    ):
        super().__init__(
            tot_samples,
            sim_buffer,
            sim_chunk_size,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
            **kwargs
        )
        numFreqBins = true_value.shape[0]
        self.lowBin = int(numFreqBins * lowFreq / sampleRate)+1
        self.highBin = int(numFreqBins * highFreq / sampleRate)

        self.true_value = np.copy(true_value[self.lowBin : self.highBin, ...])
        self.true_valuePower = np.sum(np.abs(self.true_value) ** 2)

        self.metadata["title"] = "Constant Estimate NMSE Selected Frequencies"
        self.metadata["ylabel"] = "NMSE (dB)"
        self.metadata["low frequency"] = lowFreq
        self.metadata["high frequency"] = highFreq

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx * self.sim_chunk_size + (idx - self.sim_buffer)
        self.dataBuffer[totIdx] = 10 * np.log10(
            np.sum(
                np.abs(
                    self.true_value - currentEstimate[self.lowBin : self.highBin, ...]
                )
                ** 2
            )
            / self.true_valuePower
        )



class ConstantEstimatePhaseDifferenceSelectedFrequencies(PerBlockDiagnostic):
    def __init__(
        self,
        true_value,
        lowFreq,
        highFreq,
        sampleRate,
        tot_samples,
        sim_buffer,
        sim_chunk_size,
        plot_begin=0,
        plot_frequency=1,
        **kwargs
    ):
        super().__init__(
            tot_samples,
            sim_buffer,
            sim_chunk_size,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
            **kwargs
        )

        self.dataBuffer = np.full((tot_samples), np.nan)
        numFreqBins = true_value.shape[0]
        self.lowBin = int(numFreqBins * lowFreq / sampleRate)+1
        self.highBin = int(numFreqBins * highFreq / sampleRate)
        self.truePhase = np.copy(np.angle(true_value[self.lowBin:self.highBin,...]))

        self.metadata["title"] = "Constant Estimate Phase"
        self.metadata["ylabel"] = "Average Difference (rad)"
        #self.metadata["label_suffix_channel"] = ["mean", "max"]

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx * self.sim_chunk_size + (idx - self.sim_buffer)
        phaseDiff = np.abs(self.truePhase - np.angle(currentEstimate[self.lowBin:self.highBin,...]))
        phaseDiff = phaseDiff % np.pi
        #pha = np.min((np.abs(2*np.pi-phaseDiff),np.abs(phaseDiff), np.abs(2*np.pi+phaseDiff)))
        self.dataBuffer[totIdx] = np.mean(phaseDiff)
        #self.dataBuffer[1, totIdx] = np.max(phaseDiff)



class ConstantEstimateWeightedPhaseDifference(PerBlockDiagnostic):
    def __init__(
        self,
        true_value,
        lowFreq,
        highFreq,
        sampleRate,
        tot_samples,
        sim_buffer,
        sim_chunk_size,
        plot_begin=0,
        plot_frequency=1,
        **kwargs
    ):
        super().__init__(
            tot_samples,
            sim_buffer,
            sim_chunk_size,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
            **kwargs
        )

        self.dataBuffer = np.full((tot_samples), np.nan)
        numFreqBins = true_value.shape[0]
        self.lowBin = int(numFreqBins * lowFreq / sampleRate)+1
        self.highBin = int(numFreqBins * highFreq / sampleRate)
        self.truePhase = np.copy(np.angle(true_value[self.lowBin:self.highBin,...]))

        trueTotal = np.sum(np.abs(true_value[self.lowBin:self.highBin,...]), axis=0, keepdims=True)
        self.weight = np.abs(true_value[self.lowBin:self.highBin,...]) / trueTotal

        self.metadata["title"] = "Constant Estimate Phase"
        self.metadata["ylabel"] = "Average Difference (rad)"
        #self.metadata["label_suffix_channel"] = ["mean", "max"]

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx * self.sim_chunk_size + (idx - self.sim_buffer)
        phaseDiff = np.abs(self.truePhase - np.angle(currentEstimate[self.lowBin:self.highBin,...]))
        phaseDiff = phaseDiff % np.pi
        #phaseDiff *= self.weight
        #pha = np.min((np.abs(2*np.pi-phaseDiff),np.abs(phaseDiff), np.abs(2*np.pi+phaseDiff)))
        self.dataBuffer[totIdx] = np.mean(np.sum(self.weight * phaseDiff,axis=0))
        #self.dataBuffer[1, totIdx] = np.max(phaseDiff)

class ConstantEstimateAmpDifference(PerBlockDiagnostic):
    def __init__(
        self,
        true_value,
        tot_samples,
        sim_buffer,
        sim_chunk_size,
        plot_begin=0,
        plot_frequency=1,
        **kwargs
    ):
        super().__init__(
            tot_samples,
            sim_buffer,
            sim_chunk_size,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
            **kwargs
        )

        self.trueAmplitude = np.copy(np.abs(true_value))
        self.trueAmpSum = np.sum(self.trueAmplitude)

        self.metadata["title"] = "Constant Estimate Amplitude"
        self.metadata["ylabel"] = "Normalized Sum Difference (dB)"

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx * self.sim_chunk_size + (idx - self.sim_buffer)
        self.dataBuffer[totIdx] = 20 * np.log10(
            np.sum(np.abs(self.trueAmplitude - np.abs(currentEstimate)))
            / self.trueAmpSum
        )


class ConstantEstimatePhaseDifference(PerBlockDiagnostic):
    def __init__(
        self,
        true_value,
        tot_samples,
        sim_buffer,
        sim_chunk_size,
        plot_begin=0,
        plot_frequency=1,
        **kwargs
    ):
        super().__init__(
            tot_samples,
            sim_buffer,
            sim_chunk_size,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
            **kwargs
        )

        self.truePhase = np.copy(np.angle(true_value))

        self.metadata["title"] = "Constant Estimate Phase"
        self.metadata["ylabel"] = "Average Difference (rad)"

    def saveBlockData(self, idx, currentEstimate):
        totIdx = self.bufferIdx * self.sim_chunk_size + (idx - self.sim_buffer)
        self.dataBuffer[totIdx] = np.mean(
            np.abs(self.truePhase - np.angle(currentEstimate))
        )


class SignalEstimateNMSE:
    def __init__(
        self,
        tot_samples,
        sim_buffer,
        sim_chunk_size,
        smoothing_len=1,
        plot_begin=0,
        plot_frequency=1,
        **kwargs
    ):
        """Assumes the signal to be estimated is of the form (numChannels, numSamples)"""
        self.sim_buffer = sim_buffer
        self.sim_chunk_size = sim_chunk_size
        self.info = DiagnosticInfo(
            DIAGNOSTICTYPE.perSample,
            dplot.functionOfTimePlot,
            dsum.meanNearTimeIdx,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
        )

        self.recordedValues = np.full((tot_samples), np.nan)
        self.bufferIdx = 0
        self.metadata = {
            "title": "Signal Estimate NMSE",
            "xlabel": "Samples",
            "ylabel": "NMSE (dB)",
        }
        # self.errorSmoother = Filter_IntBuffer(ir=np.ones((smoothing_len)))
        self.true_valuePowerSmoother = Filter_IntBuffer(
            ir=np.ones((smoothing_len)) / smoothing_len
        )

    def getOutput(self):
        return self.recordedValues

    def saveDiagnostics(self, startIdx, endIdx, saveStartIdx, saveEndIdx):
        pass

    def resetBuffers(self):
        self.bufferIdx += 1

    def saveBlockData(self, idx, estimate, true_value):
        numSamples = estimate.shape[-1]
        
        totIdx = self.bufferIdx * self.sim_chunk_size + (idx - self.sim_buffer)

        nmse = np.sum(
            np.abs(estimate - true_value) ** 2, axis=tuple([i for i in range(estimate.ndim-1)])
        ) / self.true_valuePowerSmoother.process(
            np.sum(np.abs(true_value) ** 2, axis=tuple([i for i in range(estimate.ndim-1)]))[None,:]
        )

        self.recordedValues[totIdx : totIdx + numSamples] = 10 * np.log10(nmse)


class SignalEstimateNMSEBlock:
    def __init__(
        self,
        tot_samples,
        sim_buffer,
        sim_chunk_size,
        plot_begin=0,
        plot_frequency=1,
        **kwargs
    ):
        """Assumes the signal to be estimated is of the form (numChannels, numSamples)"""
        self.sim_buffer = sim_buffer
        self.sim_chunk_size = sim_chunk_size
        self.info = DiagnosticInfo(
            DIAGNOSTICTYPE.perSample,
            dplot.functionOfTimePlot,
            dsum.meanNearTimeIdx,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
        )

        self.recordedValues = np.full((tot_samples), np.nan)
        self.bufferIdx = 0
        self.metadata = {
            "title": "Signal Estimate NMSE",
            "xlabel": "Samples",
            "ylabel": "NMSE (dB)",
        }
        # self.errorSmoother = Filter_IntBuffer(ir=np.ones((smoothing_len)))

    def getOutput(self):
        return self.recordedValues

    def saveDiagnostics(self, startIdx, endIdx, saveStartIdx, saveEndIdx):
        pass

    def resetBuffers(self):
        self.bufferIdx += 1

    def saveBlockData(self, idx, estimate, true_value):
        numSamples = estimate.shape[-1]
        
        totIdx = self.bufferIdx * self.sim_chunk_size + (idx - self.sim_buffer)

        nmse = np.sum(np.abs(estimate - true_value) ** 2) / np.sum(np.abs(true_value)**2)

        self.recordedValues[totIdx] = 10 * np.log10(1e-30+nmse)

#plt.plot(np.sum(np.abs(1*self.buffers["xf"][...,self.idx-self.blockSize:self.idx] - trueXf)**2,axis=(0,1,2)) / np.sum(np.abs(trueXf)**2))


class RecordScalar:
    def __init__(
        self,
        tot_samples,
        sim_buffer,
        sim_chunk_size,
        plot_begin=0,
        plot_frequency=1,
        **kwargs
    ):
        self.sim_buffer = sim_buffer
        self.sim_chunk_size = sim_chunk_size
        self.info = DiagnosticInfo(
            DIAGNOSTICTYPE.perBlock,
            dplot.functionOfTimePlot,
            dsum.meanNearTimeIdx,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
        )

        self.recordedValues = np.full((tot_samples), np.nan)
        self.bufferIdx = 0
        self.metadata = {"title": "Value of ", "xlabel": "Samples", "ylabel": "value"}

    def getOutput(self):
        return self.recordedValues

    def resetBuffers(self):
        self.bufferIdx += 1

    def saveBlockData(self, idx, value):
        totIdx = self.bufferIdx * self.sim_chunk_size + (idx - self.sim_buffer)
        self.recordedValues[totIdx] = value


class RecordVector:
    def __init__(
        self,
        vectorLength,
        tot_samples,
        sim_buffer,
        sim_chunk_size,
        plot_begin=0,
        plot_frequency=1,
        **kwargs
    ):
        self.sim_buffer = sim_buffer
        self.sim_chunk_size = sim_chunk_size
        self.info = DiagnosticInfo(
            DIAGNOSTICTYPE.perBlock,
            dplot.functionOfTimePlot,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
        )

        self.recordedValues = np.full((vectorLength, tot_samples), np.nan)
        self.bufferIdx = 0
        self.metadata = {"title": "Value of ", "xlabel": "Samples", "ylabel": "value"}

    def getOutput(self):
        return self.recordedValues

    def resetBuffers(self):
        self.bufferIdx += 1

    def saveBlockData(self, idx, value):
        totIdx = self.bufferIdx * self.sim_chunk_size + (idx - self.sim_buffer)
        self.recordedValues[:, totIdx] = value


class RecordVectorSummary:
    def __init__(
        self,
        tot_samples,
        sim_buffer,
        sim_chunk_size,
        plot_begin=0,
        plot_frequency=1,
        **kwargs
    ):
        self.sim_buffer = sim_buffer
        self.sim_chunk_size = sim_chunk_size
        self.info = DiagnosticInfo(
            DIAGNOSTICTYPE.perBlock,
            dplot.functionOfTimePlot,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
        )

        self.recordedValues = np.full((3, tot_samples), np.nan)
        #self.recordedMax = np.full((1, tot_samples), np.nan)
        #self.recordedMin = np.full((1, tot_samples), np.nan)
        #self.recordedMean = np.full((1, tot_samples), np.nan)
        self.bufferIdx = 0
        self.metadata = {"title": "Value of ", "xlabel": "Samples", "ylabel": "value", "label_suffix_channel" : ["max", "mean", "min"]}

    def getOutput(self):
        return self.recordedValues

    def resetBuffers(self):
        self.bufferIdx += 1

    def saveBlockData(self, idx, value):
        totIdx = self.bufferIdx * self.sim_chunk_size + (idx - self.sim_buffer)
        self.recordedValues[0, totIdx] = np.max(value)
        self.recordedValues[1, totIdx] = np.mean(value)
        self.recordedValues[2, totIdx] = np.min(value)





class saveTotalPower:
    def __init__(self, tot_samples, acousticPath=None):
        self.info = DiagnosticInfo(
            DIAGNOSTICTYPE.perSample,
            OUTPUTFUNCTION.savenpz,
            None,
            plot_begin=plot_begin,
            plot_frequency=plot_frequency,
        )
        self.noisePower = np.zeros(tot_samples)
        if acousticPath is not None:
            self.pathFilter = FilterSum(acousticPath)
        else:
            self.pathFilter = FilterSum(np.ones((1, 1, 1)))

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
    def __init__(self, acousticPath, samplerate, maxLength, tot_samples, sim_buffer,
                 sim_chunk_size, plot_begin=0, plot_frequency=1):
        self.pathFilter = FilterSum(ir=acousticPath)
        self.sim_buffer = sim_buffer
        self.sim_chunk_size = sim_chunk_size
        self.chunkIdx = 0

        self.withANC = np.full((self.pathFilter.numOut, tot_samples), np.nan)
        self.withoutANC = np.full((self.pathFilter.numOut, tot_samples), np.nan)

        self.primaryNoise = np.zeros((self.pathFilter.numOut, self.sim_buffer+self.sim_chunk_size))

        self.info = DiagnosticInfo(
            dplot.createAudioFiles,
            plot_begin,
            plot_frequency
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
                self.primaryNoise[:, -self.sim_buffer:],
                np.zeros((self.primaryNoise.shape[0], self.sim_chunk_size)),
            ),
            axis=-1,
        )