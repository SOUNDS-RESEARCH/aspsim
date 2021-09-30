import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from functools import partial
import itertools as it
from operator import attrgetter

import ancsim.array as ar
import ancsim.utilities as util
import ancsim.experiment.plotscripts as psc
import ancsim.signal.filterclasses as fc
import ancsim.diagnostics.diagnosticplots as dplot
import ancsim.diagnostics.diagnosticsummary as dsum


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

    def dispatch(self):
        raise NotImplementedError

    def dispatch_old(self, filters, timeIdx, bufferIdx, folder):
        for diagName, funcInfo in self.funcInfo.items():
            if self.plotThisBuffer(bufferIdx, funcInfo):
                diagSet = self.getDiagnosticSet(diagName, filters)
                self.dispatchPlot(funcInfo, diagName, diagSet, timeIdx, folder)
                self.dispatchSummary(funcInfo, diagName, diagSet, timeIdx, folder)

    def dispatchPlot_old(self, funcInfo, diagName, diagSet, timeIdx, folder):
        if not isinstance(funcInfo.outputFunction, (list, tuple)):
            funcInfo.outputFunction = [funcInfo.outputFunction]

        for i, outType in enumerate(funcInfo.outputFunction):
            if len(funcInfo.outputFunction) == 1:
                outputs = {key: diag.getOutput() for key, diag in diagSet.items()}
            elif len(funcInfo.outputFunction) > 1:
                outputs = {key: diag.getOutput()[i] for key, diag in diagSet.items()}
            plot_data = diagSet[list(diagSet.keys())[0]].plot_data
            outType(diagName, outputs, plot_data, timeIdx, folder, self.outputFormat)

    def dispatchSummary_old(self, funcInfo, diagName, diagSet, timeIdx, folder):
        if not isinstance(funcInfo.summaryFunction, (list, tuple)):
            funcInfo.summaryFunction = [funcInfo.summaryFunction]

        for i, sumType in enumerate(funcInfo.summaryFunction):
            if sumType is not None:
                if len(funcInfo.summaryFunction) == 1:
                    outputs = {key: diag.getOutput() for key, diag in diagSet.items()}
                elif len(funcInfo.summaryFunction) > 1:
                    outputs = {key: diag.getOutput()[i] for key, diag in diagSet.items()}
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
        self.diagnostics = {}

        #To keep track of which samples of data to save to each diagnostics object
        self.lastGlobalSaveIdx = {}
        self.lastSaveIdx = {}
        self.samplesUntilSave = {}

        #To keep track of when each diagnostics are available to be exported
        self.exportAfterSample = {}
        self.toBeExported = []

    def prepare(self):
        for diagName, diag in self.diagnostics.items():
            self.lastGlobalSaveIdx[diagName] = 0
            self.lastSaveIdx[diagName] = self.sim_info.sim_buffer
            self.samplesUntilSave[diagName] = self.sim_info.sim_buffer
            self.exportAfterSample[diagName] = self.sim_info.sim_chunk_size 
            diag.prepare()

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

                if globalIdx >= self.exportAfterSample[diagName]:
                    self.toBeExported.append(diagName)
                    self.exportAfterSample[diagname] = util.nextDivisible(self.sim_info.plot_frequency*self.sim_info.sim_chunk_size, 
                                                                            self.exportAfterSample[diagname])

                self.samplesUntilSave[diagName] += diag.save_frequency - numSamples
                self.lastSaveIdx[diagName] += numSamples
                self.lastGlobalSaveIdx[diagName] += numSamples



class SignalDiagnostic():
    def __init__(self):
        pass

    def saveData(self, processor, sig, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd):
        pass

    def getOutput(self):
        pass

    def getRawData(self):
        pass

class StateDiagnostic():
    def __init__(self):
        pass

class InstantDiagnostic():
    def __init__(self):
        pass









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

class Diagnostic(ABC):
    def __init__(self,
                sim_info,
                save_frequency, 
                plot_begin=None, 
                plot_frequency=None
    ):
        self.save_at = np.arange(sim_info.tot_samples) * plot_frequency

        self.info = DiagnosticInfo(
            outputFunc,
            dsum.meanNearTimeIdx,
            self.plot_begin,
            self.plot_frequency,
        )
        

        self.plot_data = {
            "title" : "",
            "xlabel" : "",
            "ylabel" : "",
        }

    def prepare(self):
        pass

    @abstractmethod
    def saveData(self, processor, sig, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd):
        pass

    @abstractmethod
    def getOutput(self):
        pass
        
class DiagnosticOverTime(Diagnostic):
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

        self.plot_data["xlabel"] = "Samples"
        
    
        



class SignalDiagnostic_(DiagnosticOverTime):
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

    def saveDiagnostics(self, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd):
        pass

    def resetBuffers(self):
        self.bufferIdx += 1

    @abstractmethod
    def saveBlockData(self, idx):
        pass