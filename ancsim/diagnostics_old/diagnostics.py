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
import ancsim.diagnostics.core as diacore


class SoundfieldPower():
    """ """
    def __init__(self, sim_info, arrays, rectangle, averagingLen, saveIndices):
        self.averaginLen = averagingLen
        self.saveIndices = saveIndices
        
        self.arrays = ar.ArrayCollection()
        self.arrays.add_array(ar.RegionArray("measure_points", rectangle))
        for src in arrays.sources():
            self.arrays.add_array(deepcopy(src))

        #check for available sessions here, as a TON of IRs are needed
        self.arrays.set_default_path_type(sim_info.reverb)
        self.arrays.setupIR(sim_info)

        self.propFilt = {}
        for src in self.arrays.sources():
            self.propFilt[src.name] = fc.createFilter(ir = self.arrays.paths[src.name]["measure_points"])

        self.soundfield = np.zeros((self.arrays["measure_points"].num))

        self.maxPropLen = np.max([self.propFilt[src.name].irLen for src in self.arrays.sources()])
        self.globIndices = [(si-self.averaginLen-self.maxPropLen,si) for si in self.saveIndices]
        self._nextSoundfield()

        assert all(gi[0] >= 0 for gi in self.globIndices)
        assert all(self.globIndices[i][1] < self.globIndices[i+1][0] for i in range(len(self.globIndices)-1))
        assert np.allclose(rectangle.point_spacing[0], rectangle.point_spacing[1])

        self.info = DiagnosticInfo(dplot.soundfieldPlot, None, 0, 1)
        

    def prepare(self):
        pass

    def _nextSoundfield(self):
        self.numSamples = 0
        self.currentGlobIdxs = self.globIndices.pop(0)
        self.soundfield.fill(0)
        
        for src in self.arrays.sources():
            self.propFilt[src.name].buffer.fill(0)


    def saveData(self, processor, sig, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd):
        if self.currentGlobIdxs[1] >= globIdxStart and self.currentGlobIdxs[0] <= globIdxEnd:
            start = max(globIdxStart, self.currentGlobIdxs[0])
            end = min(globIdxEnd, self.currentGlobIdxs[1])
            numSamples = end - start
            chunkStart = chunkIdxStart + globIdxStart - start

            if self.numSamples < self.maxPropLen:
                numToRemove = min(self.maxPropLen - self.numSamples, numSamples)
                
            for src in self.arrays.sources():
                propSig = self.propFilt[src.name].process(sig[src.name][:,chunkStart:chunkStart+numSamples])
                propSig = propSig[:,numToRemove:]
                if propSig.shape[-1] > 0:
                    meanSfPower = np.mean(propSig**2, axis=-1)
                    self.soundfield = (self.soundfield + meanSfPower) / 2

            self.numSamples += numSamples

    def getOutput(self):
        return self.soundfield

class SignalPowerRatio(diacore.SignalDiagnostic):
    def __init__(
        self,
        sim_info, 
        numeratorName,
        denominatorName,
        **kwargs
    ):
        super().__init__(sim_info, **kwargs)
        #self.info.outputFunction.append(dplot.spectrumPlot)
        #self.info.summaryFunction = [self.info.summaryFunction, None]
        self.numeratorName = numeratorName
        self.denominatorName = denominatorName
        
        self.numPower = np.zeros(self.sim_info.tot_samples)
        self.denomPower = np.zeros(self.sim_info.tot_samples)
        self.numSmoother = fc.Filter_IntBuffer(ir=np.ones((self.smoothing_len))/self.smoothing_len, numIn=1)
        self.denomSmoother = fc.Filter_IntBuffer(ir=np.ones((self.smoothing_len))/self.smoothing_len, numIn=1)
        self.powerRatio = np.full((self.sim_info.tot_samples), np.nan)

        self.plot_data["title"] = "Power Ratio: " + self.numeratorName + " / " + self.denominatorName
        self.plot_data["ylabel"] = "Ratio (dB)"
        self.plot_data["samplerate"] = sim_info.samplerate
        self.plot_data["scaling"] = "dbpow"
        

    def saveData(self, processor, sig, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd):
        numPower = np.mean(sig[self.numeratorName][..., chunkIdxStart:chunkIdxEnd] ** 2, axis=0, keepdims=True)
        denomPower = np.mean(
            sig[self.denominatorName][..., chunkIdxStart:chunkIdxEnd] ** 2, axis=0, keepdims=True
        )

        numPowerSmooth = self.numSmoother.process(numPower)
        denomPowerSmooth = self.denomSmoother.process(denomPower)

        self.powerRatio[globIdxStart:globIdxEnd] = numPowerSmooth / denomPowerSmooth
        self.numPower[globIdxStart:globIdxEnd] = numPower
        self.denomPower[globIdxStart:globIdxEnd] = denomPower

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

    def getSummaryOutput(self, globEndIdx):
        num = self.numPower[globEndIdx-self.summaryMeanLength:globEndIdx]
        denom = self.denomPower[globEndIdx-self.summaryMeanLength:globEndIdx]
        numFilt = np.logical_not(np.isnan(num))
        denomFilt = np.logical_not(np.isnan(denom))
        assert np.allclose(numFilt, denomFilt)
        print(f"number of averaged samples for SignalPowerRatio is {num[numFilt].shape}")
        
        return 10*np.log10(np.mean(num[numFilt]) / np.mean(denom[numFilt]))



class SignalSpectrumRatio(diacore.SignalDiagnostic):
    def __init__(
        self,
        sim_info, 
        numeratorName,
        numeratorChannels,
        denominatorName,
        denominatorChannels,
        **kwargs
    ):
        super().__init__(sim_info, **kwargs)
        self.info.outputFunction = dplot.spectrumRatioPlot
        self.info.summaryFunction = None
        self.numeratorName = numeratorName
        self.denominatorName = denominatorName
        
        self.num = np.full((numeratorChannels, self.sim_info.tot_samples), np.nan)
        self.denom = np.full((denominatorChannels, self.sim_info.tot_samples), np.nan)

        self.plot_data["title"] = "Spectrum Ratio: " + self.numeratorName + " / " + self.denominatorName
        self.plot_data["ylabel"] = "Ratio (dB)"
        self.plot_data["samplerate"] = sim_info.samplerate
        self.plot_data["scaling"] = "dbpow"
        

    def saveData(self, processor, sig, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd):
        self.num[:, globIdxStart:globIdxEnd] = sig[self.numeratorName][..., chunkIdxStart:chunkIdxEnd]
        self.denom[:, globIdxStart:globIdxEnd] = sig[self.denominatorName][..., chunkIdxStart:chunkIdxEnd]

    def getOutput(self):
        return [self.num, self.denom]

    def getSummaryOutput(self, globEndIdx):
        return None


class SignalSpectrum(diacore.SignalDiagnostic):
    def __init__(
        self,
        sim_info, 
        signalName,
        signalChannels,
        **kwargs
    ):
        super().__init__(sim_info, **kwargs)
        self.info.outputFunction = dplot.spectrumPlot
        self.info.summaryFunction = None
        self.signalName = signalName
        
        self.signal = np.zeros((signalChannels, self.sim_info.tot_samples))

        self.plot_data["title"] = "Spectrum: " + self.signalName
        self.plot_data["ylabel"] = "Ratio (dB)"
        self.plot_data["samplerate"] = sim_info.samplerate
        self.plot_data["scaling"] = "dbpow"
        

    def saveData(self, processor, sig, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd):
        self.signal[:, globIdxStart:globIdxEnd] = sig[self.signalName][..., chunkIdxStart:chunkIdxEnd]

    def getOutput(self):
        return self.signal

    def getSummaryOutput(self, globEndIdx):
        return None


class StateNMSE(diacore.StateDiagnostic):
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
        self.plot_data["scaling"] = "dbpow"

    def saveData(self, processor, _sig, _unusedIdx1, _unusedIdx2, _unusedIdx3, globalIdx):
        self.nmse[globalIdx] = np.mean(np.abs(self.true_value - self.get_property(processor))**2) / self.true_power

    def getOutput(self):
        return self.nmse


class SignalPower(diacore.SignalDiagnostic):
    def __init__(self, 
                sim_info, 
                signal_name,
                **kwargs):
        super().__init__(sim_info, **kwargs)
        #self.info.summaryFunction = [self.info.summaryFunction, None]
        self.signal_name = signal_name
        self.signal_power_raw = np.full(self.sim_info.tot_samples, np.nan)
        self.signal_power = np.full(self.sim_info.tot_samples, np.nan)
        self.signal_smoother = fc.Filter_IntBuffer(ir=np.ones((self.smoothing_len))/self.smoothing_len, numIn=1)

        self.plot_data["title"] = "Signal power"
        self.plot_data["ylabel"] = "Power (dB)"
        self.plot_data["scaling"] = "dbpow"

        if self.save_raw_data:
            raise NotImplementedError
    
    def saveData(self, processor, sig, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd):
        #print(chunkIdxStart, chunkIdxEnd)
        self.signal_power_raw[globIdxStart:globIdxEnd] = np.mean(
            sig[self.signal_name][:, chunkIdxStart:chunkIdxEnd] ** 2, axis=0)

        self.signal_power[globIdxStart:globIdxEnd] = self.signal_smoother.process(self.signal_power_raw[None,globIdxStart:globIdxEnd])

    def getOutput(self):
        return self.signal_power

    def getSummaryOutput(self, globEndIdx):
        selected_chunk = self.signal_power_raw[globEndIdx-self.summaryMeanLength:globEndIdx]
        filterArray = np.logical_not(np.isnan(selected_chunk))
        return 10*np.log10(np.mean(selected_chunk[filterArray]))
        #return [10*np.log10(np.mean(selected_chunk[filterArray])), None]
        #filterArray = np.logical_not(np.isnan(val))

        #return np.mean()



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
#         self.signal_smoother = fc.Filter_IntBuffer(ir=np.ones((smoothing_len))/smoothing_len, numIn=1)

#         self.plot_data["title"] = "Power"
#         self.plot_data["ylabel"] = "Power (dB)"
    
#     def saveData(self, sig, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd):
#         signal_power = np.mean(sig[self.signal_name][:, chunkIdxStart:chunkIdxEnd] ** 2, axis=0, keepdims=True)
#         self.signal_power[globIdxStart:globIdxEnd] = util.pow2db(self.signal_smoother.process(signal_power))

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
#         self.pathFilter = fc.FilterSum(acousticPath)

#         self.totalNoise = np.zeros((numPoints, self.sim_chunk_size + self.sim_buffer))
#         self.primaryNoise = np.zeros((numPoints, self.sim_chunk_size + self.sim_buffer))

#         self.totalNoisePower = np.zeros(tot_samples)
#         self.primaryNoisePower = np.zeros(tot_samples)

#         self.totalNoiseSmoother = fc.Filter_IntBuffer(ir=np.ones((smoothing_len)), numIn=1)
#         self.primaryNoiseSmoother = fc.Filter_IntBuffer(
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

#     def saveDiagnostics(self, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd, y):
#         secondaryNoise = self.pathFilter.process(y[:, chunkIdxStart:chunkIdxEnd])
#         self.totalNoise[:, chunkIdxStart:chunkIdxEnd] = (
#             self.primaryNoise[:, chunkIdxStart:chunkIdxEnd] + secondaryNoise
#         )

#         totalNoisePower = np.mean(
#             self.totalNoise[:, chunkIdxStart:chunkIdxEnd] ** 2, axis=0, keepdims=True
#         )
#         primaryNoisePower = np.mean(
#             self.primaryNoise[:, chunkIdxStart:chunkIdxEnd] ** 2, axis=0, keepdims=True
#         )

#         self.totalNoisePower[globIdxStart:globIdxEnd] = totalNoisePower
#         self.primaryNoisePower[globIdxStart:globIdxEnd] = primaryNoisePower

#         smoothTotalNoisePower = self.totalNoiseSmoother.process(totalNoisePower)
#         smoothPrimaryNoisePower = self.primaryNoiseSmoother.process(primaryNoisePower)

#         self.noiseReduction[globIdxStart:globIdxEnd] = util.pow2db(
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
#         self.totalNoiseSmoother = fc.Filter_IntBuffer(ir=np.ones((smoothing_len)), numIn=1)
#         self.primaryNoiseSmoother = fc.Filter_IntBuffer(
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

#     def saveDiagnostics(self, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd, e):
#         totalNoisePower = np.mean(e[:, chunkIdxStart:chunkIdxEnd] ** 2, axis=0, keepdims=True)
#         primaryNoisePower = np.mean(
#             self.primaryNoise[:, chunkIdxStart:chunkIdxEnd] ** 2, axis=0, keepdims=True
#         )

#         totalNoisePowerSmooth = self.totalNoiseSmoother.process(totalNoisePower)
#         primaryNoisePowerSmooth = self.primaryNoiseSmoother.process(primaryNoisePower)

#         self.noiseReduction[globIdxStart:globIdxEnd] = util.pow2db(
#             totalNoisePowerSmooth / primaryNoisePowerSmooth
#         )
#         self.totalNoisePower[globIdxStart:globIdxEnd] = totalNoisePower
#         self.primaryNoisePower[globIdxStart:globIdxEnd] = primaryNoisePower

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
        self.pathFilter = fc.FilterSum(acousticPath)

    def getOutput(self):
        return self.totalNoise

    def resetBuffers(self):
        pass

    def saveDiagnostics(self, chunkIdx, y):
        if (
            chunkIdx >= self.info.plot_begin - 1
            and (chunkIdx + 1) % self.info.plot_frequency == 0
        ):
            chunkIdxStart = self.sim_buffer
            yFiltered = self.pathFilter.process(
                y[
                    :,
                    chunkIdxStart - self.pathFilter.irLen + 1 : chunkIdxStart + self.samplesForSF,
                ]
            )
            self.totalNoise[:, :] = (
                self.primaryNoise[:, chunkIdxStart : chunkIdxStart + self.samplesForSF]
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
        self.pathFilter = fc.FilterSum(acousticPath)

    def getOutput(self):
        return self.totalNoise

    def resetBuffers(self):
        pass

    def saveDiagnostics(self, chunkIdx, y):
        if (
            chunkIdx >= self.info.plot_begin - 1
            and (chunkIdx + 1) % self.info.plot_frequency == 0
        ):
            chunkIdxStart = self.sim_buffer
            yFiltered = self.pathFilter.process(
                y[
                    :,
                    chunkIdxStart - self.pathFilter.irLen + 1 : chunkIdxStart + self.samplesForSF,
                ]
            )
            self.totalNoise[:, :] = (
                self.primaryNoise[:, chunkIdxStart : chunkIdxStart + self.samplesForSF]
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







class ConstantEstimateNMSE(diacore.PerBlockDiagnostic):
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

class ConstantEstimateNMSEAllCombinations(diacore.PerBlockDiagnostic):
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
        num_channels = np.math.factorial(self.true_value.shape[1]) * \
                        np.math.factorial(self.true_value.shape[2])
        self.dataBuffer = np.full((num_channels, tot_samples), np.nan)

        
        

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

class ConstantEstimateNMSESelectedFrequencies(diacore.PerBlockDiagnostic):
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



class ConstantEstimatePhaseDifferenceSelectedFrequencies(diacore.PerBlockDiagnostic):
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



class ConstantEstimateWeightedPhaseDifference(diacore.PerBlockDiagnostic):
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

class ConstantEstimateAmpDifference(diacore.PerBlockDiagnostic):
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


class ConstantEstimatePhaseDifference(diacore.PerBlockDiagnostic):
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
        """Assumes the signal to be estimated is of the form (num_channels, numSamples)"""
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
        self.true_valuePowerSmoother = fc.Filter_IntBuffer(
            ir=np.ones((smoothing_len)) / smoothing_len
        )

    def getOutput(self):
        return self.recordedValues

    def saveDiagnostics(self, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd):
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
        """Assumes the signal to be estimated is of the form (num_channels, numSamples)"""
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

    def saveDiagnostics(self, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd):
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
            self.pathFilter = fc.FilterSum(acousticPath)
        else:
            self.pathFilter = fc.FilterSum(np.ones((1, 1, 1)))

    def getOutput(self):
        return self.noisePower

    def saveDiagnostics(self, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd, e):
        totalNoisePower = np.mean(e[:, chunkIdxStart:chunkIdxEnd] ** 2, axis=0, keepdims=True)

        self.primaryNoisePower[globIdxStart:globIdxEnd] = primaryNoisePower

    def resetBuffers(self):
        pass

    def saveBlockData(self, idx, newPrimaryNoise):
        numSamples = newPrimaryNoise.shape[-1]
        self.primaryNoise[:, idx : idx + numSamples] = newPrimaryNoise


class RecordAudio:
    def __init__(self, acousticPath, samplerate, maxLength, tot_samples, sim_buffer,
                 sim_chunk_size, plot_begin=0, plot_frequency=1):
        self.pathFilter = fc.FilterSum(ir=acousticPath)
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
    
    def saveDiagnostics(self, chunkIdxStart, chunkIdxEnd, globIdxStart, globIdxEnd, y):
        secondaryNoise = self.pathFilter.process(y[:, chunkIdxStart:chunkIdxEnd])
        self.withANC[:, globIdxStart:globIdxEnd] = self.primaryNoise[:,chunkIdxStart:chunkIdxEnd] + secondaryNoise
        self.withoutANC[:,globIdxStart:globIdxEnd] = self.primaryNoise[:,chunkIdxStart:chunkIdxEnd]

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