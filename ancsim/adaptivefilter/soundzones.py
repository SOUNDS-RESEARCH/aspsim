import numpy as np
import scipy.linalg as linalg
import scipy.signal as spsig
from abc import abstractmethod
import ancsim.utilities as util

import matplotlib.pyplot as plt

import ancsim.adaptivefilter.base as base
import ancsim.signal.filterclasses as fc
import ancsim.diagnostics.core as diacore
import ancsim.diagnostics.diagnostics as dia
import ancsim.diagnostics.diagnosticplots as dplot
import ancsim.signal.filterdesign as fd






class SoundzoneFIR(base.AudioProcessor):
    def __init__(self, sim_info, arrays, blockSize, src, ctrlFiltLen, **kwargs):
        super().__init__(sim_info, arrays, blockSize, **kwargs)
        self.name = "Soundzone processor"
        self.src = src
        self.ctrlFiltLen = ctrlFiltLen

        assert self.sim_info.save_source_contributions

        self.createNewBuffer("orig_sig", self.src.num_channels)
        self.createNewBuffer("error_desired", self.arrays["mic"].num)
        self.createNewBuffer("desired", self.arrays["mic"].num)

        self.diag.add_diagnostic("power_mic", dia.SignalPower("mic", self.sim_info, self.blockSize))
        self.diag.add_diagnostic("power_ls", dia.SignalPower("ls", self.sim_info, self.blockSize))
        #self.diag.add_diagnostic("dark_mic_power", dia.SignalPower(self.sim_info, "mic_dark", blockSize=self.blockSize))
        
        #self.diag.add_diagnostic("acoustic_contrast_mic", dia.SignalPowerRatio(self.sim_info, "mic", "mic", blockSize=self.blockSize, numerator_channels=arrays["mic"].group_idxs[0], denom_channels=arrays["mic"].group_idxs[1]))
        self.diag.add_diagnostic("bright_mic_error", dia.SignalPowerRatio(self.sim_info, "error_bright", "desired", blockSize=self.blockSize))

        desired_dly = self.ctrlFiltLen // 2
        self.desiredFilter = fc.createFilter(ir=np.pad(self.arrays.paths["virtual_source"]["mic_bright"], 
                                                ((0,0),(0,0),(desired_dly,0))), 
                                                broadcastDim=1, 
                                                sumOverInput=False)

        if "zone" in self.arrays:
            raise NotImplementedError
            self.createNewBuffer("reg_error_bright", self.arrays["zone_bright"].num)
            self.createNewBuffer("reg_desired", self.arrays["zone_bright"].num)
            self.desiredFilterSpatial = fc.createFilter(ir=np.pad(self.arrays.paths["virtual_source"]["zone_bright"],
                                                            ((0,0),(0,0),(desired_dly,0))), 
                                                            broadcastDim=1, sumOverInput=False)
            self.diag.add_diagnostic("bright_zone_power", dia.SignalPower(self.sim_info, "zone_bright", blockSize=self.blockSize))
            self.diag.add_diagnostic("bright_zone_error", dia.SignalPowerRatio(self.sim_info, "reg_error_bright", "reg_desired", blockSize=self.blockSize))
            self.diag.add_diagnostic("bright_zone_error_spec", dia.SignalSpectrumRatio(self.sim_info, "reg_error_bright", self.sig["reg_error_bright"].shape[0], 
                                                                                                        "reg_desired", self.sig["reg_desired"].shape[0], 
                                                                                                        blockSize=self.blockSize))
            if "zone_dark" in self.arrays:
                self.diag.add_diagnostic("acoustic_contrast_spatial", dia.SignalPowerRatio(self.sim_info, "zone_bright", "zone_dark", blockSize=self.blockSize))
                self.diag.add_diagnostic("acoustic_contrast_spatial_spec", dia.SignalSpectrumRatio(self.sim_info, "zone_bright", self.arrays["zone_bright"].num, 
                                                                                                                    "zone_dark", self.arrays["zone_dark"].num, 
                                                                                                                    blockSize=self.blockSize))
                self.diag.add_diagnostic("dark_zone_power", dia.SignalPower(self.sim_info, "zone_dark", blockSize=self.blockSize))
                
        #delay_desired = 55 // 2
        #desiredPath = np.sum(np.pad(self.pathsBright,((0,0),(0,0),(delay_desired,0))),axis=0, keepdims=True)
        self.metadata["source type"] = self.src.__class__.__name__
        self.metadata["source"] = self.src.metadata
        self.metadata["control filter length"] = self.ctrlFiltLen

    def process(self, numSamples):
        x = self.src.get_samples(self.blockSize)
        self.sig["orig_sig"][:,self.idx:self.idx+self.blockSize] = x
        self.sig["ls"][:,self.idx:self.idx+self.blockSize] = self.controlFilter.process(x)

        self.sig["desired"][:,self.idx:self.idx+self.blockSize] = self.desiredFilter.process(x)#np.squeeze(self.desiredFilter.process(x), axis=-2)[0,...]
        self.sig["error_desired"][:,self.idx-self.blockSize:self.idx] = self.sig["mic"][:,self.idx-self.blockSize:self.idx] - \
                                                                        self.sig["desired"][:,self.idx-self.blockSize:self.idx]

        if "zone" in self.arrays:
            raise NotImplementedError
            self.sig["reg_desired"][:,self.idx:self.idx+self.blockSize] = np.squeeze(self.desiredFilterSpatial.process(x), axis=-2)[0,...]
            self.sig["reg_error_bright"][:,self.idx-self.blockSize:self.idx] = self.sig["zone_bright"][:,self.idx-self.blockSize:self.idx] - \
                                                                                self.sig["reg_desired"][:,self.idx-self.blockSize:self.idx]
            













class SoundzoneFIR_superposition(base.AudioProcessor):
    def __init__(self, sim_info, arrays, blockSize, src, ctrlFiltLen, **kwargs):
        super().__init__(sim_info, arrays, blockSize, **kwargs)
        self.name = "Abstract Soundzone Processor"
        self.src = src
        self.ctrlFiltLen = ctrlFiltLen

        assert self.src.num_channels == 1
        
        self.pathsBright = self.arrays.paths["ls"]["mic_bright"]
        self.pathsDark = self.arrays.paths["ls"]["mic_dark"]
        self.numls = self.arrays["ls"].num
        self.numMicBright = self.arrays["mic_bright"].num
        self.numMicDark = self.arrays["mic_dark"].num

        self.createNewBuffer("orig_sig", 1)
        self.createNewBuffer("error_bright", self.numMicBright)
        self.createNewBuffer("desired", self.numMicBright)

        self.diag.add_diagnostic("mic_bright_power", dia.SignalPower("mic_bright", self.sim_info, self.blockSize, preprocess=[[dplot.smooth(self.sim_info.output_smoothing), dplot.db_power]]))
        self.diag.add_diagnostic("mic_dark_power", dia.SignalPower("mic_dark", self.sim_info, self.blockSize, preprocess=[[dplot.smooth(self.sim_info.output_smoothing), dplot.db_power]]))
        self.diag.add_diagnostic("ls_power", dia.SignalPower("ls", self.sim_info, self.blockSize, preprocess=[[dplot.smooth(self.sim_info.output_smoothing), dplot.db_power]]))
        self.diag.add_diagnostic("acoustic_contrast_mic", dia.SignalPowerRatio("mic_bright", "mic_dark", self.sim_info, self.blockSize))
        self.diag.add_diagnostic("bright_mic_error", dia.SignalPowerRatio("error_bright", "desired", self.sim_info, self.blockSize, preprocess=dplot.db_power))

        desired_dly = self.ctrlFiltLen // 2
        self.desiredFilter = fc.createFilter(ir=np.pad(self.arrays.paths["virtual_source"]["mic_bright"], 
                                                ((0,0),(0,0),(desired_dly,0))), 
                                                broadcastDim=1, 
                                                sumOverInput=False)

        if "zone_bright" in self.arrays:
            self.createNewBuffer("reg_error_bright", self.arrays["zone_bright"].num)
            self.createNewBuffer("reg_desired", self.arrays["zone_bright"].num)
            self.desiredFilterSpatial = fc.createFilter(ir=np.pad(self.arrays.paths["virtual_source"]["zone_bright"],
                                                            ((0,0),(0,0),(desired_dly,0))), 
                                                            broadcastDim=1, sumOverInput=False)
            self.diag.add_diagnostic("bright_zone_power", dia.SignalPower(self.sim_info, "zone_bright", blockSize=self.blockSize))
            self.diag.add_diagnostic("bright_zone_error", dia.SignalPowerRatio(self.sim_info, "reg_error_bright", "reg_desired", blockSize=self.blockSize))
            self.diag.add_diagnostic("bright_zone_error_spec", dia.SignalSpectrumRatio(self.sim_info, "reg_error_bright", self.sig["reg_error_bright"].shape[0], 
                                                                                                        "reg_desired", self.sig["reg_desired"].shape[0], 
                                                                                                        blockSize=self.blockSize))
            if "zone_dark" in self.arrays:
                self.diag.add_diagnostic("acoustic_contrast_spatial", dia.SignalPowerRatio(self.sim_info, "zone_bright", "zone_dark", blockSize=self.blockSize))
                self.diag.add_diagnostic("acoustic_contrast_spatial_spec", dia.SignalSpectrumRatio(self.sim_info, "zone_bright", self.arrays["zone_bright"].num, 
                                                                                                                    "zone_dark", self.arrays["zone_dark"].num, 
                                                                                                                    blockSize=self.blockSize))
                self.diag.add_diagnostic("dark_zone_power", dia.SignalPower(self.sim_info, "zone_dark", blockSize=self.blockSize))
                
        #delay_desired = 55 // 2
        #desiredPath = np.sum(np.pad(self.pathsBright,((0,0),(0,0),(delay_desired,0))),axis=0, keepdims=True)
        self.metadata["source type"] = self.src.__class__.__name__
        self.metadata["source"] = self.src.metadata
        self.metadata["control filter length"] = self.ctrlFiltLen

    def process(self, numSamples):
        x = self.src.get_samples(self.blockSize)
        self.sig["orig_sig"][:,self.idx:self.idx+self.blockSize] = x
        self.sig["ls"][:,self.idx:self.idx+self.blockSize] = self.controlFilter.process(x)

        self.sig["desired"][:,self.idx:self.idx+self.blockSize] = np.squeeze(self.desiredFilter.process(x), axis=-2)[0,...]
        self.sig["error_bright"][:,self.idx-self.blockSize:self.idx] = self.sig["mic_bright"][:,self.idx-self.blockSize:self.idx] - \
                                                                        self.sig["desired"][:,self.idx-self.blockSize:self.idx]

        if "zone_bright" in self.arrays:
            self.sig["reg_desired"][:,self.idx:self.idx+self.blockSize] = np.squeeze(self.desiredFilterSpatial.process(x), axis=-2)[0,...]
            self.sig["reg_error_bright"][:,self.idx-self.blockSize:self.idx] = self.sig["zone_bright"][:,self.idx-self.blockSize:self.idx] - \
                                                                                self.sig["reg_desired"][:,self.idx-self.blockSize:self.idx]
            



class SoundzoneFixedFIR(SoundzoneFIR_superposition):
    def __init__(self, sim_info, arrays, blockSize, src, controlFilter, **kwargs):
        self.controlFilter = fc.createFilter(ir=controlFilter)
        super().__init__ (sim_info, arrays, blockSize, src, self.controlFilter.irLen, **kwargs)
        self.name = "Fixed FIR sound zones"

        assert self.controlFilter.numIn == 1
        assert self.controlFilter.numOut == self.numls

