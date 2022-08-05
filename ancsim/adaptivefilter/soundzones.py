from email.mime import audio
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




def diag_preset_power_of_signals(processor):
    """Must be called from processor.prepare(), not
        processor.__init__()
    """
    for sig_name in processor.sig.keys():
        processor.diag.add_diagnostic(f"power_{sig_name}", 
                dia.SignalPower(sig_name, processor.sim_info, processor.blockSize, 
                preprocess=[[dplot.smooth(processor.sim_info.output_smoothing), dplot.db_power]]))
    



class SoundzoneFIR(base.AudioProcessor):
    def __init__(self, sim_info, arrays, blockSize, audio_sources, control_filter_ir, **kwargs):
        super().__init__(sim_info, arrays, blockSize, **kwargs)
        self.name = "Soundzone FIR processor"
        self.audio_sources = audio_sources
        self.num_zones = len(audio_sources)
        assert len(control_filter_ir) == self.num_zones
        assert all([f"mic_{z}" in self.arrays for z in range(self.num_zones)])
        assert all([f"ls_{z}" in self.arrays for z in range(self.num_zones)])
        assert self.sim_info.save_source_contributions is True
        for k in range(self.num_zones):
            for i in range(1, self.num_zones):
                assert np.allclose(self.arrays.paths[f"ls_0"][f"mic_{k}"], self.arrays.paths[f"ls_{i}"][f"mic_{k}"])

        self.createNewBuffer(f"ls_tot", self.arrays[f"ls_{0}"].num)
        for k in range(self.num_zones):
            self.createNewBuffer(f"audio_sig_{k}", 1)
            self.createNewBuffer(f"interference_{k}", self.arrays[f"mic_{k}"].num)
            
            num_other_mics = int(np.sum([self.arrays[f"mic_{i}"].num for i in range(self.num_zones) if i != k]))
            self.createNewBuffer(f"bleed_ls_{k}", num_other_mics)
            #self.createNewBuffer(f"error_desired", self.arrays["mic"].num)
            #self.createNewBuffer(f"desired", self.arrays["mic"].num)

        self.control_filters = [fc.createFilter(ir) for ir in control_filter_ir]

        #diag_preset_power_of_signals(self)
        self.diag.add_diagnostic(f"power_ls_tot", dia.SignalPower(f"ls_tot", self.sim_info, self.blockSize, preprocess=[[dplot.smooth(self.sim_info.output_smoothing), dplot.db_power, dplot.clip(-5, 10)]]))
        for k in range(self.num_zones):
            self.diag.add_diagnostic(f"power_audio_signal_{k}", dia.SignalPower(f"audio_sig_{k}", self.sim_info, self.blockSize, preprocess=[[dplot.smooth(self.sim_info.output_smoothing), dplot.db_power]]))
            self.diag.add_diagnostic(f"power_ls_{k}", dia.SignalPower(f"ls_{k}", self.sim_info, self.blockSize, preprocess=[[dplot.smooth(self.sim_info.output_smoothing), dplot.db_power]]))

            self.diag.add_diagnostic(f"power_desired_sig_{k}", dia.SignalPower(f"ls_{k}~mic_{k}", self.sim_info, self.blockSize, preprocess=[[dplot.smooth(self.sim_info.output_smoothing)]]))
            self.diag.add_diagnostic(f"power_interference_{k}", dia.SignalPower(f"interference_{k}", self.sim_info, self.blockSize, preprocess=[[dplot.smooth(self.sim_info.output_smoothing)]]))
            self.diag.add_diagnostic(f"power_bleed_ls_{k}", dia.SignalPower(f"bleed_ls_{k}", self.sim_info, self.blockSize, preprocess=[[dplot.smooth(self.sim_info.output_smoothing)]]))
            self.diag.add_diagnostic(f"power_mic_{k}", dia.SignalPower(f"mic_{k}", self.sim_info, self.blockSize, preprocess=[[dplot.smooth(self.sim_info.output_smoothing)]]))
            self.diag.add_diagnostic(f"power_noise_src_{k}", dia.SignalPower(f"noise_src_{k}", self.sim_info, self.blockSize, preprocess=[[dplot.smooth(self.sim_info.output_smoothing)]]))
            self.diag.add_diagnostic(f"power_noise_at_mic_{k}", dia.SignalPower(f"noise_src_{k}~mic_{k}", self.sim_info, self.blockSize, preprocess=[[dplot.smooth(self.sim_info.output_smoothing)]]))

            self.diag.add_diagnostic(f"sinr_{k}", dia.SignalPowerRatio(f"ls_{k}~mic_{k}", f"interference_{k}", self.sim_info, self.blockSize))
            self.diag.add_diagnostic(f"acoustic_contrast_{k}", dia.SignalPowerRatio(f"ls_{k}~mic_{k}", f"bleed_ls_{k}", self.sim_info, self.blockSize, preprocess=dplot.db_power))

            self.diag.add_diagnostic(f"rir_{k}", dia.RecordFilter(f"arrays.paths['ls_{k}']['mic_{k}']", self.sim_info, self.blockSize, save_at=[1]))
            self.diag.add_diagnostic(f"ir_control_filter_{k}", dia.RecordFilter(f"control_filters[{k}].ir", self.sim_info, self.blockSize, save_at=[1]))

        #self.diag.add_diagnostic("acoustic_contrast_mic", dia.SignalPowerRatio(self.sim_info, "mic", "mic", blockSize=self.blockSize, numerator_channels=arrays["mic"].group_idxs[0], denom_channels=arrays["mic"].group_idxs[1]))
        #self.diag.add_diagnostic("bright_mic_error", dia.SignalPowerRatio(self.sim_info, "error_bright", "desired", blockSize=self.blockSize))

        #desired_dly = self.ctrlFiltLen // 2
        #self.desiredFilter = fc.createFilter(ir=np.pad(self.arrays.paths["virtual_source"]["mic_bright"], 
        #                                        ((0,0),(0,0),(desired_dly,0))), 
        #                                        broadcastDim=1, 
        #                                        sumOverInput=False)

        #delay_desired = 55 // 2
        #desiredPath = np.sum(np.pad(self.pathsBright,((0,0),(0,0),(delay_desired,0))),axis=0, keepdims=True)
        self.metadata["source type"] = [asrc.__class__.__name__ for asrc in self.audio_sources]
        self.metadata["source"] = [asrc.metadata for asrc in self.audio_sources]
        self.metadata["num zones"] = self.num_zones


    def process(self, num_samples):
        assert num_samples == self.blockSize

        for k in range(self.num_zones):
            audio_sig = self.audio_sources[k].get_samples(num_samples)
            self.sig[f"audio_sig_{k}"][:,self.idx-num_samples:self.idx] = audio_sig
            self.sig[f"ls_{k}"][:,self.idx:self.idx+num_samples] = self.control_filters[k].process(audio_sig)

        self.calc_diagnostic_signals(num_samples)
            

    def calc_diagnostic_signals(self, num_samples):
        for k in range(self.num_zones):
            self.sig[f"ls_tot"][:,self.idx-num_samples:self.idx] += self.sig[f"ls_{k}"][:,self.idx-num_samples:self.idx]

            self.sig[f"interference_{k}"][:,self.idx-num_samples:self.idx] = self.sig[f"noise_src_{k}~mic_{k}"][:,self.idx-num_samples:self.idx]
            for i in range(self.num_zones):
                if i != k:
                    self.sig[f"interference_{k}"][:,self.idx-num_samples:self.idx] += self.sig[f"ls_{i}~mic_{k}"][:,self.idx-num_samples:self.idx]

            channel_idx = 0
            for i in range(self.num_zones):
                if i != k:
                    num_channels = self.arrays[f"mic_{i}"].num
                    self.sig[f"bleed_ls_{k}"][channel_idx:channel_idx+num_channels,self.idx-num_samples:self.idx] += \
                                                    self.sig[f"ls_{k}~mic_{i}"][:,self.idx-num_samples:self.idx]
                    channel_idx += num_channels

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

