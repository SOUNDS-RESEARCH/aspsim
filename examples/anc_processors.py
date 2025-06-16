import numpy as np
from abc import ABC, abstractmethod


import aspcore.filter as fc
import aspcore.fouriertransform as ft
import aspcore.utilities as utils

import aspsim.diagnostics.diagnostics as dia
import aspsim.diagnostics.preprocessing as pp
import aspsim.processor as afbase



def add_anc_diagnostics(sim):
    sim.diag.add_diagnostic("noise_reduction", dia.SignalPowerRatio("error", "noise~error", sim.sim_info, preprocess = pp.db_power))
    sim.diag.add_diagnostic("power_ls", dia.SignalPower("loudspeaker", sim.sim_info, preprocess = [[pp.smooth(sim.sim_info.output_smoothing), pp.db_power]]))
    sim.diag.add_diagnostic("power_error", dia.SignalPower("error", sim.sim_info, preprocess = [[pp.smooth(sim.sim_info.output_smoothing), pp.db_power]]))
    sim.diag.add_diagnostic("power_primary_noise", dia.SignalPower("noise~error", sim.sim_info, preprocess = [[pp.smooth(sim.sim_info.output_smoothing), pp.db_power]]))
    sim.diag.add_diagnostic("power_secondary_noise", dia.SignalPower("loudspeaker~error", sim.sim_info, preprocess = [[pp.smooth(sim.sim_info.output_smoothing), pp.db_power]]))
    sim.diag.add_diagnostic("power_ref", dia.SignalPower("ref", sim.sim_info, preprocess = [[pp.smooth(sim.sim_info.output_smoothing), pp.db_power]]))

    sim.diag.add_diagnostic("sig_ls", dia.RecordSignal("loudspeaker", sim.sim_info, num_channels = sim.arrays["loudspeaker"].num))
    sim.diag.add_diagnostic("sig_error", dia.RecordSignal("error", sim.sim_info, num_channels = sim.arrays["error"].num))
    sim.diag.add_diagnostic("sig_ref", dia.RecordSignal("ref", sim.sim_info, num_channels = sim.arrays["ref"].num))
    sim.diag.add_diagnostic("sig_primary_noise", dia.RecordSignal("noise~error", sim.sim_info, num_channels = sim.arrays["error"].num))
    sim.diag.add_diagnostic("sig_secondary_noise", dia.RecordSignal("loudspeaker~error", sim.sim_info, num_channels = sim.arrays["error"].num))

    sim.diag.add_diagnostic("secondary_paths", dia.RecordFilter("arrays.paths['loudspeaker']['error']", sim.sim_info))

    if "eval" in sim.arrays:
        sim.diag.add_diagnostic("spatial_noise_reduction", dia.SignalPowerRatio("eval", "noise~eval", sim.sim_info, preprocess = pp.db_power))


class ActiveNoiseControlProcessor(afbase.AudioProcessor):
    def __init__(self, sim_info, arrays, block_size, update_block_size, **kwargs):
        super().__init__(sim_info, arrays, block_size, **kwargs)
        self.update_block_size = update_block_size
        self.num_ref = self.arrays["ref"].num
        self.num_error = self.arrays["error"].num
        self.num_speaker = self.arrays["loudspeaker"].num

        assert update_block_size % block_size == 0, "Update block size must be a multiple of block size"
        assert update_block_size >= block_size, "Block size must be smaller or equal to update block size"

        self.counter = utils.EventCounter({"update_filter" : (update_block_size, update_block_size)})
        self.metadata["update block size"] = update_block_size

    def process(self, num_samples):
        self.gen_speaker_signals(num_samples)
        if "update_filter" in self.counter.event:
            self.update_filter()

        for i in range(self.block_size):
            self.counter.progress()

    @abstractmethod
    def gen_speaker_signals(self, num_samples):
        pass

    @abstractmethod
    def update_filter(self):
        pass



class TDANCProcessor(ActiveNoiseControlProcessor):
    """Base class for an ANC processor with a time-domain control filter. 
    
    This class implements the basic structure of an ANC processor, including the generation of speaker signals
    and the update of the control filter. It is designed to be subclassed to implement the adaptation algorithm
    for the control filter, such as FxLMS or FastBlockFxLMS.
    """
    def __init__(self, sim_info, arrays, block_size, update_block_size, control_filt_len, **kwargs):
        super().__init__(sim_info, arrays, block_size, update_block_size, **kwargs)
        self.filt_len = control_filt_len
        self.control_filter = fc.create_filter(num_in=self.num_ref, num_out=self.num_speaker, ir_len=self.filt_len)

        self.metadata["control filter length"] = control_filt_len

    def gen_speaker_signals(self, num_samples):
        """"""
        self.sig["loudspeaker"][:,self.sig.idx:self.sig.idx+num_samples] = \
            self.control_filter.process(self.sig["ref"][:,self.sig.idx-num_samples:self.sig.idx])


class FxLMS(TDANCProcessor):
    """Filtered-x least men squares adaptive filter

    Parameters
    ----------
    block_size : int
        The number of samples to process in each block, meaning the number of samples of the loudspeaker signal
        that is generated each block. A higher block size increases the latency of the system, so will generally
        decrease the performance of the ANC system. Set it to 1 unless you have a good reason to use a larger block size.

    normalization : str, optional, one of {'sample', 'block', None}
    if normalization is selected as 'block', then  this is exactly equivalent to FastBlockFxLMS
    However, 'block' can be unstable for very small block sizes (for example in sample-by-sample processing)
    in which case 'sample' normalization will work much better
    
    
    """
    def __init__(self, sim_info, arrays, block_size, update_block_size, control_filt_len, mu, beta, sec_path_ir, normalization="sample", **kwargs):
        super().__init__(sim_info, arrays, block_size, update_block_size, control_filt_len, **kwargs)
        self.name = "Block FxLMS"
        self.mu = mu
        self.beta = beta
        self.normalization = normalization
        assert self.normalization in ("sample", "block", None)

        sec_path_ir = np.concatenate((np.zeros((sec_path_ir.shape[0], sec_path_ir.shape[1], self.block_size)), sec_path_ir),axis=-1)
        self.sec_path_filt = fc.create_filter(ir=sec_path_ir, broadcast_dim=self.num_ref, sum_over_input=False)
        self.sig.create_signal("xf", (self.num_ref, self.num_speaker, self.num_error))
        self.metadata["step size"] = self.mu
        self.metadata["beta"] = self.beta

    def prepare(self):
        super().prepare()
        self.sig["xf"][..., : self.sig.idx] = np.transpose(
            self.sec_path_filt.process(self.sig["ref"][:, :self.sig.idx]), (2, 0, 1, 3)
        )

    def update_filter(self):
        start_idx = self.sig.idx - self.update_block_size
        end_idx = self.sig.idx

        self.sig["xf"][..., start_idx : end_idx] = np.transpose(
            self.sec_path_filt.process(self.sig["ref"][:, start_idx : end_idx]),
            (2, 0, 1, 3),
        )

        grad = np.zeros_like(self.control_filter.ir)
        for n in range(start_idx, end_idx):
            Xf = np.flip(
                self.sig["xf"][..., n - self.filt_len + 1 : n + 1], axis=-1
            )
            grad_contribution = np.sum(Xf * self.sig["error"][None, None, :, n:n+1], axis=2)
            if self.normalization == "sample":
                norm = 1 / (np.sum(Xf**2) + self.beta)
                grad_contribution *= norm
            grad += grad_contribution


        if self.normalization == "block":
            norm = 1 / (
            np.sum(self.sig["xf"][:, :, :, start_idx : end_idx] ** 2)
            + self.beta
            )
            grad *= norm

        self.control_filter.ir -= grad * self.mu




class FastBlockFxLMS(TDANCProcessor):
    """Fully block based operation using overlap save and FFT. 
        updateBlockSize is equal to the control filter length"""
    def __init__(self, sim_info, arrays, block_size, control_filt_len, mu, beta, sec_path, **kwargs):
        super().__init__(sim_info, arrays, block_size, control_filt_len, control_filt_len, **kwargs)
        self.name = "Fast Block FxLMS"
        self.mu = mu
        self.beta = beta

        #add thee block size delay to the secondary path
        sec_path_ir = np.concatenate((np.zeros((*sec_path.shape[:2], self.block_size)), sec_path), axis=-1)
        if sec_path_ir.shape[-1] < 2 * self.filt_len:
            sec_path_ir = np.concatenate((sec_path_ir, np.zeros((*sec_path_ir.shape[:2], 2 * self.filt_len - sec_path_ir.shape[-1]))), axis=-1)
        elif sec_path_ir.shape[-1] > 2 * self.filt_len:
            sec_path_ir = sec_path_ir[..., :2*self.filt_len]



                                    #np.zeros((sec_path.shape[0], sec_path.shape[1], 2*self.filt_len-sec_path.shape[-1]-self.block_size))),
                                    #axis=-1)
        sec_path_ir[...,self.filt_len:] = 0
        sec_path_tf = ft.fft(sec_path_ir)

        self.sec_path_filt = fc.FilterBroadcastFreq(data_dims=self.num_ref, tf=sec_path_tf)
        self.sig.create_signal("xf", (self.num_ref, self.num_speaker, self.num_error))

        self.metadata["step size"] = self.mu
        self.metadata["beta"] = self.beta

    def prepare(self):
        super().prepare()
        for start_idx in utils.block_process_idxs(self.sig.idx, self.filt_len, 0, 0):
            end_idx = start_idx + self.filt_len
            self.sig["xf"][:, :, :, start_idx:end_idx] = np.transpose(
                self.sec_path_filt.process(self.sig["ref"][:, start_idx:end_idx]), (2, 0, 1, 3)
            )

    def update_filter(self):
        start_idx = self.sig.idx - self.update_block_size
        end_idx = self.sig.idx
        #assert self.sig.idx - self.updateIdx == self.filtLen
        self.sig["xf"][:, :, :, start_idx : end_idx] = \
            np.moveaxis(self.sec_path_filt.process(self.sig["ref"][:, start_idx : end_idx]), 2, 0)

        td_grad = ft.correlate_sum_tt(
            self.sig["error"][None, None, :, start_idx : end_idx],
            self.sig["xf"][:, :, :, end_idx - (2 * self.filt_len) : end_idx])

        norm = 1 / (
            np.sum(self.sig["xf"][:, :, :, start_idx : end_idx] ** 2)
            + self.beta
        )
        self.control_filter.ir -= self.mu * norm * td_grad







def equivalent_delay(ir):
    """Defined according to the paper with theoretical bounds for
    FxLMS for single channel"""
    eq_delay = np.zeros((ir.shape[0], ir.shape[1]))

    for i in range(ir.shape[0]):
        for j in range(ir.shape[1]):
            this_ir = ir[i, j, :]
            eq_delay[i, j] = np.sum(
                np.arange(this_ir.shape[-1]) * (this_ir ** 2)
            ) / np.sum(this_ir ** 2)
    return eq_delay