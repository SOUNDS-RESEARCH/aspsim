import numpy as np
from abc import ABC, abstractmethod
import itertools as it
import operator as op

import ancsim.array as ar
import ancsim.utilities as util
import ancsim.experiment.plotscripts as psc
import ancsim.signal.filterclasses as fc
import ancsim.diagnostics.diagnosticplots as dplot
import ancsim.diagnostics.diagnosticsummary as dsum


"""
===== DIAGNOSTICS OVERVIEW =====
All diagnostics should inherit from Diagnostic, but it is suggested to use the 
intermediate SignalDiagnostic, StateDiagnostic, and Instantdiagnostic. 

===== PROCESSOR.IDX =====
The processor.idx should be interpreted as the next (local, in reference to the internal signal buffer)
time index it will process. So if processor.idx == 1, then processor.sig['signame'][:,0] is a processed sample
that can be saved to a diagnostic, but processor.sig['signame'][:,1] is not. 

The processor.idx is increased after propagating all signals. During its .process method, the processing can
add source signals between idx:idx+block_size. These new signals are propagated to the microphones, and the 
index is increased by block_size. Directly after that, the diagnostics are saved. 

===== GLOBAL VS LOCAL TIME INDEX =====
At the time when data is saved to diagnostics, the global_idx == (local_idx-buffer) + K*chunk_size for some 
integer K. Meaning that the two indices are in sync, without any small offset. The very first chunk the local
index is merely offset by the buffer_size. 


===== SAVE_AT AND EXPORT_AT =====
All diagnostics needs an IntervalCounter as the member save_at
which dictates when data should be saved from the processor to the diagnostic

All diagnostics needs an IndexCounter as the member export_at
which dictates when data should be saved from the diagnostic to file. 

Both export_at and save_at will be indexes compared to the global time index. 

save_at should be/represent a sequence of non-overlapping intervals, where
each interval is (start, end), and the relevant signal should be saved
to the diagnostics between start:end (meaning end-exclusive).
To save state or instant diagnostics (where a continuous signal is not available)
the interval should be of length 1, so (start, start+1). 

When specifying save frequency, or giving just a list of indices (as you would for state or instant diagnostics)
it is interpreted as "save each X samples" or "save after X samples has passed". 
This means that with a save frequency of 200, the first save would be after globalIdx 199 has been processed; 
and as explained above, by the time data is saved to diagnostics, the globalIdx would then be 200.  


"""






class DiagnosticExporter:
    def __init__(self, sim_info, processors):
        self.sim_info = sim_info
        
        self.upcoming_export = 0
        self.update_next_export(processors)

    def update_next_export(self, processors):
        try:
            self.upcoming_export = np.amin([dg.next_export() for proc in processors 
                                            for dg in proc.diag])
        except ValueError:
            self.upcoming_export = np.inf

        #self.upcoming_export = int(np.amin(np.array([dg.next_export() for proc in processors 
        #                                    for dg in proc.diag], dtype=np.float64), initial=np.inf))

    def ready_for_export(self, processors):
        for proc in processors:
            for _, dg in proc.diag.items():
                start, end = dg.save_at.upcoming()
                if start <= self.upcoming_export:
                    return False
        return True

    def next_export(self):
        return self.upcoming_export
        
    def export_this_idx(self, processors, idx_to_check):
        diag_names = []
        for proc in processors:
            for dg_name, dg in proc.diag.items():
                if dg.next_export() <= idx_to_check:
                    if dg.next_save()[0] >= idx_to_check:
                        if dg_name not in diag_names:
                            diag_names.append(dg_name)
        return diag_names
                
    def verify_same_export_settings(self, diag_dict):
        first_diag = diag_dict[list(diag_dict.keys())[0]]
        for dg in diag_dict.values():
            assert first_diag.export_function == dg.export_function
            assert first_diag.next_export() == dg.next_export()
            assert first_diag.keep_only_last_export == dg.keep_only_last_export
            assert first_diag.export_kwargs == dg.export_kwargs
            for prep1, prep2 in zip(first_diag.preprocess, dg.preprocess):
                assert len(prep1) == len(prep2)
                for pp_func1, pp_func2 in zip(prep1, prep2):
                    assert pp_func1.__name__ == pp_func2.__name__
            #assert first_diag.preprocess == dg.preprocess


    def export_single_diag(self, diag_name, processors, fldr):
        diag_dict = {proc.name : proc.diag[diag_name] for proc in processors if diag_name in proc.diag}
        self.verify_same_export_settings(diag_dict)

        one_diag_object = diag_dict[list(diag_dict.keys())[0]]
        exp_funcs = one_diag_object.export_function
        exp_kwargs = one_diag_object.export_kwargs
        preproc = one_diag_object.preprocess
        export_time_idx = one_diag_object.next_export()
        for exp_func, exp_kwarg, pp in zip(exp_funcs, exp_kwargs, preproc):
            exp_func(diag_name, diag_dict, export_time_idx, fldr, pp, **exp_kwarg)

        for diag in diag_dict.values():
            diag.progress_export()


    def dispatch(self, processors, time_idx, fldr):
        """
        processors is list of the processor objects
        time_idx is the global time index
        fldr is Path to figure folder
        """
        #if time_idx == self.next_export():
        for diag_name in self.export_this_idx(processors, time_idx):
            self.export_single_diag(diag_name, processors, fldr)
        self.update_next_export(processors)


class DiagnosticHandler:
    def __init__(self, sim_info, block_size):
        self.sim_info = sim_info
        #self.block_size = block_size
        self.diagnostics = {}

        #To keep track of when each diagnostics are available to be exported
        #self.exportAfterSample = {}
        #self.toBeExported = []

    def prepare(self):
        pass
        #for diagName, diag in self.diagnostics.items():
        #    diag.prepare()
        #    self.exportAfterSample[diagName] = self.sim_info.sim_chunk_size
            

    def __contains__(self, key):
        return key in self.diagnostics

    def __iter__(self):
        for diag_obj in self.diagnostics.values():
            yield diag_obj

    def items(self):
        for diag_name, diag_obj in self.diagnostics.items():
            yield diag_name, diag_obj

    def __getitem__(self, key):
        return self.diagnostics[key]

    def __setitem__(self, name, diagnostic):
        self.add_diagnostic(name, diagnostic)
    
    def add_diagnostic(self, name, diagnostic):
        assert name not in self.diagnostics
        self.diagnostics[name] = diagnostic

    def reset(self):
        raise NotImplementedError
        for diag in self.diagnostics.values():
            diag.reset()

    #def add_diagnostic(self, name, diagnostic):
    #    self.diagnostics[name] = diagnostic

    def saveData(self, processor, idx, global_idx, last_block_on_chunk):
        for diagName, diag in self.diagnostics.items():
            start, end = diag.next_save()

            if global_idx >= end:
                end_lcl = idx - (global_idx - end)
                start_lcl = end_lcl - (end-start)
                diag.save(processor, (start_lcl, end_lcl), (start, end))
                diag.progress_save(end)
            elif last_block_on_chunk and global_idx >= start:
                start_lcl = idx - (global_idx-start)
                diag.save(processor, (start_lcl, idx), (start, global_idx))
                diag.progress_save(global_idx)



class IntervalCounter:
    def __init__(self, intervals, num_values = None):
        """
        intervals is an iterable or iterator where each entry is a tuple or list 
        of length 2, with start (inclusive) and end (exclusive) points of each interval
        np.ndarray of shape (num_intervals, 2) is also valid

        It is assumed that the intervals are strictly increasing, with no overlap. 
        meaning that ivs[i+1][0]>ivs[i][1] for all i. 
        """
        if isinstance(intervals, (list, tuple, np.ndarray)):
            if isinstance(intervals[0], (list, tuple, np.ndarray)):
                self.num_values = np.sum([iv[1]-iv[0] for iv in intervals])
            else: 
                self.num_values = len(intervals)
                intervals = [[idx-1, idx] for idx in intervals]
            self.intervals = iter(intervals)
            assert num_values is None
        else:
            self.intervals = intervals
            self.num_values = num_values

        self.start, self.end = next(self.intervals)

    @classmethod
    def from_frequency(cls, frequency, max_value, include_zero=False):
        num_values = int(np.ceil(max_value / frequency))

        start_value = 0
        if not include_zero:
            start_value += frequency
        return cls(zip(range(start_value-1, max_value, frequency), range(start_value, max_value+1, frequency)), num_values)


    # @classmethod
    # def from_frequency(cls, frequency, max_value, include_zero=True):
    #     num_values = int(np.ceil(max_value / frequency))

    #     start_value = 0
    #     if not include_zero:
    #         start_value += frequency
    #     return cls(zip(range(start_value, max_value, frequency), range(start_value+1, max_value+1, frequency)), num_values)

    def upcoming(self):
        return self.start, self.end

    def progress(self, progress_until):
        self.start = progress_until
        assert self.end >= self.start
        if self.start == self.end:
            self.start, self.end = next(self.intervals, (np.inf, np.inf))




class IndexCounter():
    def __init__(self, idx_selection, tot_samples=None):
        """
        idx_selection is either an iterable of indices, or a single number
                        which is the interval between adjacent desired indices. 
        
        """
        try:
            self.orig_iterable = idx_selection
            self.idx_selection = iter(idx_selection)
        except TypeError:
            assert tot_samples is not None
            self.interval = idx_selection
            #self.idx_selection = it.count(idx_selection-1, idx_selection)
            self.idx_selection = iter(range(idx_selection, tot_samples+1, idx_selection))
            #self.idx_selection = it.count(idx_selection, idx_selection)
        self.upcoming_idx = next(self.idx_selection, np.inf)

    def progress(self):
        self.upcoming_idx = next(self.idx_selection, np.inf)

    def upcoming (self):
        return self.upcoming_idx

    def num_idx_until(self, until_value):
        raise ValueError
        #TODO check that this one works as expected
        try:
            return int(np.ceil(until_value / self.interval))
        except NameError:
            for i, idx in self.orig_iterable:
                if idx > until_value:
                    return i

        
class Diagnostic:
    export_functions = {}
    def __init__(
        self, 
        sim_info, 
        block_size, 
        export_at,
        save_at,
        export_func,
        keep_only_last_export,
        export_kwargs,
        preprocess,
        ):
        """
        save_at_idx is an iterable which gives all indices for which to save data. 
                    Must be an integer multiple of the block size. (maybe change to 
                    must be equal or larger than the block size) 
        """
        self.sim_info = sim_info
        self.block_size = block_size

        assert isinstance(save_at, IntervalCounter)
        self.save_at = save_at

        if export_at is None:
            export_at = sim_info.sim_chunk_size * sim_info.chunk_per_export
        if isinstance(export_at, (list, tuple, np.ndarray)):
            assert all([exp_at >= block_size for exp_at in export_at])
        else:
            assert export_at >= block_size
        
        self.export_at = IndexCounter(export_at, self.sim_info.tot_samples)

        if isinstance(export_func, str):
            export_func = [export_func]
        self.export_function = [type(self).export_functions[func_choice] for func_choice in export_func]

        if export_kwargs is None:
            export_kwargs = {}
        if isinstance(export_kwargs, dict):
            export_kwargs = [export_kwargs]
        assert len(export_kwargs) == len(self.export_function)
        self.export_kwargs = export_kwargs

        if preprocess is None:
            preprocess = [[] for _ in range(len(self.export_function))]
        elif callable(preprocess):
            preprocess = [preprocess]
        self.preprocess = [[pp] if callable(pp) else pp for pp in preprocess]
        assert len(self.preprocess) == len(self.export_function)

        self.keep_only_last_export = keep_only_last_export

        self.plot_data = {}

    def next_export(self):
        return self.export_at.upcoming()

    def next_save(self):
        return self.save_at.upcoming()

    def progress_save(self, progress_until):
        self.save_at.progress(progress_until)

    def progress_export(self):
        self.export_at.progress()


    @abstractmethod
    def save(self, processor, chunkInterval, globInterval):
        pass
        # self.get_property = op.attrgetter(property_name)
        # prop = self.get_property(processor)

        # signal = processor.sig[signal_name]

    @abstractmethod
    def get_output(self):
        self.export_at.progress()

    def get_processed_output(self, time_idx, preprocess):
        """
        """
        output = self.get_output()
        for pp in preprocess:
            output = pp(output)
        return output


class SignalDiagnostic(Diagnostic):
    export_functions = {
        "plot" : dplot.functionOfTimePlot,
        "npz" : dplot.savenpz,
    }
    def __init__ (
        self,
        sim_info,
        block_size,
        export_at=None,
        save_at = None, 
        export_func = "plot",
        keep_only_last_export = None,
        export_kwargs = None,
        preprocess = None,
    ):
        if save_at is None:
            save_at = IntervalCounter(((0,sim_info.tot_samples),))
            if keep_only_last_export is None:
                keep_only_last_export = True
        else:
            if keep_only_last_export is None:
                keep_only_last_export = False
        super().__init__(sim_info, block_size, export_at, save_at, export_func, keep_only_last_export, export_kwargs, preprocess)

        self.plot_data["xlabel"] = "Samples"
        self.plot_data["ylabel"] = ""
        self.plot_data["title"] = ""

    def get_processed_output(self, time_idx, preprocess):
        output, time_indices = get_values_up_to_idx(self.get_output(), time_idx)
        for pp in preprocess:
            output = pp(output)
        return output, time_indices


class StateDiagnostic(Diagnostic):
    export_functions = {
        "plot" : dplot.functionOfTimePlot,
        "npz" : dplot.savenpz,
    }
    def __init__ (
        self,
        sim_info,
        block_size,
        export_at=None,
        save_frequency=None,
        export_func = "plot",
        keep_only_last_export=True,
        export_kwargs = None,
        preprocess = None,
    ):
        if save_frequency is None:
            save_frequency = block_size
        super().__init__(sim_info, block_size, export_at, 
                        IntervalCounter.from_frequency(save_frequency, sim_info.tot_samples,include_zero=True),
                        export_func, 
                        keep_only_last_export,
                        export_kwargs,
                        preprocess)
        self.plot_data["xlabel"] = "Samples"
        self.plot_data["ylabel"] = ""
        self.plot_data["title"] = ""

    def get_processed_output(self, time_idx, preprocess):
        output, time_indices = get_values_from_selection(self.get_output(), self.time_indices, time_idx)
        for pp in preprocess:
            output = pp(output)
        return output, time_indices


class InstantDiagnostic(Diagnostic):
    export_functions = {
        "plot" : dplot.plotIR,
        "matshow" : dplot.matshow, 
        "npz" : dplot.savenpz,
    }
    def __init__ (
        self,
        sim_info,
        block_size,
        save_at = None, 
        export_func = "plot",
        keep_only_last_export = False,
        export_kwargs = None,
        preprocess = None,
    ):
        if save_at is None:
            save_freq = sim_info.sim_chunk_size * sim_info.chunk_per_export
            save_at = IntervalCounter.from_frequency(save_freq, sim_info.tot_samples, include_zero=False)
            export_at = save_freq
            #export_at = IndexCounter(save_freq)
        else:
            # Only a single list, or a scalar. save_at can't be intervals
            export_at = save_at
            if isinstance(save_at, (tuple, list, np.ndarray)):
                assert not isinstance(save_at[0], (tuple, list, np.ndarray))
                save_at = IntervalCounter(save_at)
            else:
                save_at = IntervalCounter.from_frequency(save_at, sim_info.tot_samples, include_zero=False)
        super().__init__(sim_info, block_size, export_at, save_at, export_func, keep_only_last_export, export_kwargs, preprocess)



def get_values_up_to_idx(signal, max_idx):
    """
    gives back signal values that correspond to time_values less than max_idx, 
    and signal values that are not nan

    max_idx is exlusive
    """
    signal = np.atleast_2d(signal)
    assert signal.ndim == 2

    time_indices = np.arange(max_idx)
    signal = signal[:,:max_idx]

    nan_filter = np.logical_not(np.isnan(signal))
    assert np.isclose(nan_filter, nan_filter[0, :]).all()
    nan_filter = nan_filter[0, :]

    time_indices = time_indices[nan_filter]
    signal = signal[:,nan_filter]
    return signal, time_indices

def get_values_from_selection(signal, time_indices, max_idx):
    """
    gives back signal values that correspond to time_values less than max_idx, 
    and signal values that are not nan

    max_idx is exlusive
    """
    signal = np.atleast_2d(signal)
    assert signal.ndim == 2
    nan_filter = np.logical_not(np.isnan(signal))
    assert np.isclose(nan_filter, nan_filter[0, :]).all()
    nan_filter = nan_filter[0, :]

    assert time_indices.shape[-1] == signal.shape[-1]
    time_indices = time_indices[nan_filter]
    signal = signal[:,nan_filter]

    if len(time_indices) == 0:
        assert signal.shape[-1] == 0
        return signal, time_indices

    above_max_idx = np.argmax(time_indices >= max_idx)
    if above_max_idx == 0:
        if np.logical_not(above_max_idx).all():
            above_max_idx = len(time_indices)

    time_indices = time_indices[:above_max_idx]
    signal = signal[:,:above_max_idx]
    return signal, time_indices
