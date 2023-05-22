import numpy as np
from abc import abstractmethod

import aspsim.diagnostics.plot as dplot


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



class Logger:
    def __init__(self, sim_info):
        self.sim_info = sim_info
        self.diagnostics = {}

        self.upcoming_export = 0

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

    def prepare(self):
        self.update_next_export()
    
    def add_diagnostic(self, name, diagnostic):
        assert name not in self.diagnostics
        self.diagnostics[name] = diagnostic

    def save_data(self, processors, sig, idx, global_idx, last_block_on_chunk):
        for diag_name, diag in self.diagnostics.items():
            start, end = diag.next_save()
            num_samples = end - start

            if global_idx >= end:
                end_lcl = idx - (global_idx - end) - 1# or maybe start_lcl should be +1
                start_lcl = end_lcl - num_samples
                diag.save(processors, sig, (start_lcl, end_lcl), (start, end))
                diag.progress_save(end)
            elif last_block_on_chunk and global_idx > start:
                end_lcl = idx - 1
                start_lcl = idx - (global_idx-start) - 1
                diag.save(processors, sig, (start_lcl, end_lcl), (start, global_idx))
                diag.progress_save(global_idx)


    def update_next_export(self): 
        try:
            self.upcoming_export = np.amin([dg.next_export() for dg in self.diagnostics.values()])
        except ValueError:
            self.upcoming_export = np.inf

    def next_export(self):
        return self.upcoming_export
    
    def export_this_idx(self):
        diag_names = []
        for dg_name, dg in self.diagnostics.items():
            if dg.next_export() <= dg.save_at.saved_until: # all relevant data have been saved
                assert dg_name not in diag_names
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


    def export_single_diag(self, diag_name, fldr):
        diag_dict = {diag_name : self.diagnostics[diag_name]}
        #diag_dict = {proc.name : proc.diag[diag_name] for proc in processors if diag_name in proc.diag}
        self.verify_same_export_settings(diag_dict)

        one_diag_object = diag_dict[list(diag_dict.keys())[0]]
        exp_funcs = one_diag_object.export_function
        exp_kwargs = one_diag_object.export_kwargs
        preproc = one_diag_object.preprocess
        #export_time_idx = one_diag_object.next_save()[1]
        export_time_idx = one_diag_object.next_export()
        for exp_func, exp_kwarg, pp in zip(exp_funcs, exp_kwargs, preproc):
            exp_func(diag_name, diag_dict, export_time_idx, fldr, pp, print_method=self.sim_info.plot_output, **exp_kwarg)

        for diag in diag_dict.values():
            diag.progress_export()


    def dispatch(self, fldr):
        """
        processors is list of the processor objects
        time_idx is the global time index
        fldr is Path to figure folder
        """
        #if time_idx == self.next_export():
        if self.sim_info.plot_output != "none":
            while self.export_this_idx():
                for diag_name in self.export_this_idx():
                    self.export_single_diag(diag_name, fldr)
            self.update_next_export()


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

        self.saved_until = 0
        self.start, self.end = next(self.intervals, (np.inf, np.inf))
        

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
        self.saved_until = progress_until
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

        assert isinstance(save_at, IntervalCounter)
        self.save_at = save_at

        if export_at is None:
            export_at = sim_info.export_frequency
        #if isinstance(export_at, (list, tuple, np.ndarray)):
        #    assert all([exp_at >= block_size for exp_at in export_at])
        #else:
            #assert export_at >= block_size
        
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
    def save(self, processor, sig, chunk_interval, glob_interval):
        pass
        # self.get_property = op.attrgetter(property_name)
        # prop = self.get_property(processor)

        # signal = sig[signal_name]

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
        "plot" : dplot.function_of_time_plot,
        "npz" : dplot.savenpz,
        "text" : dplot.txt,
        "wav" : dplot.create_audio_files,
        "spectrum" : dplot.spectrum_plot, 
    }
    def __init__ (
        self,
        sim_info,
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
            keep_only_last_export = False
        super().__init__(sim_info, export_at, save_at, export_func, keep_only_last_export, export_kwargs, preprocess)

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
        "plot" : dplot.function_of_time_plot,
        "npz" : dplot.savenpz,
    }
    def __init__ (
        self,
        sim_info,
        export_at=None,
        save_frequency=None,
        export_func = "plot",
        keep_only_last_export=False,
        export_kwargs = None,
        preprocess = None,
    ):
        if save_frequency is None:
            raise NotImplementedError
            #save_frequency = block_size
        super().__init__(sim_info, export_at, 
                        IntervalCounter.from_frequency(save_frequency, sim_info.tot_samples,include_zero=False),
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
        "plot" : dplot.plot_ir,
        "matshow" : dplot.matshow, 
        "npz" : dplot.savenpz,
        "text" : dplot.txt, 
    }
    def __init__ (
        self,
        sim_info,
        save_at = None, 
        export_func = "plot",
        keep_only_last_export = False,
        export_kwargs = None,
        preprocess = None,
    ):
        if save_at is None:
            save_freq = sim_info.export_frequency
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
        super().__init__(sim_info, export_at, save_at, export_func, keep_only_last_export, export_kwargs, preprocess)



# def attritemgetter(name):
#     assert name[0] != "["
#     attributes = name.replace("]", "").replace("'", "")
#     attributes = attributes.split(".")
#     attributes = [attr.split("[") for attr in attributes]

#     def getter (obj):
#         for sub_list in attributes:
#             obj = getattr(obj, sub_list[0])
#             for item in sub_list[1:]:
#                 obj = obj[item]
#         return obj
#     return getter

def attritemgetter(name):
    """
    If you have a dictionary with strings, use the name "dict_obj['key']"
    Without apostrophes, the key is assumed to be a list index, and is converted to integer.

    TODO: Possibly allow for objects other than integers, such as indexing arrays
    
    """
    assert name[0] != "["
    attributes = name.replace("]", "")
    attributes = attributes.split(".")
    attributes = [attr.split("[") for attr in attributes]

    def getter (obj):
        for sub_list in attributes:
            obj = getattr(obj, sub_list[0])
            for item in sub_list[1:]:
                if item[0] == "'":
                    if not item[-1] == "'":
                        raise ValueError("Unmatched apostrophes")
                    item = item.replace("'", "")
                else:
                    item = int(item)
                obj = obj[item]
        return obj
    return getter




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
