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
                    if dg_name not in diag_names:
                        diag_names.append(dg_name)
        return diag_names
                
    def verify_same_export_settings(self, diag_dict):
        first_diag = diag_dict[list(diag_dict.keys())[0]]
        for dg in diag_dict.values():
            assert first_diag.export_function == dg.export_function
            assert first_diag.next_export() == dg.next_export()
            assert first_diag.keep_only_last_export == dg.keep_only_last_export


    def export_single_diag(self, diag_name, processors, fldr):
        diag_dict = {proc.name : proc.diag[diag_name] for proc in processors if diag_name in proc.diag}
        self.verify_same_export_settings(diag_dict)

        exp_funcs = processors[0].diag[diag_name].export_function
        export_time_idx = processors[0].diag[diag_name].next_export()
        for exp_func in exp_funcs:
            exp_func(diag_name, diag_dict, export_time_idx, fldr)

        for proc in processors:
            proc.diag[diag_name].progress_export()


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

    def saveData(self, processor, idx, globalIdx, last_block_on_chunk):
        for diagName, diag in self.diagnostics.items():
            start, end = diag.next_save()

            if globalIdx >= end:
                end_lcl = idx - (globalIdx - end)
                start_lcl = end_lcl - (end-start)
                diag.save(processor, (start_lcl, end_lcl), (start, end))
                diag.progress_save(end)
            elif last_block_on_chunk and globalIdx >= start:
                start_lcl = idx - (globalIdx-start)
                diag.save(processor, (start_lcl, idx), (start, globalIdx))
                diag.progress_save(globalIdx)
                pass


    # def saveData(self, processor, idx, globalIdx):
    #     for diagName, diag in self.diagnostics.items():
    #         if globalIdx >= self.lastGlobalSaveIdx[diagName] + self.samplesUntilSave[diagName]:
                
    #             if idx < self.lastSaveIdx[diagName]:
    #                 self.lastSaveIdx[diagName] -= self.sim_info.sim_chunk_size
    #             numSamples = idx - self.lastSaveIdx[diagName]

    #             diag.saveData(processor, self.lastSaveIdx[diagName], idx, self.lastGlobalSaveIdx[diagName], globalIdx)

    #             if globalIdx >= self.exportAfterSample[diagName]:
    #                 self.toBeExported.append(diagName)
    #                 self.exportAfterSample[diagname] = util.nextDivisible(self.sim_info.plot_frequency*self.sim_info.sim_chunk_size, 
    #                                                                         self.exportAfterSample[diagname])

    #             self.samplesUntilSave[diagName] += diag.save_frequency - numSamples
    #             self.lastSaveIdx[diagName] += numSamples
    #             self.lastGlobalSaveIdx[diagName] += numSamples


class IntervalCounter:
    def __init__(self, intervals, num_values = None):
        """
        intervals is an iterable or iterator where each entry is a tuple or list 
        of length 2, with start (inclusive) and end (exclusive) points of each interval
        np.ndarray of shape (num_intervals, 2) is also valid

        It is assumed that the intervals are strictly increasing, with no overlap. 
        meaning that ivs[i+1][0]>ivs[i][1] for all i. 
        """
        # try: #if is iterable
        #     self.intervals = iter(intervals)
        #     self.num_values = np.sum([iv[1]-iv[0] for iv in intervals])
        # except TypeError: #else is iterator
        #     self.intervals = intervals
        #     self.num_values = num_values
        #     assert num_values is not None
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
    def from_frequency(cls, frequency, max_value):
        num_values = int(np.ceil(max_value / frequency))
        return cls(zip(range(0, max_value, frequency), range(1, max_value+1, frequency)), num_values)

    # @classmethod
    # def from_list(cls, lst):
    #     """
    #     If you want freely chosen discrete times instead of intervals,
    #     supply a list as [time_1, time_2, time_3, ...]
    #     """
    #     num_values = len(lst)
    #     interval_lst = [[idx, idx+1] for idx in lst]
    #     return cls(interval_lst, num_values)

    def upcoming(self):
        return self.start, self.end

    def progress(self, progress_until):
        self.start = progress_until
        assert self.end >= self.start
        if self.start == self.end:
            self.start, self.end = next(self.intervals, (np.inf, np.inf))




class IndexCounter():
    def __init__(self, idx_selection):
        """
        idx_selection is either an iterable of indices, or a single number
                        which is the interval between adjacent desired indices. 
        
        """
        try:
            self.orig_iterable = idx_selection
            self.idx_selection = iter(idx_selection)
        except TypeError:
            self.interval = idx_selection
            self.idx_selection = it.count(idx_selection-1, idx_selection)
        self.upcoming_idx = next(self.idx_selection, np.inf)
        #self.progress()

    # @classmethod
    # def from_intervals(cls, intervals, max_spacing):
    #     """
    #         Intervals is a list of lists or list of tuples, 
    #         where len(intervals[i]) == 2, and the intervals 
    #         are sorted according to end value. 
    #         numpy array is also fine, with the shape (num_intervals, 2)
    #     """

    #     assert all([intervals[i,1] < intervals[i+1,1] for i in range(len(intervals)-1)])

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
        ):
        """
        save_at_idx is an iterable which gives all indices for which to save data. 
                    Must be an integer multiple of the block size. (maybe change to 
                    must be equal or larger than the block size) 
        """

        assert isinstance(save_at, IntervalCounter)
        self.save_at = save_at

        if export_at is None:
            export_at = sim_info.sim_chunk_size * sim_info.chunk_per_export
        if not isinstance(export_at, (list, tuple, np.ndarray)):
            assert export_at >= block_size
        self.export_at = IndexCounter(export_at)

        if isinstance(export_func, str):
            export_func = [export_func]
        self.export_function = [type(self).export_functions[func_choice] for func_choice in export_func]

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
    ):
        if save_at is None:
            save_at = IntervalCounter(((0,sim_info.tot_samples),))
            if keep_only_last_export is None:
                keep_only_last_export = True
        else:
            if keep_only_last_export is None:
                keep_only_last_export = False
        super().__init__(sim_info, block_size, export_at, save_at, export_func, keep_only_last_export)

        self.plot_data["xlabel"] = "Samples"
        self.plot_data["ylabel"] = ""
        self.plot_data["title"] = ""

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
    ):
        if save_frequency is None:
            save_frequency = block_size
        super().__init__(sim_info, block_size, export_at, 
                        IntervalCounter.from_frequency(save_frequency, sim_info.tot_samples),
                        export_func, 
                        keep_only_last_export)
        self.plot_data["xlabel"] = "Samples"
        self.plot_data["ylabel"] = ""
        self.plot_data["title"] = ""


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
    ):
        if save_at is None:
            save_freq = sim_info.sim_chunk_size * sim_info.chunk_per_export
            save_at = IntervalCounter.from_frequency(save_freq, sim_info.tot_samples)
            export_at = save_freq
            #export_at = IndexCounter(save_freq)
        else:
            # Only a single list, or a scalar. save_at can't be intervals
            if isinstance(save_at, (tuple, list, np.ndarray)):
                assert not isinstance(save_at[0], (tuple, list, np.ndarray))
            export_at = save_at
            save_at = IntervalCounter(save_at)
        super().__init__(sim_info, block_size, export_at, save_at, export_func, keep_only_last_export)




