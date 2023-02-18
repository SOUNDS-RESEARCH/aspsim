import numpy as np
import copy
import ancsim.signal.filterclasses as fc
import ancsim.diagnostics.core as diacore
import ancsim.diagnostics.preprocessing as pp
import ancsim.diagnostics.plot as dplot

class RecordFilter(diacore.InstantDiagnostic):
    """
        Remember to include .ir in the property name
        if the property is a filter object
    """
    def __init__ (
        self, 
        prop_name, 
        *args,
        **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.prop_name = prop_name
        self.get_prop = diacore.attritemgetter(prop_name)
        self.prop = None

    def save(self, processor, chunkInterval, globInterval):
        self.prop = copy.deepcopy(self.get_prop(processor))

    def get_output(self):
        return self.prop


class RecordFilterDifference(RecordFilter):
    """
    """
    def __init__(self, 
        state_name,
        state_name_subtract,
        *args,
        **kwargs):
        super().__init__(state_name, *args, **kwargs)
        self.get_state_subtract = diacore.attritemgetter(state_name_subtract)

    def save(self, processor, chunkInterval, globInterval):
        self.prop = copy.deepcopy(self.get_prop(processor)) - copy.deepcopy(self.get_state_subtract(processor))





class RecordSignal(diacore.SignalDiagnostic):
    def __init__(self, 
        sig_name,
        *args,
        num_channels = 1, 
        **kwargs):
        super().__init__(*args, **kwargs)
        self.sig_name = sig_name
        self.num_channels = num_channels
        self.signal = np.full((self.num_channels, self.sim_info.tot_samples), np.nan)
        
    def save(self, processor, chunkInterval, globInterval):
        assert processor.sig[self.sig_name].ndim == 2
        assert processor.sig[self.sig_name].shape[0] == self.num_channels
        #if processor.sig[self.sig_name].shape[0] > 1:
        #    raise NotImplementedError

        self.signal[:, globInterval[0]:globInterval[1]] = \
            processor.sig[self.sig_name][:,chunkInterval[0]:chunkInterval[1]]

    def get_output(self):
        return self.signal


class RecordState(diacore.StateDiagnostic):
    """
    If state_dim is 1, the state is a scalar calue
    If the number is higher, the state is a vector, and the value of each component
        will be plotted as each line by the default plot function
    
    If state_dim is a tuple (i.e. the state is a matrix/tensor), 
    then the default plot function will have trouble
    """
    def __init__(self, 
        state_name,
        state_dim,
        *args,
        label_suffix_channel = None,
        **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(state_dim, int):
            state_dim = (state_dim,)

        self.state_values = np.full((*state_dim, self.save_at.num_values), np.nan)
        self.time_indices = np.full((self.save_at.num_values), np.nan, dtype=int)

        self.get_prop = diacore.attritemgetter(state_name)
        self.diag_idx = 0

        if label_suffix_channel is not None:
            self.plot_data["label_suffix_channel"] = label_suffix_channel
        

    def save(self, processor, chunkInterval, globInterval):
        assert globInterval[1] - globInterval[0] == 1
        self.state_values[:, self.diag_idx] = self.get_prop(processor)
        self.time_indices[self.diag_idx] = globInterval[0]

        self.diag_idx += 1

    def get_output(self):
        return self.state_values




class SignalPower(diacore.SignalDiagnostic):
    def __init__(self, sig_name,
                        sim_info, 
                        block_size, 
                        export_at=None,
                        sig_channels =slice(None),
                        **kwargs):
        super().__init__(sim_info, block_size, export_at, **kwargs)
        self.sig_name = sig_name
        self.sig_channels = sig_channels
        self.power = np.full((sim_info.tot_samples), np.nan)
    
    def save(self, processor, chunkInterval, globInterval):
        self.power[globInterval[0]:globInterval[1]] = np.mean(np.abs(
            processor.sig[self.sig_name][self.sig_channels,chunkInterval[0]:chunkInterval[1]])**2, axis=0)

    def get_output(self):
        return self.power


class SignalPowerRatio(diacore.SignalDiagnostic):
    def __init__(self, 
        numerator_name,
        denom_name,
        sim_info, 
        block_size, 
        export_at=None,
        numerator_channels=slice(None),
        denom_channels = slice(None),
        **kwargs
        ):
        super().__init__(sim_info, block_size, export_at, **kwargs)
        self.numerator_name = numerator_name
        self.denom_name = denom_name
        self.power_ratio = np.full((sim_info.tot_samples), np.nan)

        self.numerator_channels = numerator_channels
        self.denom_channels = denom_channels
        
    def save(self, processor, chunkInterval, globInterval):
        smoother_num = fc.create_filter(ir=np.ones((1,1,self.sim_info.output_smoothing)) / self.sim_info.output_smoothing)
        smoother_denom = fc.create_filter(ir=np.ones((1,1,self.sim_info.output_smoothing)) / self.sim_info.output_smoothing)

        num = smoother_num.process(np.mean(np.abs(processor.sig[self.numerator_name][self.numerator_channels, chunkInterval[0]:chunkInterval[1]])**2,axis=0, keepdims=True))
        denom = smoother_denom.process(np.mean(np.abs(processor.sig[self.denom_name][self.denom_channels, chunkInterval[0]:chunkInterval[1]])**2,axis=0, keepdims=True))

        self.power_ratio[globInterval[0]:globInterval[1]] = num / denom

    def get_output(self):
        return self.power_ratio



class StatePower(diacore.StateDiagnostic):
    def __init__(self, prop_name, 
                        sim_info, 
                        block_size, 
                        export_at=None,
                        save_frequency=None, 
                        **kwargs):
        super().__init__(sim_info, block_size, save_frequency, export_at, **kwargs)
        
        self.power = np.full((self.save_at.num_values), np.nan)
        self.time_indices = np.full((self.save_at.num_values), np.nan, dtype=int)

        self.get_prop = diacore.attritemgetter(prop_name)
        self.diag_idx = 0
        

    def save(self, processor, chunkInterval, globInterval):
        assert globInterval[1] - globInterval[0] == 1
        prop_val = self.get_prop(processor)
        self.power[self.diag_idx] = np.mean(np.abs(prop_val)**2)
        self.time_indices[self.diag_idx] = globInterval[0]

        self.diag_idx += 1

    def get_output(self):
        return self.power


class StateComparison(diacore.StateDiagnostic):
    """
        compare_func should take the two states as argument, and 
        return a single value representing the comparison (distance or MSE for example)
    """
    def __init__(self, compare_func, name_state1, name_state2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compare_func = compare_func
        self.get_state1 = diacore.attritemgetter(name_state1)
        self.get_state2 = diacore.attritemgetter(name_state2)

        self.compare_value = np.full((self.save_at.num_values), np.nan)
        self.time_indices = np.full((self.save_at.num_values), np.nan, dtype=int)
        self.diag_idx = 0

        self.plot_data["title"] = compare_func.__name__

    def save(self, processor, chunkInterval, globInterval):
        assert globInterval[1] - globInterval[0] == 1
        state1 = self.get_state1(processor)
        state2 = self.get_state2(processor)

        self.compare_value[self.diag_idx] = self.compare_func(state1, state2)
        self.time_indices[self.diag_idx] = globInterval[0]
        self.diag_idx += 1
        
    def get_output(self):
        return self.compare_value



class StateSummary(diacore.StateDiagnostic):
    """
        summary_func should take the state as argument and return a single scalar
        representing the state (norm or power for example)
    """
    def __init__(self, summary_func, state_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.summary_func = summary_func
        self.get_state = diacore.attritemgetter(state_name)

        self.summary_value = np.full((self.save_at.num_values), np.nan)
        self.time_indices = np.full((self.save_at.num_values), np.nan, dtype=int)
        self.diag_idx = 0

        self.plot_data["title"] = summary_func.__name__
        

    def save(self, processor, chunkInterval, globInterval):
        assert globInterval[1] - globInterval[0] == 1
        state = self.get_state(processor)

        self.summary_value[self.diag_idx] = self.summary_func(state)
        self.time_indices[self.diag_idx] = globInterval[0]
        self.diag_idx += 1

    def get_output(self):
        return self.summary_value





def power_of_all_signals(processor):
    """Must be called from processor.prepare(), not
        processor.__init__()
    """
    for sig_name in processor.sig.keys():
        processor.diag.add_diagnostic(f"power_{sig_name}", 
                SignalPower(sig_name, processor.sim_info, processor.block_size, 
                preprocess=[[pp.smooth(processor.sim_info.output_smoothing), pp.db_power]]))





class SoundfieldPower(diacore.Diagnostic):
    export_functions = {
        "image" : dplot.soundfield, 
        "npz" : dplot.savenpz, 
    }
    def __init__(
        self, 
        sig_name, 
        pos_mic, 
        use_samples, 
        sim_info, 
        block_size, 
        plot_arrays = None, 
        export_func = "image", 
        export_kwargs = None, 
        preprocess = None, 
        ):
        """
        
        use_samples : tuple[int, int]
            Uses the samples from use_samples[0] (inclusive) to use_samples[1] (exclusive)
            to compute the average soundfield power. 
        
        """
        save_at = diacore.IntervalCounter((use_samples,))
        export_at = (use_samples[1],)

        #save_at_idx = [exp_idx - num_avg]
        keep_only_last_export = False
        super().__init__(sim_info, block_size, export_at, save_at, export_func, keep_only_last_export, export_kwargs, preprocess)
        self.sig_name = sig_name
        self.pos_mic = pos_mic
        self.num_mic = self.pos_mic.shape[0]
        self.plot_arrays = plot_arrays

        self.num_avg = use_samples[1] - use_samples[0]
        assert self.num_avg >= 1
        

        self.power = np.zeros(self.num_mic)

        #src_sig = {src_name : np.zeros((arrays[src_name].num)) for src_name in source_names}

    def save(self, processor, chunkInterval, globInterval):
        self.power[:] += np.sum(np.abs(processor.sig[self.sig_name][:,chunkInterval[0]:chunkInterval[1]])**2, axis=-1) / self.num_avg

    def get_output(self):
        return self.power
        
