import numpy as np
import scipy.linalg as splin
import operator as op
import copy

import ancsim.array as ar
import ancsim.utilities as util
import ancsim.experiment.plotscripts as psc
import ancsim.signal.filterclasses as fc
import ancsim.signal.correlation as corr
import ancsim.diagnostics.diagnosticplots as dplot
import ancsim.diagnostics.diagnosticsummary as dsum


import ancsim.diagnostics.core as diacore


def attritemgetter(name):
    assert name[0] != "["
    attributes = name.replace("]", "").replace("'", "")
    attributes = attributes.split(".")
    attributes = [attr.split("[") for attr in attributes]

    def getter (obj):
        for sub_list in attributes:
            obj = getattr(obj, sub_list[0])
            for item in sub_list[1:]:
                obj = obj[item]
        return obj
    return getter


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
        self.get_prop = attritemgetter(prop_name)
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
        self.get_state_subtract = attritemgetter(state_name_subtract)

    def save(self, processor, chunkInterval, globInterval):
        self.prop = copy.deepcopy(self.get_prop(processor)) - copy.deepcopy(self.get_state_subtract(processor))




class RecordSignal(diacore.SignalDiagnostic):
    def __init__(self, 
        sig_name, 
        *args,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.sig_name = sig_name
        self.signal = np.full((self.sim_info.tot_samples), np.nan)
        
    def save(self, processor, chunkInterval, globInterval):
        if processor.sig[self.sig_name].shape[0] > 1:
            raise NotImplementedError

        self.signal[globInterval[0]:globInterval[1]] = \
            processor.sig[self.sig_name][0,chunkInterval[0]:chunkInterval[1]]

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

        self.get_prop = attritemgetter(state_name)
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
        smoother_num = fc.createFilter(ir=np.ones((1,1,self.sim_info.output_smoothing)) / self.sim_info.output_smoothing)
        smoother_denom = fc.createFilter(ir=np.ones((1,1,self.sim_info.output_smoothing)) / self.sim_info.output_smoothing)

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

        self.get_prop = attritemgetter(prop_name)
        self.diag_idx = 0
        

    def save(self, processor, chunkInterval, globInterval):
        assert globInterval[1] - globInterval[0] == 1
        prop_val = self.get_prop(processor)
        self.power[self.diag_idx] = np.mean(np.abs(prop_val)**2)
        self.time_indices[self.diag_idx] = globInterval[0]

        self.diag_idx += 1

    def get_output(self):
        return self.power




class StateMSE(diacore.StateDiagnostic):
    def __init__(self, est_state_name,
                        true_state_name,
                        *args,
                        **kwargs):
        super().__init__(*args, **kwargs)
        
        self.mse = np.full((self.save_at.num_values), np.nan)
        self.time_indices = np.full((self.save_at.num_values), np.nan, dtype=int)

        self.get_est_state = attritemgetter(est_state_name)
        self.get_true_state = attritemgetter(true_state_name)
        self.diag_idx = 0
        

    def save(self, processor, chunkInterval, globInterval):
        assert globInterval[1] - globInterval[0] == 1
        est_val = self.get_est_state(processor)
        true_val = self.get_true_state(processor)
        self.mse[self.diag_idx] = np.sum(np.abs(est_val - true_val)**2) / np.sum(np.abs(true_val)**2)
        self.time_indices[self.diag_idx] = globInterval[0]

        self.diag_idx += 1

    def get_output(self):
        return self.mse



class EigenvaluesOverTime(diacore.StateDiagnostic):
    """
    Matrix must be square, otherwise EVD doesn't work
    For now assumes hermitian matrix as well

    eigval_idx should be a tuple with the indices of the desired eigenvalues
    ascending order, zero indexed, and top inclusive

    The first value of num_eigvals is how many of the lowest eigenvalues that should be recorded
    The seconds value is how many of the largest eigenvalues that should be recorded
    """
    def __init__(self, matrix_name, eigval_idx, *args, abs_value=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(eigval_idx, (list, tuple, np.ndarray))
        assert len(eigval_idx) == 2
        self.get_matrix = attritemgetter(matrix_name)
        self.eigval_idx = eigval_idx
        self.abs_value = abs_value

        self.eigvals = np.full((eigval_idx[1]-eigval_idx[0]+1, self.save_at.num_values), np.nan)
        self.time_indices = np.full((self.save_at.num_values), np.nan, dtype=int)
        self.diag_idx = 0

        if self.abs_value:
            self.plot_data["title"] = "Size of Eigenvalues"
        else:
            self.plot_data["title"] = "Eigenvalues"

    def save(self, processor, chunkInterval, globInterval):
        assert globInterval[1] - globInterval[0] == 1
        mat = self.get_matrix(processor)
        assert np.allclose(mat, mat.T.conj())
        evs = splin.eigh(mat, eigvals_only=True, subset_by_index=self.eigval_idx)
        if self.abs_value:
            evs = np.abs(evs)
        self.eigvals[:, self.diag_idx] = evs
        self.time_indices[self.diag_idx] = globInterval[0]

        self.diag_idx += 1
        
    def get_output(self):
        return self.eigvals


class Eigenvalues(diacore.InstantDiagnostic):
    def __init__ (
        self, 
        matrix_name,
        *args,
        abs_value = False,
        **kwargs,
        ):
        super().__init__(*args, **kwargs)
        #self.matrix_name = matrix_name
        self.get_mat = attritemgetter(matrix_name)
        self.evs = None
        self.abs_value = abs_value

        if self.abs_value:
            self.plot_data["title"] = "Size of Eigenvalues"
        else:
            self.plot_data["title"] = "Eigenvalues"

    def save(self, processor, chunkInterval, globInterval):
        mat = self.get_mat(processor)
        assert np.allclose(mat, mat.T.conj())
        self.evs = splin.eigh(mat, eigvals_only=True)[None,None,:]

        if self.abs_value:
            self.evs = np.abs(self.evs)

    def get_output(self):
        return self.evs



class CorrMatrixDistance(diacore.StateDiagnostic):
    """
        Records the Correlation Matrix Distance between two matrices
        See corr.corr_matrix_distance for more info
    """
    def __init__(self, name_mat1, name_mat2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_mat1 = attritemgetter(name_mat1)
        self.get_mat2 = attritemgetter(name_mat2)

        self.cmd = np.full((self.save_at.num_values), np.nan)
        self.time_indices = np.full((self.save_at.num_values), np.nan, dtype=int)
        self.diag_idx = 0

        self.plot_data["title"] = "Correlation matrix distance"

    def save(self, processor, chunkInterval, globInterval):
        assert globInterval[1] - globInterval[0] == 1
        mat1 = self.get_mat1(processor)
        mat2 = self.get_mat2(processor)

        self.cmd[self.diag_idx] = corr.corr_matrix_distance(mat1, mat2)
        self.time_indices[self.diag_idx] = globInterval[0]
        self.diag_idx += 1
        
    def get_output(self):
        return self.cmd

class CosSimilarity(diacore.StateDiagnostic):
    """
        Records the cosine similarity between two vectors
        see corr.cos_similarity for more info
    """
    def __init__(self, name_vec1, name_vec2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.get_vec1 = attritemgetter(name_vec1)
        self.get_vec2 = attritemgetter(name_vec2)

        self.sim = np.full((self.save_at.num_values), np.nan)
        self.time_indices = np.full((self.save_at.num_values), np.nan, dtype=int)
        self.diag_idx = 0

        self.plot_data["title"] = "Cosine Similiarity"

    def save(self, processor, chunkInterval, globInterval):
        assert globInterval[1] - globInterval[0] == 1
        vec1 = self.get_vec1(processor)
        vec2 = self.get_vec2(processor)

        self.sim[self.diag_idx] = corr.cos_similary(vec1, vec2)
        self.time_indices[self.diag_idx] = globInterval[0]
        self.diag_idx += 1
        
    def get_output(self):
        return self.sim



class StateComparison(diacore.StateDiagnostic):
    """
        compare_func should take the two states as argument, and 
        return a single value representing the comparison (distance or MSE for example)
    """
    def __init__(self, compare_func, name_state1, name_state2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compare_func = compare_func
        self.get_state1 = attritemgetter(name_state1)
        self.get_state2 = attritemgetter(name_state2)

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
        self.get_state = attritemgetter(state_name)

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






def mse(state1, state2):
    """Normalized with size of state 2"""
    return np.sum(np.abs(state1 - state2)**2) / np.sum(np.abs(state2)**2)












class SoundfieldPower(diacore.Diagnostic):
    def __init__(
        self, 
        source_names,
        arrays, 
        num_avg,
        sim_info, 
        block_size, 
        export_at_idx
        ):

        save_at_idx = [exp_idx - num_avg]
        super().__init__(sim_info, block_size, export_at_idx)
        

        self.num_avg = num_avg

        src_sig = {src_name : np.full((self.num_avg, arrays[src_name].num), np.nan) for src_name in source_names}

    def save(self, processor, chunkInterval, globInterval):
        pass

    def get_output(self):
        #... actually get output

        self.reset()
        