import numpy as np
import scipy.linalg as splin
import operator as op
import copy

import ancsim.array as ar
import ancsim.utilities as util
import ancsim.experiment.plotscripts as psc
import ancsim.signal.filterclasses as fc
import ancsim.diagnostics.diagnosticplots as dplot
import ancsim.diagnostics.diagnosticsummary as dsum


import ancsim.diagnostics.core as diacore

# class RecordMatrix(diacore.InstantDiagnostic):
#     """
#     """
#     def __init__ (
#         self, 
#         prop_name, 
#         sim_info, 
#         block_size, 
#         save_at = None, 
#         export_func = "plot",
#         **kwargs,
#         ):
#         super().__init__(sim_info, block_size, save_at, export_func, **kwargs)
#         self.prop_name = prop_name
#         self.get_prop = op.attrgetter(prop_name)
#         self.prop = None

#     def save(self, processor, chunkInterval, globInterval):
#         self.prop = copy.deepcopy(self.get_prop(processor))

#     def get_output(self):
#         return self.prop



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
        self.get_prop = op.attrgetter(prop_name)
        self.prop = None

    def save(self, processor, chunkInterval, globInterval):
        self.prop = copy.deepcopy(self.get_prop(processor))

    def get_output(self):
        return self.prop


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

        self.scalar_values = np.full((*state_dim, self.save_at.num_values), np.nan)
        self.time_indices = np.full((self.save_at.num_values), np.nan, dtype=int)

        self.get_prop = op.attrgetter(state_name)
        self.diag_idx = 0

        if label_suffix_channel is not None:
            self.plot_data["label_suffix_channel"] = label_suffix_channel
        

    def save(self, processor, chunkInterval, globInterval):
        self.state_values[:, self.diag_idx] = self.get_prop(processor)
        self.time_indices[self.diag_idx] = globInterval[1]

        self.diag_idx += 1

    def get_output(self):
        return self.state_values



class SignalPower(diacore.SignalDiagnostic):
    def __init__(self, sig_name,
                        sim_info, 
                        block_size, 
                        export_at=None,
                        **kwargs):
        super().__init__(sim_info, block_size, export_at, **kwargs)
        self.sig_name = sig_name
        self.power = np.full((sim_info.tot_samples), np.nan)
    
    def save(self, processor, chunkInterval, globInterval):
        self.power[globInterval[0]:globInterval[1]] = np.mean(np.abs(
            processor.sig[self.sig_name][:,chunkInterval[0]:chunkInterval[1]])**2, axis=0)

    def get_output(self):
        return self.power



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

        self.get_prop = op.attrgetter(prop_name)
        self.diag_idx = 0
        

    def save(self, processor, chunkInterval, globInterval):
        prop_val = self.get_prop(processor)
        self.power[self.diag_idx] = np.mean(np.abs(prop_val)**2)
        self.time_indices[self.diag_idx] = globInterval[1]

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

        self.get_est_state = op.attrgetter(est_state_name)
        self.get_true_state = op.attrgetter(true_state_name)
        self.diag_idx = 0
        

    def save(self, processor, chunkInterval, globInterval):
        est_val = self.get_est_state(processor)
        true_val = self.get_true_state(processor)
        self.mse[self.diag_idx] = 10*np.log10(1e-60+ np.sum(np.abs(est_val - true_val)**2) / np.sum(np.abs(true_val)**2))
        self.time_indices[self.diag_idx] = globInterval[1]

        self.diag_idx += 1

    def get_output(self):
        return self.mse



class Eigenvalues(diacore.StateDiagnostic):
    """
    Matrix must be square, otherwise EVD doesn't work
    For now assumes hermitian matrix as well

    eigval_idx should be a tuple with the indices of the desired eigenvalues
    ascending order, zero indexed, and top inclusive

    The first value of num_eigvals is how many of the lowest eigenvalues that should be recorded
    The seconds value is how many of the largest eigenvalues that should be recorded
    """
    def __init__(self, matrix_name, eigval_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(eigval_idx, (list, tuple, np.ndarray))
        assert len(eigval_idx) == 2
        self.get_matrix = op.attrgetter(matrix_name)
        self.eigval_idx = eigval_idx

        self.eigvals = np.full((eigval_idx[1]-eigval_idx[0]+1, self.save_at.num_values), np.nan)
        self.time_indices = np.full((self.save_at.num_values), np.nan, dtype=int)
        self.diag_idx = 0

    def save(self, processor, chunkInterval, globInterval):
        mat = self.get_matrix(processor)
        assert np.allclose(mat, mat.T.conj())
        evs = splin.eigh(mat, eigvals_only=True, subset_by_index=self.eigval_idx)
        self.eigvals[:, self.diag_idx] = evs
        self.time_indices[self.diag_idx] = globInterval[1]

        self.diag_idx += 1
        
    def get_output(self):
        return self.eigvals





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
        