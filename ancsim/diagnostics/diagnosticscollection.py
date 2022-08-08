import numpy as np
import scipy.linalg as splin
import copy


import ancsim.signal.filterclasses as fc
import ancsim.diagnostics.core as diacore

import aspcol.correlation as corr






class StateMSE(diacore.StateDiagnostic):
    def __init__(self, est_state_name,
                        true_state_name,
                        *args,
                        **kwargs):
        super().__init__(*args, **kwargs)
        
        self.mse = np.full((self.save_at.num_values), np.nan)
        self.time_indices = np.full((self.save_at.num_values), np.nan, dtype=int)

        self.get_est_state = diacore.attritemgetter(est_state_name)
        self.get_true_state = diacore.attritemgetter(true_state_name)
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
        self.get_matrix = diacore.attritemgetter(matrix_name)
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
        self.get_mat = diacore.attritemgetter(matrix_name)
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
        self.get_mat1 = diacore.attritemgetter(name_mat1)
        self.get_mat2 = diacore.attritemgetter(name_mat2)

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
        self.get_vec1 = diacore.attritemgetter(name_vec1)
        self.get_vec2 = diacore.attritemgetter(name_vec2)

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
        