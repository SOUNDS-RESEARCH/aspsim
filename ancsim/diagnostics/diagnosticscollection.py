import numpy as np
import scipy.linalg as splin
import copy

import ancsim.diagnostics.core as diacore


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
        