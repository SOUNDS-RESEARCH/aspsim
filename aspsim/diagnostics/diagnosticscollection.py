import numpy as np
import scipy.linalg as splin
import scipy.signal as spsig

import aspsim.diagnostics.core as diacore
import aspsim.diagnostics.plot as dplot


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

    def save(self, processor, sig, chunkInterval, globInterval):
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

    def save(self, processor, sig, chunkInterval, globInterval):
        mat = self.get_mat(processor)
        assert np.allclose(mat, mat.T.conj())
        self.evs = splin.eigh(mat, eigvals_only=True)[None,None,:]

        if self.abs_value:
            self.evs = np.abs(self.evs)

    def get_output(self):
        return self.evs





class SummaryDiagnostic(diacore.Diagnostic):
    export_functions = {
        "npz" : dplot.savenpz,
        "text" : dplot.txt,
        "spectrum" : dplot.spectrum_plot,
    }
    def __init__ (
        self,
        sim_info,
        block_size,
        save_at,
        export_func = "text",
        keep_only_last_export = False,
        export_kwargs = None,
        preprocess = None,
        ):
        """
        save_at should be a tuple (start_sample, end_sample)
            it will use the samples between start_samle (inclusive) and end_sample (exclusive)
        
        """
        if isinstance(save_at, diacore.IntervalCounter):
            raise NotImplementedError
        else:
            export_at = [save_at[1]]
            save_at = diacore.IntervalCounter(((save_at[0], save_at[1]),))

        super().__init__(sim_info, block_size, export_at, save_at, export_func, keep_only_last_export, export_kwargs, preprocess)


class SignalPowerSpectrum(SummaryDiagnostic):
    def __init__(self, 
        sig_name,
        sim_info, 
        block_size, 
        save_range,
        num_channels,
        nperseg, 
        sig_channels=slice(None),
        **kwargs
        ):
        """
        Will output the wrong value if it is exported in the middle of the save_range
        """
        self.save_range = save_range
        self.num_samples = save_range[1] - save_range[0]
        #save_at = diacore.IntervalCounter((save_range,))
        super().__init__(sim_info, block_size, save_at=save_range, export_func = "spectrum", **kwargs)
        self.sig_name = sig_name

        self.samplerate = self.sim_info.samplerate
        self.num_channels = num_channels
        self.nperseg = nperseg
        self.sig_channels = sig_channels
        self.signal = np.full((self.num_channels, self.num_samples), fill_value=np.nan)

        self.sample_counter = 0

        

        #self.plot_data["title"] = f"Power of {self.sig_name}. Samples: {self.save_range}"
        
    def save(self, processor, sig, chunkInterval, globInterval):
        num_samples = chunkInterval[1] - chunkInterval[0]
        self.signal[:,self.sample_counter:self.sample_counter+num_samples] = processor.sig[self.sig_name][self.sig_channels, chunkInterval[0]:chunkInterval[1]]

        self.sample_counter += num_samples
        #self.power_ratio[globInterval[0]:globInterval[1]] = num / denom

    def get_output(self):
        f, spec = spsig.welch(self.signal, self.samplerate, nperseg=self.nperseg, scaling="spectrum", axis=-1)
        spec = np.mean(spec, axis=0)
        return spec



class SignalSummary(SummaryDiagnostic):
    def __init__(self, 
        sig_name,
        sim_info, 
        block_size, 
        save_range,
        summary_func = None,
        **kwargs
        ):
        self.save_range = save_range
        self.num_samples = save_range[1] - save_range[0]
        #save_at = diacore.IntervalCounter((save_range,))
        super().__init__(sim_info, block_size, save_at=save_range, export_func = "text", **kwargs)
        self.sig_name = sig_name
        self.summary_func = summary_func

        self.mean = 0

        #self.plot_data["title"] = f"Power of {self.sig_name}. Samples: {self.save_range}"
        
    def save(self, processor, sig, chunkInterval, globInterval):
        if self.summary_func is None:
            self.mean += np.sum(processor.sig[self.sig_name][:, chunkInterval[0]:chunkInterval[1]]) / self.num_samples
        else:
            self.mean += self.summary_func(processor.sig[self.sig_name][:, chunkInterval[0]:chunkInterval[1]]) / self.num_samples

    def get_output(self):
        return self.mean
    
# class SignalComparisonSummary(SummaryDiagnostic):
#     def __init__(self, 
#         sig_name,
#         sig_name2,
#         sim_info, 
#         block_size, 
#         save_range,
#         comparison_func,
#         **kwargs
#         ):
#         self.save_range = save_range
#         self.num_samples = save_range[1] - save_range[0]
#         #save_at = diacore.IntervalCounter((save_range,))
#         super().__init__(sim_info, block_size, save_at=save_range, export_func = "text", **kwargs)
#         self.sig_name = sig_name
#         self.sig_name2 = sig_name2
#         self.comparison_func = comparison_func

#         self.mean = 0

#         #self.plot_data["title"] = f"Power of {self.sig_name}. Samples: {self.save_range}"
        
#     def save(self, processor, chunkInterval, globInterval):
#         sig1 = processor.sig[self.sig_name][:, chunkInterval[0]:chunkInterval[1]]
#         sig2 = processor.sig[self.sig_name2][:, chunkInterval[0]:chunkInterval[1]]
#         self.mean += self.comparison_func() / self.num_samples

#     def get_output(self):
#         return self.mean
